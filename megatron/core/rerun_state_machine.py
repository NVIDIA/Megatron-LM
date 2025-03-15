# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import datetime
import inspect
import logging
import math
import os
import random
import re
from collections import defaultdict
from enum import Enum
from typing import Any, Callable, Iterable, NamedTuple, Optional, Set, Tuple, Union

import numpy as np
import torch

import megatron.core.parallel_state as mpu
from megatron.core.dist_checkpointing.mapping import ShardedObject

"""DISCLAIMER: THIS IS AN EXPERIMENTAL FEATURE.

The rerun state machine implementation in this file is alpha-level code to help
with attribution of unexpected results (e.g. NaN, spiky loss, etc.). This code
has not been tested at scale so should not be assumed to be accurate. Nodes
flagged by this code as potentially faulty should be subjected to standard
diagnostic test suites for a definitive diagnosis.

Also note that experimental features may break existing APIs.
"""

logger = logging.getLogger(__name__)

_GLOBAL_RERUN_STATE_MACHINE: Optional["RerunStateMachine"] = None

# Exit code returned when job needs to be restarted to disambiguate the results.
EXIT_CODE_RESUME_TO_DISAMBIGUATE: int = 16

# Exit code returned when job failed on result validation.
EXIT_CODE_FAILED_ON_RESULT_VALIDATION: int = 17

SerializableStateType = Union[list, dict]
DataIteratorArgType = Optional[Union["RerunDataIterator", list["RerunDataIterator"]]]


class Caller(NamedTuple):
    """Class capturing the code and rank calling a function."""

    filename: str
    lineno: int
    rank: int


class Call(NamedTuple):
    """Class capturing a function call."""

    caller: Caller
    sequence: int


class RerunDiagnostic(str, Enum):
    """Enum representing the different diagnostic attributions.

    CORRECT_RESULT: the result was the expected result given the input.
    TRANSIENT_ERROR: the result could not be reproduced on the same GPU.
    PERSISTENT_ERROR: the result could be reproduced on the same GPU, but
        not on a different GPU.
    """

    CORRECT_RESULT = 'correct_result'
    TRANSIENT_ERROR = 'transient_error'
    PERSISTENT_ERROR = 'persistent_error'


class RerunMode(str, Enum):
    """Enum representing the different run mode for the rerun state machine."""

    DISABLED = 'disabled'
    VALIDATE_RESULTS = 'validate_results'
    REPORT_DETERMINISM_STATS = 'report_determinism_stats'


class RerunState(Enum):
    """Enum representing the different states of the rerun state machine.

    Description of states (would benefit from a diagram):
    - NOT_RUNNING_YET
        State before the should_rerun_forward_and_backward while loop has been entered (and
        not restarting from a checkpoint for a 2nd re-run), and after it has been successfully
        completed (all validation succeeded).
    - INITIAL_RUN
        State during the initial run of the should_rerun_forward_and_backward while loop.
    - RERUNNING_IN_PLACE
        State during the second run of the should_rerun_forward_and_backward (1+ validation has
        failed).
    - WILL_RERUN_FROM_CHECKPOINT
        State after the should_rerun_forward_and_backward while loop has exited (on initial job run)
        and before the while loop has been entered (on the second job run restarted from the
        checkpoint) when the 1st re-run yielded the same result than on the initial run.
    - RERUNNING_FROM_CHECKPOINT
        State during first (and only) run of the should_rerun_forward_and_backward while loop when
        the job was restarted from a checkpoint.
    - RERUNNING_AGAIN_FROM_CHECKPOINT
        State when the re-run from checkpoint was rescheduled on the same potentially faulty GPU.
    """

    NOT_RUNNING_YET = 0
    INITIAL_RUN = 1
    RERUNNING_IN_PLACE = 2
    WILL_RERUN_FROM_CHECKPOINT = 3
    RERUNNING_FROM_CHECKPOINT = 4
    RERUNNING_AGAIN_FROM_CHECKPOINT = 5


class RerunValidationStatus(str, Enum):
    """Enum representing the status of a record in the tracker log file"""

    RERUN_DISABLED = 'rerun_disabled'
    INITIAL_RUN = 'initial_run'
    FIRST_RERUN_NOT_REPRODUCIBLE = 'first_rerun_not_reproducible'
    FIRST_RERUN_REPRODUCIBLE = "first_rerun_reproducible"
    SECOND_RERUN_NOT_REPRODUCIBLE = "second_rerun_not_reproducible"
    SECOND_RERUN_REPRODUCIBLE = "second_rerun_reproducible"


COMPARISON_MATCH: float = 0.0
COMPARISON_MISMATCH: float = math.inf


class RerunStateMachine:
    """Class implementing the re-run state machine used to validate calculations.

    This class is a singleton and should not be instantiated directly. The instance
    should be initialized by calling the initialize_rerun_state_machine() helper function instead.

    Args:
        state_save_func: optional function to save any additional state that needs
                    to be restore to rerun the iteration.
        state_restore_func: optional function to restore the state saved by state_save_func.
        mode: operating mode for the rerun state machine, default is disabled.
        error_injector: optional result injection engine, default is no result injection.
        result_rejected_tracker_filename: optional name of file tracking `result rejected` events.

    Example usage:

        def state_save_func():
            # save any custom state that may change during the
            # forward-backward pass and that needs to be saved/restored
            # when re-running the iteration (Python/NumPy/Pytorch/CUDA
            # RNG states already taken care of)
            return {
                'mystate': get_state(...)
            }

        def state_restore_func(state_dict):
            restore_state(state_dict['mystate'])

        initialize_rerun_state_machine(
            state_save_func=state_save_func,
            state_restore_func=state_restore_func,
            error_injector=RerunErrorInjector(
                error_injection_rate=100000,
                error_injection_type=RerunDiagnostic.TRANSIENT_ERROR,
            ),
        )

    To use the rerun state machine, the training code needs to be modified as described in the
    documentation for each of the public methods.

    Caveats and assumptions:
    1) A core assumption of the rerun state machine is that execution (flow control) of the
    iteration is deterministic w.r.t. the state captured by the rerun state (_save_state() and
    _restore_state() methods below). More specifically, the requirement is that a re-run of the
    iteration yields the same calls to validate_results() as in the initial run.
    On the other hand, computations are NOT required to be deterministic, i.e. results may vary
    slightly across re-runs of the iteration.

    2) The re-run logic is currently only able to re-run the current step. It may be that an
    unexpected result (e.g. spiky loss) is the result of a calculation that happened at a previous
    iteration. The current implementation will not catch such issues. We're planning to add the
    capability to re-run multiple steps in a future implementation.
    """

    REPORTING_INTERVAL_ITERATIONS: int = 2

    def __init__(
        self,
        state_save_func: Optional[Callable[[], SerializableStateType]] = None,
        state_restore_func: Optional[Callable[[SerializableStateType], None]] = None,
        mode: RerunMode = RerunMode.DISABLED,
        error_injector: Optional["RerunErrorInjector"] = None,
        result_rejected_tracker_filename: Optional[str] = None,
    ) -> None:
        self.mode: RerunMode = mode
        self.state: RerunState = RerunState.NOT_RUNNING_YET
        self.current_iteration: int = -1
        # The flags below are per-rank flags that get all-reduced across all ranks
        # request to rerun iteration  because validation failed (1st re-run).
        self.rerun_requested: bool = False
        # Request to checkpoint to re-run iteration on different GPU (2nd re-run).
        self.checkpoint_requested: bool = False
        # Request to restart job again from checkpoint because got the same GPU (3rd+ re-run).
        self.restart_again_requested: bool = False
        # Request to resume normal execution when no HW fault was detected.
        self.continue_requested: bool = False
        self.logged_sdc_enabled: bool = False

        self.error_injector: RerunErrorInjector = error_injector or RerunErrorInjector()
        self.validation_counts: dict[Caller, int] = defaultdict(int)
        self.failed_validation_call: Optional[Call] = None
        self.initial_result: Any = None
        self.suspicious_node: str = None
        self.suspicious_device: int = None

        # Keep track of `result_rejected` events.
        # Make sure the file can be written to and abort if not.
        self.result_rejected_tracker_filename = result_rejected_tracker_filename
        if self.result_rejected_tracker_filename is not None:
            try:
                with open(self.result_rejected_tracker_filename, 'a'):
                    pass
            except Exception as e:
                raise RuntimeError(
                    f"RerunStateMachine result validation log cannot be appended to! ({e})"
                )

        self.saved_state: Optional[SerializableStateType] = None
        self.state_save_func: Optional[Callable[[], SerializableStateType]] = state_save_func
        self.state_restore_func: Optional[Callable[[SerializableStateType], None]] = (
            state_restore_func
        )
        self.data_iterator_checkpoints: Optional[list[SerializableStateType]] = None

        self.large_value_counts: dict[str, int] = {}
        self.max_values: dict[str, float] = {}

        self.saved_results: dict[Call, Any] = {}
        self.stats: dict[Caller, QuickStats] = defaultdict(lambda: QuickStats())
        if _safe_get_rank() == 0:
            logger.warning(f"RerunStateMachine initialized in mode {mode}")

    def set_mode(self, mode: RerunMode) -> None:
        """Method to set the operating mode"""

        if _safe_get_rank() == 0:
            logger.warning(f"Setting RerunStateMachine mode {mode}")
        self.mode = mode

    def get_mode(self) -> RerunMode:
        """Method to get the operating mode"""

        return self.mode

    def should_run_forward_backward(self, data_iterator: DataIteratorArgType) -> bool:
        """Method instructing whether to (re)run the forward-backward pass.

        Args:
            data_iterator: data iterator or list of data iterators used in this step,
                or None if no data iterator
        Returns:
            A boolean telling whether the forward-backward pass should be (re)run.

        Example usage:

            def train_step(data_iterator, ...):
                rerun_state_machine = get_rerun_state_machine()
                while rerun_state_machine.should_rerun_forward_and_backward(data_iterator):
                    optimizer.zero_grad()
                    data = next(data)
                    outputs = model(data)
                    loss = loss_fn(outputs)
                    loss.backward()
                ...
                optimizer.step()
        """

        self.validation_counts = defaultdict(int)

        data_iterators: list[RerunDataIterator] = self._sanitize_data_iterators(data_iterator)

        # Are we about to start the initial run?
        if self.state == RerunState.NOT_RUNNING_YET:
            if self.mode == RerunMode.DISABLED:
                self.state = RerunState.INITIAL_RUN
                self.current_iteration += 1  # Increment self.current_iteration for reporting.
                return True
            if self.data_iterator_checkpoints is not None:
                assert len(self.data_iterator_checkpoints) == len(
                    data_iterators
                ), "data iterator has different length than checkpointed data iterator"
                for i, d in enumerate(data_iterators):
                    d.load_state_dict(self.data_iterator_checkpoints[i])
                self.data_iterator_checkpoints = None
            self._save_state()
            if data_iterators:
                for d in data_iterators:
                    d.advance()
            self.rerun_requested = False
            self.checkpoint_requested = False
            self.restart_again_requested = False
            self.continue_requested = False
            self.injected_result = None
            self.current_iteration += 1
            self.state = RerunState.INITIAL_RUN
            return True
        # Are we done with the initial run?
        elif self.state == RerunState.INITIAL_RUN:
            if self.mode == RerunMode.DISABLED:
                self.state = RerunState.NOT_RUNNING_YET
                return False
            will_rerun_tensor: torch.Tensor = torch.tensor(
                [self.rerun_requested], dtype=torch.int32, device='cuda'
            )
            torch.distributed.all_reduce(will_rerun_tensor)
            if will_rerun_tensor.item() == 0:
                self.state = RerunState.NOT_RUNNING_YET
                return False
            if self.mode == RerunMode.VALIDATE_RESULTS and _safe_get_rank() == 0:
                logger.warning("Need to rerun step to check reproducibility of initial result")
            self.state = RerunState.RERUNNING_IN_PLACE
            self._restore_state()
            if data_iterators:
                for d in data_iterators:
                    d.rewind()
            return True
        # Are we done with the 1st re-run?
        elif self.state == RerunState.RERUNNING_IN_PLACE:
            # If we are reporting stats rather than validating results, we just continue with
            # normal execution after re-running the step once to compare results.
            if self.mode == RerunMode.REPORT_DETERMINISM_STATS:
                self.state = RerunState.NOT_RUNNING_YET
                self._maybe_report_stats()
                self.saved_results = defaultdict(list)
                return False
            will_checkpoint_tensor: torch.Tensor = torch.tensor(
                [self.checkpoint_requested], dtype=torch.int32, device='cuda'
            )
            torch.distributed.all_reduce(will_checkpoint_tensor)
            if will_checkpoint_tensor.item() > 0:
                self.state = RerunState.WILL_RERUN_FROM_CHECKPOINT
            self._restore_state()
            if data_iterators:
                for d in data_iterators:
                    d.rewind()
            return False
        # Are we about to re-run from a checkpoint?
        elif self.state == RerunState.WILL_RERUN_FROM_CHECKPOINT:
            self.state = RerunState.RERUNNING_FROM_CHECKPOINT
            return True
        # Are we done re-running from a checkpoint?
        elif self.state == RerunState.RERUNNING_FROM_CHECKPOINT:
            will_restart_again_tensor: torch.Tensor = torch.tensor(
                [self.restart_again_requested], dtype=torch.int32, device='cuda'
            )
            torch.distributed.all_reduce(will_restart_again_tensor)
            if will_restart_again_tensor.item() > 0:
                if _safe_get_rank() == 0:
                    logger.warning(
                        "Need to restart job from the same checkpoint "
                        "because it was scheduled on the same node/GPU"
                    )
                self.state = RerunState.RERUNNING_AGAIN_FROM_CHECKPOINT
            else:
                will_continue_tensor: torch.Tensor = torch.tensor(
                    [self.continue_requested], dtype=torch.int32, device='cuda'
                )
                torch.distributed.all_reduce(will_continue_tensor)
                if will_continue_tensor.item() > 0:
                    if _safe_get_rank() == 0:
                        logger.warning(
                            "Continuing normal execution because failed validation was not fatal"
                        )
                    self.state = RerunState.NOT_RUNNING_YET
            return False
        raise RuntimeError("Should not be here")

    def should_checkpoint_and_exit(self) -> Tuple[bool, bool, int]:
        """Method instructing whether to checkpoint and/or abort the job.

        Args:
            None
        Returns:
            A tuple formed of:
            - a boolean telling whether a checkpoint should be taken.
            - a boolean telling whether the job should be aborted.
            - an exit code (int) to return if aborting (0 if not aborting).

        Example usage:

            def train_step(data_iterator, ...):
                rerun_state_machine = get_rerun_state_machine()
                while rerun_state_machine.should_rerun_forward_and_backward(data_iterator):
                    ...
                should_checkpoint, should_exit, exit_code = (
                    rerun_state_machine.should_checkpoint_and_exit()
                )
                if should_checkpoint:
                    save_checkpoint()
                if should_exit:
                    sys.exit(exit_code)
                optimizer.step()
        """

        if self.mode in [RerunMode.DISABLED, RerunMode.REPORT_DETERMINISM_STATS]:
            return False, False, 0
        if self.state == RerunState.RERUNNING_IN_PLACE:
            if _safe_get_rank() == 0:
                logger.warning(
                    "Exiting now. A checkpoint at the last iteration is being saved "
                    "if further examination is needed"
                )
            return True, True, EXIT_CODE_FAILED_ON_RESULT_VALIDATION
        elif self.state == RerunState.WILL_RERUN_FROM_CHECKPOINT:
            if _safe_get_rank() == 0:
                logger.warning(
                    "Saving a checkpoint and exiting now. Please resume the job "
                    "from the checkpoint to rerun the last iteration "
                    "and establish a diagnostic"
                )
            return True, True, EXIT_CODE_RESUME_TO_DISAMBIGUATE
        elif self.state == RerunState.RERUNNING_FROM_CHECKPOINT:
            if _safe_get_rank() == 0:
                logger.warning(
                    "Exiting now. A checkpoint at the last iteration already exists "
                    "if further examination is needed"
                )
            return False, True, EXIT_CODE_FAILED_ON_RESULT_VALIDATION
        elif self.state == RerunState.RERUNNING_AGAIN_FROM_CHECKPOINT:
            if _safe_get_rank() == 0:
                logger.warning(
                    "Exiting now. Please resume the job from the same checkpoint "
                    "to rerun the last iteration and establish a diagnostic"
                )
            return False, True, EXIT_CODE_RESUME_TO_DISAMBIGUATE
        return False, False, 0

    def validate_result(
        self,
        result: Any,
        rejection_func: Callable[[Any], bool],
        message: str = "unexpected result",
        comparison_func: Optional[Callable[[Any, Any], float]] = None,
        tolerance: float = 0.0,
        fatal: bool = True,
    ) -> None:
        """This method verifies a result and possibly triggers a re-run.

        Args:
            result: result to verify.
            rejection_func: function taking a result as input and returning whether the result fails
                validation (e.g. torch.isnan, returns True if result is NaN).
            message: message describing the validation test (e.g. "spiky loss").
            comparison_func: optional function used to compare the results of the original run and
                of a rerun. It should return a float representing the relative difference between
                the 2. The default implementation is for 0-dim float tensors.
            tolerance: tolerance used in combination with comparison_func to determine
                reproducibility of results. Default is no tolerance (deterministic calculations).
            fatal: whether to abort the job when no HW fault was identified (unexpected result is
                reproducible and correct).
        Returns:
            None

        Example usage:

            def train_step(data_iterator, ...):
                rerun_state_machine = get_rerun_state_machine()
                while rerun_state_machine.should_rerun_forward_and_backward(data_iterator):
                    optimizer.zero_grad()
                    data = next(data)
                    outputs = model(data)
                    loss = loss_fn(outputs)
                    rerun_state_machine.validate_result(
                        result=loss,
                        rejection_func=torch.is_nan,    # rejects result if NaN
                        message="loss is NaN",
                        tolerance=0.001,    # max 0.1% difference in results due to non-determinism
                        fatal=True,         # abort job if validation fails
                    )
                    loss.backward()

        We establish the diagnostic using this overall flow:
        - an irreproducible result is detected by rerunning the iteration locally (same GPU) and
          verifying the result is different.
        - a mismatching result is detected by rerunning the iteration on a different GPU by
          verifying the result is different.
        - an expected result is detected by rerunning the iteration on a different GPU and
          verifying the result is the same.
        """

        # If reruns are disabled, still validate the result and throw a RuntimeError if it is
        # rejected. This is a backward-compatible behavior.
        if self.mode == RerunMode.DISABLED:
            result_rejected: bool = rejection_func(result)
            if result_rejected:
                self._log_validation_error_to_file(
                    status=RerunValidationStatus.RERUN_DISABLED, result=result, message=message
                )
                rank: int = _safe_get_rank()
                node: str = os.uname()[1]
                device: int = torch.cuda.current_device()
                full_message: str = (
                    f"Rank {rank}, node {node}, device {device}, "
                    f"iteration {self.current_iteration}: "
                    f"Unexpected result {result} (message='{message}')"
                )
                raise RuntimeError(full_message)
            return

        # Skip the validation on the first iteration, as we cannot guarantee a checkpoint can be
        # taken before the optimizer has been stepped at least once.
        if self.current_iteration < 1:
            return

        if comparison_func is None:
            comparison_func = _compare_floats

        assert (
            self.state != RerunState.NOT_RUNNING_YET
        ), "validate_result should not be called outside of the forward-backward pass"

        validation_call: Call = self._get_validation_call_info()

        # Handle the stats reporting mode. In that mode, we rerun every iteration once to collect
        # stats about any non-determinism in the calculations (as a relative difference between the
        # calculations in the initial run and in the re-run). The only assumption here is that the
        # control flow is deterministic (so that the results corresponding to the nth invokation of
        # validate_result() can be compared).

        if self.mode == RerunMode.REPORT_DETERMINISM_STATS:
            if self.state == RerunState.INITIAL_RUN:
                self.rerun_requested = True
                self.saved_results[validation_call] = result
            elif self.state == RerunState.RERUNNING_IN_PLACE:
                initial_result = self.saved_results.get(validation_call)
                assert initial_result is not None, "Result from initial run missing"
                diff = comparison_func(initial_result, result)
                caller: Caller = Caller(
                    filename=validation_call.caller.filename,
                    lineno=validation_call.caller.lineno,
                    rank=0,
                )
                self.stats[caller].record(diff)
            return

        def log_failure(message: str) -> None:
            rank: int = _safe_get_rank()
            node: str = os.uname()[1]
            device: int = torch.cuda.current_device()
            logger.error(f"Rank {rank}, node {node}, device {device}: {message}!")

        # Emit message in log so that we can identify which jobs have this instrumentation
        # enabled. We do this from the validate_result() method because some jobs may run with
        # the check_for_nan_in_loss_and_grad option but never call validate_result.
        if not self.logged_sdc_enabled:
            self.logged_sdc_enabled = True
            if _safe_get_rank() == 0:
                logger.warning("Result validation enabled")

        # If this the initial run of the iteration, and no unexpected result has already been
        # identified?
        if self.state == RerunState.INITIAL_RUN and not self.rerun_requested:
            result_rejected: bool = self.error_injector.maybe_inject() or rejection_func(result)
            if result_rejected:
                self.failed_validation_call = validation_call
                self.initial_result = result
                self.rerun_requested = True
                self._log_validation_error_to_file(
                    status=RerunValidationStatus.INITIAL_RUN, result=result, message=message
                )
                logger.error(
                    f"Unexpected result {result} at {validation_call.caller.filename} "
                    f"line {validation_call.caller.lineno}, "
                    f"invokation #{validation_call.sequence} "
                    f"at iteration #{self.current_iteration} "
                    f"(message='{message}')"
                )
        # If this the first rerun (same GPU) or second 2nd rerun (different GPU), and have we
        # reached the validation call that failed during the initial run?
        elif (
            self.state in [RerunState.RERUNNING_IN_PLACE, RerunState.RERUNNING_FROM_CHECKPOINT]
            and validation_call == self.failed_validation_call
        ):

            comparison: float = self.error_injector.maybe_miscompare(
                comparison_func, self.initial_result, result, self.state
            )
            # This is the first re-run.
            if self.state == RerunState.RERUNNING_IN_PLACE:
                if comparison > tolerance:
                    logger.warning(
                        "First rerun: unexpected result is not reproducible within the tolerance "
                        f"({result} != {self.initial_result})"
                    )
                    self._log_validation_error_to_file(
                        status=RerunValidationStatus.FIRST_RERUN_NOT_REPRODUCIBLE,
                        result=result,
                        message=message,
                    )
                    log_failure("Possible transient error!")
                else:
                    self.checkpoint_requested = True
                    # Remember the node and device we're running on so that we can check we're not
                    # rerunning on the same GPU when we resume from the checkpoint.
                    self.suspicious_node = os.uname()[1]
                    self.suspicious_device = torch.cuda.current_device()
                    self._log_validation_error_to_file(
                        status=RerunValidationStatus.FIRST_RERUN_REPRODUCIBLE,
                        result=result,
                        message=message,
                    )
                    logger.warning(
                        "First rerun: unexpected result is reproducible within the tolerance "
                        f"({result} = {self.initial_result}). "
                        "Need to rerun on a different GPU to verify correctness"
                    )
            # This is the second re-run.
            elif self.state == RerunState.RERUNNING_FROM_CHECKPOINT:
                # Ensure we're not on the same GPU as the first rerun.
                node: str = os.uname()[1]
                device: int = torch.cuda.current_device()
                if node == self.suspicious_node and device == self.suspicious_device:
                    logger.error(
                        f"Got rescheduled on the same GPU. Need to resume again from the same "
                        f"checkpoint (node: {self.suspicious_node}, gpu: {self.suspicious_device})"
                    )
                    self.restart_again_requested = True
                elif comparison > tolerance:
                    self._log_validation_error_to_file(
                        status=RerunValidationStatus.SECOND_RERUN_NOT_REPRODUCIBLE,
                        result=result,
                        message=message,
                    )
                    logger.warning(
                        "Second rerun: unexpected result is not reproducible on a different GPU, "
                        f"therefore was likely incorrect ({result} != {self.initial_result})"
                    )
                    log_failure("Possible persistent error!")
                else:
                    self._log_validation_error_to_file(
                        status=RerunValidationStatus.SECOND_RERUN_REPRODUCIBLE,
                        result=result,
                        message=message,
                    )
                    logger.warning(
                        "Second rerun: unexpected result is reproducible on a different GPU, "
                        f"therefore it was likely correct ({result} = {self.initial_result})"
                    )
                    log_failure(f"Correct result (but possible Application error) ({message})")
                    if not fatal:
                        self.continue_requested = True
            else:
                raise RuntimeError("Should not be here")

    def is_unexpectedly_large(
        self,
        result: torch.Tensor,
        threshold: float,
        context: str,
        num_samples: int = 100,
        resample: bool = False,
    ) -> bool:
        """Helper method to estimate whether a result is unexpectedly large.

        Some calculation errors manifest themselves as results with unexpectedly large
        exponents, e.g. spiky loss or grads. This method keeps track of a value over time
        and flags it if it exceeds a certain threshold expressed as a multiple factor of
        the max value observed.

        Args:
            loss_tensor: a zero-dim tensor containing the current loss.
            threshold: a float representing the minimum trigger threshold
                e.g. 10 means > 10x max absolute value observed.
            context: a string identifying the value. This is used to differentiate
                between different invokations of validate_results targetting different
                values, e.g. loss and grads.
            num_samples: the sample size used to estimate the max value.
                Default is 100 value samples.
            reset: whether to resample the max value. Default is False.
        Returns:
            A boolean telling whether the current loss deviates from the previous
            loss by a factor greater than the threshold

        This method can be passed as a rejection function to the validate_result()
        method.

        Example usage:

            def train_step(data_iterator, ...):
                rerun_machine = get_rerun_machine()
                while rerun_machine.should_rerun_forward_and_backward(data_iterator):
                    optimizer.zero_grad()
                    data = next(data)
                    outputs = model(data)
                    loss = loss_fn(outputs)
                    rerun_machine.validate_result(
                        result=loss,
                        rejection_func=partial(
                            rerun_machine.is_unexpectedly_large,
                            threshold=10,
                            context="loss",
                        ),
                        message="Spiky loss",
                        tolerance=0.0,
                        fatal=False,
                    )
        """

        value: float = math.fabs(result.item())
        # Ignore NaNs and Infs. They should be checked separately.
        if math.isnan(value) or math.isinf(value):
            return False

        if resample or context not in self.large_value_counts:
            self.large_value_counts[context] = 0
        if self.large_value_counts[context] < num_samples:
            self.large_value_counts[context] += 1
            self.max_values[context] = max(self.max_values.get(context, 0.0), value)
            if self.large_value_counts[context] == num_samples:
                logger.warning(f"Max value for {context}: {self.max_values[context]}")
            return False

        return value >= self.max_values[context] * threshold

    def state_dict(self, data_iterator: DataIteratorArgType, ckpt_format: str) -> dict[str, Any]:
        """Method that returns a state dict to be checkpointed.

        Args:
            data_iterator: the data iterator that needs to be checkpointed (or None
                if this checkpoint is not requested by the rerun state machine).
            ckpt_format: the checkpoint format to use.
        Returns:
            A state dict representing the rerun state machine.

        Example usage:

            def save_my_model_checkpoint(data_iterator, ...):
                checkpoint = {}
                ...
                rerun_state_machine = get_rerun_state_machine()
                checkpoint['rerun_state_machine'] = (
                    rerun_state_machine.state_dict(data_iterator, "torch_dist")
                )
                ...
                return checkpoint
        """

        data_iterators: list[RerunDataIterator] = self._sanitize_data_iterators(data_iterator)

        # The RerunStateMachine state is different across all ranks. Therefore it needs to be
        # checkpointed using a ShardedObject. However, we keep the common state in the non-sharded
        # (common) checkpoint. This allows us to verify whether a checkpoint contains a
        # RerunStateMachine state by checking the common checkpoint.
        state_dict: dict[str, Any] = {
            'mode': self.mode,
            'sharded': {
                'state': self.state,
                'current_iteration': self.current_iteration,
                'rerun_requested': self.rerun_requested,
                'checkpoint_requested': self.checkpoint_requested,
                'restart_again_requested': self.restart_again_requested,
                'continue_requested': self.continue_requested,
                # logged_sdc_enabled should not be saved (set at the job startup time).
                'error_injector_checkpoint': self.error_injector.state_dict(),
                # validation_counts should not be saved (reset at start of training loop).
                'failed_validation_call': self.failed_validation_call,
                'initial_result': self.initial_result,
                'suspicious_node': self.suspicious_node,
                'suspicious_device': self.suspicious_device,
                # No need to save saved_state (RNG state  already captured in checkpoint).
                'data_iterator_checkpoints': (
                    [d.state_dict() for d in data_iterators] if data_iterators else None
                ),
                'large_value_counts': self.large_value_counts,
                'max_values': self.max_values,
                # No need to save saved_results and stats (resets when job resumes).
            },
        }
        if ckpt_format == "torch_dist":
            pp_rank = mpu.get_pipeline_model_parallel_rank()
            pp_size = mpu.get_pipeline_model_parallel_world_size()
            tp_rank = mpu.get_tensor_model_parallel_rank()
            tp_size = mpu.get_tensor_model_parallel_world_size()
            state_dict['sharded'] = ShardedObject(
                'rerun_state_machine_state',
                state_dict['sharded'],
                (pp_size, tp_size),
                (pp_rank, tp_rank),
                replica_id=mpu.get_data_parallel_rank(with_context_parallel=True),
            )
        return state_dict

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Method that restores the state from a checkpoint.

        Args:
            state_dict: the state dict saved in the checkpoint and originally
                obtained from state_dict().
        Returns:
            None

        Example usage:

            def load_checkpoint(checkpoint, ...)
                ...
                if 'rerun_state_machine' in checkpoint:
                    rerun_state_machine = get_rerun_state_machine()
                    rerun_state_machine.load_state_dict(checkpoint['rerun_state_machine'])
        """

        if self.mode == RerunMode.DISABLED:
            if _safe_get_rank() == 0:
                logger.warning(
                    "RerunStateMachine disabled via CLI, ignoring machine state saved in checkpoint"
                )
            return
        if state_dict['mode'] == RerunMode.DISABLED:
            if _safe_get_rank() == 0:
                logger.warning(
                    "RerunStateMachine disabled in checkpoint but enabled via CLI, "
                    "ignoring machine state saved in checkpoint"
                )
            return
        if _safe_get_rank() == 0:
            logger.warning(
                "Getting RerunStateMachine state from checkpoint, CLI rerun args ignored"
            )
        self.mode = state_dict['mode']
        sharded_dict = state_dict['sharded']
        self.state = sharded_dict['state']
        self.current_iteration = sharded_dict['current_iteration']
        self.rerun_requested = sharded_dict['rerun_requested']
        self.checkpoint_requested = sharded_dict['checkpoint_requested']
        self.restart_again_requested = sharded_dict['restart_again_requested']
        self.continue_requested = sharded_dict['continue_requested']
        self.error_injector.load_state_dict(sharded_dict['error_injector_checkpoint'])
        self.failed_validation_call = sharded_dict['failed_validation_call']
        self.initial_result = sharded_dict['initial_result']
        self.suspicious_node = sharded_dict['suspicious_node']
        self.suspicious_device = sharded_dict['suspicious_device']
        self.data_iterator_checkpoints = sharded_dict['data_iterator_checkpoints']
        self.large_value_counts = sharded_dict['large_value_counts']
        self.max_values = sharded_dict['max_values']

    def _sanitize_data_iterators(
        self, data_iterator: DataIteratorArgType
    ) -> list["RerunDataIterator"]:
        data_iterators: list[RerunDataIterator]
        if self.mode == RerunMode.DISABLED:
            data_iterators = []
        elif not isinstance(data_iterator, list):
            data_iterators = [data_iterator]
        else:
            data_iterators = data_iterator
        data_iterators = [d for d in data_iterators if d is not None]
        for d in data_iterators:
            assert isinstance(
                d, RerunDataIterator
            ), "data iterator is not wrapped with RerunDataIterator"
        return data_iterators

    def _get_validation_call_info(self) -> Call:
        """Internal method to get the context about the caller to validate_result()."""

        frame: inspect.frame = inspect.currentframe()
        frame = frame.f_back.f_back
        filename: str = inspect.getframeinfo(frame).filename
        lineno: int = frame.f_lineno
        rank: int = _safe_get_rank()
        caller = Caller(filename=filename, lineno=lineno, rank=rank)
        self.validation_counts[caller] += 1
        sequence: int = self.validation_counts[caller]
        return Call(caller=caller, sequence=sequence)

    def _save_state(self) -> None:
        """Internal method that saves the state that needs to be restored when rewound.

        Any state that may change during the execution of a step before the optimizer is updated,
        e.g. RNG state, should be saved here. The state of the data iterator is taken care
        separately by the RerunDataIterator class.

        At this point, this only consists in the RNG state.
        """

        self.saved_state = {
            'rng_state': {
                'random_rng_state': random.getstate(),
                'np_rng_state': np.random.get_state(),
                'torch_rng_state': torch.get_rng_state(),
                'cuda_rng_state': torch.cuda.get_rng_state(),
            },
            'other_state': self.state_save_func() if self.state_save_func else None,
            # any other state to save to guarantee deterministic execution?
        }

    def _restore_state(self) -> None:
        """Internal method that restores the state that was saved in _save_state()."""

        rng_state = self.saved_state['rng_state']
        random.setstate(rng_state['random_rng_state'])
        np.random.set_state(rng_state['np_rng_state'])
        torch.set_rng_state(rng_state['torch_rng_state'])
        torch.cuda.set_rng_state(rng_state['cuda_rng_state'])
        if self.saved_state['other_state'] and self.state_restore_func:
            self.state_restore_func(self.saved_state['other_state'])

    def _maybe_report_stats(self) -> None:
        """Internal method that reports stats if needed."""

        if self.current_iteration % RerunStateMachine.REPORTING_INTERVAL_ITERATIONS == 0:
            if torch.distributed.is_initialized():
                world_size: int = torch.distributed.get_world_size()
                stats_list = [None for _ in range(world_size)]
                rank = torch.distributed.get_rank()
                torch.distributed.gather_object(dict(self.stats), stats_list if rank == 0 else None)
                if rank == 0:
                    callers: Set[Caller] = {c for s in stats_list for c in s.keys()}
                    logger.info("Stats on computation determinism in validation calls")
                    for caller in callers:
                        self.stats[caller].combine(
                            [s.get(caller) for s in stats_list[1:] if s.get(caller)]
                        )
                        logger.info(f"  From {caller.filename}, line {caller.lineno}:")
                        logger.info(f"    {self.stats[caller].print_stats()}")
                else:
                    for caller, stats in self.stats.items():
                        stats.reset()
            else:
                logger.info("Stats on computation determinism in validation calls")
                for caller, stats in self.stats.items():
                    logger.info(f"  From {caller.filename}, line {caller.lineno}:")
                    logger.info(f"    {stats.print_stats()}")

    def _log_validation_error_to_file(
        self, status: RerunValidationStatus, result: Any, message: str
    ) -> None:
        if self.result_rejected_tracker_filename is not None:
            # Append to log.
            try:
                rank: int = _safe_get_rank()
                node: str = os.uname()[1]
                device: int = torch.cuda.current_device()
                with open(self.result_rejected_tracker_filename, 'a') as f:
                    print(
                        f"ts={datetime.datetime.now()} node={node} device={device} "
                        f"jobID={os.getenv('SLURM_JOBID', 'N/A')} rank={rank} "
                        f"iteration={self.current_iteration} status={status} result={result} "
                        f"message='{message}'",
                        file=f,
                    )
            except Exception as e:
                logger.error(f"Could not log validation error! ({e})")

    @classmethod
    def get_skipped_iterations_from_tracker_file(cls, tracker_file_name: str) -> list[int]:
        """Get list of iterations to skip from results recorded in tracker file. If an
        "abnormality" (e.g., NaN or infinity in gradient) is seen more than once on a
        given rank and iteration, the corresponding iteration is skipped.

        Args:
            tracker_file_name (str): Name of tracker file.

        Returns:
            list[int]: List of iterations to skip.
        """
        iterations_to_skip: set[int] = set()
        seen: set[Tuple[int, int]]
        regex = r"ts=.+ node=.+ device=.+ jobID=.+ rank=(.+) iteration=(.+) status=(.+) .+"
        try:
            with open(tracker_file_name, 'r') as f:
                for line in f.readlines():
                    match = re.search(regex, line)
                    if match:
                        rank = int(match[1])
                        iteration = int(match[2])
                        status = match[3]
                        # Skip an iteration if:
                        # - Reruns were disabled and it has failed on the same rank twice.
                        # or
                        # - Reruns were enabled and it was reproducible on the 2nd rerun
                        if status == RerunValidationStatus.RERUN_DISABLED:
                            if (rank, iteration) in seen:
                                iterations_to_skip.add(iteration)
                            else:
                                seen.add((rank, iteration))
                        elif status == RerunValidationStatus.SECOND_RERUN_REPRODUCIBLE:
                            iterations_to_skip.add(iteration)
        except Exception as e:
            logger.error(f"Could not parse iterations to skip in tracker file! ({e})")
        return sorted(iterations_to_skip)


class RerunDataIterator:
    """A wrapper class for data iterators that adds replay capability.

    Args:
        iterable: data iterator that needs the replay capability.
        make_iterable: if set, iterator is created by calling iter() on iterable.

    The RerunState class below uses the rewind capability to replay all the microbatches
    fetched during an iteration.

    Example usage:

        class MyDataIterator:
            ...

        data_iterator = MyDataIterator(...)
        replay_data_iterator = RerunDataIterator(data_iterator)
    """

    def __init__(self, iterable: Iterable[Any]) -> None:
        self.iterable: Iterable[Any] = iterable
        self.saved_microbatches: list[Any] = []
        self.replaying: bool = False
        self.replay_pos: int = 0

    def __next__(self) -> Any:
        """__next__ method override adding replay capability."""

        if self.replaying:
            # we should not read past the saved batches if execution is deterministic,
            # as the number of calls to get_batch() should remain the same across reruns
            assert len(self.saved_microbatches) > self.replay_pos, "No more batches to replay"
            n = self.saved_microbatches[self.replay_pos]
            self.replay_pos += 1
            return n
        n: Any = next(self.iterable)
        if get_rerun_state_machine().get_mode() != RerunMode.DISABLED:
            self.saved_microbatches.append(n)
        return n

    def rewind(self) -> None:
        """Method to rewind the data iterator to the first microbatch of the iteration."""

        self.replaying = True
        self.replay_pos = 0

    def advance(self) -> None:
        """Method to drop all the buffered microbatches and jump to the next iteration."""

        self.replaying = False
        self.saved_microbatches = []

    def state_dict(self) -> SerializableStateType:
        """Method to capture the state of the iterator as a serializable dict."""

        return {
            'saved_microbatches': self.saved_microbatches,
            'replaying': self.replaying,
            'replay_pos': self.replay_pos,
        }

    def load_state_dict(self, state_dict: SerializableStateType) -> None:
        """Method to restore the state saved as a serializable dict."""

        self.saved_microbatches = state_dict['saved_microbatches']
        self.replaying = state_dict['replaying']
        self.replay_pos = state_dict['replay_pos']


class QuickStats:
    """Simple class to keep track of distribution of a statistic.

    Args:
        max_size: maximum number of samples to keep.
    """

    def __init__(self, max_size: int = 100000) -> None:
        self.samples: list[float] = []
        self.pos: int = 0
        self.zero_cnt: int = 0
        self.max: float = 0.0
        self.max_size: int = max_size

    def record(self, data: float) -> None:
        """Record a new sample."""

        if data == 0.0:
            self.zero_cnt += 1
        else:
            if self.pos < self.max_size:
                self.samples.append(data)
            else:
                self.samples[self.pos % self.self.max_size] = data
            self.pos += 1
            if data > self.max:
                self.max = data

    def combine(self, others: list["QuickStats"]) -> None:
        """Append the samples from multiple instances into one object."""

        if len(others) == 0:
            return
        n = len(self.samples) + sum(len(o.samples) for o in others)
        if n <= self.max_size:
            for o in others:
                self.samples.extend(o.samples)
            self.pos = n
        self.zero_cnt += sum(o.zero_cnt for o in others)
        self.max = max(self.max, max(o.max for o in others))

    def reset(self) -> None:
        """Forget all data."""

        self.samples = []
        self.pos = 0
        self.zero_cnt = 0
        self.max = 0.0

    def print_stats(self) -> str:
        """Return a string describing the data distribution."""

        self.samples.sort()
        z = self.zero_cnt
        n = len(self.samples)
        if n > 0:
            t = z + n
            s = sum(self.samples)
            a = s / t
            ps = {}
            for p in [0.5, 0.9, 0.99, 0.999]:
                ps[p] = f"{self.samples[int(t * p) - z]:.3E}" if int(t * p) - z >= 0 else "0.0"
            mx = self.max
            return (
                f"{t:,}/{z:,} total/identical samples, rel. variability: avg= {a:.3E}, "
                f"p50= {ps[0.5]}, p90= {ps[0.9]}, p99= {ps[0.99]}, p99.9= {ps[0.999]}, "
                f"max: {mx:.3E}"
            )
        else:
            return f"{z:,} samples, all identical"

    def __getstate_(self) -> Any:
        """Pickle method, used by torch.distributed.gather_object."""

        return vars(self)

    def __setstate(self, state: Any) -> Any:
        """Unpickle method, used by torch.distributed.gather_object."""

        self.samples = state['samples']
        self.pos = state['pos']
        self.zero_cnt = state['zero_cnt']
        self.max = state['max']


class RerunErrorInjector:
    """A class to manage error injection into the rerun state machine."""

    _ERROR_NAMES: dict[RerunDiagnostic, str] = {
        RerunDiagnostic.CORRECT_RESULT: "Expected result",
        RerunDiagnostic.TRANSIENT_ERROR: "Transient error",
        RerunDiagnostic.PERSISTENT_ERROR: "Persistent error",
    }

    def __init__(
        self,
        error_injection_rate: int = 0,
        error_injection_type: RerunDiagnostic = RerunDiagnostic.TRANSIENT_ERROR,
    ) -> None:
        assert isinstance(
            error_injection_type, RerunDiagnostic
        ), "Injected result type must be a valid RerunDiagnostic"
        self.error_injection_rate: int = error_injection_rate
        self.error_injection_type: RerunDiagnostic = error_injection_type
        self.should_inject_errors: bool = error_injection_rate > 0
        self.injected_error_type: Optional[RerunDiagnostic] = (
            None  # set to a non-None value when a result is injected
        )

    def maybe_inject(self) -> bool:
        """Method that decides whether to inject an error."""

        # Do not inject an error if error injection is turned off or if an error was
        # already injected in this iteration.
        if not self.should_inject_errors or self.injected_error_type is not None:
            return False
        r: int = (
            random.randint(0, self.error_injection_rate - 1) + _safe_get_rank()
        ) % self.error_injection_rate
        if r != 0:
            return False
        self.injected_error_type = self.error_injection_type
        logger.warning(
            f"Injecting error type {RerunErrorInjector._ERROR_NAMES[self.error_injection_type]}"
        )
        return True

    def maybe_miscompare(
        self,
        comparison_func: Callable[[Any, Any], float],
        initial_result: Any,
        result: Any,
        state: RerunState,
    ) -> float:
        """Method that introduces mismatching results during reruns when an error is injected.

        When no error is injected, this method defers to the user-provided comparison function.
        When an error is injected, it returns matching or mismatching results depending on the type
        of error being injected and on the re-run state."""

        if self.injected_error_type is None:
            return comparison_func(initial_result, result)
        # On the first re-run, return a different results and mark the injection processed when
        # injecting an irreproducible result.
        if state == RerunState.RERUNNING_IN_PLACE:
            if self.injected_error_type == RerunDiagnostic.TRANSIENT_ERROR:
                self.injected_error_type = None
                return COMPARISON_MISMATCH
            else:
                return COMPARISON_MATCH
        # On the second re-run, mark the injection processed and, when injecting a mismatching
        # result return a different result.
        elif state == RerunState.RERUNNING_FROM_CHECKPOINT:
            if self.injected_error_type == RerunDiagnostic.PERSISTENT_ERROR:
                self.injected_error_type = None
                return COMPARISON_MISMATCH
            elif self.injected_error_type == RerunDiagnostic.CORRECT_RESULT:
                self.injected_error_type = None
                return COMPARISON_MATCH
            else:
                raise RuntimeError("Should not be here")
        else:
            raise RuntimeError("Should not be here")

    def state_dict(self) -> SerializableStateType:
        """Method to capture the state of the error injector as a serializable dict."""

        return {
            'error_injection_rate': self.error_injection_rate,
            'error_injection_type': self.error_injection_type,
            # No need to checkpoint should_inject_errors (inferred from error_injection_rate).
            'injected_error_type': self.injected_error_type,
        }

    def load_state_dict(self, state_dict: SerializableStateType) -> None:
        """Method to restore the state saved as a serializable dict."""

        self.error_injection_rate = state_dict['error_injection_rate']
        self.error_injection_type = state_dict['error_injection_type']
        self.should_inject_errors = self.error_injection_rate > 0
        self.injected_error_type = state_dict['injected_error_type']


def initialize_rerun_state_machine(**kwargs) -> None:
    """Helper function to initialize the rerun machine instance.

    Check the RerunStateMachine class for the details.
    """

    rerun_state_machine: RerunStateMachine = RerunStateMachine(**kwargs)
    _set_rerun_state_machine(rerun_state_machine)


def destroy_rerun_state_machine() -> None:
    """Helper function to shut down the rerun machine instance."""

    global _GLOBAL_RERUN_STATE_MACHINE
    _GLOBAL_RERUN_STATE_MACHINE = None


def get_rerun_state_machine() -> RerunStateMachine:
    """Helper function to return the singleton instance of the rerun machine."""

    if _GLOBAL_RERUN_STATE_MACHINE is None:
        logger.warning("Implicit initialization of Rerun State Machine!")
        initialize_rerun_state_machine()
    return _GLOBAL_RERUN_STATE_MACHINE


def _set_rerun_state_machine(rerun_state_machine) -> None:
    """Internal function to set the singleton instance of the rerun machine."""

    global _GLOBAL_RERUN_STATE_MACHINE
    assert _GLOBAL_RERUN_STATE_MACHINE is None, 'Rerun state machine is already initialized'
    _GLOBAL_RERUN_STATE_MACHINE = rerun_state_machine


def _safe_get_rank() -> int:
    """Internal function that safely checks and returns the rank of the caller."""

    if torch.distributed.is_initialized():
        return torch.distributed.get_rank()

    # If torch.distributed is not initialized, try to read environment variables.
    try:
        return int(os.environ.get("RANK", 0))
    except (ValueError, TypeError):
        return 0


def _compare_floats(a: torch.Tensor, b: torch.Tensor) -> float:
    """Internal function that implements the default compare_func.

    Check the validate_result() method of the RerunStateMachine class for details.
    """

    af: float = a.item()
    bf: float = b.item()
    if (af == bf) or (math.isnan(af) and math.isnan(bf)):
        return COMPARISON_MATCH
    if (
        (math.isnan(af) and not math.isnan(bf))
        or (not math.isnan(af) and math.isnan(bf))
        or (math.isinf(af) and not math.isinf(bf))
        or (not math.isinf(af) and math.isinf(bf))
        or (math.isnan(af) and math.isinf(bf))
        or (math.isinf(af) and math.isnan(bf))
    ):
        return COMPARISON_MISMATCH
    return math.fabs((af - bf) / (af + bf) * 2)
