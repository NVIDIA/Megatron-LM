from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union
import copy
import os
import torch
from torch.profiler import ProfilerAction
from transformers.trainer_callback import CallbackHandler, TrainerCallback, TrainerControl, TrainerState


@contextmanager
def clear_environment():
    """
    provide a blank environment with empty os envs,
    the previous envs will be resumed when exist the context
    """
    _old_os_env_backups = os.environ.copy()
    os.environ.clear()

    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(_old_os_env_backups)

class DataclassMixin:
    """
    provide a default to_dict and to_kwargs helper functions for dataclass
    """

    def to_json_string(self):
        """
        Serializes this instance to a JSON string.
        """
        return json.dumps(self.to_dict(), indent=2)

    def to_dict(self):
        """
        Serializes this instance while replace `Enum` by their values and `Callable` by dictionaries (for JSON
        serialization support). It obfuscates the token values by removing their value.
        """

        def serialize_dict(d: dict):
            args_to_dict = {}
            for k, v in d.items():
                if isinstance(v, Callable):  # type: ignore[arg-type]
                    args_to_dict[k] = v.__name__ if hasattr(v, "__name__") else str(v)
                elif isinstance(v, (list, tuple)) and len(v) > 0:
                    if isinstance(v[0], Enum):
                        args_to_dict[k] = [x.value for x in v]
                        if isinstance(v, tuple):
                            args_to_dict[k] = tuple(d[k])
                    elif isinstance(v[0], Callable):  # type: ignore[arg-type]
                        args_to_dict[k] = [x.__name__ if hasattr(x, "__name__") else str(x) for x in v]
                        if isinstance(v, tuple):
                            args_to_dict[k] = tuple(d[k])
                elif isinstance(v, dict):
                    args_to_dict[k] = serialize_dict(v)
                else:
                    if is_json_serializable(v):
                        args_to_dict[k] = v
                    elif hasattr(v, "__name__"):
                        args_to_dict[k] = v.__name__
                    else:
                        args_to_dict[k] = str(v)

            return args_to_dict

        d = {field.name: getattr(self, field.name) for field in fields(self) if field.init}

        return serialize_dict(d)

    def to_kwargs(self):
        """
        Returns a dictionary containing the attributes with values different from the default of this class.
        """
        with clear_environment():
            default_dict = self.__class__().to_dict()
        this_dict = self.to_dict()
        return {k: v for k, v in this_dict.items() if default_dict[k] != v}


@dataclass
class AutoMapperExtraConfigs:
    extra_configs: dict = field(default_factory=dict)

    def do_auto_mapping(self, name_map: dict, overwrite_extra_configs=False):
        """

        Args:
            name_map:
            overwrite_extra_configs: whether to overwrite `extra_configs` attribute. Be very careful to set to True,
            since once you overwrite it, the auto_mapping will not be able to change the value inside the extra_configs
            from the outside args wrapper

        Returns:

        """
        if overwrite_extra_configs:
            target_dict = self.extra_configs
        else:
            target_dict = copy.copy(self.extra_configs)

        # Compat megatron args and atorch args
        # If there is an argument named A in Megatron and an argument named B in atorch,
        # with same meaning but different name, A is given higher priority.
        for atorch_args_name, megatron_args_name in name_map.items():
            # For example, logging interval is named "log_interval" in megatron but "logging_steps" in atorch,
            # you can set "logging_steps" in TrainingArgs or "log_interval" in TrainingArgs.extra_configs .
            # Just select one of them. If they are both set, give priority to "log_interval".
            if megatron_args_name not in target_dict:
                target_dict[megatron_args_name] = getattr(self, atorch_args_name)
            else:
                if target_dict[megatron_args_name] is None and getattr(self, atorch_args_name) is not None:
                    target_dict[megatron_args_name] = getattr(self, atorch_args_name)
                elif target_dict[megatron_args_name] != getattr(self, atorch_args_name):
                    logger.warning(
                        f"{atorch_args_name}:{getattr(self, atorch_args_name)} in TrainingArgs will "
                        f"be overridden by {megatron_args_name}:{target_dict[megatron_args_name]} in Megatron args."
                    )
                    setattr(self, atorch_args_name, target_dict[megatron_args_name])
        return target_dict


@dataclass
class TrainingArgs(DataclassMixin, AutoMapperExtraConfigs):
    # Profiling
    profiler_type: Optional[str] = field(
        default=None,
        metadata={
            "help": 'Select a profiler platform. "hw": torch NPU profiler; "hw_dp": torch NPU dynamic profiler; '
            '"nv": origin torch profiler; "nv_dp": torch NPU dynamic profiler.',
            "choices": [None, "hw", "hw_dp", "nv", "nv_dp", "nsys"],
        },
    )

    # torch profiler
    profiler_file_path: Optional[str] = field(
        default=None,
    )
    profiler_config: dict = field(default_factory=dict)
    profiler_schedule_skip_first: int = field(
        default=20, metadata={"help": "Torch profiler schedule 'skip_first' arg."}
    )
    profiler_schedule_wait: int = field(default=1, metadata={"help": "Torch profiler schedule 'wait' arg."})
    profiler_schedule_warmup: int = field(default=1, metadata={"help": "Torch profiler schedule 'warmup' arg."})
    profiler_schedule_active: int = field(default=1, metadata={"help": "Torch profiler schedule 'active' arg."})
    profiler_schedule_repeat: int = field(default=1, metadata={"help": "Torch profiler schedule 'repeat' arg."})
    profile_use_gzip: bool = field(
        default=False,
        metadata={
            "help": "Whether to save torch profile in gzip format. Only valid when using torch profile on NV GPU."
        },
    )
    profiler_record_function: bool = field(
        default=True,
        metadata={
            "help": "Whether to use record_function to catch function name when doing torch profiler or NVTX profiling."
        },
    )

    # dynamic torch profiler
    dynamic_profiler_config_path: Optional[str] = field(
        default=None, metadata={"help": "The config directory when using torch dynamic profiler."}
    )

    # nsys profiling
    profile_step_start: int = field(default=10, metadata={"help": "Global step to start profiling."})
    profile_step_end: int = field(default=12, metadata={"help": "Global step to stop profiling."})
    profile_ranks: List[int] = field(default_factory=lambda: [-1], metadata={"help": "Global ranks to profile."})
    enable_auto_nvtx: bool = field(
        default=False,
        metadata={"help": "Whether to make every autograd operation emit an NVTX range when dosing nsys profiling."},
    )


def get_profiler(args: TrainingArgs):
    profiler_type = args.profiler_type
    profiler_file_path = args.profiler_file_path
    profiler_config = args.profiler_config

    if profiler_type is None or profiler_type == "nsys":
        return nullcontext()
    

    elif profiler_type == "nv":

        def trace_handler():
            if profiler_file_path is not None and profiler_type is not None:
                os.makedirs(profiler_file_path, exist_ok=True)

            rank = torch.distributed.get_rank()
            if args.profile_ranks == [-1] or rank in args.profile_ranks:
                return torch.profiler.tensorboard_trace_handler(
                    profiler_file_path,
                    worker_name=f"torch_profiler_rank{rank}_{socket.gethostname()}_{os.getpid()}",
                    use_gzip=args.profile_use_gzip,
                )
            else:

                def _dummy_writer(p):
                    # Do nothing
                    pass

                return _dummy_writer

        default_profiler_config = dict(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            with_stack=False,
            record_shapes=False,
            profile_memory=False,
            schedule=torch.profiler.schedule(
                wait=args.profiler_schedule_wait,
                warmup=args.profiler_schedule_warmup,
                active=args.profiler_schedule_active,
                repeat=args.profiler_schedule_repeat,
                skip_first=args.profiler_schedule_skip_first,
            ),
            on_trace_ready=trace_handler(),
        )
        default_profiler_config.update(profiler_config)
        return torch.profiler.profile(**default_profiler_config)

    else:
        logger.warning(
            f"Unsupported profiler_type:{profiler_type}. Please use one of ['hw', 'hw_dp', 'nv', 'nv_dp', 'nsys']."
        )
        return nullcontext()

class ProfilerCallback(TrainerCallback):
    """
    A [`TrainerCallback`] that execute profiling.
    """

    def on_train_begin(
        self,
        args: TrainingArgs,
        **kwargs,
    ):
        self.prof = get_profiler(args)
        if hasattr(self.prof, "start"):
            self.prof.start()

        self.waiting_tag = [ProfilerAction.NONE]
        self.working_tag = [ProfilerAction.WARMUP, ProfilerAction.RECORD, ProfilerAction.RECORD_AND_SAVE]

    def on_train_end(
        self,
        args: TrainingArgs,
        **kwargs,
    ):
        if hasattr(self.prof, "stop"):
            self.prof.stop()

    def on_step_begin(
        self,
        args: TrainingArgs,
        **kwargs,
    ):
        if (
            args.profiler_type == "nsys"
            and state.global_step == args.profile_step_start
            and (args.profile_ranks == [-1] or args.process_index in args.profile_ranks)
        ):
            torch.cuda.cudart().cudaProfilerStart()
            if args.enable_auto_nvtx:
                # Auto NVTX will cause more profiling file size.
                self.nvtx_ctx = torch.autograd.profiler.emit_nvtx(record_shapes=True)
                self.nvtx_ctx.__enter__()

            # Set NVTX range.
            configure_profiling(enable_nvtx=True)

            control.should_profiling = True
            
        elif self.prof is not None and hasattr(self.prof, "step"):
            prev_action: ProfilerAction = self._get_profiler_status(args.profiler_type == "nv_dp")
            self.prof.step()
            current_action: ProfilerAction = self._get_profiler_status(args.profiler_type == "nv_dp")

            if args.profiler_record_function and args.distributed_state.distributed_type == DistributedType.MEGATRON:
                model = kwargs["model"]
                assert isinstance(model, List)

                if torch.autograd.profiler._is_profiler_enabled:
                    # Set record_function
                    configure_profiling(enable_record_function=True, module_list=model)
                else:
                    clear_profiling_hooks()

            # Dump timeline when prev_action == ProfilerAction.RECORD_AND_SAVE
            if current_action in self.working_tag or prev_action == ProfilerAction.RECORD_AND_SAVE:
                control.should_profiling = True
            else:
                control.should_profiling = False

    def on_step_end(
        self,
        args: TrainingArgs,

        **kwargs,
    ):
        if (
            args.profiler_type == "nsys"
            and state.global_step == args.profile_step_end
            and (args.profile_ranks == [-1] or args.process_index in args.profile_ranks)
        ):
            torch.cuda.cudart().cudaProfilerStop()
            if args.enable_auto_nvtx:
                self.nvtx_ctx.__exit__(None, None, None)

            clear_profiling_hooks()

            control.should_profiling = False

    def _get_profiler_status(self, is_dynamic_profiler: bool):
        if is_dynamic_profiler:
            _torch_profiler = self.prof.get_torch_profiler()
            return _torch_profiler.current_action if _torch_profiler is not None else None
        else:
            return self.prof.current_action
