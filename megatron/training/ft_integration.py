# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""
FT Package Integration

This file is part of the integration process for the FT package, a custom heartbeat-based
system developed by NVIDIA. The FT package monitors the ranks to detect hangs, gracefully
terminates the workload, and respawns it from the last checkpoints. It includes an auto
config feature that automatically sets up timeouts based on the observed time of iterations.

Note: This tool is an internal NVIDIA tool and is not open source. This file does not
contain the FT package itself but supports its integration.
"""

import types
from enum import Enum, auto
from . import global_vars

class StateMachineActions(Enum):
    NONE = auto()
    SAVE_CHECKPOINT = auto()
    TRAIN_HEARTBEAT = auto() 
    EVAL_HEARTBEAT = auto()
    UPDATE_TIMEOUT = auto()

class _TrainingStateMachine:
    """
    This class encapsulates logic for determining when:
    - FT timeouts can be updated (`.can_update_timeouts` property)

    `on_ ...` methods update the state and should be called from the corresponding places.
    """

    MIN_ITERS_FOR_TIMEOUT_UPDATE = 2

    def __init__(self):
        self.num_tr_iters_total = 0
        self.num_tr_iter_at_last_save = None
        self.seen_checkpointing = False
        self.timeouts_updated = False

    def on_save_checkpoint(self):
        self.num_tr_iter_at_last_save = self.num_tr_iters_total

    def on_train_heartbeat(self):
        self.num_tr_iters_total += 1
        if not self.seen_checkpointing and self.num_tr_iter_at_last_save is not None:
            # detect mid-epoch checkpointing that makes hearbeat interval longer
            iters_pre_save = self.num_tr_iter_at_last_save
            iters_post_save = self.num_tr_iters_total - self.num_tr_iter_at_last_save
            self.seen_checkpointing = iters_pre_save > 0 and iters_post_save > 0

    def on_eval_heartbeat(self):
        pass

    def on_timeouts_updated(self):
        self.timeouts_updated = True

    @property
    def can_update_timeouts(self) -> bool:
        """
        Returns True if new timeouts can be computed.
        `.on_timeouts_updated()` resets this property back to False.
        """
        if self.timeouts_updated:
            # timeouts are updated at most once per training run
            return False
        if self.num_tr_iters_total < self.MIN_ITERS_FOR_TIMEOUT_UPDATE:
            # need a few training iters
            return False
        # check if there was checkoint saving
        # this makes heartbeat iterval longer than usual.
        return self.seen_checkpointing

    def perform_action(self, action: StateMachineActions):
        if action == StateMachineActions.TRAIN_HEARTBEAT:
            self.on_train_heartbeat()
        elif action == StateMachineActions.SAVE_CHECKPOINT:
            self.on_save_checkpoint()
        elif action == StateMachineActions.EVAL_HEARTBEAT:
            self.on_eval_heartbeat()
        elif action == StateMachineActions.UPDATE_TIMEOUT:
            self.on_timeouts_updated()
            assert not self.can_update_timeouts
        # No action for StateMachineActions.NONE


_GLOBAL_RANK_MONITOR_CLIENT = None
_GLOBAL_STATE_MACHINE = _TrainingStateMachine()

def _set_rank_monitor_client():
    from fault_tolerance import RankMonitorClient
    cli = RankMonitorClient()
    global _GLOBAL_RANK_MONITOR_CLIENT
    global_vars._ensure_var_is_not_initialized(_GLOBAL_RANK_MONITOR_CLIENT, 'rank monitor client')
    _GLOBAL_RANK_MONITOR_CLIENT = cli

def get_rank_monitor_client(action=StateMachineActions.NONE):
    global _GLOBAL_RANK_MONITOR_CLIENT, _GLOBAL_STATE_MACHINE
    if _GLOBAL_RANK_MONITOR_CLIENT is None:
        try:
            _set_rank_monitor_client()
        except ImportError:
            _GLOBAL_RANK_MONITOR_CLIENT = None
    _GLOBAL_STATE_MACHINE.perform_action(action)
    return _GLOBAL_RANK_MONITOR_CLIENT

def can_update_timeouts():
    global _GLOBAL_STATE_MACHINE
    return _GLOBAL_STATE_MACHINE.can_update_timeouts
