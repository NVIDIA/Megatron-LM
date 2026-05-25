# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from types import SimpleNamespace

from megatron.training import ft_integration


class FakeRankMonitorClient:
    section_timeouts = {"setup": 1}

    def __init__(self):
        self.calls = []

    def start_section(self, name):
        self.calls.append(("start", name))

    def end_section(self, name):
        self.calls.append(("end", name))

    def calculate_and_set_section_timeouts(self, selected_sections, calc_out_of_section):
        self.calls.append(("update", tuple(selected_sections), calc_out_of_section))

    def state_dict(self):
        return {"section_timeouts": self.section_timeouts}

    def load_state_dict(self, state):
        self.calls.append(("load", state))

    def shutdown_workload_monitoring(self):
        self.calls.append("shutdown")


def test_ft_setup_noops_when_disabled(monkeypatch):
    monkeypatch.setattr(
        ft_integration.arguments,
        "parse_args",
        lambda ignore_unknown_args=True: SimpleNamespace(enable_ft_package=False),
    )
    monkeypatch.setattr(ft_integration, "_GLOBAL_RANK_MONITOR_CLIENT", None)

    ft_integration.setup()

    assert ft_integration.get_rank_monitor_client() is None


def test_ft_section_lifecycle_and_timeout_updates(monkeypatch, tmp_path):
    client = FakeRankMonitorClient()
    printed = []
    monkeypatch.setattr(ft_integration, "_GLOBAL_RANK_MONITOR_CLIENT", client)
    monkeypatch.setattr(ft_integration, "_ft_state_path", str(tmp_path / "ft_state.json"))
    monkeypatch.setattr(ft_integration, "_is_setup_section_open", True)
    monkeypatch.setattr(ft_integration, "_NUM_WARMUP_ITERS", 1)
    monkeypatch.setattr(ft_integration, "_seen_tr_iters_cnt", 1)
    monkeypatch.setattr(ft_integration, "_curr_eval_iter_idx", 1)
    monkeypatch.setattr(ft_integration, "_seen_checkpoints_cnt", 0)
    monkeypatch.setattr(ft_integration, "_is_persistent_chkpt_loaded", True)
    monkeypatch.setattr(ft_integration, "_is_async_chkpt_enabled", False)
    monkeypatch.setattr(ft_integration, "_is_calculating_timeouts", True)
    monkeypatch.setattr(ft_integration, "_MIN_ITERS_FOR_STEP_TIMEOUT_UPDATE", 1)
    monkeypatch.setattr(ft_integration, "is_rank0", lambda: True)
    monkeypatch.setattr(ft_integration, "print_rank_0", lambda message: printed.append(message))

    ft_integration.on_training_step_start()
    ft_integration.on_training_step_end()
    ft_integration.on_eval_step_start()
    ft_integration.on_eval_step_end()
    ft_integration.on_checkpointing_start()
    ft_integration.on_checkpointing_end(is_async_finalization=False)
    ft_integration.on_checkpoint_loaded(is_local_chkpt=False)
    ft_integration.shutdown()

    assert ("end", "setup") in client.calls
    assert ("start", "step") in client.calls
    assert ("end", "step") in client.calls
    assert ("start", "checkpointing") in client.calls
    assert ("end", "checkpointing") in client.calls
    assert any(call[0] == "update" for call in client.calls if isinstance(call, tuple))
    assert "shutdown" in client.calls
    assert ft_integration.get_rank_monitor_client() is None
    assert (tmp_path / "ft_state.json").exists()
    assert any("FT: closing" in item for item in printed)


def test_ft_load_state_if_exists(monkeypatch, tmp_path):
    client = FakeRankMonitorClient()
    state_path = tmp_path / "ft_state.json"
    state_path.write_text('{"loaded": true}', encoding="utf-8")
    printed = []
    monkeypatch.setattr(ft_integration, "_GLOBAL_RANK_MONITOR_CLIENT", client)
    monkeypatch.setattr(ft_integration, "_ft_state_path", str(state_path))
    monkeypatch.setattr(ft_integration, "print_rank_0", lambda message: printed.append(message))

    ft_integration._load_state_if_exists()

    assert ("load", {"loaded": True}) in client.calls
    assert printed
