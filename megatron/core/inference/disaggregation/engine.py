# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Dynamic inference engine composed with the KV/state hand-off behavior."""

import msgpack

from megatron.core.inference.disaggregation.inference_state_handoff import (
    InferenceStateHandoffMixin,
)
from megatron.core.inference.engines.dynamic_engine import DynamicInferenceEngine
from megatron.core.inference.headers import Headers
from megatron.core.utils import internal_api


class StateHandoffDynamicInferenceEngine(InferenceStateHandoffMixin, DynamicInferenceEngine):
    """Dynamic engine with control-plane-neutral KV/state handoff behavior.

    External control planes can use this class directly and configure the
    transfer role with :meth:`setup_kv_transfer`. It intentionally does not
    install the coordinator-native registration protocol below.
    """


class DisaggDynamicInferenceEngine(StateHandoffDynamicInferenceEngine):
    """Dynamic inference engine with prefill/decode state hand-off support.

    This subclass adds the coordinator-native registration callbacks. External
    control planes should use :class:`StateHandoffDynamicInferenceEngine`.
    """

    @internal_api
    def set_disaggregation_config(
        self, *, role, identity, spawn_coordinator, coordinator_group
    ) -> None:
        """Configure one coordinator-native prefill or decode instance."""

        if role not in ("prefill", "decode"):
            raise ValueError(f"invalid disaggregation role {role!r}")
        self._disagg_config = {
            "role": role,
            "identity": identity,
            "spawn_coordinator": spawn_coordinator,
            "coordinator_group": coordinator_group,
        }
        self.setup_kv_transfer(role)
        self._instance_transfer_meta = self._build_instance_transfer_meta()

    def _build_instance_transfer_meta(self) -> list[dict]:
        """Flatten this instance's PP/TP transport descriptors for registration."""

        instance_meta = []
        # Keep PP stages aligned for models without SSM state.
        pp_ssm_metas = self._pp_ssm_peer_metas or [{}] * len(self._pp_kv_peer_metas)
        ssm_capacity = self.context.max_requests if self.context.is_hybrid_model else None
        for kv_stage, ssm_stage in zip(self._pp_kv_peer_metas, pp_ssm_metas):
            # TP=1 exports one descriptor; TP>1 exports one per rank.
            kv_rank_metas = kv_stage if isinstance(kv_stage, list) else [kv_stage]
            for tp_index, kv_meta in enumerate(kv_rank_metas):
                rank_meta = dict(kv_meta)
                # Flow control uses the lowest capacity advertised by the MP ranks.
                rank_meta["request_capacity"] = self.context.max_requests
                rank_ssm_meta = {}
                # Pair each KV descriptor with the SSM buffers on the same TP rank.
                for state_kind, state_metas in ssm_stage.items():
                    if isinstance(state_metas, list):
                        rank_ssm_meta[state_kind] = state_metas[tp_index]
                    else:
                        rank_ssm_meta[state_kind] = state_metas
                if rank_ssm_meta:
                    rank_meta["ssm"] = rank_ssm_meta
                if ssm_capacity is not None:
                    rank_meta["ssm_slot_capacity"] = ssm_capacity
                    # One hybrid handoff owns one live slot across all SSM state buffers.
                    rank_meta["ssm_handoff_max_slots"] = 1
                instance_meta.append(rank_meta)
        return instance_meta

    def _notify_kv_read_done(self, request_id: int) -> None:
        """Tell the native coordinator that decode no longer needs source storage."""

        if self._disagg_config is not None and self.is_mp_coordinator:
            self.socket_for_receiving_requests.send(
                msgpack.packb([Headers.KV_READ_DONE.value, int(request_id)], use_bin_type=True)
            )

    def _notify_kv_transfer_ready(self, request_id: int, cached_prefix_blocks: int) -> None:
        """Tell the native coordinator that NCCL destination storage is committed."""

        if self._disagg_config is not None and self.is_mp_coordinator:
            self.socket_for_receiving_requests.send(
                msgpack.packb(
                    [Headers.KV_TRANSFER_READY.value, int(request_id), int(cached_prefix_blocks)],
                    use_bin_type=True,
                )
            )
