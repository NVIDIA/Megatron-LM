# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Reserved HybridEP chunk-dispatch/combine backend; no communication yet."""

from megatron.core.transformer.moe.replica_weight_transport import (
    UnimplementedReplicaWeightTransport,
)


class HybridEPWeightTransport(UnimplementedReplicaWeightTransport):
    """Future NVLink/RDMA transport using private chunk routing and wire layouts.

    Implement plan compilation, weight dispatch and replica-gradient combine
    here. Keep TE wrappers, GTP and optimizer accumulation in the runtime.
    Validate MXFP8 encoding and FP32 gradient accumulation before advertising
    support. Do not reuse the old autograd weight-dispatch wrapper wholesale.
    """

    transport_name = "replica_hybridep"
