# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Small semantic table/norm annotations for mixed Muon/Sinkhorn/Adam groups."""


def table_role(param, name):
    """Identify table roles independently of tensor rank and fused storage."""
    if getattr(param, "is_engram_embedding", False):
        return "engram"
    if getattr(param, "is_embedding_or_output_parameter", False):
        if name.endswith("output_layer.weight"):
            return "output_layer"
        return "embedding"
    return None


def annotate_optimizer_parameters(model_chunks, config, pg_collection):
    """Attach immutable logical geometry before masters replace model parameters.

    Engram supplies its own hash-region ranges. Vocabulary tables use the declared
    matrix row dimension; TP row shards are reconstructed only through statistics.
    Norm parameters are tagged by their owning modules, including fused norms.
    """
    for chunk in model_chunks:
        for name, module in chunk.named_modules():
            if getattr(module, "share_embeddings_and_output_weights", False):
                fallback = getattr(config, "muon_scalar_optimizer", "adam")
                embedding = getattr(config, "embedding_optimizer", None) or fallback
                output = getattr(config, "output_layer_optimizer", None) or fallback
                if embedding != output:
                    raise ValueError(
                        "Tied embedding/output weights require identical optimizer selections"
                    )
            for field, param in module.named_parameters(recurse=False):
                if field == "layer_norm_weight" or (
                    field == "weight"
                    and param.ndim == 1
                    and "norm" in type(module).__name__.lower()
                ):
                    param.is_normalization_parameter = True
        for name, param in chunk.named_parameters():
            if (
                getattr(config, "muon_update_rms", None) is not None
                and param.ndim > 2
                and "experts" in name
            ):
                raise ValueError(
                    "Muon prototype requires per-expert 2D weights, not packed expert tensors"
                )
            role = table_role(param, name)
            if role is None:
                continue
            param.optimizer_role = role
            if getattr(config, role + "_optimizer", None) != "sinkhorn":
                continue
            if getattr(param, "sinkhorn_table_ranges", None) is not None:
                continue
            if role == "engram":
                raise ValueError(f"Engram parameter {name} is missing logical hash-table metadata")
            tp = pg_collection.tp
            tp_size = 1 if tp is None else tp.size()
            tp_rank = 0 if tp is None else tp.rank()
            partition = getattr(param, "partition_dim", -1)
            sharded = getattr(param, "tensor_model_parallel", False) and tp_size > 1
            if sharded and partition != 0:
                raise ValueError("Sinkhorn vocabulary tables require row-parallel storage")
            rows = param.shape[0] * (tp_size if sharded else 1)
            param.sinkhorn_table_ranges = ((0, rows),)
            param.sinkhorn_row_start = tp_rank * param.shape[0] if sharded else 0
            param.sinkhorn_group = tp if sharded else None


def table_group_override(config, param, name):
    """Return explicit per-role routing and schedule values; preserve defaults otherwise."""
    role = table_role(param, name)
    if role is None:
        return {}
    selected = getattr(config, role + "_optimizer", None)
    override = {}
    if selected is not None:
        override["optimizer"] = selected
        if selected == "sinkhorn":
            override["wd_mult"] = 0.0
    if role == "engram" and getattr(config, "engram_lr_multiplier", 1.0) != 1.0:
        override["schedule_lr_multiplier"] = config.engram_lr_multiplier
        override["default_config"] = False
    return override
