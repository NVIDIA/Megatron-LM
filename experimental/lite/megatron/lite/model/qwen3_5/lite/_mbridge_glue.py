"""Verbatim mbridge helpers for weight splitting and VL model hooks.

Sources:
  - mbridge/mbridge/models/qwen3_5/base_bridge.py  (weight-split)
  - mbridge/mbridge/models/qwen3_5/model.py        (VL hooks)
Only change from source: self. → bridge. / standalone functions (per R24).
"""
from __future__ import annotations

import torch  # pyright: ignore[reportMissingImports]
import torch.nn as nn  # pyright: ignore[reportMissingImports]


def _weight_split_across_tp(
    bridge,
    mcore_weights_name: str,
    mcore_weights: torch.Tensor,
    param: torch.Tensor,
    tp_split_size: int,
) -> list[torch.Tensor]:
    # VERBATIM from mbridge base_bridge.py::_weight_split_across_tp (L656-762)
    # Only change: self. → bridge.
    if tp_split_size == 1:
        return [mcore_weights]

    if (
        "self_attention.linear_qkv." in mcore_weights_name
        and "layer_norm" not in mcore_weights_name
    ):
        return mcore_weights.chunk(tp_split_size)
    elif "vision_model" not in mcore_weights_name and (
        "linear_fc1.weight" in mcore_weights_name
        or "linear_fc1.bias" in mcore_weights_name
    ):
        mcore_config = bridge._get_mcore_config_by_name(mcore_weights_name)
        if not mcore_config.gated_linear_unit:
            return mcore_weights.chunk(tp_split_size)

        gate, up = mcore_weights.chunk(2)
        gates = gate.chunk(tp_split_size)
        ups = up.chunk(tp_split_size)
        ret = [torch.cat([g, u], dim=0) for g, u in zip(gates, ups, strict=False)]
    elif "mlp.experts.linear_fc2.weight" in mcore_weights_name:  # moe
        ret = mcore_weights.chunk(tp_split_size, dim=1)
    elif "self_attention.in_proj.weight" in mcore_weights_name:
        mcore_config = bridge._get_mcore_config_by_name(mcore_weights_name)
        k_dim = mcore_config.linear_num_key_heads * mcore_config.linear_key_head_dim
        v_dim = (
            mcore_config.linear_num_value_heads * mcore_config.linear_value_head_dim
        )
        split_shape = [
            k_dim,
            k_dim,
            v_dim,
            v_dim,
            mcore_config.linear_num_value_heads,
            mcore_config.linear_num_value_heads,
        ]
        split_w_lst = mcore_weights.split(split_shape, dim=0)
        # split_w_lst: [wq, wk, wv, wz, wb, wa]
        assert len(split_w_lst) == 6, f"split_shape {split_shape} not supported"
        weight_list = []
        for weight in split_w_lst:
            weight_list.append(weight.chunk(tp_split_size))
        ret = [
            torch.cat(
                [wq_slice, wk_slice, wv_slice, wz_slice, wb_slice, wa_slice], dim=0
            )
            for wq_slice, wk_slice, wv_slice, wz_slice, wb_slice, wa_slice in zip(
                *weight_list, strict=False
            )
        ]
    elif "self_attention.conv1d" in mcore_weights_name:
        if "weight" in mcore_weights_name:
            mcore_config = bridge._get_mcore_config_by_name(mcore_weights_name)
            k_dim = (
                mcore_config.linear_num_key_heads * mcore_config.linear_key_head_dim
            )
            v_dim = (
                mcore_config.linear_num_value_heads
                * mcore_config.linear_value_head_dim
            )
            split_shape = [
                k_dim,
                k_dim,
                v_dim,
            ]
            split_w_lst = mcore_weights.split(split_shape, dim=0)
            # split_w_lst: [X, B, C]
            assert len(split_w_lst) == 3, f"split_shape {split_shape} not supported"
            weight_list = []
            for weight in split_w_lst:
                weight_list.append(weight.chunk(tp_split_size))
            ret = [
                torch.cat([x_slice, b_slice, c_slice], dim=0)
                for x_slice, b_slice, c_slice in zip(*weight_list, strict=False)
            ]
        else:
            raise NotImplementedError(f"{mcore_weights_name} not supported yet")
    else:
        if param.shape == mcore_weights.shape:
            return [mcore_weights for _ in range(tp_split_size)]
        assert len(param.shape) == len(mcore_weights.shape)
        for partition_dim, (s1, s2) in enumerate(  # noqa: B007
            zip(param.shape, mcore_weights.shape, strict=False)
        ):
            if s1 != s2:
                break

        ret = mcore_weights.chunk(tp_split_size, dim=partition_dim)
    return ret


def _split_weight_by_size_and_merge_across_tp(
    mcore_weights: list[torch.Tensor],
    split_shape: list[int],
) -> torch.Tensor:
    # VERBATIM from mbridge base_bridge.py::_split_weight_by_size_and_merge_across_tp (L764-785)
    tp_size = len(mcore_weights)

    weight_lst = [[] for _ in range(len(split_shape))]
    for mcore_weight in mcore_weights:
        split_w_lst = mcore_weight.split(split_shape, dim=0)
        assert len(split_w_lst) == len(weight_lst)
        for wi, split_w in enumerate(split_w_lst):
            weight_lst[wi].append(split_w)
    for weight in weight_lst:
        assert len(weight) == tp_size
    ret = torch.cat([torch.cat(w_split, dim=0) for w_split in weight_lst], dim=0)
    return ret


def _hook_fp32_rotary_emb_verbatim(module: nn.Module) -> None:
    # VERBATIM from mbridge/.../qwen3_5/model.py Qwen3_5VLModel._hook_fp32_rotary_emb (L122-139)
    for submodule in module.modules():
        if hasattr(submodule, "inv_freq") and submodule.inv_freq is not None:
            submodule._inv_freq_fp32_original = (
                submodule.inv_freq.detach().clone().float()
            )

            def _hook(mod, args):
                if hasattr(mod, "_inv_freq_fp32_original"):
                    mod.inv_freq = mod._inv_freq_fp32_original.to(
                        device=mod.inv_freq.device
                    )

            submodule.register_forward_pre_hook(_hook)


def _hook_vision_params_avg_grad_across_tp_verbatim(module: nn.Module) -> None:
    # VERBATIM from mbridge/.../qwen3_5/model.py Qwen3_5VLModel._hook_vision_params_avg_grad_across_tp (L141-152)
    if module is None:
        return
    for param in module.parameters(recurse=True):
        param.average_gradients_across_tp_domain = True  # type: ignore[assignment]
