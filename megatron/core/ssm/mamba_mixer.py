# Copyright (c) 2024, Tri Dao, Albert Gu.
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

# Some of this code was adopted from https://github.com/state-spaces/mamba/
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from megatron.core.parallel_state import get_tensor_model_parallel_world_size
from megatron.core.tensor_parallel import get_cuda_rng_tracker
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig

try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn = None
    causal_conv1d_update = None

try:
    from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as RMSNormGated
    from mamba_ssm.ops.triton.ssd_combined import (
        mamba_chunk_scan_combined,
        mamba_split_conv1d_scan_combined,
    )
except ImportError:
    raise ImportError("mamba-ssm is required by the Mamba model but cannot be imported")

try:
    from einops import rearrange, repeat
except ImportError:
    raise ImportError("einops is required by the Mamba model but cannot be imported")


@dataclass
class MambaMixerSubmodules:
    in_proj: Union[ModuleSpec, type] = None
    out_proj: Union[ModuleSpec, type] = None


class MambaMixer(MegatronModule):
    def __init__(
        self,
        config: TransformerConfig,
        submodules: MambaMixerSubmodules,
        d_model,
        d_state=128,
        d_conv=4,
        conv_init=None,
        expand=2,
        headdim=64,
        ngroups=8,
        A_init_range=(1, 16),
        D_has_hdim=False,
        rmsnorm=True,
        norm_before_gate=False,
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        bias=False,
        conv_bias=True,
        # Fused kernel and sharding options
        chunk_size=128,
        use_mem_eff_path=True,
        layer_idx=None,
    ):
        super().__init__(config)
        self.config = config
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.conv_init = conv_init
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.headdim = headdim
        self.ngroups = ngroups
        assert self.d_inner % self.headdim == 0
        self.nheads = self.d_inner // self.headdim
        self.D_has_hdim = D_has_hdim
        self.rmsnorm = rmsnorm
        self.norm_before_gate = norm_before_gate
        self.chunk_size = chunk_size
        self.use_mem_eff_path = use_mem_eff_path
        self.layer_idx = layer_idx

        self.tensor_model_parallel_size = get_tensor_model_parallel_world_size()
        assert self.d_inner % self.tensor_model_parallel_size == 0
        assert self.ngroups % self.tensor_model_parallel_size == 0
        assert self.nheads % self.tensor_model_parallel_size == 0
        assert not bias
        assert not self.norm_before_gate

        self.d_inner_local = self.d_inner // self.tensor_model_parallel_size
        self.ngroups_local = self.ngroups // self.tensor_model_parallel_size
        self.nheads_local = self.nheads // self.tensor_model_parallel_size

        assert self.d_inner_local % self.ngroups_local == 0

        # Assume sequence parallelism: input is already partitioned along the
        # sequence dimension
        self.in_proj = build_module(
            submodules.in_proj,
            self.d_model,
            self.d_inner * 2 + 2 * self.ngroups * self.d_state + self.nheads,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=bias,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name='fc1',
        )

        conv_dim = self.d_inner_local + 2 * self.ngroups_local * self.d_state
        with get_cuda_rng_tracker().fork():
            self.conv1d = nn.Conv1d(
                in_channels=conv_dim,
                out_channels=conv_dim,
                bias=conv_bias,
                kernel_size=d_conv,
                groups=conv_dim,
                padding=d_conv - 1,
                device=torch.cuda.current_device(),
                dtype=config.params_dtype,
            )
            setattr(self.conv1d.weight, 'tensor_model_parallel', True)
            setattr(self.conv1d.bias, 'tensor_model_parallel', True)

            if self.conv_init is not None:
                nn.init.uniform_(self.conv1d.weight, -self.conv_init, self.conv_init)

        self.activation = "silu"
        self.act = nn.SiLU()

        with get_cuda_rng_tracker().fork():
            # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
            dt = torch.exp(
                torch.rand(
                    self.nheads_local, device=torch.cuda.current_device(), dtype=config.params_dtype
                )
                * (math.log(dt_max) - math.log(dt_min))
                + math.log(dt_min)
            ).clamp(min=dt_init_floor)
            # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
            inv_dt = dt + torch.log(-torch.expm1(-dt))
            with torch.no_grad():
                self.dt_bias = nn.Parameter(inv_dt)
            # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
            self.dt_bias._no_reinit = True
            # Just to be explicit. Without this we already don't put wd on dt_bias because of the check
            # name.endswith("bias") in param_grouping.py
            self.dt_bias._no_weight_decay = True

            assert A_init_range[0] > 0 and A_init_range[1] >= A_init_range[0]
            A = torch.empty(
                self.nheads_local, dtype=torch.float32, device=torch.cuda.current_device()
            ).uniform_(*A_init_range)
            A_log = torch.log(A)  # Keep A_log in fp32
            self.A_log = nn.Parameter(A_log)
            self.A_log._no_weight_decay = True
            setattr(self.A_log, 'tensor_model_parallel', True)

        # D "skip" parameter
        self.D = nn.Parameter(
            torch.ones(
                self.d_inner_local if self.D_has_hdim else self.nheads_local,
                device=torch.cuda.current_device(),
            )
        )  # Keep in fp32
        self.D._no_weight_decay = True
        setattr(self.D, 'tensor_model_parallel', True)

        if self.rmsnorm:
            assert RMSNormGated is not None
            self.norm = RMSNormGated(
                self.d_inner_local,
                eps=1e-5,
                group_size=self.d_inner_local // self.ngroups_local,
                norm_before_gate=self.norm_before_gate,
                device=torch.cuda.current_device(),
                dtype=config.params_dtype,
            )

        # Assume sequence parallelism: input is partitioned along d_inner and
        # output is partitioned along the sequence dimension
        self.out_proj = build_module(
            submodules.out_proj,
            self.d_inner,
            self.d_model,
            config=self.config,
            init_method=self.config.output_layer_init_method,
            bias=bias,
            input_is_parallel=True,
            skip_bias_add=True,
            is_expert=False,
            tp_comm_buffer_name='fc2',
        )

    def forward(self, hidden_states, inference_params=None):
        """
        hidden_states: (nL, B, D) / (L B D)
        Returns: same shape as hidden_states
        """
        _, batch, dim = hidden_states.shape

        conv_state, ssm_state = None, None
        if inference_params is not None:
            assert not self.config.sequence_parallel
            conv_state, ssm_state = self._get_states_from_cache(inference_params, batch)
            if inference_params.seqlen_offset > 0:
                # The states are updated inplace
                out, out_bias, _, _ = self.step(hidden_states, conv_state, ssm_state)
                return out, out_bias

        # (nheads_local)
        A = -torch.exp(self.A_log.float())

        xz, _ = self.in_proj(hidden_states)

        # transpose: l b pd --> b l pd
        xz = rearrange(xz, "l b d -> b l d").contiguous()

        if self.use_mem_eff_path and inference_params is None:
            assert ssm_state is None

            if self.conv1d.bias is not None:
                self.conv1d.bias.data_ptr()

            y = mamba_split_conv1d_scan_combined(
                xz,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.dt_bias.float(),
                A,
                D=(
                    rearrange(self.D.float(), "(h p) -> h p", p=self.headdim)
                    if self.D_has_hdim
                    else self.D
                ),
                chunk_size=self.chunk_size,
                activation=self.activation,
                headdim=None if self.D_has_hdim else self.headdim,
                ngroups=self.ngroups_local,
                norm_before_gate=self.norm_before_gate,
            )

            if self.rmsnorm:
                y = self.norm(y)
        else:
            z, xBC, dt = torch.split(
                xz,
                [
                    self.d_inner_local,
                    self.d_inner_local + 2 * self.ngroups_local * self.d_state,
                    self.nheads_local,
                ],
                dim=-1,
            )

            # transpose: b l pd --> b pd l
            xBC = rearrange(xBC, "b l d -> b d l").contiguous()

            # Compute short convolution
            if conv_state is not None:
                # If we just take x[:, :, -self.d_conv :], it will error if seqlen < self.d_conv
                # Instead F.pad will pad with zeros if seqlen < self.d_conv, and truncate otherwise.
                conv_state.copy_(
                    F.pad(xBC, (self.d_conv - xBC.shape[-1], 0))
                )  # Update state (B D W)

            seqlen = xBC.size(2)
            if causal_conv1d_fn is None:
                xBC = self.act(self.conv1d(xBC)[..., :seqlen])
            else:
                assert self.activation in ["silu", "swish"]
                xBC = causal_conv1d_fn(
                    x=xBC,
                    weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                )

            # transpose b pd l --> b l pd
            xBC = rearrange(xBC, "b d l ->  b l d").contiguous()

            x, B, C = torch.split(
                xBC,
                [
                    self.d_inner_local,
                    self.ngroups_local * self.d_state,
                    self.ngroups_local * self.d_state,
                ],
                dim=-1,
            )

            # TODO Vijay: fuse most of the transposes with the GEMMS
            x = rearrange(x, "b l (h p) -> b l h p", p=self.headdim).contiguous()
            dt = dt.contiguous()
            B = rearrange(B, "b l (g n) -> b l g n", n=self.d_state).contiguous()
            C = rearrange(C, "b l (g n) -> b l g n", n=self.d_state).contiguous()
            z = rearrange(z, "b l (h p) -> b l h p", p=self.headdim).contiguous()
            y = mamba_chunk_scan_combined(
                x,
                dt,
                A,
                B,
                C,
                self.chunk_size,
                D=(
                    rearrange(self.D.float(), "(h p) -> h p", p=self.headdim)
                    if self.D_has_hdim
                    else self.D
                ),
                z=z if not self.rmsnorm else None,
                dt_bias=self.dt_bias.float(),
                dt_softplus=True,
                return_final_states=ssm_state is not None,
            )

            if ssm_state is not None:
                y, last_state = y
                ssm_state.copy_(last_state)

            if self.rmsnorm:
                y = rearrange(y, "b l h p -> b l (h p)").contiguous()
                z = rearrange(z, "b l h p -> b l (h p)").contiguous()
                y = self.norm(y, z)
            else:
                y = rearrange(y, "b l h p -> b l (h p)").contiguous()

        y = rearrange(y, "b l d -> l b d").contiguous()
        out, out_bias = self.out_proj(y)

        return out, out_bias

    def step(self, hidden_states, conv_state, ssm_state):
        # assert self.ngroups_local == 1, "Only support ngroups=1 for inference for now"
        dtype = hidden_states.dtype
        assert hidden_states.shape[0] == 1, "Only support decoding with 1 token at a time for now"

        # l b d --> b d
        hidden_states = hidden_states.squeeze(0)

        #  b d_model --> b p(2d)
        xz, _ = self.in_proj(hidden_states)

        z, xBC, dt = torch.split(
            xz,
            [
                self.d_inner_local,
                self.d_inner_local + 2 * self.ngroups_local * self.d_state,
                self.nheads_local,
            ],
            dim=-1,
        )

        # Conv step
        if causal_conv1d_update is None:
            conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))  # Update state (B D W)
            conv_state[:, :, -1] = xBC
            xBC = torch.sum(
                conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1
            )  # (B D)
            if self.conv1d.bias is not None:
                xBC = xBC + self.conv1d.bias
            xBC = self.act(xBC).to(dtype=dtype)
        else:
            xBC = causal_conv1d_update(
                xBC,
                conv_state,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.activation,
            )

        x, B, C = torch.split(
            xBC,
            [
                self.d_inner_local,
                self.ngroups_local * self.d_state,
                self.ngroups_local * self.d_state,
            ],
            dim=-1,
        )
        A = -torch.exp(self.A_log.float())

        # SSM step
        if selective_state_update is None:
            if self.ngroups_local > 1:
                B = rearrange(B, "b (g n) -> b g n", n=self.d_state)
                C = rearrange(C, "b (g n) -> b g n", n=self.d_state)
                B = repeat(B, "b g n -> b (g h) n", h=self.d_inner_local // self.ngroups_local)
                C = repeat(C, "b g n -> b (g h) n", h=self.d_inner_local // self.ngroups_local)

                dt = repeat(dt, "b h -> b (h p)", p=self.headdim)
                dt_bias = repeat(self.dt_bias, "h -> (h p)", p=self.headdim)
                A = repeat(A, "h -> (h p) n", p=self.headdim, n=self.d_state)
                D = repeat(self.D, "h -> (h p)", p=self.headdim)

                dt = F.softplus(dt + dt_bias.to(dtype=dt.dtype))
                dA = torch.exp(torch.einsum("bd,dn->bdn", dt, A))

                dB_x = torch.einsum('bd,bdn,bd->bdn', dt, B, x)
                ssm_state.copy_(
                    ssm_state * rearrange(dA, "b (h p) n -> b h p n", p=self.headdim)
                    + rearrange(dB_x, "b (h p) n -> b h p n", p=self.headdim)
                )

                y = torch.einsum(
                    "bdn,bdn->bd",
                    rearrange(ssm_state.to(dtype), "b h p n -> b (h p) n", p=self.headdim),
                    C,
                )
                y = y + D.to(dtype) * x
                if not self.rmsnorm:
                    y = y * self.act(z)  # (B D)
            else:
                # Discretize A and B (b (g n))
                dt = F.softplus(dt + self.dt_bias.to(dtype=dt.dtype))  # (batch, nheads)
                dA = torch.exp(dt * A)
                x = rearrange(x, "b (h p) -> b h p", p=self.headdim)
                dBx = torch.einsum("bh,bn,bhp->bhpn", dt, B, x)
                ssm_state.copy_(ssm_state * rearrange(dA, "b h -> b h 1 1") + dBx)
                y = torch.einsum("bhpn,bn->bhp", ssm_state.to(dtype), C)
                y = y + rearrange(self.D.to(dtype), "h -> h 1") * x
                y = rearrange(y, "b h p -> b (h p)")
                if not self.rmsnorm:
                    y = y * self.act(z)  # (B D)
        else:
            A = repeat(A, "h -> h p n", p=self.headdim, n=self.d_state).to(dtype=torch.float32)
            dt = repeat(dt, "b h -> b h p", p=self.headdim)
            dt_bias = repeat(self.dt_bias, "h -> h p", p=self.headdim)
            D = repeat(self.D, "h -> h p", p=self.headdim)
            B = rearrange(B, "b (g n) -> b g n", g=self.ngroups_local)
            C = rearrange(C, "b (g n) -> b g n", g=self.ngroups_local)
            x_reshaped = rearrange(x, "b (h p) -> b h p", p=self.headdim)
            if not self.rmsnorm:
                z = rearrange(z, "b (h p) -> b h p", p=self.headdim)
            y = selective_state_update(
                ssm_state,
                x_reshaped,
                dt,
                A,
                B,
                C,
                D,
                z=z if not self.rmsnorm else None,
                dt_bias=dt_bias,
                dt_softplus=True,
            )
            y = rearrange(y, "b h p -> b (h p)")

        if self.rmsnorm:
            y = self.norm(y, z)

        # b pd --> b d
        out, out_bias = self.out_proj(y)
        return out.unsqueeze(0), out_bias, conv_state, ssm_state

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None):
        device = self.out_proj.weight.device
        conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
        conv_state = torch.zeros(
            batch_size, self.conv1d.weight.shape[0], self.d_conv, device=device, dtype=conv_dtype
        )
        ssm_dtype = self.in_proj.weight.dtype if dtype is None else dtype
        # ssm_dtype = torch.float32
        ssm_state = torch.zeros(
            batch_size,
            self.nheads_local,
            self.headdim,
            self.d_state,
            device=device,
            dtype=ssm_dtype,
        )
        return conv_state, ssm_state

    def _get_states_from_cache(self, inference_params, batch_size, initialize_states=False):
        assert self.layer_idx is not None
        if self.layer_idx not in inference_params.key_value_memory_dict:
            conv_state = torch.zeros(
                batch_size,
                self.conv1d.weight.shape[0],
                self.d_conv,
                device=self.conv1d.weight.device,
                dtype=self.conv1d.weight.dtype,
            )
            ssm_state = torch.zeros(
                batch_size,
                self.nheads_local,
                self.headdim,
                self.d_state,
                device=self.in_proj.weight.device,
                dtype=self.in_proj.weight.dtype,
            )
            inference_params.key_value_memory_dict[self.layer_idx] = (conv_state, ssm_state)
        else:
            conv_state, ssm_state = inference_params.key_value_memory_dict[self.layer_idx]
            # TODO: What if batch size changes between generation, and we reuse the same states?
            if initialize_states:
                conv_state.zero_()
                ssm_state.zero_()
        return conv_state, ssm_state
