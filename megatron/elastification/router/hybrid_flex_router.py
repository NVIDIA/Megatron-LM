# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import random
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from megatron.core import parallel_state
from megatron.core.num_microbatches_calculator import (
    get_current_global_batch_size,
    get_micro_batch_size,
)
from megatron.core.parallel_state import (
    get_data_parallel_rank,
    get_data_parallel_world_size,
    get_pipeline_model_parallel_rank,
    get_pipeline_model_parallel_world_size,
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_rank,
)

# Remove top-level import to avoid circular imports
# from megatron.training import get_args, print_rank_0
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import init_method_normal

# Import TE parallel linear layers
try:
    from megatron.core.extensions.transformer_engine import (
        TEColumnParallelLinear,
        TERowParallelLinear,
    )
    HAVE_TE = True
except ImportError:
    HAVE_TE = False
    # Fallback to regular tensor parallel layers
    from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear

# Router implemation for pre-gating router.
# Use router to determine #heads, MLP sizes, and layers to skip (Router_v2)
# Only takes the budget as input
class FlextronRouter(MegatronModule):
    def __init__(self, config: TransformerConfig):
        super().__init__(config=config)
        
        self.config = config
        self.input_dim = len(self.config.budget_list)
        self.n_dim = self.config.router_inter_dim
        self.budget_map = {
            item: torch.tensor(idx) for idx, item in enumerate(self.config.budget_list)
        }

        # Initialize DP-aware Gumbel softmax
        self._init_dp_gumbel_softmax()

        # Create init method for router layers
        self.init_method = init_method_normal(self.config.router_std)

        self.add_router_for_mlp()
        self.add_router_for_emb()
        self.add_router_for_mamba()
        self.add_router_for_head()
        self.add_router_for_moe_expert()
        if self.config.add_skipping:
            self.add_router_for_skipping()

        # Synchronize router weights across all pipeline parallel ranks
        self._sync_router_weights()
        self._mark_router_params_for_pp_sync()
        self.hard_sample_th = config.hard_sample_th

        self.add_scaler_schedule()

        self.dp_size = get_data_parallel_world_size()
        self.grad_accumulation_steps = get_current_global_batch_size() // (get_micro_batch_size() * self.dp_size)
        self.fwd_pass_count = 0


    def _init_dp_gumbel_softmax(self, base_seed=42):
        """Initialize DP-aware Gumbel softmax functionality"""
        self.dp_rank = get_data_parallel_rank()
        self.gumbel_base_seed = base_seed
        
    def _sync_router_weights(self):
        """
        Synchronize router weights across all pipeline parallel groups by broadcasting
        from global rank 0 to all other ranks.
        """
        if not torch.distributed.is_initialized():
            return
            
        # Get global rank 0 as the source
        source_rank = 0
        
        # Broadcast all router parameters from rank 0
        for name, param in self.named_parameters():
            if param is not None:
                torch.distributed.broadcast(param.data, src=source_rank)
                

    def _mark_router_params_for_pp_sync(self):
        """
        Mark all router parameters to be synchronized across pipeline parallel ranks.
        This ensures they get handled by the main gradient synchronization system.
        """
        for param in self.parameters():
            if param.requires_grad:
                # Mark parameter for pipeline parallel synchronization
                setattr(param, 'pipeline_parallel', True)
                
                
    def _dp_gumbel_softmax(self, logits, tau=1.0, hard=False, curr_iteration=0):
        """DP-aware Gumbel softmax that uses different random seeds per DP rank and iteration"""
        # Create unique seed for this iteration and DP rank

        seed = self.gumbel_base_seed + (self.dp_rank + self.fwd_pass_count * self.dp_size) % self.config.router_gbs + curr_iteration * 1000
        # Save and set random state
        old_state = torch.get_rng_state()
        torch.manual_seed(seed)
        
        try:
            result = F.gumbel_softmax(logits, tau=tau, hard=hard)
            return result
        finally:
            torch.set_rng_state(old_state)

    def _create_linear_layer(self, input_size, output_size, bias=False, is_first_layer=True):
        """Helper method to create appropriate linear layer (TE or fallback)"""
        if HAVE_TE:
            if is_first_layer:
                # First layer: TEColumnParallelLinear
                return TEColumnParallelLinear(
                    input_size=input_size,
                    output_size=output_size,
                    config=self.config,
                    init_method=self.init_method,
                    gather_output=False,
                    bias=bias,
                    skip_bias_add=False,
                    is_expert=False,
                )
            else:
                # Second layer: TERowParallelLinear
                return TERowParallelLinear(
                    input_size=input_size,
                    output_size=output_size,
                    config=self.config,
                    init_method=self.init_method,
                    bias=bias,
                    input_is_parallel=True,
                    skip_bias_add=False,
                    is_expert=False,
                )
        else:
            # Fallback to regular tensor parallel layers
            if is_first_layer:
                return ColumnParallelLinear(
                    input_size=input_size,
                    output_size=output_size,
                    config=self.config,
                    init_method=self.init_method,
                    gather_output=False,
                    bias=bias,
                    skip_bias_add=False,
                    is_expert=False,
                )
            else:
                return RowParallelLinear(
                    input_size=input_size,
                    output_size=output_size,
                    config=self.config,
                    init_method=self.init_method,
                    bias=bias,
                    input_is_parallel=True,
                    skip_bias_add=False,
                    is_expert=False,
                )

    def add_router_for_mlp(self):
        mlp_list = self.config.mlp_int_list
        if self.config.flex_hetero_ffn:
            num_mlp = self.config.hybrid_layer_pattern.count("E")
            gate_mlp_layer_list = [
                self._create_linear_layer(self.input_dim, self.n_dim, bias=False, is_first_layer=True),
                nn.LeakyReLU(0.1),
                self._create_linear_layer(self.n_dim, len(mlp_list) * num_mlp, bias=False, is_first_layer=False)
            ]
            # Set bias for the last layer
            if hasattr(gate_mlp_layer_list[-1], 'bias') and gate_mlp_layer_list[-1].bias is not None:
                last_layer_bias  = [0.00 for _ in range(len(mlp_list))]
                last_layer_bias[-1] = 1.00
                gate_mlp_layer_list[-1].bias.data = torch.tensor(
                    last_layer_bias, 
                    dtype=gate_mlp_layer_list[-1].weight.dtype, 
                    device=gate_mlp_layer_list[-1].weight.device
                ).repeat(num_mlp)
        else:
            gate_mlp_layer_list = [
                self._create_linear_layer(self.input_dim, self.n_dim, bias=False, is_first_layer=True),
                nn.LeakyReLU(0.1),
                self._create_linear_layer(self.n_dim, len(mlp_list), bias=False, is_first_layer=False)
            ]
        self.gate_mlp = nn.Sequential(*gate_mlp_layer_list)

    def add_router_for_moe_expert(self):
        moe_expert_list = self.config.moe_expert_int_list
        if self.config.flex_hetero_moe_expert:
            num_moe_expert = self.config.hybrid_layer_pattern.count("E")
            gate_moe_expert_layer_list = [
                self._create_linear_layer(self.input_dim, self.n_dim, bias=False, is_first_layer=True), 
                nn.LeakyReLU(0.1),
                self._create_linear_layer(self.n_dim, len(moe_expert_list) * num_moe_expert, bias=False, is_first_layer=False)
            ]
        else:
            gate_moe_expert_layer_list = [
                self._create_linear_layer(self.input_dim, self.n_dim, bias=False, is_first_layer=True), 
                nn.LeakyReLU(0.1),
                self._create_linear_layer(self.n_dim, len(moe_expert_list), bias=False, is_first_layer=False)
            ]
        self.gate_moe_expert = nn.Sequential(*gate_moe_expert_layer_list)

    def add_router_for_emb(self):
        emb_list = self.config.emb_int_list
        gate_emb_layer_list = [
            self._create_linear_layer(self.input_dim, self.n_dim, bias=False, is_first_layer=True), 
            nn.LeakyReLU(0.1),
            self._create_linear_layer(self.n_dim, len(emb_list), bias=False, is_first_layer=False)
        ]
        self.gate_emb = nn.Sequential(*gate_emb_layer_list)

    def add_router_for_head(self):
        head_list = self.config.head_int_list
        if self.config.flex_hetero_head:
            num_head = self.config.hybrid_layer_pattern.count("*")
            gate_head_layer_list = [
                self._create_linear_layer(self.input_dim, self.n_dim, bias=False, is_first_layer=True), 
                nn.LeakyReLU(0.1),
                self._create_linear_layer(self.n_dim, len(head_list) * num_head, bias=False, is_first_layer=False)
            ]
            # Set bias for the last layer
            if hasattr(gate_head_layer_list[-1], 'bias') and gate_head_layer_list[-1].bias is not None:
                last_layer_bias  = [0.00 for _ in range(len(head_list))]
                last_layer_bias[-1] = 1.00
                gate_head_layer_list[-1].bias.data = torch.tensor(last_layer_bias).repeat(num_head)
        else:
            gate_head_layer_list = [
                self._create_linear_layer(self.input_dim, self.n_dim, bias=False, is_first_layer=True), 
                nn.LeakyReLU(0.1),
                self._create_linear_layer(self.n_dim, len(head_list), bias=False, is_first_layer=False)
            ]
        self.gate_head = nn.Sequential(*gate_head_layer_list)

    def add_router_for_skipping(self): 

        self.output_dim = int(len(self.config.layer_ranking_list)+1)
        
        gate_skip_mlp_layer_list = [
            self._create_linear_layer(self.input_dim, self.n_dim, bias=False, is_first_layer=True), 
            nn.LeakyReLU(0.1),
            self._create_linear_layer(self.n_dim, self.output_dim, bias=False, is_first_layer=False)
        ]

        self.gate_skip_layer = nn.Sequential(*gate_skip_mlp_layer_list)

    def add_router_for_mamba(self):
        mamba_list = self.config.mamba_int_list
        if self.config.flex_hetero_mamba:
            num_mamba = self.config.hybrid_layer_pattern.count("M")
            gate_mamba_layer_list = [
                self._create_linear_layer(self.input_dim, self.n_dim, bias=False, is_first_layer=True), 
                nn.LeakyReLU(0.1),
                self._create_linear_layer(self.n_dim, len(mamba_list) * num_mamba, bias=False, is_first_layer=False)
            ]
            # Set bias for the last layer
            if hasattr(gate_mamba_layer_list[-1], 'bias') and gate_mamba_layer_list[-1].bias is not None:
                last_layer_bias  = [0.00 for _ in range(len(mamba_list))]
                last_layer_bias[-1] = 1.00
                gate_mamba_layer_list[-1].bias.data = torch.tensor(last_layer_bias).repeat(num_mamba)
        else:
            gate_mamba_layer_list = [
                self._create_linear_layer(self.input_dim, self.n_dim, bias=False, is_first_layer=True), 
                nn.LeakyReLU(0.1),
                self._create_linear_layer(self.n_dim, len(mamba_list), bias=False, is_first_layer=False)
            ]
        self.gate_mamba = nn.Sequential(*gate_mamba_layer_list)

    def mamba_forward(self, args, budget_tensor, device, dtype, tau, hard_sample):
        
        # TODO @ataghibakhsh: check router out of sync on TP ranks

        router_mamba_logits1 = self.gate_mamba[0](budget_tensor)
        router_mamba_logits2 = self.gate_mamba[1](router_mamba_logits1[0])
        router_mamba_logits = self.gate_mamba[2](router_mamba_logits2)[0].flatten()
        # torch.distributed.all_reduce(router_mamba_logits, group=get_tensor_model_parallel_group(), op=torch.distributed.ReduceOp.AVG)
        if self.scaler is not None:
            scale = self.scaler[args.curr_iteration].to(device=device, dtype=dtype) 

        if self.config.flex_hetero_mamba:
            mamba_n = len(self.config.mamba_int_list)
            router_mamba_logits = router_mamba_logits.reshape(-1, mamba_n)
            if self.config.normalize_router_logits:
                router_mamba_logits = scale * router_mamba_logits / router_mamba_logits.std(dim=1, keepdim=True).clamp(min=1e-6)
            else:
                router_mamba_logits = scale * router_mamba_logits
            router_mamba_logits = self._dp_gumbel_softmax(router_mamba_logits, tau=tau, hard=hard_sample, curr_iteration=args.curr_iteration)
            _, choices_mamba = torch.topk(router_mamba_logits, 1, dim=-1)
            return (router_mamba_logits, [self.config.mamba_int_list[i] for i in choices_mamba.flatten().tolist()])
        else:
            if self.config.normalize_router_logits:
                if len(self.config.mamba_int_list) > 1:
                    router_mamba_logits = scale * router_mamba_logits / router_mamba_logits.std(dim=0, keepdim=True).clamp(min=1e-6)
            else:
                router_mamba_logits = scale * router_mamba_logits
            router_mamba_logits = self._dp_gumbel_softmax(router_mamba_logits, tau=tau, hard=hard_sample, curr_iteration=args.curr_iteration)
            _, choices_mamba = torch.topk(router_mamba_logits, 1, dim=-1)
            return (router_mamba_logits, self.config.mamba_int_list[choices_mamba.item()])

    def mlp_forward(self, args, budget_tensor, device, dtype, tau, hard_sample):
        
        # TODO @ataghibakhsh: check router out of sync on TP ranks
        router_mlp_logits1 = self.gate_mlp[0](budget_tensor)
        router_mlp_logits2 = self.gate_mlp[1](router_mlp_logits1[0])
        router_mlp_logits = self.gate_mlp[2](router_mlp_logits2)[0].flatten()
        # torch.distributed.all_reduce(router_mlp_logits, group=get_tensor_model_parallel_group(), op=torch.distributed.ReduceOp.AVG)
        if self.scaler is not None:
            scale = self.scaler[args.curr_iteration].to(device=device, dtype=dtype) 
        if self.config.flex_hetero_ffn:
            mlp_n = len(self.config.mlp_int_list)
            router_mlp_logits = router_mlp_logits.reshape(-1, mlp_n)
            if self.config.normalize_router_logits:
                router_mlp_logits = scale * router_mlp_logits / router_mlp_logits.std(dim=1, keepdim=True).clamp(min=1e-6)
            else:
                router_mlp_logits = scale * router_mlp_logits
            router_mlp_logits = self._dp_gumbel_softmax(router_mlp_logits, tau=tau, hard=hard_sample, curr_iteration=args.curr_iteration)
            _, choices_mlp = torch.topk(router_mlp_logits, 1, dim=-1)
            return (router_mlp_logits, [self.config.mlp_int_list[i] for i in choices_mlp.flatten().tolist()])
        else:
            if self.config.normalize_router_logits:
                router_mlp_logits = scale * router_mlp_logits / router_mlp_logits.std(dim=0, keepdim=True).clamp(min=1e-6)
            else:
                router_mlp_logits = scale * router_mlp_logits
            router_mlp_logits = self._dp_gumbel_softmax(router_mlp_logits, tau=tau, hard=hard_sample, curr_iteration=args.curr_iteration)
            _, choices_mlp = torch.topk(router_mlp_logits, 1, dim=-1)
            return (router_mlp_logits, self.config.mlp_int_list[choices_mlp.item()])
    
    def moe_expert_forward(self, args, budget_tensor, device, dtype, tau, hard_sample):
        router_moe_expert_logits1 = self.gate_moe_expert[0](budget_tensor)
        router_moe_expert_logits2 = self.gate_moe_expert[1](router_moe_expert_logits1[0])
        router_moe_expert_logits = self.gate_moe_expert[2](router_moe_expert_logits2)[0].flatten()
        # torch.distributed.all_reduce(router_moe_expert_logits, group=get_tensor_model_parallel_group(), op=torch.distributed.ReduceOp.AVG)
        if self.scaler is not None:
            scale = self.scaler[args.curr_iteration].to(device=device, dtype=dtype) 
        if self.config.flex_hetero_moe_expert:
            moe_expert_n = len(self.config.moe_expert_int_list)
            router_moe_expert_logits = router_moe_expert_logits.reshape(-1, moe_expert_n)
            if self.config.normalize_router_logits:
                router_moe_expert_logits = scale * router_moe_expert_logits / router_moe_expert_logits.std(dim=1, keepdim=True).clamp(min=1e-6)
            else:
                router_moe_expert_logits = scale * router_moe_expert_logits
            router_moe_expert_logits = self._dp_gumbel_softmax(router_moe_expert_logits, tau=tau, hard=hard_sample, curr_iteration=args.curr_iteration)
            _, choices_moe_expert = torch.topk(router_moe_expert_logits, 1, dim=-1)
            return (router_moe_expert_logits, [self.config.moe_expert_int_list[i] for i in choices_moe_expert.flatten().tolist()])
        else:
            if self.config.normalize_router_logits:
                router_moe_expert_logits = scale * router_moe_expert_logits / router_moe_expert_logits.std(dim=0, keepdim=True).clamp(min=1e-6)
            else:
                router_moe_expert_logits = scale * router_moe_expert_logits
            router_moe_expert_logits = self._dp_gumbel_softmax(router_moe_expert_logits, tau=tau, hard=hard_sample, curr_iteration=args.curr_iteration)
            _, choices_moe_expert = torch.topk(router_moe_expert_logits, 1, dim=-1)
            return (router_moe_expert_logits, self.config.moe_expert_int_list[choices_moe_expert.item()])
    
    def emb_forward(self, args, budget_tensor, device, dtype, tau, hard_sample):

        router_emb_logits1 = self.gate_emb[0](budget_tensor)
        router_emb_logits2 = self.gate_emb[1](router_emb_logits1[0])
        router_emb_logits = self.gate_emb[2](router_emb_logits2)[0].flatten()
        # torch.distributed.all_reduce(router_emb_logits, group=get_tensor_model_parallel_group(), op=torch.distributed.ReduceOp.AVG)
        if self.scaler is not None:
            scale = self.scaler[args.curr_iteration].to(device=device, dtype=dtype) 
            router_emb_logits = scale * router_emb_logits

        # router_emb_logits = F.gumbel_softmax(router_emb_logits, tau=tau, hard=hard_sample)
        router_emb_logits = self._dp_gumbel_softmax(router_emb_logits, tau=tau, hard=hard_sample, curr_iteration=args.curr_iteration)
        _, choices_emb = torch.topk(router_emb_logits, 1, dim=-1)

        return (router_emb_logits, self.config.emb_int_list[choices_emb.item()])

    def head_forward(self, args, budget_tensor, device, dtype, tau, hard_sample):

        router_head_logits1 = self.gate_head[0](budget_tensor)
        router_head_logits2 = self.gate_head[1](router_head_logits1[0])
        router_head_logits = self.gate_head[2](router_head_logits2)[0].flatten()
        # torch.distributed.all_reduce(router_head_logits, group=get_tensor_model_parallel_group(), op=torch.distributed.ReduceOp.AVG)
        if self.scaler is not None:
            scale = self.scaler[args.curr_iteration].to(device=device, dtype=dtype) 
        if self.config.flex_hetero_head:
            head_n = len(self.config.head_int_list)
            router_head_logits = router_head_logits.reshape(-1, head_n)
            router_head_logits = scale * router_head_logits
            router_head_logits = self._dp_gumbel_softmax(router_head_logits, tau=tau, hard=hard_sample, curr_iteration=args.curr_iteration)
            _, choices_head = torch.topk(router_head_logits, 1, dim=-1)
            return (router_head_logits, [self.config.head_int_list[i] for i in choices_head.flatten().tolist()])
        else:
            router_head_logits = scale * router_head_logits
            router_head_logits = self._dp_gumbel_softmax(router_head_logits, tau=tau, hard=hard_sample, curr_iteration=args.curr_iteration)
            _, choices_head = torch.topk(router_head_logits, 1, dim=-1)
            return (router_head_logits, self.config.head_int_list[choices_head.item()])

    def skipping_forward(self, args, budget_tensor, device, dtype, tau, hard_sample):

        # for layer skipping, skipping MLP layers
        router_skip_layer_logits1 = self.gate_skip_layer[0](budget_tensor)
        router_skip_layer_logits2 = self.gate_skip_layer[1](router_skip_layer_logits1[0])
        router_skip_layer_logits = self.gate_skip_layer[2](router_skip_layer_logits2)[0].flatten()
        # torch.distributed.all_reduce(router_skip_layer_logits, group=get_tensor_model_parallel_group(), op=torch.distributed.ReduceOp.AVG)
        router_skip_layer_logits = torch.repeat_interleave(router_skip_layer_logits, repeats=1, dim=0)
        if self.scaler is not None:
            router_skip_layer_logits = router_skip_layer_logits * self.scaler[args.curr_iteration].to(device=device, dtype=dtype)

        # router_skip_layer_logits = F.gumbel_softmax(router_skip_layer_logits, tau=tau, hard=hard_sample)
        router_skip_layer_logits = self._dp_gumbel_softmax(router_skip_layer_logits, tau=tau, hard=hard_sample, curr_iteration=args.curr_iteration)
        _, choices_skip_layer = torch.topk(router_skip_layer_logits, 1, dim=-1)
        if choices_skip_layer.item() != 0:
            selected_to_drop = self.config.layer_ranking_list[:choices_skip_layer.item()]
            choices_skip_layer = torch.zeros(self.config.num_layers).to(device=device, dtype=dtype)
            choices_skip_layer[selected_to_drop] = 1
        else:
            choices_skip_layer = torch.zeros(self.config.num_layers).to(device=device, dtype=dtype)
        return (router_skip_layer_logits, choices_skip_layer)


    def get_curr_tau(self, curr_iteration):
        tau = self.config.tau_init * torch.pow(torch.tensor(self.config.tau_decay), curr_iteration)
        return tau

    def add_scaler_schedule(self):

        if self.config.linear_scaler_start is not None and self.config.linear_scaler_end is not None:
            from megatron.training import get_args
            args = get_args()
            self.scaler = torch.linspace(
                start=self.config.linear_scaler_start, 
                end=self.config.linear_scaler_end, 
                steps=args.train_iters if args.train_iters is not None else (args.train_samples // args.global_batch_size))
        else:
            self.scaler = None

    def forward(self, budget):
        
        from megatron.training import get_args
        args = get_args()

        hard_sample = random.random() > self.hard_sample_th

        tau = self.get_curr_tau(args.curr_iteration)
        
        device, dtype = next(self.parameters()).device, next(self.parameters()).dtype

        if budget in self.budget_map.keys():
            budget_tensor = torch.nn.functional.one_hot(self.budget_map[budget], len(self.config.budget_list)).to(
                    device=device, dtype=dtype)
        elif budget == 1.0:
            budget_tensor = torch.nn.functional.one_hot(self.budget_map[list(self.budget_map.keys())[0]], 
            len(self.config.budget_list)).to(device=device, dtype=dtype)
        else:
            # budgets must be sorted ascending for bucketize
            budget_values = torch.tensor(sorted(self.config.budget_list), device=device, dtype=dtype)
            budget_t = torch.as_tensor(budget, device=device, dtype=dtype)

            # idx2 = first index where budget_values[idx] > budget (right=False gives >= behavior with floats)
            idx2 = torch.bucketize(budget_t, budget_values, right=False)
            # Clamp to valid interior so we always have a left neighbor
            idx2 = idx2.clamp(min=1, max=len(self.config.budget_list) - 1)
            idx1 = idx2 - 1

            b1 = budget_values.index_select(0, idx1.to(torch.long))
            b2 = budget_values.index_select(0, idx2.to(torch.long))
            denom = (b2 - b1)#.clamp_min(1e-12)
            weight = (budget_t - b1) / denom  # in [0,1] when budget is between b1 and b2

            num_classes = len(self.config.budget_list)
            one_hot_1 = torch.nn.functional.one_hot(idx1.to(torch.long), num_classes=num_classes).to(device=device, dtype=dtype)
            one_hot_2 = torch.nn.functional.one_hot(idx2.to(torch.long), num_classes=num_classes).to(device=device, dtype=dtype)

            # If weight is scalar, broadcasting works; if vector, it blends per-sample
            budget_tensor = (1 - weight).unsqueeze(-1) * one_hot_1 + weight.unsqueeze(-1) * one_hot_2
            budget_tensor = budget_tensor.squeeze(0).flip(0)

        budget_tensor = budget_tensor.unsqueeze(0)
        mlp_forward_outputs = self.mlp_forward(args, budget_tensor, device, dtype, tau, hard_sample)
        mamba_forward_outputs = self.mamba_forward(args, budget_tensor, device, dtype, tau, hard_sample)
        moe_expert_forward_outputs = self.moe_expert_forward(args, budget_tensor, device, dtype, tau, hard_sample)

        if self.config.add_skipping:
            skipping_forward_outputs = self.skipping_forward(args, budget_tensor, device, dtype, tau, hard_sample)
        else:
            skipping_forward_outputs = None
        
        emb_forward_outputs = self.emb_forward(args, budget_tensor, device, dtype, tau, hard_sample)
        head_forward_outputs = self.head_forward(args, budget_tensor, device, dtype, tau, hard_sample)
        self.fwd_pass_count += 1
        return mlp_forward_outputs, skipping_forward_outputs, emb_forward_outputs, mamba_forward_outputs, head_forward_outputs, moe_expert_forward_outputs
