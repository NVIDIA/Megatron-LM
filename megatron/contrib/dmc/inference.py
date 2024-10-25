# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import torch


class DMCPagedKVCache:

    def __init__(self, B, H, D, layers, block_size=256, max_len=4096, cache_size=20.0):
        kw_bf16 = {'dtype': torch.bfloat16, 'device': torch.cuda.current_device()}
        kw_int = {'dtype': torch.int, 'device': torch.cuda.current_device()}
        self.H = H
        self.B = B

        self.block_size = block_size
        self.num_layers = layers
        max_blocks_per_head = (max_len + block_size - 1) // block_size
        # GiB -> B -> num_blocks
        # *2 bytes in BF16 and *2 for K/V
        self.num_all_blocks = int((cache_size * 1024**3) // (D * block_size * 2 * 2))

        self.k_cache = torch.empty(self.num_all_blocks, block_size, 1, D, **kw_bf16)
        self.v_cache = torch.empty(self.num_all_blocks, block_size, 1, D, **kw_bf16)
        self.block_table = torch.full((layers * B * H, max_blocks_per_head), **kw_int,
                                      fill_value=layers*B*H*max_blocks_per_head-1)

        self.head_alloc = torch.full((layers * B * H,), fill_value=block_size, **kw_int)
        self.block_table[:, 0] = torch.arange(layers * B * H, **kw_int)
        self.used_blocks = layers * B * H

    def get_block_table(self, layer_idx):
        return self.block_table[layer_idx*self.B*self.H:(layer_idx+1)*self.B*self.H]

    def allocate_blocks(self, dmc_params_dict, layer_idx=None):
        if layer_idx is not None:
            head_lens = dmc_params_dict[layer_idx+1].lens[0].view(-1) + 1
            head_alloc = self.head_alloc[layer_idx*self.B*self.H:(layer_idx+1)*self.B*self.H]
            block_table = self.get_block_table(layer_idx)
        else:
            # Collect head lens from all layers and add 1 to get actual length
            lens_all_layers = [dmc_params.lens[0] for dmc_params in dmc_params_dict.values()]
            head_lens = torch.stack(lens_all_layers, dim=0)
            head_lens = head_lens.view(-1) + 1
            head_alloc = self.head_alloc
            block_table = self.block_table

        inds = torch.where(head_alloc - head_lens == 0)[0]
        new_block_pos = head_lens[inds] // self.block_size
        new_block_ids = (
            torch.arange(inds.size(0), device=inds.device, dtype=torch.int)
            + self.used_blocks
        )
        block_table[inds, new_block_pos] = new_block_ids
        head_alloc[inds] += self.block_size
        self.used_blocks += inds.size(0)
        assert self.used_blocks <= self.num_all_blocks


class InferencePoolingParamsTriton:
    """DMC state container optimized for OpenAI Triton state update fun"""
    def __init__(self, B, H, D, pooling_window_size=12, precision=torch.float16):
        kw_int = {'device': torch.cuda.current_device(), 'dtype': torch.int32}
        kw_fp = {'device': torch.cuda.current_device(), 'dtype': precision}
        self.kw_int = kw_int
        self.kw_fp = kw_fp
        # First elem serves as "kv_sum"
        self.kv_win = torch.zeros(pooling_window_size + 1, 2, B, H, D, **kw_fp)
        self.w_win = torch.ones(pooling_window_size + 1, B, H, 1, **kw_fp)
        self.w_win[0] = 0
        self.lens = torch.zeros(2, B, H, **kw_int)  # [total_lens, head_lens]
        self.lens[0] = -1  # Keep (lens - 1) for later computation of flashattn
        self.win_sz = pooling_window_size
        self.win_ptr = 0
        self.num_pooled = 0
