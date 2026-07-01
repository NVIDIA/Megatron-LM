# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""
Utility helpers for broadcasting nested dictionaries of tensors across tensor-parallel ranks.

"""

from typing import Any, Dict, List, Tuple, Union, Optional
import math
from megatron.core.models.bagel.mot_packed_seq_params import MoTPackedSeqParams
from megatron.core import parallel_state

import torch

from megatron.core import mpu 
from megatron.core.tensor_parallel.data import _build_key_size_numel_dictionaries, _check_data_types

# Special suffixes to mark original types
_LIST_MARKER = "__was_list__"
_INT_MARKER = "__was_int__"
_FLOAT_MARKER = "__was_float__"
_BOOL_MARKER = "__was_bool__"

# Types that should be skipped during tensor broadcast (they'll be broadcast via object_list)
_SKIP_TENSOR_BROADCAST_TYPES = (type(None),)

# Try to import BlockMask for flex attention (optional)
try:
    from torch.nn.attention.flex_attention import BlockMask
    _SKIP_TENSOR_BROADCAST_TYPES = (type(None), BlockMask)
except ImportError:
    BlockMask = None



def flatten(
    nested: Dict[str, Any], prefix: Tuple[str, ...] = ()
) -> Tuple[List[Tuple[Tuple[str, ...], torch.Tensor]], Dict[str, Any]]:
    """Recursively flatten nested dict into [(key_path, tensor), …] and non-tensor items.
    
    Returns:
        Tuple of (flat_tensors, non_tensor_items)
        - flat_tensors: List of (path, tensor) tuples for tensorizable items
        - non_tensor_items: Dict of items that cannot be converted to tensors
    """
    flat = []
    non_tensor = {}
    
    for k, v in nested.items():
        path = prefix + (k,)
        path_str = ".".join(path)
        
        if isinstance(v, dict):
            sub_flat, sub_non_tensor = flatten(v, path)
            flat.extend(sub_flat)
            non_tensor.update(sub_non_tensor)
        elif isinstance(v, torch.Tensor):
            flat.append((path, v))
        elif isinstance(v, list):
            # Convert list to tensor and mark with special suffix
            try:
                tensor_v = torch.tensor(v)
                marked_path = prefix + (k + _LIST_MARKER,)
                flat.append((marked_path, tensor_v))
            except (ValueError, TypeError):
                # List contains non-numeric items, store as non-tensor
                non_tensor[path_str] = v
        elif isinstance(v, int):
            # Convert int to tensor and mark
            tensor_v = torch.tensor([v], dtype=torch.long)
            marked_path = prefix + (k + _INT_MARKER,)
            flat.append((marked_path, tensor_v))
        elif isinstance(v, float):
            # Convert float to tensor and mark
            tensor_v = torch.tensor([v], dtype=torch.float32)
            marked_path = prefix + (k + _FLOAT_MARKER,)
            flat.append((marked_path, tensor_v))
        elif isinstance(v, bool):
            # Convert bool to tensor and mark (must check before int since bool is subclass of int)
            tensor_v = torch.tensor([v], dtype=torch.bool)
            marked_path = prefix + (k + _BOOL_MARKER,)
            flat.append((marked_path, tensor_v))
        elif isinstance(v, _SKIP_TENSOR_BROADCAST_TYPES):
            # These types will be broadcast via object_list
            non_tensor[path_str] = v
        else:
            # Unknown type - try to handle, or store as non-tensor
            non_tensor[path_str] = v
    
    return flat, non_tensor


def regroup(flat: List[Tuple[Tuple[str, ...], torch.Tensor]], non_tensor: Dict[str, Any]) -> Dict[str, Any]:
    """Rebuild the nested dict from [(key_path, tensor), …] and non-tensor items."""
    root = {}
    
    # First, add all tensor items
    for path, tensor in flat:
        cur = root
        for k in path[:-1]:
            cur = cur.setdefault(k, {})
        final_key = path[-1]
        
        # Check for type markers and convert back
        if final_key.endswith(_LIST_MARKER):
            final_key = final_key[:-len(_LIST_MARKER)]
            cur[final_key] = tensor.tolist()
        elif final_key.endswith(_INT_MARKER):
            final_key = final_key[:-len(_INT_MARKER)]
            cur[final_key] = tensor.item()
        elif final_key.endswith(_FLOAT_MARKER):
            final_key = final_key[:-len(_FLOAT_MARKER)]
            cur[final_key] = tensor.item()
        elif final_key.endswith(_BOOL_MARKER):
            final_key = final_key[:-len(_BOOL_MARKER)]
            cur[final_key] = tensor.item()
        else:
            cur[final_key] = tensor
    
    # Then, add all non-tensor items
    for path_str, value in non_tensor.items():
        path = path_str.split(".")
        cur = root
        for k in path[:-1]:
            cur = cur.setdefault(k, {})
        cur[path[-1]] = value
    
    return root

def broadcast_data(keys, data, datatype, group):
    """Broadcast data from rank zero of each model parallel group to the
    members of the same model parallel group.

    Args:
        keys: list of keys in the data disctionary to be broadcasted
        data: data dictionary of string keys and cpu tensor values.
        datatype: torch data type of all tensors in data associated
                  with keys.
        tp_group: the tensor model parallel group to broadcast to.
    """
    assert group is not None, "group must be provided"
    # Build (key, size) and (key, number of elements) dictionaries along
    # with the total number of elements on all ranks.
    key_size, key_numel, total_numel = _build_key_size_numel_dictionaries(keys, data, group)
    # Pack on rank zero.
    if group.rank() == 0:
        # Check that all keys have the same data type.
        _check_data_types(keys, data, datatype)
        # Flatten the data associated with the keys
        flatten_data = torch.cat([data[key].cuda().contiguous().view(-1) for key in keys], dim=0)
    else:
        flatten_data = torch.empty(total_numel, device=torch.cuda.current_device(), dtype=datatype)

    # Broadcast
    group_ranks = torch.distributed.get_process_group_ranks(group=group)
    torch.distributed.broadcast(flatten_data, group_ranks[0], group=group)

    # Unpack
    output = {}
    offset = 0
    for key in keys:
        size = key_size[key]
        numel = key_numel[key]
        output[key] = flatten_data.narrow(0, offset, numel).view(size)
        offset += numel

    return output


def broadcast_nested_data_batch(nested_dict: Dict[str, Any], group='tp') -> Dict[str, Any]:
    """Recursively broadcast nested dictionaries of tensors using each tensor's own dtype.
    
    Handles:
    - Tensors: broadcast via tensor_parallel.broadcast_data
    - Lists: converted to tensors, broadcast, converted back
    - Scalars (int, float, bool): converted to tensors, broadcast, converted back
    - Other types (None, BlockMask, etc.): broadcast via object_list
    """
    assert group in ('tp', 'cp'), "group must be 'tp' or 'cp'"

    if group == 'tp':
        src = mpu.get_tensor_model_parallel_src_rank()
        broadcast_group = mpu.get_tensor_model_parallel_group()
    elif group == 'cp':
        src = mpu.get_context_parallel_global_ranks()[0]
        broadcast_group = mpu.get_context_parallel_group()
    else:
        raise ValueError(f"Invalid group: {group}")

    # ---------- rank-0 prepares metadata ----------
    # if mpu.get_tensor_model_parallel_rank() == 0:
    if broadcast_group.rank() == 0:
        flat, non_tensor = flatten(nested_dict)
        paths, tensors = zip(*flat) if flat else ([], [])
        dtypes = [t.dtype for t in tensors]
    else:
        paths, dtypes = [], []
        tensors = []
        non_tensor = {}

    # ---------- 1. broadcast schema (paths + dtypes + non_tensor keys) ----------
    meta = [paths, dtypes, non_tensor]
    obj_list = [meta]
    torch.distributed.broadcast_object_list(obj_list, src=src, group=broadcast_group)
    paths, dtypes, non_tensor = obj_list[0]

    # ---------- 2. group tensors by dtype and broadcast ----------
    dtype_to_keys = {}
    for p, dt in zip(paths, dtypes):
        dtype_to_keys.setdefault(dt, []).append(".".join(p))

    # On src rank: make a dict {joined_path: tensor}
    # if mpu.get_tensor_model_parallel_rank() == 0:
    if broadcast_group.rank() == 0:
        data_dict = {".".join(p): t.cuda() for p, t in zip(paths, tensors)}
    else:
        data_dict = {}

    flat_out = []
    for dt, keys in dtype_to_keys.items():
        out = broadcast_data(keys, data_dict, dt, broadcast_group)
        flat_out.extend([(tuple(k.split(".")), out[k]) for k in keys])

    # ---------- 3. rebuild nested structure ----------
    return regroup(flat_out, non_tensor)

from megatron.core.models.bagel import gather_pad_to_length  # noqa: E402,F401


def get_packed_seq_params(
        # packed_text_indexes: torch.Tensor,
        # packed_vit_token_indexes: Optional[torch.Tensor],
        # packed_vae_token_indexes: Optional[torch.Tensor],
        mimo_batch: dict,
    ) -> MoTPackedSeqParams:
        """Build MoTPackedSeqParams with CP-aware local index slices.

        For CP=1 local == global; for CP=N each rank receives ceil(U/N) und tokens
        and ceil(G/N) gen tokens.  The last rank may have fewer real tokens.
        local_*_token_indexes stores the actual (unpadded) slice; len() gives the
        real token count.  Callers must pad tensors to padded_*_seqlen for A2A.

        Args:
            packed_text_indexes: Global positions of text tokens  [T]
            packed_vit_token_indexes: Global positions of ViT tokens  [V] or None
            packed_vae_token_indexes: Global positions of VAE tokens  [G] or None

        Returns:
            MoTPackedSeqParams with global und/gen indexes and per-rank local slices.
        """
        packed_text_indexes = mimo_batch['packed_text_indexes'].cuda()
        packed_vit_token_indexes = mimo_batch.get('packed_vit_token_indexes', None)
        if packed_vit_token_indexes is not None:
            packed_vit_token_indexes = packed_vit_token_indexes.cuda()
        packed_vae_token_indexes = mimo_batch.get('packed_vae_token_indexes', None)
        if packed_vae_token_indexes is not None:
            packed_vae_token_indexes = packed_vae_token_indexes.cuda()

        torch.cuda.synchronize()
        device = packed_text_indexes.device

        # Build global und/gen index arrays
        und_idx = packed_text_indexes
        if packed_vit_token_indexes is not None:
            und_idx = torch.cat([und_idx, packed_vit_token_indexes.to(device)], dim=0)
        gen_idx = (
            packed_vae_token_indexes.to(device)
            if packed_vae_token_indexes is not None
            else torch.zeros(0, dtype=torch.long, device=device)
        )

        U, G = len(und_idx), len(gen_idx)
        cp_size = parallel_state.get_context_parallel_world_size()
        cp_rank = parallel_state.get_context_parallel_rank()

        Lund = math.ceil(U / cp_size) if U > 0 else 0
        Lgen = math.ceil(G / cp_size) if G > 0 else 0

        # Actual (unpadded) per-rank slices; len() gives real token count on this rank
        actual_lund = min(Lund, max(0, U - cp_rank * Lund))
        actual_lgen = min(Lgen, max(0, G - cp_rank * Lgen))
        local_und_idx = und_idx[cp_rank * Lund : cp_rank * Lund + actual_lund]
        local_gen_idx = gen_idx[cp_rank * Lgen : cp_rank * Lgen + actual_lgen]

        mimo_batch['packed_seq_params'] = MoTPackedSeqParams(
            packed_text_indexes=packed_text_indexes,
            packed_vit_token_indexes=packed_vit_token_indexes,
            packed_vae_token_indexes=packed_vae_token_indexes,
            packed_und_token_indexes=und_idx,
            packed_gen_token_indexes=gen_idx,
            local_und_token_indexes=local_und_idx,
            local_gen_token_indexes=local_gen_idx,
            padded_und_seqlen=Lund,
            padded_gen_seqlen=Lgen,
            vit_tokens_encoded_per_cp=None,
        )
        return mimo_batch

def bagel_process_gen_data(
    batch_dict: dict,
    diffusion_wrapper,
    *,
    run_vae_encode: bool = True,
) -> tuple:
    """Process visual generation data: VAE-encode images, add diffusion noise.

    Args:
        batch_dict: Packed batch dict from ``BagelPacker.to_tensor``.
        diffusion_wrapper: DiffusionWrapper with ``shift_timesteps``,
            ``vae_encode``, and ``add_noise`` methods.
        run_vae_encode: When False, skip ``vae_encode`` + ``add_noise`` and
            emit zero-length ``[0, c*p^2]`` placeholders for ``latents`` and
            ``vis_gen_target``. The diffusion modality submodule (which
            consumes ``latents``) only runs on the first PP stage and the
            MSE loss (which consumes ``vis_gen_target``) only fires on the
            last PP stage; middle stages (PP>=3) discard both, so the
            tensors are unused after dataloading. The VAE itself runs on
            every rank that calls this function, so skipping saves real
            compute. Default True keeps PP=1 / PP=2 behaviour exact.

    Returns:
        Tuple of ``(loss_inputs, modality_inputs)`` dicts.
    """
    assert diffusion_wrapper is not None, "diffusion_wrapper must be provided"

    loss_inputs: dict = {}
    modality_inputs: dict = {}

    if "packed_timesteps" in batch_dict:
        packed_timesteps = batch_dict['packed_timesteps'].cuda()
        shifted_timesteps = diffusion_wrapper.shift_timesteps(packed_timesteps)
        shifted_timesteps.requires_grad = True  # for fsdp backward hook
        modality_inputs['shifted_timesteps'] = shifted_timesteps
        loss_inputs['mse_loss_indexes'] = batch_dict['mse_loss_indexes']

    loss_inputs['packed_vae_token_indexes'] = batch_dict['packed_vae_token_indexes']

    if run_vae_encode:
        padded_images = batch_dict['padded_images'].cuda()
        latents = diffusion_wrapper.vae_encode(padded_images, batch_dict['patchified_vae_latent_shapes'])

        if 'packed_timesteps' in batch_dict:
            latents, _, target = diffusion_wrapper.add_noise(latents, shifted_timesteps)
            loss_inputs['vis_gen_target'] = target

        modality_inputs['latents'] = latents

    # ``latent_position_ids`` is dataset-derived and always emitted; needed
    # to keep ``modality_inputs['diffusion']`` non-empty so the schema
    # broadcast within the TP/CP group sees a consistent dict structure.
    modality_inputs['latent_position_ids'] = batch_dict['packed_latent_position_ids']

    # When ``run_vae_encode`` is False, ``latents`` and ``vis_gen_target``
    # are simply absent from the returned dicts. Both are consumed only on
    # PP boundary stages (latents on first via the diffusion submodule;
    # vis_gen_target on last via the MSE head) and ``shard_data_for_cp``
    # gracefully handles vis_gen_target being absent (falls through to the
    # zero-fill ``elif vae_dim is not None`` branch). Skipping the keys —
    # rather than emitting 0-size placeholders — also keeps the TP/CP
    # broadcast happy: ``_build_key_size_numel_dictionaries`` encodes
    # tensor shapes with a 0 sentinel and miscounts numel for any tensor
    # whose first dim is 0, which corrupts the receive-side narrow.

    return loss_inputs, modality_inputs


def shard_data_for_cp(mimo_batch, *, cp_group, vae_dim=None):
    """Shard the mimo batch for context parallelism using type-balanced sharding.

    Und tokens (text + ViT) and gen tokens (VAE/diffusion) are sharded
    independently so every CP rank processes exactly Lund = ceil(U/cp_size) und
    tokens and Lgen = ceil(G/cp_size) gen tokens.

    Tensors sharded here:
    - ``labels [S]``      → indexed by ``local_und_idx`` → ``[1, actual_lund]``
    - ``loss_mask [S]``   → indexed by ``local_und_idx`` → ``[1, actual_lund]``
    - ``packed_position_ids [S]`` → compact cat of und + gen local positions
    - diffusion modality inputs (latents, latent_position_ids, shifted_timesteps)
      → contiguous ``[Lgen]`` slice
    - ``vis_gen_target [L_mse, C·p²]`` → [G] → ``[Lgen, C·p²]`` slice  (dense CP form)
    - ``gen_loss_mask [L_mse]`` → [G] → ``[Lgen]`` slice               (dense CP form)

    Tensors NOT sharded (kept intact on all ranks):
    - ``input_ids``, ``position_ids`` — all ranks compute full text embeddings
    - ``packed_text_indexes``, ``packed_vit_token_indexes``, ``packed_vae_token_indexes``
      — global index arrays; ``BagelMimoModel.get_packed_seq_params()`` recomputes
      local slices from these at model-forward time

    Args:
        mimo_batch: Full MIMO batch dict from ``bagel_packed_batch_to_mimo_batch``.
        cp_group: Context-parallel ``torch.distributed`` process group. Rank and
            world size are derived from this group.

    Returns:
        Sharded MIMO batch dict for this CP rank.
    """
    cp_rank = torch.distributed.get_rank(cp_group)
    cp_size = torch.distributed.get_world_size(cp_group)
    # ------------------------------------------------------------------ #
    # 1. Compute type-balanced sharding indices                           #
    # ------------------------------------------------------------------ #
    packed_text_indexes = mimo_batch['packed_text_indexes']
    packed_vit_token_indexes = mimo_batch.get(
        'packed_vit_token_indexes', torch.empty(0, dtype=torch.long)
    )
    packed_vae_token_indexes = mimo_batch.get(
        'packed_vae_token_indexes', torch.empty(0, dtype=torch.long)
    )

    # und = text + ViT (no sort; Ulysses A2A is order-agnostic for load balance)
    und_idx = torch.cat([packed_text_indexes, packed_vit_token_indexes])  # [U]
    gen_idx = packed_vae_token_indexes                                     # [G]

    U = len(und_idx)
    G = len(gen_idx)
    Lund = math.ceil(U / cp_size)
    Lgen = math.ceil(G / cp_size) if G > 0 else 0

    und_start = cp_rank * Lund
    und_end   = min((cp_rank + 1) * Lund, U)
    gen_start = cp_rank * Lgen
    gen_end   = min((cp_rank + 1) * Lgen, G)

    local_und_idx = und_idx[und_start:und_end]  # [actual_lund]
    local_gen_idx = gen_idx[gen_start:gen_end]  # [actual_lgen]

    sharded = dict(mimo_batch)

    # ------------------------------------------------------------------ #
    # 2. Shard und-indexed loss tensors                                   #
    # ------------------------------------------------------------------ #
    if mimo_batch.get('labels') is not None:
        labels_full = mimo_batch['labels']  # [S]
        sharded['labels'] = labels_full[local_und_idx]  # [actual_lund]
    else:
        # No text supervision — provide ignore-index labels so output_layer always
        # participates in backward (FSDP correctness). loss_mask=0 so CE contributes 0.
        sharded['labels'] = torch.full((len(local_und_idx),), fill_value=-100, dtype=torch.long)

    if mimo_batch.get('loss_mask') is not None:
        loss_mask_full = mimo_batch['loss_mask']  # [S]
        sharded['loss_mask'] = loss_mask_full[local_und_idx]  # [actual_lund]
    else:
        sharded['loss_mask'] = torch.zeros(len(local_und_idx), dtype=torch.float)

    # ------------------------------------------------------------------ #
    # 3. Compact packed_position_ids for local tokens                     #
    # ------------------------------------------------------------------ #
    if 'packed_position_ids' in mimo_batch:
        packed_pos = mimo_batch['packed_position_ids']  # [S]
        sharded['packed_position_ids'] = torch.cat(
            [gather_pad_to_length(packed_pos, local_und_idx, Lund),
            gather_pad_to_length(packed_pos, local_gen_idx, Lgen)], dim=0)  # [Lund + Lgen]

    # we don't shard the diffusion modality inputs, because we need to reconstruct the full sequence in bagel mimo
    # ------------------------------------------------------------------ #
    # 4. Shard gen diffusion modality inputs                              #
    # ------------------------------------------------------------------ #
    # modality_inputs = mimo_batch.get('modality_inputs', {})
    # if 'diffusion' in modality_inputs:
    #     diffusion = modality_inputs['diffusion']
    #     sharded_diffusion = {}
    #     for key in ('latents', 'latent_position_ids', 'shifted_timesteps'):
    #         if key in diffusion:
    #             sharded_diffusion[key] = diffusion[key][gen_start:gen_end]
    #     sharded['modality_inputs'] = dict(modality_inputs)
    #     sharded['modality_inputs']['diffusion'] = sharded_diffusion

    # ------------------------------------------------------------------ #
    # 5. Shard dense CP loss tensors (Gap 2 — full-seq forms)             #
    # ------------------------------------------------------------------ #
    # vis_gen_target: should be dense [G, C·p²] form expected for CP .
    if 'vis_gen_target' in mimo_batch and mimo_batch['vis_gen_target'] is not None:
        # 
        assert 'mse_loss_indexes' in mimo_batch, "mse_loss_indexes must be in mimo_batch"
        mse_loss_indexes = mimo_batch['mse_loss_indexes']
        target = mimo_batch['vis_gen_target'] #[L_mse, C·p²]
        # mse_loss_mask[i] == 1 iff the i-th VAE token (in packed_vae_token_indexes) has MSE loss
        mse_loss_mask = torch.isin(mimo_batch['packed_vae_token_indexes'], mse_loss_indexes).float() #[G]
        # Expand target from [L_mse, C·p²] to dense [G, C·p²] using the boolean mask
        target_full = torch.zeros(G, target.shape[1], dtype=target.dtype, device=target.device)
        target_full[mse_loss_mask.bool()] = target
        sharded['vis_gen_target'] = target_full[gen_start:gen_end]
        sharded['gen_loss_mask'] = mse_loss_mask[gen_start:gen_end]
    elif vae_dim is not None:
        # No MSE targets but diffusion branch exists — provide zero-weight tensors so
        # llm2vae always participates in backward (FSDP correctness).
        G_local = gen_end - gen_start
        sharded['gen_loss_mask'] = torch.zeros(G_local, dtype=torch.float)
        sharded['vis_gen_target'] = torch.zeros(G_local, vae_dim, dtype=torch.float)

    # ------------------------------------------------------------------ #
    # 6. Shard ViT images by token-count-balanced image boundaries        #
    # ------------------------------------------------------------------ #
    # do nothing for vit on batch slice, since there would no enough images in bagel data.
    # Images are assigned to CP ranks in contiguous whole-image slices.
    # cur_modality = sharded.get('modality_inputs', mimo_batch.get('modality_inputs', {}))
    # if cur_modality and 'images' in cur_modality:
    #     vision_encoder = cur_modality['images'].get('vision_encoder', {})
    #     vit_token_seqlens = vision_encoder.get('vit_token_seqlens')
    #     if vit_token_seqlens is not None and len(vit_token_seqlens) > 0:
    #         N_img = len(vit_token_seqlens)

    #         img_per_cp = math.ceil(N_img / cp_size)
    #         # prefix[i] = total tokens for images 0..i-1  (prefix[0]=0, prefix[N_img]=V)
    #         accum_seqlen = torch.zeros(N_img + 1, dtype=torch.long, device=vit_token_seqlens.device)
    #         accum_seqlen[1:] = torch.cumsum(vit_token_seqlens, dim=0)

    #         vit_tokens_encoded_per_cp = []
    #         for i in range(cp_size):
    #             img_start = i * img_per_cp
    #             img_end = min((i + 1) * img_per_cp, N_img)
    #             tok_start = accum_seqlen[i * img_per_cp]
    #             tok_end = accum_seqlen[min((i + 1) * img_per_cp, N_img)]
    #             vit_tokens_encoded_per_cp.append(tok_end- tok_start)
    #             if i == cp_rank:
    #                 # get cp local vit packed_tokens
    #                 sharded_ve = {
    #                     'packed_vit_tokens':      vision_encoder['packed_vit_tokens'][tok_start:tok_end],
    #                     'vit_token_seqlens':      vit_token_seqlens[img_start:img_end],
    #                     'packed_vit_position_ids': vision_encoder['packed_vit_position_ids'][tok_start:tok_end],
    #                 }
    #         # Ensure modality_inputs is a fresh copy before mutating
    #         if 'modality_inputs' not in sharded or sharded['modality_inputs'] is mimo_batch.get('modality_inputs'):
    #             sharded['modality_inputs'] = dict(cur_modality)
    #         sharded['modality_inputs']['images'] = dict(cur_modality['images'])
    #         sharded['modality_inputs']['images']['vision_encoder'] = sharded_ve

    #         sharded['packed_seq_params'].vit_tokens_encoded_per_cp = vit_tokens_encoded_per_cp

    for _key in ('mse_loss_indexes', 'packed_vae_token_indexes',
                 'packed_text_indexes', 'packed_vit_token_indexes'):
        sharded.pop(_key, None)

    # Update packed_seq_params to reflect this CP rank's local token slices.
    # packed_seq_params is broadcast from CP rank 0 as an object (non-tensor),
    # so all ranks initially hold the rank-0 local_*_token_indexes.  Each rank
    # must replace the local slices with its own.
    if 'packed_seq_params' in sharded and sharded['packed_seq_params'] is not None:
        old_params = sharded['packed_seq_params']
        sharded['packed_seq_params'] = MoTPackedSeqParams(
            packed_text_indexes=old_params.packed_text_indexes,
            packed_vit_token_indexes=old_params.packed_vit_token_indexes,
            packed_vae_token_indexes=old_params.packed_vae_token_indexes,
            packed_und_token_indexes=old_params.packed_und_token_indexes,
            packed_gen_token_indexes=old_params.packed_gen_token_indexes,
            local_und_token_indexes=local_und_idx,
            local_gen_token_indexes=local_gen_idx,
            padded_und_seqlen=Lund,
            padded_gen_seqlen=Lgen,
            vit_tokens_encoded_per_cp=old_params.vit_tokens_encoded_per_cp,
        )

    return sharded


def bagel_packed_batch_to_mimo_batch(
    packed_batch,
    diffusion_wrapper=None,
    *,
    run_vae_encode: bool = True,
) -> dict:
    """Convert a packed batch dict (from ``BagelPacker.to_tensor``) to MIMO model input format.

    Args:
        packed_batch: Dict or SimpleCustomBatch from the packer / PackedDataset.
        diffusion_wrapper: Optional DiffusionWrapper. When provided the visual
            generation branch is processed (VAE encode + noise addition).
        run_vae_encode: Forwarded to ``bagel_process_gen_data``. When False,
            VAE encode + add_noise are skipped and zero-length placeholders
            are emitted for ``latents`` and ``vis_gen_target``. Caller (PP
            ``get_batch``) sets this to False on PP middle stages where the
            diffusion modality submodule and MSE loss never run, saving VAE
            compute.

    Returns:
        Dict with keys expected by the BAGEL MIMO model forward pass.
    """
    batch_dict = packed_batch.to_dict() if hasattr(packed_batch, 'to_dict') else packed_batch

    seq_len = batch_dict['sequence_length']

    input_ids = batch_dict['packed_text_ids'].unsqueeze(0)  # (1, num_text_tokens)
    text_seq_len = input_ids.shape[1]
    position_ids = torch.arange(text_seq_len, dtype=torch.long).unsqueeze(0)

    if 'packed_label_ids' in batch_dict:
        labels = torch.full((1, seq_len), fill_value=-100, dtype=torch.long)
        ce_loss_indexes = batch_dict['ce_loss_indexes']
        labels[:, ce_loss_indexes] = batch_dict['packed_label_ids']
        loss_mask = torch.zeros((1, seq_len), dtype=torch.float)
        loss_mask[:, ce_loss_indexes] = torch.tensor(batch_dict['ce_loss_weights'])
    else:
        labels = None
        loss_mask = None

    mimo_batch = {
        'input_ids': input_ids,
        'position_ids': position_ids,
        'labels': labels,
        'loss_mask': loss_mask,
    }

    modality_inputs = {}

    if 'packed_vit_tokens' in batch_dict:
        modality_inputs['images'] = {
            'vision_encoder': {
                'packed_vit_tokens': batch_dict['packed_vit_tokens'],
                'vit_token_seqlens': batch_dict['vit_token_seqlens'],
                'packed_vit_position_ids': batch_dict['packed_vit_position_ids'],
            }
        }

    if 'packed_vae_token_indexes' in batch_dict:
        vis_gen_loss_inputs, vis_gen_modality_inputs = bagel_process_gen_data(
            batch_dict, diffusion_wrapper, run_vae_encode=run_vae_encode,
        )
        modality_inputs['diffusion'] = vis_gen_modality_inputs
        mimo_batch.update(vis_gen_loss_inputs)
    else:
        print("packed_vae_token_indexes not in batch_dict, skipping visual gen data")

    if modality_inputs:
        mimo_batch['modality_inputs'] = modality_inputs

    mimo_batch['sample_lens'] = batch_dict['sample_lens']
    mimo_batch['sequence_length'] = seq_len
    mimo_batch['packed_position_ids'] = batch_dict['packed_position_ids']
    mimo_batch['packed_text_indexes'] = batch_dict['packed_text_indexes']

    if 'packed_vit_token_indexes' in batch_dict:
        mimo_batch['packed_vit_token_indexes'] = batch_dict['packed_vit_token_indexes']
    else:
        print("packed_vit_token_indexes not in batch_dict")


    # calculate full label and full loss mask
    if 'ce_loss_indexes' in batch_dict:
        ce_loss_indexes= batch_dict['ce_loss_indexes'] #[L_ce]
        assert 'packed_label_ids' in batch_dict, "packed_label_ids must be in batch_dict"
        packed_label_ids = batch_dict['packed_label_ids'] #[L_ce]
        labels_full = torch.zeros(seq_len, dtype=torch.long, device=input_ids.device)
        loss_mask_full = torch.zeros(seq_len, dtype=torch.float, device=input_ids.device)
        labels_full[ce_loss_indexes] = packed_label_ids
        loss_mask_full[ce_loss_indexes] = batch_dict['ce_loss_weights']
        mimo_batch['labels'] = labels_full
        mimo_batch['loss_mask'] = loss_mask_full


    if 'split_lens' in batch_dict:
        mimo_batch['split_lens'] = batch_dict['split_lens']
    if 'attn_modes' in batch_dict:
        mimo_batch['attn_modes'] = batch_dict['attn_modes']

    
    # mimo_batch['packed_seq_params'] = get_packed_seq_params(
    #     packed_text_indexes=mimo_batch['packed_text_indexes'],
    #     packed_vit_token_indexes=mimo_batch.get('packed_vit_token_indexes'),
    #     packed_vae_token_indexes=mimo_batch.get('packed_vae_token_indexes'),
    # )



    return mimo_batch