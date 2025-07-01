import torch
import triton
import triton.language as tl

# grouped forward
@triton.jit
def softmax_presoft_grouped_forward_kernel(
    # temp
    scores_temp_ptr,
    top_indices_temp_ptr,
    group_mask_temp_ptr,
    # input
    logits_ptr,
    # output
    topk_masked_gates_ptr,
    topk_map_ptr,
    top_indices_ptr,
    top_scores_ptr,
    # param
    num_experts: tl.constexpr,
    num_groups: tl.constexpr,
    experts_per_group: tl.constexpr,
    topk: tl.constexpr,
    group_topk: tl.constexpr,
    topk_per_group: tl.constexpr,
    scaling_factor: tl.constexpr,
    BLOCK_SIZE_NUM_EXPERTS: tl.constexpr,
    BLOCK_SIZE_NUM_GROUPS: tl.constexpr,
    BLOCK_SIZE_EXPERTS_PER_GROUP: tl.constexpr,
    BLOCK_SIZE_TOPK: tl.constexpr,
    BLOCK_SIZE_GROUP_TOPK: tl.constexpr,
    BLOCK_SIZE_TOPK_PER_GROUP: tl.constexpr
):
    pid = tl.program_id(axis=0)

    # offs & mask
    expert_offs = tl.arange(0, BLOCK_SIZE_NUM_EXPERTS)
    expert_mask = expert_offs < num_experts

    group_offs = tl.arange(0, BLOCK_SIZE_NUM_GROUPS)

    group_topk_offs = tl.arange(0, BLOCK_SIZE_GROUP_TOPK)
    group_topk_mask = group_topk_offs < group_topk

    topk_offs = tl.arange(0, BLOCK_SIZE_TOPK)
    topk_mask = topk_offs < topk

    # load input
    logits = tl.load(logits_ptr + pid * num_experts + expert_offs, mask=expert_mask, other=float('-inf')) # other=0 leads to wrong softmax

    # compute scores
    scores = tl.softmax(logits.to(tl.float32)).to(logits_ptr.dtype.element_ty)
    tl.store(scores_temp_ptr + pid * num_experts + expert_offs, scores, mask=expert_mask)

    # compute group_scores
    # group_view = tl.reshape(scores, num_groups, experts_per_group) # only if num_groups & experts_per_group are "power_of_2"
    
    tl.debug_barrier()

    offs_m = tl.arange(0, BLOCK_SIZE_NUM_GROUPS)[:, None]
    offs_n = tl.arange(0, BLOCK_SIZE_EXPERTS_PER_GROUP)[None, :]
    indices = offs_m * experts_per_group + offs_n
    mask = (offs_m < num_groups) & (offs_n < experts_per_group)

    group_view = tl.load(scores_temp_ptr + pid * num_experts + indices, mask=mask)

    # get top in group
    sorted_group_view = tl.sort(group_view, descending=True)
    group_top_mask = (offs_m < num_groups) & (offs_n < topk_per_group)
    group_top_values = tl.where(group_top_mask, sorted_group_view, 0.0)

    group_scores = tl.sum(group_top_values, axis=-1)

    # compute group_idx (topk)
    data = group_scores
    for i in range(group_topk):
        max_idx = tl.argmax(data, axis=0)
        tl.store(top_indices_temp_ptr + pid * group_topk + i, max_idx)
        data = tl.where(group_offs == max_idx, -float('inf'), data)
    tl.debug_barrier()
    group_idx = tl.load(top_indices_temp_ptr + pid * group_topk + group_topk_offs, mask=group_topk_mask)
    
    # compute group_mask
    ones = tl.full([BLOCK_SIZE_GROUP_TOPK], 1, logits_ptr.dtype.element_ty)
    tl.store(group_mask_temp_ptr + pid * num_groups + group_idx, ones, mask=group_topk_mask)

    tl.debug_barrier()

    # compute score_mask
    expert_group_idx = expert_offs // experts_per_group # mapping experts to group_id
    score_mask = tl.load(group_mask_temp_ptr + pid * num_groups + expert_group_idx, mask=expert_mask)

    # compute masked_scores
    score_mask_bool = score_mask != 0
    masked_scores = tl.where(score_mask_bool, scores, -float('inf'))

    # compute top_indices (topk)
    data = masked_scores
    for i in range(topk):
        max_val = tl.max(data, axis=0)
        max_idx = tl.argmax(data, axis=0)
        tl.store(top_scores_ptr + pid * topk + i, max_val)
        tl.store(top_indices_ptr + pid * topk + i, max_idx)
        data = tl.where(expert_offs == max_idx, -float('inf'), data)
    tl.debug_barrier()
    top_scores = tl.load(top_scores_ptr + pid * topk + topk_offs, mask=topk_mask)
    top_indices = tl.load(top_indices_ptr + pid * topk + topk_offs, mask=topk_mask)

    # compute probs
    probs = top_scores
    probs *= scaling_factor

    # compute topk_masked_gates
    tl.store(topk_masked_gates_ptr + pid * num_experts + top_indices, probs, mask=topk_mask)

    # compute topk_map
    ones = tl.full([BLOCK_SIZE_TOPK], 1, logits_ptr.dtype.element_ty)
    tl.store(topk_map_ptr + pid * num_experts + top_indices, ones, mask=topk_mask)

@triton.jit
def softmax_postsoft_grouped_forward_kernel(
    # temp
    scores_temp_ptr,
    top_indices_temp_ptr,
    group_mask_temp_ptr,
    # input
    logits_ptr,
    # output
    topk_masked_gates_ptr,
    topk_map_ptr,
    top_indices_ptr,
    top_scores_ptr,
    # param
    num_experts: tl.constexpr,
    num_groups: tl.constexpr,
    experts_per_group: tl.constexpr,
    topk: tl.constexpr,
    group_topk: tl.constexpr,
    topk_per_group: tl.constexpr,
    scaling_factor: tl.constexpr,
    BLOCK_SIZE_NUM_EXPERTS: tl.constexpr,
    BLOCK_SIZE_NUM_GROUPS: tl.constexpr,
    BLOCK_SIZE_EXPERTS_PER_GROUP: tl.constexpr,
    BLOCK_SIZE_TOPK: tl.constexpr,
    BLOCK_SIZE_GROUP_TOPK: tl.constexpr,
    BLOCK_SIZE_TOPK_PER_GROUP: tl.constexpr
):
    pid = tl.program_id(axis=0)

    # offs & mask
    expert_offs = tl.arange(0, BLOCK_SIZE_NUM_EXPERTS)
    expert_mask = expert_offs < num_experts

    group_offs = tl.arange(0, BLOCK_SIZE_NUM_GROUPS)

    group_topk_offs = tl.arange(0, BLOCK_SIZE_GROUP_TOPK)
    group_topk_mask = group_topk_offs < group_topk

    topk_offs = tl.arange(0, BLOCK_SIZE_TOPK)
    topk_mask = topk_offs < topk

    # load input
    logits = tl.load(logits_ptr + pid * num_experts + expert_offs, mask=expert_mask)

    # compute scores
    scores = logits

    # compute group_scores
    # group_view = tl.reshape(scores, num_groups, experts_per_group) # only if num_groups & experts_per_group are "power_of_2"
    
    tl.debug_barrier()

    offs_m = tl.arange(0, BLOCK_SIZE_NUM_GROUPS)[:, None]
    offs_n = tl.arange(0, BLOCK_SIZE_EXPERTS_PER_GROUP)[None, :]
    indices = offs_m * experts_per_group + offs_n
    mask = (offs_m < num_groups) & (offs_n < experts_per_group)

    group_view = tl.load(logits_ptr + pid * num_experts + indices, mask=mask)

    # get top in group
    sorted_group_view = tl.sort(group_view, descending=True)
    group_top_mask = (offs_m < num_groups) & (offs_n < topk_per_group)
    group_top_values = tl.where(group_top_mask, sorted_group_view, 0.0)

    group_scores = tl.sum(group_top_values, axis=-1)

    # compute group_idx (topk)
    data = group_scores
    for i in range(group_topk):
        max_idx = tl.argmax(data, axis=0)
        tl.store(top_indices_temp_ptr + pid * group_topk + i, max_idx)
        data = tl.where(group_offs == max_idx, -float('inf'), data)
    tl.debug_barrier()
    group_idx = tl.load(top_indices_temp_ptr + pid * group_topk + group_topk_offs, mask=group_topk_mask)
    
    # compute group_mask
    ones = tl.full([BLOCK_SIZE_GROUP_TOPK], 1, logits_ptr.dtype.element_ty)
    tl.store(group_mask_temp_ptr + pid * num_groups + group_idx, ones, mask=group_topk_mask)

    tl.debug_barrier()

    # compute score_mask
    expert_group_idx = expert_offs // experts_per_group # mapping experts to group_id
    score_mask = tl.load(group_mask_temp_ptr + pid * num_groups + expert_group_idx, mask=expert_mask)

    # compute masked_scores
    score_mask_bool = score_mask != 0
    masked_scores = tl.where(score_mask_bool, scores, -float('inf'))

    # compute top_indices (topk)
    data = masked_scores
    for i in range(topk):
        max_val = tl.max(data, axis=0)
        max_idx = tl.argmax(data, axis=0)
        tl.store(top_scores_ptr + pid * topk + i, max_val)
        tl.store(top_indices_ptr + pid * topk + i, max_idx)
        data = tl.where(expert_offs == max_idx, -float('inf'), data)
    tl.debug_barrier()
    top_scores = tl.load(top_scores_ptr + pid * topk + topk_offs, mask=topk_mask, other=float('-inf')) # other=0 leads to wrong softmax
    top_indices = tl.load(top_indices_ptr + pid * topk + topk_offs, mask=topk_mask)

    # compute probs
    probs = tl.softmax(top_scores.to(tl.float32)).to(logits_ptr.dtype.element_ty)
    probs *= scaling_factor

    # compute topk_masked_gates
    tl.store(topk_masked_gates_ptr + pid * num_experts + top_indices, probs, mask=topk_mask)

    # compute topk_map
    ones = tl.full([BLOCK_SIZE_TOPK], 1, logits_ptr.dtype.element_ty)
    tl.store(topk_map_ptr + pid * num_experts + top_indices, ones, mask=topk_mask)

@triton.jit
def sigmoid_withbias_grouped_forward_kernel(
    # temp
    scores_temp_ptr,
    group_view_temp_ptr,
    top_indices_temp_ptr,
    group_mask_temp_ptr,
    # input
    logits_ptr,
    expert_bias_ptr,
    # output
    topk_masked_gates_ptr,
    topk_map_ptr,
    top_indices_ptr,
    top_scores_ptr,
    # param
    num_experts: tl.constexpr,
    num_groups: tl.constexpr,
    experts_per_group: tl.constexpr,
    topk: tl.constexpr,
    group_topk: tl.constexpr,
    topk_per_group: tl.constexpr,
    scaling_factor: tl.constexpr,
    BLOCK_SIZE_NUM_EXPERTS: tl.constexpr,
    BLOCK_SIZE_NUM_GROUPS: tl.constexpr,
    BLOCK_SIZE_EXPERTS_PER_GROUP: tl.constexpr,
    BLOCK_SIZE_TOPK: tl.constexpr,
    BLOCK_SIZE_GROUP_TOPK: tl.constexpr,
    BLOCK_SIZE_TOPK_PER_GROUP: tl.constexpr
):
    pid = tl.program_id(axis=0)

    # offs & mask
    expert_offs = tl.arange(0, BLOCK_SIZE_NUM_EXPERTS)
    expert_mask = expert_offs < num_experts

    group_offs = tl.arange(0, BLOCK_SIZE_NUM_GROUPS)

    group_topk_offs = tl.arange(0, BLOCK_SIZE_GROUP_TOPK)
    group_topk_mask = group_topk_offs < group_topk

    topk_offs = tl.arange(0, BLOCK_SIZE_TOPK)
    topk_mask = topk_offs < topk

    # load input
    logits = tl.load(logits_ptr + pid * num_experts + expert_offs, mask=expert_mask)
    expert_bias = tl.load(expert_bias_ptr + expert_offs, mask=expert_mask)

    # compute scores
    scores = tl.sigmoid(logits.to(tl.float32)).to(logits_ptr.dtype.element_ty)
    scores_for_routing = scores + expert_bias
    tl.store(scores_temp_ptr + pid * num_experts + expert_offs, scores, mask=expert_mask)

    # compute group_scores
    # group_view = tl.reshape(scores_for_routing, num_groups, experts_per_group) # only if num_groups & experts_per_group are "power_of_2"

    tl.store(group_view_temp_ptr + pid * num_experts + expert_offs, scores_for_routing, mask=expert_mask)
    tl.debug_barrier()

    offs_m = tl.arange(0, BLOCK_SIZE_NUM_GROUPS)[:, None]
    offs_n = tl.arange(0, BLOCK_SIZE_EXPERTS_PER_GROUP)[None, :]
    indices = offs_m * experts_per_group + offs_n
    mask = (offs_m < num_groups) & (offs_n < experts_per_group)

    group_view = tl.load(group_view_temp_ptr + pid * num_experts + indices, mask=mask)

    # get top in group
    sorted_group_view = tl.sort(group_view, descending=True)
    group_top_mask = (offs_m < num_groups) & (offs_n < topk_per_group)
    group_top_values = tl.where(group_top_mask, sorted_group_view, 0.0)

    group_scores = tl.sum(group_top_values, axis=-1)

    # compute group_idx (topk)
    data = group_scores
    for i in range(group_topk):
        max_idx = tl.argmax(data, axis=0)
        tl.store(top_indices_temp_ptr + pid * group_topk + i, max_idx)
        data = tl.where(group_offs == max_idx, -float('inf'), data)
    tl.debug_barrier()
    group_idx = tl.load(top_indices_temp_ptr + pid * group_topk + group_topk_offs, mask=group_topk_mask)
    
    # compute group_mask
    ones = tl.full([BLOCK_SIZE_GROUP_TOPK], 1, logits_ptr.dtype.element_ty)
    tl.store(group_mask_temp_ptr + pid * num_groups + group_idx, ones, mask=group_topk_mask)

    tl.debug_barrier()

    # compute score_mask
    expert_group_idx = expert_offs // experts_per_group # mapping experts to group_id
    score_mask = tl.load(group_mask_temp_ptr + pid * num_groups + expert_group_idx, mask=expert_mask)

    # compute masked_scores
    score_mask_bool = score_mask != 0
    masked_scores = tl.where(score_mask_bool, scores_for_routing, -float('inf'))

    # compute top_indices (topk)
    data = masked_scores
    for i in range(topk):
        max_idx = tl.argmax(data, axis=0)
        tl.store(top_indices_ptr + pid * topk + i, max_idx)
        data = tl.where(expert_offs == max_idx, -float('inf'), data)
    tl.debug_barrier()
    top_indices = tl.load(top_indices_ptr + pid * topk + topk_offs, mask=topk_mask)

    # compute top_scores
    top_scores = tl.load(scores_temp_ptr + pid * num_experts + top_indices, mask=topk_mask)
    tl.store(top_scores_ptr + pid * topk + topk_offs, top_scores, mask=topk_mask)

    # compute probs
    probs = top_scores / (tl.sum(top_scores) + 1e-20) if topk > 1 else top_scores
    probs *= scaling_factor

    # compute topk_masked_gates
    tl.store(topk_masked_gates_ptr + pid * num_experts + top_indices, probs, mask=topk_mask)

    # compute topk_map
    ones = tl.full([BLOCK_SIZE_TOPK], 1, logits_ptr.dtype.element_ty)
    tl.store(topk_map_ptr + pid * num_experts + top_indices, ones, mask=topk_mask)

@triton.jit
def sigmoid_nobias_grouped_forward_kernel(
    # temp
    scores_temp_ptr,
    top_indices_temp_ptr,
    group_mask_temp_ptr,
    # input
    logits_ptr,
    # output
    topk_masked_gates_ptr,
    topk_map_ptr,
    top_indices_ptr,
    top_scores_ptr,
    # param
    num_experts: tl.constexpr,
    num_groups: tl.constexpr,
    experts_per_group: tl.constexpr,
    topk: tl.constexpr,
    group_topk: tl.constexpr,
    topk_per_group: tl.constexpr,
    scaling_factor: tl.constexpr,
    BLOCK_SIZE_NUM_EXPERTS: tl.constexpr,
    BLOCK_SIZE_NUM_GROUPS: tl.constexpr,
    BLOCK_SIZE_EXPERTS_PER_GROUP: tl.constexpr,
    BLOCK_SIZE_TOPK: tl.constexpr,
    BLOCK_SIZE_GROUP_TOPK: tl.constexpr,
    BLOCK_SIZE_TOPK_PER_GROUP: tl.constexpr
):
    pid = tl.program_id(axis=0)

    # offs & mask
    expert_offs = tl.arange(0, BLOCK_SIZE_NUM_EXPERTS)
    expert_mask = expert_offs < num_experts

    group_offs = tl.arange(0, BLOCK_SIZE_NUM_GROUPS)

    group_topk_offs = tl.arange(0, BLOCK_SIZE_GROUP_TOPK)
    group_topk_mask = group_topk_offs < group_topk

    topk_offs = tl.arange(0, BLOCK_SIZE_TOPK)
    topk_mask = topk_offs < topk

    # load input
    logits = tl.load(logits_ptr + pid * num_experts + expert_offs, mask=expert_mask)

    # compute scores
    scores = tl.sigmoid(logits.to(tl.float32)).to(logits_ptr.dtype.element_ty)
    tl.store(scores_temp_ptr + pid * num_experts + expert_offs, scores, mask=expert_mask)

    # compute group_scores
    # group_view = tl.reshape(scores, num_groups, experts_per_group) # only if num_groups & experts_per_group are "power_of_2"
    
    tl.debug_barrier()

    offs_m = tl.arange(0, BLOCK_SIZE_NUM_GROUPS)[:, None]
    offs_n = tl.arange(0, BLOCK_SIZE_EXPERTS_PER_GROUP)[None, :]
    indices = offs_m * experts_per_group + offs_n
    mask = (offs_m < num_groups) & (offs_n < experts_per_group)

    group_view = tl.load(scores_temp_ptr + pid * num_experts + indices, mask=mask)

    # get top in group
    sorted_group_view = tl.sort(group_view, descending=True)
    group_top_mask = (offs_m < num_groups) & (offs_n < topk_per_group)
    group_top_values = tl.where(group_top_mask, sorted_group_view, 0.0)

    group_scores = tl.sum(group_top_values, axis=-1)

    # compute group_idx (topk)
    data = group_scores
    for i in range(group_topk):
        max_idx = tl.argmax(data, axis=0)
        tl.store(top_indices_temp_ptr + pid * group_topk + i, max_idx)
        data = tl.where(group_offs == max_idx, -float('inf'), data)
    tl.debug_barrier()
    group_idx = tl.load(top_indices_temp_ptr + pid * group_topk + group_topk_offs, mask=group_topk_mask)
    
    # compute group_mask
    ones = tl.full([BLOCK_SIZE_GROUP_TOPK], 1, logits_ptr.dtype.element_ty)
    tl.store(group_mask_temp_ptr + pid * num_groups + group_idx, ones, mask=group_topk_mask)

    tl.debug_barrier()

    # compute score_mask
    expert_group_idx = expert_offs // experts_per_group # mapping experts to group_id
    score_mask = tl.load(group_mask_temp_ptr + pid * num_groups + expert_group_idx, mask=expert_mask)

    # compute masked_scores
    score_mask_bool = score_mask != 0
    masked_scores = tl.where(score_mask_bool, scores, -float('inf'))

    # compute top_indices & top_scores (topk)
    data = masked_scores
    for i in range(topk):
        max_val = tl.max(data, axis=0)
        max_idx = tl.argmax(data, axis=0)
        tl.store(top_scores_ptr + pid * topk + i, max_val)
        tl.store(top_indices_ptr + pid * topk + i, max_idx)
        data = tl.where(expert_offs == max_idx, -float('inf'), data)
    tl.debug_barrier()
    top_scores = tl.load(top_scores_ptr + pid * topk + topk_offs, mask=topk_mask)
    top_indices = tl.load(top_indices_ptr + pid * topk + topk_offs, mask=topk_mask)

    # compute probs
    probs = top_scores / (tl.sum(top_scores) + 1e-20) if topk > 1 else top_scores
    probs *= scaling_factor

    # compute topk_masked_gates
    tl.store(topk_masked_gates_ptr + pid * num_experts + top_indices, probs, mask=topk_mask)

    # compute topk_map
    ones = tl.full([BLOCK_SIZE_TOPK], 1, logits_ptr.dtype.element_ty)
    tl.store(topk_map_ptr + pid * num_experts + top_indices, ones, mask=topk_mask)


# nogroup forward
@triton.jit
def softmax_presoft_nogroup_forward_kernel(
    # input
    logits_ptr,
    # output
    topk_masked_gates_ptr,
    topk_map_ptr,
    top_indices_ptr,
    top_scores_ptr,
    # param
    num_experts: tl.constexpr,
    topk: tl.constexpr,
    scaling_factor: tl.constexpr,
    BLOCK_SIZE_NUM_EXPERTS: tl.constexpr,
    BLOCK_SIZE_TOPK: tl.constexpr
):
    pid = tl.program_id(axis=0)

    # offs & mask
    expert_offs = tl.arange(0, BLOCK_SIZE_NUM_EXPERTS)
    expert_mask = expert_offs < num_experts

    topk_offs = tl.arange(0, BLOCK_SIZE_TOPK)
    topk_mask = topk_offs < topk

    # load logits
    logits = tl.load(logits_ptr + pid * num_experts + expert_offs, mask=expert_mask, other=float('-inf')) # other=0 leads to wrong softmax

    # compute scores
    scores = tl.softmax(logits.to(tl.float32)).to(logits_ptr.dtype.element_ty)

    # compute top_indices & top_scores (topk)
    data = scores
    for i in range(topk):
        max_val = tl.max(data, axis=0)
        max_idx = tl.argmax(data, axis=0)
        tl.store(top_scores_ptr + pid * topk + i, max_val)
        tl.store(top_indices_ptr + pid * topk + i, max_idx)
        data = tl.where(expert_offs == max_idx, -float('inf'), data)
    tl.debug_barrier()
    top_scores = tl.load(top_scores_ptr + pid * topk + topk_offs, mask=topk_mask)
    top_indices = tl.load(top_indices_ptr + pid * topk + topk_offs, mask=topk_mask)

    # compute probs
    probs = top_scores
    probs *= scaling_factor

    # compute topk_masked_gates
    tl.store(topk_masked_gates_ptr + pid * num_experts + top_indices, probs, mask=topk_mask)
    
    # compute topk_map
    ones = tl.full([BLOCK_SIZE_TOPK], 1, logits_ptr.dtype.element_ty)
    tl.store(topk_map_ptr + pid * num_experts + top_indices, ones, mask=topk_mask)

@triton.jit
def softmax_postsoft_nogroup_forward_kernel(
    # input
    logits_ptr,
    # output
    topk_masked_gates_ptr,
    topk_map_ptr,
    top_indices_ptr,
    top_scores_ptr,
    # param
    num_experts: tl.constexpr,
    topk: tl.constexpr,
    scaling_factor: tl.constexpr,
    BLOCK_SIZE_NUM_EXPERTS: tl.constexpr,
    BLOCK_SIZE_TOPK: tl.constexpr
):
    pid = tl.program_id(axis=0)

    # offs & mask
    expert_offs = tl.arange(0, BLOCK_SIZE_NUM_EXPERTS)
    expert_mask = expert_offs < num_experts

    topk_offs = tl.arange(0, BLOCK_SIZE_TOPK)
    topk_mask = topk_offs < topk

    # load logits
    logits = tl.load(logits_ptr + pid * num_experts + expert_offs, mask=expert_mask)

    # compute scores
    # scores = logits

    # compute top_indices & top_scores (topk)
    # data = scores
    data = logits
    for i in range(topk):
        max_val = tl.max(data, axis=0)
        max_idx = tl.argmax(data, axis=0)
        tl.store(top_scores_ptr + pid * topk + i, max_val)
        tl.store(top_indices_ptr + pid * topk + i, max_idx)
        data = tl.where(expert_offs == max_idx, -float('inf'), data)
    tl.debug_barrier()
    top_indices = tl.load(top_indices_ptr + pid * topk + topk_offs, mask=topk_mask)
    top_scores = tl.load(top_scores_ptr + pid * topk + topk_offs, mask=topk_mask, other=float('-inf')) # other=0 leads to wrong softmax

    # compute probs
    probs = tl.softmax(top_scores.to(tl.float32)).to(logits_ptr.dtype.element_ty)
    probs *= scaling_factor

    # compute topk_masked_gates
    tl.store(topk_masked_gates_ptr + pid * num_experts + top_indices, probs, mask=topk_mask)
    
    # compute topk_map
    ones = tl.full([BLOCK_SIZE_TOPK], 1, logits_ptr.dtype.element_ty)
    tl.store(topk_map_ptr + pid * num_experts + top_indices, ones, mask=topk_mask)

@triton.jit
def sigmoid_withbias_nogroup_forward_kernel(
    # input
    logits_ptr,
    expert_bias_ptr,
    # temp
    scores_temp_ptr,
    # output
    topk_masked_gates_ptr,
    topk_map_ptr,
    top_indices_ptr,
    top_scores_ptr,
    # param
    num_experts: tl.constexpr,
    topk: tl.constexpr,
    scaling_factor: tl.constexpr,
    BLOCK_SIZE_NUM_EXPERTS: tl.constexpr,
    BLOCK_SIZE_TOPK: tl.constexpr
):
    pid = tl.program_id(axis=0)

    # offs & mask
    expert_offs = tl.arange(0, BLOCK_SIZE_NUM_EXPERTS)
    expert_mask = expert_offs < num_experts

    topk_offs = tl.arange(0, BLOCK_SIZE_TOPK)
    topk_mask = topk_offs < topk

    # load logits
    logits = tl.load(logits_ptr + pid * num_experts + expert_offs, mask=expert_mask)
    expert_bias = tl.load(expert_bias_ptr + expert_offs, mask=expert_mask)
    
    # compute scores
    scores = tl.sigmoid(logits.to(tl.float32)).to(logits_ptr.dtype.element_ty)
    scores_for_routing = scores + expert_bias

    tl.store(scores_temp_ptr + pid * num_experts + expert_offs, scores, mask=expert_mask)

    # compute top_indices (topk)
    data = scores_for_routing
    for i in range(topk):
        max_idx = tl.argmax(data, axis=0)
        tl.store(top_indices_ptr + pid * topk + i, max_idx)
        data = tl.where(expert_offs == max_idx, -float('inf'), data)
    tl.debug_barrier()
    top_indices = tl.load(top_indices_ptr + pid * topk + topk_offs, mask=topk_mask)

    # compute top_scores
    top_scores = tl.load(scores_temp_ptr + pid * num_experts + top_indices, mask=topk_mask)
    tl.store(top_scores_ptr + pid * topk + topk_offs, top_scores, mask=topk_mask)

    # compute probs
    probs = top_scores / (tl.sum(top_scores) + 1e-20) if topk > 1 else top_scores
    probs *= scaling_factor

    # compute topk_masked_gates
    tl.store(topk_masked_gates_ptr + pid * num_experts + top_indices, probs, mask=topk_mask)
    
    # compute topk_map
    ones = tl.full([BLOCK_SIZE_TOPK], 1, logits_ptr.dtype.element_ty)
    tl.store(topk_map_ptr + pid * num_experts + top_indices, ones, mask=topk_mask)

@triton.jit
def sigmoid_nobias_nogroup_forward_kernel(
    # input
    logits_ptr,
    # output
    topk_masked_gates_ptr,
    topk_map_ptr,
    top_indices_ptr,
    top_scores_ptr,
    # param
    num_experts: tl.constexpr,
    topk: tl.constexpr,
    scaling_factor: tl.constexpr,
    BLOCK_SIZE_NUM_EXPERTS: tl.constexpr,
    BLOCK_SIZE_TOPK: tl.constexpr
):
    pid = tl.program_id(axis=0)

    # offs & mask
    expert_offs = tl.arange(0, BLOCK_SIZE_NUM_EXPERTS)
    expert_mask = expert_offs < num_experts

    topk_offs = tl.arange(0, BLOCK_SIZE_TOPK)
    topk_mask = topk_offs < topk

    # load logits
    logits = tl.load(logits_ptr + pid * num_experts + expert_offs, mask=expert_mask)
    
    # compute scores
    scores = tl.sigmoid(logits.to(tl.float32)).to(logits_ptr.dtype.element_ty)

    # compute top_indices & top_scores (topk)
    data = scores
    for i in range(topk):
        max_val = tl.max(data, axis=0)
        max_idx = tl.argmax(data, axis=0)
        tl.store(top_scores_ptr + pid * topk + i, max_val)
        tl.store(top_indices_ptr + pid * topk + i, max_idx)
        data = tl.where(expert_offs == max_idx, -float('inf'), data)
    tl.debug_barrier()
    top_scores = tl.load(top_scores_ptr + pid * topk + topk_offs, mask=topk_mask)
    top_indices = tl.load(top_indices_ptr + pid * topk + topk_offs, mask=topk_mask)

    # compute probs
    probs = top_scores / (tl.sum(top_scores) + 1e-20) if topk > 1 else top_scores
    probs *= scaling_factor

    # compute topk_masked_gates
    tl.store(topk_masked_gates_ptr + pid * num_experts + top_indices, probs, mask=topk_mask)
    
    # compute topk_map
    ones = tl.full([BLOCK_SIZE_TOPK], 1, logits_ptr.dtype.element_ty)
    tl.store(topk_map_ptr + pid * num_experts + top_indices, ones, mask=topk_mask)


# backward
@triton.jit
def pre_softmax_backward_kernel(
    # input
    logits_ptr,
    top_indices_ptr,
    top_scores_ptr,
    grad_topk_masked_gates_ptr,
    # output
    grad_logits_ptr,
    # param
    num_experts: tl.constexpr,
    topk: tl.constexpr,
    scaling_factor: tl.constexpr,
    BLOCK_SIZE_NUM_EXPERTS: tl.constexpr,
    BLOCK_SIZE_TOPK: tl.constexpr
):
    pid = tl.program_id(axis=0)

    # offs & mask
    expert_offs = tl.arange(0, BLOCK_SIZE_NUM_EXPERTS)
    expert_mask = expert_offs < num_experts

    # compute grad_probs
    grad_probs = tl.load(grad_topk_masked_gates_ptr + pid * num_experts + expert_offs, mask=expert_mask)

    grad_probs *= scaling_factor

    # compute softmax
    logits = tl.load(logits_ptr + pid * num_experts + expert_offs, mask=expert_mask, other=float('-inf')) # other=0 leads to wrong softmax
    probs = tl.softmax(logits)
    
    # compute grad_logits(2)
    grad_logits = grad_probs
    sum_grad = tl.sum(probs * grad_logits, axis=0)
    grad_logits = probs * (grad_logits - sum_grad)

    tl.store(grad_logits_ptr + pid * num_experts + expert_offs, grad_logits, mask=expert_mask)

@triton.jit
def post_softmax_backward_kernel(
    # input
    logits_ptr,
    top_indices_ptr,
    top_scores_ptr,
    grad_topk_masked_gates_ptr,
    # output
    grad_logits_ptr,
    # param
    num_experts: tl.constexpr,
    topk: tl.constexpr,
    scaling_factor: tl.constexpr,
    BLOCK_SIZE_NUM_EXPERTS: tl.constexpr,
    BLOCK_SIZE_TOPK: tl.constexpr
):
    pid = tl.program_id(axis=0)

    # offs & mask
    topk_offs = tl.arange(0, BLOCK_SIZE_TOPK)
    topk_mask = topk_offs < topk

    top_indices = tl.load(top_indices_ptr + pid * topk + topk_offs, mask=topk_mask)

    # compute grad_probs
    grad_probs = tl.load(grad_topk_masked_gates_ptr + pid * num_experts + top_indices, mask=topk_mask)

    grad_probs *= scaling_factor

    # compute softmax
    top_scores = tl.load(top_scores_ptr + pid * topk + topk_offs, mask=topk_mask, other=float('-inf')) # other=0 leads to wrong softmax
    probs = tl.softmax(top_scores)
    
    # compute grad_logits(2)
    grad_logits = grad_probs
    sum_grad = tl.sum(probs * grad_logits, axis=0)
    grad_logits = probs * (grad_logits - sum_grad)

    tl.store(grad_logits_ptr + pid * num_experts + top_indices, grad_logits, mask=topk_mask)

@triton.jit
def sigmoid_backward_kernel(
    # input
    logits_ptr,
    top_indices_ptr,
    top_scores_ptr,
    grad_topk_masked_gates_ptr,
    # output
    grad_logits_ptr,
    # param
    num_experts: tl.constexpr,
    topk: tl.constexpr,
    scaling_factor: tl.constexpr,
    BLOCK_SIZE_NUM_EXPERTS: tl.constexpr,
    BLOCK_SIZE_TOPK: tl.constexpr
):
    pid = tl.program_id(axis=0)

    # offs & mask
    expert_offs = tl.arange(0, BLOCK_SIZE_NUM_EXPERTS)
    expert_mask = expert_offs < num_experts

    topk_offs = tl.arange(0, BLOCK_SIZE_TOPK)
    topk_mask = topk_offs < topk

    # load top_indices
    top_indices = tl.load(top_indices_ptr + pid * topk + topk_offs, mask=topk_mask)

    # compute grad_probs
    grad_probs = tl.load(grad_topk_masked_gates_ptr + pid * num_experts + top_indices, mask=topk_mask)

    grad_probs *= scaling_factor

    # compute grad_scores
    if topk > 1:
        top_scores = tl.load(top_scores_ptr + pid * topk + topk_offs, mask=topk_mask)
        sum_top_scores = tl.sum(top_scores) + 1e-20
        grad_scores = (grad_probs * sum_top_scores - tl.sum(top_scores * grad_probs)) / (sum_top_scores * sum_top_scores)
    else:
        grad_scores = grad_probs

    # compute grad_logits(1)
    tl.store(grad_logits_ptr + pid * num_experts + top_indices, grad_scores, mask=topk_mask)

    # compute sig
    logits = tl.load(logits_ptr + pid * num_experts + expert_offs, mask=expert_mask)
    sig = tl.sigmoid(logits)

    # compute grad_logits(2)
    tl.debug_barrier()
    grad_logits = tl.load(grad_logits_ptr + pid * num_experts + expert_offs, mask=expert_mask)
    grad_logits *= sig * (1 - sig)
    tl.store(grad_logits_ptr + pid * num_experts + expert_offs, grad_logits, mask=expert_mask)



class TopkSoftmax(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        logits,
        topk,
        capacity_factor,
        pad_to_capacity,
        drop_policy,
        use_pre_softmax,
        num_groups,
        group_topk,
        scaling_factor,
        deterministic_mode,
        score_function,
        expert_bias
    ):
        assert logits.dim() == 2, f"Expected 2D logits [num_tokens, num_experts], got {logits.dim()}."

        if group_topk:
            # params
            num_tokens, num_experts = logits.shape
            experts_per_group = num_experts // num_groups
            topk_per_group = topk // group_topk

            scaling_factor = 1.0 if scaling_factor is None else scaling_factor

            BLOCK_SIZE_NUM_EXPERTS = triton.next_power_of_2(num_experts)
            BLOCK_SIZE_NUM_GROUPS = triton.next_power_of_2(num_groups)
            BLOCK_SIZE_EXPERTS_PER_GROUP = triton.next_power_of_2(experts_per_group)
            BLOCK_SIZE_TOPK = triton.next_power_of_2(topk)
            BLOCK_SIZE_GROUP_TOPK = triton.next_power_of_2(group_topk)
            BLOCK_SIZE_TOPK_PER_GROUP = triton.next_power_of_2(topk_per_group)

            # output
            topk_masked_gates = torch.zeros_like(logits)
            topk_map = torch.zeros_like(logits, dtype=torch.bool)
            top_indices = torch.empty([num_tokens, topk], device=logits.device, dtype=torch.int64)
            top_scores = torch.empty([num_tokens, topk], device=logits.device, dtype=logits.dtype)

            if score_function == 'softmax':

                if use_pre_softmax: # softmax pre group

                    # temp
                    scores_temp = torch.empty([num_tokens, num_experts], device=logits.device, dtype=logits.dtype)
                    top_indices_temp = torch.empty([num_tokens, group_topk], device=logits.device, dtype=torch.int64)
                    group_mask_temp = torch.zeros([num_tokens, num_groups], device=logits.device, dtype=logits.dtype)

                    grid = lambda meta: (num_tokens,)
                    softmax_presoft_grouped_forward_kernel[grid](
                        # temp
                        scores_temp,
                        top_indices_temp,
                        group_mask_temp,
                        # input
                        logits,
                        # output
                        topk_masked_gates,
                        topk_map,
                        top_indices,
                        top_scores,
                        # params
                        num_experts,
                        num_groups,
                        experts_per_group,
                        topk,
                        group_topk,
                        topk_per_group,
                        scaling_factor,
                        BLOCK_SIZE_NUM_EXPERTS,
                        BLOCK_SIZE_NUM_GROUPS,
                        BLOCK_SIZE_EXPERTS_PER_GROUP,
                        BLOCK_SIZE_TOPK,
                        BLOCK_SIZE_GROUP_TOPK,
                        BLOCK_SIZE_TOPK_PER_GROUP
                    )
                    tokens_per_expert = topk_map.sum(dim=0)

                else: # softmax post group

                    # temp
                    scores_temp = torch.empty([num_tokens, num_experts], device=logits.device, dtype=logits.dtype)
                    top_indices_temp = torch.empty([num_tokens, group_topk], device=logits.device, dtype=torch.int64)
                    group_mask_temp = torch.zeros([num_tokens, num_groups], device=logits.device, dtype=logits.dtype)

                    grid = lambda meta: (num_tokens,)
                    softmax_postsoft_grouped_forward_kernel[grid](
                        # temp
                        scores_temp,
                        top_indices_temp,
                        group_mask_temp,
                        # input
                        logits,
                        # output
                        topk_masked_gates,
                        topk_map,
                        top_indices,
                        top_scores,
                        # params
                        num_experts,
                        num_groups,
                        experts_per_group,
                        topk,
                        group_topk,
                        topk_per_group,
                        scaling_factor,
                        BLOCK_SIZE_NUM_EXPERTS,
                        BLOCK_SIZE_NUM_GROUPS,
                        BLOCK_SIZE_EXPERTS_PER_GROUP,
                        BLOCK_SIZE_TOPK,
                        BLOCK_SIZE_GROUP_TOPK,
                        BLOCK_SIZE_TOPK_PER_GROUP
                    )
                    tokens_per_expert = topk_map.sum(dim=0)

            elif score_function == 'sigmoid':

                if expert_bias is not None: # sigmoid bias group

                    # temp
                    scores_temp = torch.empty([num_tokens, num_experts], device=logits.device, dtype=logits.dtype)
                    group_view_temp = torch.empty([num_tokens, num_groups, experts_per_group], device=logits.device, dtype=logits.dtype)
                    top_indices_temp = torch.empty([num_tokens, group_topk], device=logits.device, dtype=torch.int64)
                    group_mask_temp = torch.zeros([num_tokens, num_groups], device=logits.device, dtype=logits.dtype)

                    grid = lambda meta: (num_tokens,)
                    sigmoid_withbias_grouped_forward_kernel[grid](
                        # temp
                        scores_temp,
                        group_view_temp,
                        top_indices_temp,
                        group_mask_temp,
                        # input
                        logits,
                        expert_bias,
                        # output
                        topk_masked_gates,
                        topk_map,
                        top_indices,
                        top_scores,
                        # params
                        num_experts,
                        num_groups,
                        experts_per_group,
                        topk,
                        group_topk,
                        topk_per_group,
                        scaling_factor,
                        BLOCK_SIZE_NUM_EXPERTS,
                        BLOCK_SIZE_NUM_GROUPS,
                        BLOCK_SIZE_EXPERTS_PER_GROUP,
                        BLOCK_SIZE_TOPK,
                        BLOCK_SIZE_GROUP_TOPK,
                        BLOCK_SIZE_TOPK_PER_GROUP
                    )
                    tokens_per_expert = topk_map.sum(dim=0)

                else: # sigmoid nobias group

                    # temp
                    scores_temp = torch.empty([num_tokens, num_experts], device=logits.device, dtype=logits.dtype)
                    top_indices_temp = torch.empty([num_tokens, group_topk], device=logits.device, dtype=torch.int64)
                    group_mask_temp = torch.zeros([num_tokens, num_groups], device=logits.device, dtype=logits.dtype)

                    grid = lambda meta: (num_tokens,)
                    sigmoid_nobias_grouped_forward_kernel[grid](
                        # temp
                        scores_temp,
                        top_indices_temp,
                        group_mask_temp,
                        # input
                        logits,
                        # output
                        topk_masked_gates,
                        topk_map,
                        top_indices,
                        top_scores,
                        # params
                        num_experts,
                        num_groups,
                        experts_per_group,
                        topk,
                        group_topk,
                        topk_per_group,
                        scaling_factor,
                        BLOCK_SIZE_NUM_EXPERTS,
                        BLOCK_SIZE_NUM_GROUPS,
                        BLOCK_SIZE_EXPERTS_PER_GROUP,
                        BLOCK_SIZE_TOPK,
                        BLOCK_SIZE_GROUP_TOPK,
                        BLOCK_SIZE_TOPK_PER_GROUP
                    )
                    tokens_per_expert = topk_map.sum(dim=0)

        else: # nogroup
            # param
            num_tokens, num_experts = logits.shape

            scaling_factor = 1.0 if scaling_factor is None else scaling_factor

            BLOCK_SIZE_NUM_EXPERTS = triton.next_power_of_2(num_experts)
            BLOCK_SIZE_TOPK = triton.next_power_of_2(topk)

            # output
            topk_masked_gates = torch.zeros_like(logits)
            topk_map = torch.zeros_like(logits, dtype=torch.bool)
            top_indices = torch.zeros([num_tokens, topk], dtype=torch.int64, device=logits.device)
            top_scores = torch.zeros([num_tokens, topk], dtype=logits.dtype, device=logits.device)

            if score_function == 'softmax':

                if use_pre_softmax: # softmax pre nogroup

                    grid = lambda meta: (num_tokens,)
                    softmax_presoft_nogroup_forward_kernel[grid](
                        # input
                        logits,
                        # output
                        topk_masked_gates,
                        topk_map,
                        top_indices,
                        top_scores,
                        # param
                        num_experts,
                        topk,
                        scaling_factor,
                        BLOCK_SIZE_NUM_EXPERTS,
                        BLOCK_SIZE_TOPK
                    )
                    tokens_per_expert = topk_map.sum(dim=0)
                    
                else: # softmax post nogroup

                    grid = lambda meta: (num_tokens,)
                    softmax_postsoft_nogroup_forward_kernel[grid](
                        # input
                        logits,
                        # output
                        topk_masked_gates,
                        topk_map,
                        top_indices,
                        top_scores,
                        # param
                        num_experts,
                        topk,
                        scaling_factor,
                        BLOCK_SIZE_NUM_EXPERTS,
                        BLOCK_SIZE_TOPK
                    )
                    tokens_per_expert = topk_map.sum(dim=0)

            elif score_function == 'sigmoid':

                if expert_bias is not None: # sigmoid bias nogroup

                    # temp
                    scores_temp = torch.empty([num_tokens, num_experts], dtype=logits.dtype, device=logits.device)

                    grid = lambda meta: (num_tokens,)
                    sigmoid_withbias_nogroup_forward_kernel[grid](
                        # input
                        logits,
                        expert_bias,
                        # temp
                        scores_temp,
                        # output
                        topk_masked_gates,
                        topk_map,
                        top_indices,
                        top_scores,
                        # param
                        num_experts,
                        topk,
                        scaling_factor,
                        BLOCK_SIZE_NUM_EXPERTS,
                        BLOCK_SIZE_TOPK
                    )
                    tokens_per_expert = topk_map.sum(dim=0)

                else: # sigmoid nobias nogroup

                    grid = lambda meta: (num_tokens,)
                    sigmoid_nobias_nogroup_forward_kernel[grid](
                        # input
                        logits,
                        # output
                        topk_masked_gates,
                        topk_map,
                        top_indices,
                        top_scores,
                        # param
                        num_experts,
                        topk,
                        scaling_factor,
                        BLOCK_SIZE_NUM_EXPERTS,
                        BLOCK_SIZE_TOPK
                    )
                    tokens_per_expert = topk_map.sum(dim=0)
            
            else:
                print("wrong score_function")

        # save for backward
        ctx.topk = topk
        ctx.scaling_factor = scaling_factor
        ctx.score_function = score_function
        ctx.use_pre_softmax = use_pre_softmax
        ctx.save_for_backward(logits, top_indices, top_scores)
        return topk_masked_gates, topk_map, tokens_per_expert

    @staticmethod
    def backward(
        ctx,
        grad_topk_masked_gates,
        grad_topk_map,
        grad_tokens_per_expert
    ):
        # load saved
        logits, top_indices, top_scores = ctx.saved_tensors
        topk = ctx.topk
        scaling_factor = ctx.scaling_factor
        score_function = ctx.score_function
        use_pre_softmax = ctx.use_pre_softmax

        # param
        num_tokens, num_experts = logits.shape

        BLOCK_SIZE_NUM_EXPERTS = triton.next_power_of_2(num_experts)
        BLOCK_SIZE_TOPK = triton.next_power_of_2(topk)

        # output
        grad_logits = torch.zeros_like(logits)

        if score_function == "softmax":

            if use_pre_softmax:

                grid = lambda meta: (num_tokens,)
                pre_softmax_backward_kernel[grid](
                    # input
                    logits,
                    # output
                    top_indices,
                    top_scores,
                    grad_topk_masked_gates,
                    grad_logits,
                    # param
                    num_experts,
                    topk,
                    scaling_factor,
                    BLOCK_SIZE_NUM_EXPERTS,
                    BLOCK_SIZE_TOPK,
                )

            else:

                grid = lambda meta: (num_tokens,)
                post_softmax_backward_kernel[grid](
                    # input
                    logits,
                    # output
                    top_indices,
                    top_scores,
                    grad_topk_masked_gates,
                    grad_logits,
                    # param
                    num_experts,
                    topk,
                    scaling_factor,
                    BLOCK_SIZE_NUM_EXPERTS,
                    BLOCK_SIZE_TOPK,
                )
                
        elif score_function == "sigmoid":
            
            grid = lambda meta: (num_tokens,)
            sigmoid_backward_kernel[grid](
                # input
                logits,
                # output
                top_indices,
                top_scores,
                grad_topk_masked_gates,
                grad_logits,
                # param
                num_experts,
                topk,
                scaling_factor,
                BLOCK_SIZE_NUM_EXPERTS,
                BLOCK_SIZE_TOPK,
            )

        else:
            print("wrong score_function")

        return grad_logits, None, None, None, None, None, None, None, None, None, None, None


def fused_topk_softmax_without_capacity(
    logits,
    topk,
    capacity_factor,
    pad_to_capacity,
    drop_policy,
    use_pre_softmax,
    num_groups,
    group_topk,
    scaling_factor,
    deterministic_mode,
    score_function,
    expert_bias
):
    return TopkSoftmax.apply(
        logits,
        topk,
        capacity_factor,
        pad_to_capacity,
        drop_policy,
        use_pre_softmax,
        num_groups,
        group_topk,
        scaling_factor,
        deterministic_mode,
        score_function,
        expert_bias
    )