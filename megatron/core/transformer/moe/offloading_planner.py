import torch
import triton
import triton.language as tl
from typing import Union

def one_shot_greedy_assignment(
    token_chunks: torch.Tensor,
    buckets: torch.Tensor
) -> torch.Tensor:
    """
    Perform one-shot greedy assignment between token chunks and buckets.

    This function implements a greedy algorithm that assigns token chunks to buckets
    in a single pass. It calculates the overlap between cumulative token chunks and
    cumulative bucket capacities to determine an assignment strategy.

    The algorithm works by:
    1. Computing cumulative sums for both token chunks and buckets
    2. Finding overlapping ranges between token chunk intervals and bucket intervals
    3. Calculating the overlap amount as the assignment

    Args:
        token_chunks: Token chunks to be assigned
            Shape: [num_token_chunks]
            Type: torch.Tensor (int)
            Description: Each element represents the number of tokens in a chunk
                       that needs to be assigned to buckets

        buckets: Bucket capacities available for assignment
            Shape: [num_buckets]
            Type: torch.Tensor (int)
            Description: Each element represents the capacity of a bucket that can
                       receive token chunks

    Returns:
        overlap: Assignment matrix showing token chunk to bucket assignments
            Shape: [num_token_chunks, num_buckets]
            Type: torch.Tensor (same dtype as inputs)
            Description: Element [i, j] represents how many tokens from chunk i
                       are assigned to bucket j. Zero if no assignment.
    """
    token_chunks_cumsum = torch.cumsum(token_chunks, dim=0)
    buckets_cumsum = torch.cumsum(buckets, dim=0)
    token_chunks_start = token_chunks_cumsum - token_chunks
    buckets_start = buckets_cumsum - buckets
    token_chunks_start = token_chunks_start.unsqueeze(1)  # (num_token_chunks, 1)
    token_chunks_end = token_chunks_cumsum.unsqueeze(1)   # (num_token_chunks, 1)
    buckets_start = buckets_start.unsqueeze(0)  # (1, num_buckets)
    buckets_end = buckets_cumsum.unsqueeze(0)   # (1, num_buckets)
    overlap_start = torch.maximum(token_chunks_start, buckets_start)
    overlap_end = torch.minimum(token_chunks_end, buckets_end)
    overlap = (overlap_end - overlap_start).clamp(min=0)
    return overlap

def reclaim_spare_experts(
    num_token_to_ep_rank: torch.Tensor,
    avg_token_to_ep_rank: float,
    matched_assignment: torch.Tensor,
    threshold_multiplier: float
) -> torch.Tensor:
    """
    Reclaim spare experts by selectively disabling some spare experts.

    This function implements a threshold-based selection algorithm to reduce the
    number of active spare experts by disabling some offloading assignments based 
    on load balancing criteria. It helps prevent over-allocation of spare experts
    when the load is already well-balanced.

    The algorithm works by:
    1. Calculating current token loads per EP rank after offloading
    2. Determining a threshold based on average load and multiplier
    3. Sorting offloading assignments by size for each EP rank
    4. Using binary search to find which assignments can be safely disabled
    5. Creating a mask to disable assignments that exceed capacity thresholds

    Args:
        num_token_to_ep_rank: Current token distribution per EP rank
            Shape: [num_ep_ranks]
            Type: torch.Tensor (int)
            Description: Number of tokens currently assigned to each EP rank

        avg_token_to_ep_rank: Average tokens per EP rank
            Type: int
            Description: Average token load across all EP ranks, used as baseline
                       for threshold calculations

        matched_assignment: Current offloading assignment matrix
            Shape: [num_home_experts, num_spare_experts]
            Type: torch.Tensor (int)
            Description: Matrix where [i, j] indicates tokens offloaded from home
                       expert i to spare expert j

        threshold_multiplier: Multiplier for threshold calculation
            Type: float
            Description: Controls how aggressive the filtering is. Lower values
                       result in more spare experts being disabled. If <= 0,
                       no filtering is applied.

    Returns:
        matched_assignment: Updated assignment matrix with some assignments disabled
            Shape: [num_home_experts, num_spare_experts]
            Type: torch.Tensor (int)
            Description: Same shape as input but with some offloading assignments
                       set to zero based on threshold criteria

    """
    # Calculate current total tokens per home expert (without offloading)
    ep = num_token_to_ep_rank.shape[0]
    num_home_expert, num_spare_expert = matched_assignment.shape
    current_tokens_home_rank = num_token_to_ep_rank.view(ep, -1).sum(dim=1)
    threshold = threshold_multiplier * avg_token_to_ep_rank
    max_allowed_load = avg_token_to_ep_rank + threshold
    
    # Calculate current home expert loads after offloading
    current_tokens_after_offloading = current_tokens_home_rank - matched_assignment.sum(dim=1).view(ep, -1).sum(dim=1)  
    
    # Sort each row in ascending order (zeros will naturally be at the beginning)
    matched_assignment_ep_rank_view = matched_assignment.view(ep, num_home_expert//ep, -1).sum(dim=1)
    sorted_tokens, sorted_indices = torch.sort(matched_assignment_ep_rank_view, dim=1)
    cumulative_tokens = torch.cumsum(sorted_tokens, dim=1)
    
    # Use searchsorted to find where cumulative tokens exceed capacity for each home expert
    # We need to find where cumulative_tokens > (max_allowed_load - current_tokens_after_offloading)
    # safe_steps contains the last spare expert index needed for each EP rank (after sorting)
    capacity_remaining = max_allowed_load - current_tokens_after_offloading 
    safe_steps = torch.searchsorted(cumulative_tokens, capacity_remaining.unsqueeze(1), side='right').squeeze(1)
    
    # Create a boolean mask based on safe_steps with the same shape as sorted_tokens
    # True indicates positions that need to be enabled (after the boundary)
    step_indices = torch.arange(sorted_tokens.shape[1], device=matched_assignment.device).unsqueeze(0).expand(sorted_tokens.shape[0], -1)
    safe_steps_expanded = safe_steps.unsqueeze(1).expand(-1, sorted_tokens.shape[1])
    sorted_mask = step_indices >= safe_steps_expanded  
    
    # Recover the mask to the original order using sorted_indices
    # Use scatter to map sorted_mask back to original order
    original_order_mask = torch.zeros_like(sorted_mask)
    original_order_mask.scatter_(1, sorted_indices, sorted_mask)
    
    # Do OR operation along dim=0 to get a 1D tensor indicating which columns to mask
    column_mask = original_order_mask.any(dim=0) 
    
    # Apply the column mask directly to matched_assignment
    matched_assignment = torch.where(column_mask.unsqueeze(0), matched_assignment, torch.zeros_like(matched_assignment))
    return matched_assignment

def breadth_first_allocation(transport, reroute_map, device='cuda'):
    """
    Ultra-compact loop-free version with over-allocation protection
    """
    transport, reroute_map = transport.to(device).float(), reroute_map.to(device).float()
    # Initial allocation
    supplier = reroute_map.argmax(0)
    active = (reroute_map > 0).sum(0) > 0
    capacity = reroute_map[supplier, torch.arange(reroute_map.shape[1], device=device)]
    
    x_rel = transport[:, supplier]
    # Use safe division to avoid NaN and division by zero
    denominator = x_rel.sum(0, keepdim=True)
    props = torch.where(denominator > 0, x_rel / denominator, torch.zeros_like(x_rel))
    ideal = props * capacity
    floors = torch.floor(ideal).int()*active

    offloaded = torch.zeros_like(transport, dtype=torch.int32)
    supplier_expanded = supplier.unsqueeze(0).expand(transport.shape[0], -1)  # (num_src, num_new_dst)
    
    # For each new destination k, add t[:, k] to the reduction for old destination supplier[k]
    offloaded.scatter_add_(1, supplier_expanded, floors)
    transport_rerouted = transport - offloaded
    leftover_spare_space = reroute_map.clone()
    leftover_spare_space[supplier, torch.arange(reroute_map.shape[1], device=device)] -= floors.sum(dim=0)
    return floors, transport_rerouted, leftover_spare_space

def depth_first_allocation(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Vectorized batched version of one_shot_greedy_assignment for token routing composition.
    """
    S, J = x.shape
    J2, K = y.shape
    assert J == J2, "Dimension mismatch"
    
    # For each old destination j, x[:,j] are the token chunks
    # Compute cumulative intervals for token chunks along source dimension
    x_cumsum = torch.cumsum(x, dim=0)  # (S, J)
    x_start = x_cumsum - x             # (S, J) - start positions for each chunk
    x_end = x_cumsum                   # (S, J) - end positions for each chunk
    
    # Compute bucket intervals for each old destination
    y_cumsum = torch.cumsum(y, dim=1)  # (J, K)
    y_start = y_cumsum - y             # (J, K) - bucket start positions
    y_end = y_cumsum                   # (J, K) - bucket end positions
    
    # Expand dimensions for broadcasting
    x_start = x_start.unsqueeze(2)     # (S, J, 1)
    x_end = x_end.unsqueeze(2)         # (S, J, 1)
    y_start = y_start.unsqueeze(0)     # (1, J, K)
    y_end = y_end.unsqueeze(0)         # (1, J, K)
    
    # Compute overlaps for all (source, old_dst, new_dst) combinations
    overlap_start = torch.maximum(x_start, y_start)  # (S, J, K)
    overlap_end = torch.minimum(x_end, y_end)        # (S, J, K)
    overlap = (overlap_end - overlap_start).clamp(min=0)  # (S, J, K)
    
    # Sum over intermediate destinations to get final source->new_dst flows
    z = overlap.sum(dim=1)  # (S, K)
    supplier = y.argmax(0)
    supplier_expanded = supplier.unsqueeze(0).expand(x.shape[0], -1)  # (num_src, num_new_dst)
    x_rerouted = x.scatter_add(1, supplier_expanded, -z)
    return z, x_rerouted

@triton.jit
def reroute_tokens_w_permute_map_kernel(
    x_indices_sorted_ptr, 
    expert_for_offload_ptr,
    num_tokens_to_route_ptr,
    cumulative_offsets_ptr,
    y_ptr,
    permute_map_ptr,
    num_tokens: tl.constexpr,
    num_experts: tl.constexpr,
    num_offloading_experts: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offload_expert_id = tl.program_id(0)
    
    source_expert_id = tl.load(expert_for_offload_ptr + offload_expert_id).to(tl.int64)
    if source_expert_id < 0:
        return
    num_tokens_to_route = tl.load(num_tokens_to_route_ptr + offload_expert_id).to(tl.int64)
    offset = tl.load(cumulative_offsets_ptr + offload_expert_id).to(tl.int64)
    
    token_positions = tl.arange(0, BLOCK_SIZE)
    valid_mask = (token_positions < num_tokens_to_route)
    
    # No conditional check needed - masked operations handle empty cases
    base_offset = source_expert_id * num_tokens + offset
    token_indices = tl.load(x_indices_sorted_ptr + base_offset + token_positions, 
                           mask=valid_mask, other=0)
    
    total_experts = num_experts + num_offloading_experts
    
    # These stores are safe even with all-False masks
    x_flat_idx = token_indices * total_experts + source_expert_id
    tl.store(y_ptr + x_flat_idx, False, mask=valid_mask)
    
    # Store source_expert_id to permute_map for valid tokens
    permute_map_idx = token_indices * num_offloading_experts + offload_expert_id
    tl.store(permute_map_ptr + permute_map_idx, source_expert_id, mask=valid_mask)
    
    offload_col = num_experts + offload_expert_id
    y_flat_idx = token_indices * total_experts + offload_col
    tl.store(y_ptr + y_flat_idx, True, mask=valid_mask)


@triton.jit
def reroute_tokens_kernel(
    x_indices_sorted_ptr, 
    expert_for_offload_ptr,
    num_tokens_to_route_ptr,
    cumulative_offsets_ptr,
    y_ptr,
    rerouted_probs_ptr,
    num_tokens: tl.constexpr,
    num_experts: tl.constexpr,
    num_offloading_experts: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offload_expert_id = tl.program_id(0)
    
    source_expert_id = tl.load(expert_for_offload_ptr + offload_expert_id).to(tl.int64)
    if source_expert_id < 0:
        return
    num_tokens_to_route = tl.load(num_tokens_to_route_ptr + offload_expert_id).to(tl.int64)
    offset = tl.load(cumulative_offsets_ptr + offload_expert_id).to(tl.int64)
    
    token_positions = tl.arange(0, BLOCK_SIZE)
    valid_mask = (token_positions < num_tokens_to_route)
    
    # No conditional check needed - masked operations handle empty cases
    base_offset = source_expert_id * num_tokens + offset
    token_indices = tl.load(x_indices_sorted_ptr + base_offset + token_positions, 
                           mask=valid_mask, other=0)
    
    total_experts = num_experts + num_offloading_experts
    
    # These stores are safe even with all-False masks
    x_flat_idx = token_indices * total_experts + source_expert_id
    tl.store(y_ptr + x_flat_idx, False, mask=valid_mask)
    
    # Handle rerouted_probs: read from x_flat_idx and write to y_flat_idx, zero out x_flat_idx
    prob_value = tl.load(rerouted_probs_ptr + x_flat_idx, mask=valid_mask, other=0.0)
    tl.store(rerouted_probs_ptr + x_flat_idx, 0.0, mask=valid_mask)
    
    offload_col = num_experts + offload_expert_id
    y_flat_idx = token_indices * total_experts + offload_col
    tl.store(y_ptr + y_flat_idx, True, mask=valid_mask)
    tl.store(rerouted_probs_ptr + y_flat_idx, prob_value, mask=valid_mask)


def reroute_tokens_triton(x, probs, num_offloading_from, num_offloading_to, reroute_map):
    """
    Triton-based token rerouting with static shapes for CUDA graph compatibility.
    """
    device = x.device
    num_tokens, num_experts = x.shape
    num_offloading_experts = num_offloading_to.shape[0]
    num_offloading_to = num_offloading_to.to(torch.int64)
    # Step 1: Generate reroute_map2
    reroute_map2 = reroute_map * num_offloading_to.unsqueeze(0)
    
    # Step 2: Compute cumulative offsets
    cumsum_routes = torch.cumsum(reroute_map2, dim=1)
    cumulative_starts = cumsum_routes - reroute_map2
    
    # Step 3: Extract expert mapping 
    has_routing = reroute_map.any(dim=0)
    expert_indices = torch.argmax(reroute_map.float(), dim=0)
    expert_for_offload = torch.where(has_routing, expert_indices, -1)
    oe_indices = torch.arange(num_offloading_experts, device=device)
    
    # Clamp expert indices to valid range for safe indexing (invalid entries will be ignored anyway)
    safe_expert_indices = torch.clamp(expert_for_offload, 0, num_experts - 1)
    # Extract offsets for ALL offloading experts using static indexing
    all_offsets = cumulative_starts[safe_expert_indices, oe_indices]
    
    # Use torch.where to zero out offsets for invalid offloading experts
    cumulative_offsets = torch.where(expert_for_offload >= 0, all_offsets, 0)
    
    # Convert expert_for_offload to float for Triton compatibility
    expert_for_offload = expert_for_offload
    
    # Step 5: Create sorted indices
    x_indices_sorted = x.argsort(dim=0, descending=True).transpose(0, 1).contiguous()
    
    # Step 6: Initialize output tensor
    y = torch.zeros(num_tokens, num_experts + num_offloading_experts, 
                    dtype=torch.bool, device=device)
    y[:, :num_experts] = x.clone()
    
    # Step 6b: Initialize permute_map tensor with 0
    permute_map = torch.zeros((num_tokens, num_offloading_experts), 
                             dtype=torch.int32, device=device)
    
    # Step 7: Launch Triton kernel with permute map
    max_tokens = num_tokens
    BLOCK_SIZE = triton.next_power_of_2(max_tokens)
    grid = (num_offloading_experts,)
    
    reroute_tokens_w_permute_map_kernel[grid](
        x_indices_sorted, expert_for_offload, num_offloading_to,
        cumulative_offsets, y, permute_map, num_tokens, num_experts, num_offloading_experts, BLOCK_SIZE
    )
    
    # Step 8: Apply permute map to probabilities using gather
    # Gather probabilities from source experts for offloading tokens
    gathered_probs = torch.gather(probs, 1, permute_map)
    
    # Concatenate original probs with gathered probs
    rerouted_probs = torch.cat([probs, gathered_probs], dim=1)
    
    # Mask with y to zero out probabilities for tokens that are not routed
    rerouted_probs = rerouted_probs * y
    
    return y, rerouted_probs

def gen_offloading_plan(
    routing_map: torch.Tensor,
    probs: torch.Tensor,
    tokens_per_expert_from_ep_rank: torch.Tensor,
    ep_rank: Union[torch.Tensor, int],
    ep: int,
    spare_expert_per_ep_rank: int = 1,
    threshold_multiplier: float = 0.0,
    index_dtype: torch.dtype = torch.int32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate an offloading plan to redistribute tokens from overloaded experts to spare experts.

    This function implements a balanced routing algorithm that redistributes tokens
    from overloaded experts to spare experts to achieve better load balancing.
    The algorithm works in phases:
    1. Calculate spillover tokens that need to be offloaded from each expert
    2. Use greedy assignment to match spillover tokens to spare expert capacity
    3. Apply threshold-based filtering to disable some offloading assignments
    4. Execute the offloading plan using breadth-first and depth-first allocation
    5. Generate the final rerouting map with token movements

    Args:
        routing_map: Binary routing map for current EP rank
            Shape: [num_tokens_in_ep_rank, num_experts]
            Type: torch.Tensor (bool)
            Description: Binary tensor indicating which tokens are routed to which
                       experts within the current EP rank

        probs: Routing probabilities for current EP rank
            Shape: [num_tokens_in_ep_rank, num_experts]
            Type: torch.Tensor (float)
            Description: Probability values for each token-expert assignment within
                       the current EP rank

        tokens_per_expert_from_ep_rank: Token distribution per expert from all EP ranks
            Shape: [num_ep_ranks, num_experts]
            Type: torch.Tensor (int)
            Description: Number of tokens assigned to each expert across all EP ranks

        ep_rank: Current EP rank index
            Type: Union[torch.Tensor, int]
            Description: Index of the current expert parallel rank (0 to ep-1)

        ep: Number of expert parallel ranks
            Type: int
            Description: Total number of expert parallel ranks

        spare_expert_per_ep_rank: Number of spare experts per EP rank
            Type: int
            Default: 1
            Description: How many spare experts are available for offloading per
                       EP rank

        threshold_multiplier: Multiplier for threshold-based expert selection
            Type: float
            Default: 0.0
            Description: If > 0, enables threshold-based selection to disable
                       some spare experts based on average token load. Lower
                       values = more aggressive filtering.

        index_dtype: Data type for index tensors
            Type: torch.dtype
            Default: torch.int32
            Description: Data type used for index tensors in the computation

    Returns:
        tuple containing:
            - rerouting_map: Updated routing map after offloading
                Shape: [num_tokens_in_ep_rank, num_experts + num_spare_experts]
                Type: torch.Tensor (bool)
                Description: Binary tensor showing final token assignments including
                           offloaded tokens to spare experts

            - rerouted_probs: Updated routing probabilities after offloading
                Shape: [num_tokens_in_ep_rank, num_experts + num_spare_experts]
                Type: torch.Tensor (float)
                Description: Probability values for final token assignments including
                           offloaded tokens to spare experts

            - expert_offloading_map: Mapping of home experts to spare experts
                Shape: [num_home_experts, num_spare_experts]
                Type: torch.Tensor (bool)
                Description: Boolean matrix where [i,j] = True indicates home expert
                           i offloads tokens to spare expert j
    """
    # Phase 1: calculate how many tokens need to be offloaded from home experts
    device = routing_map.device
    num_tokens_to_expert = tokens_per_expert_from_ep_rank.sum(dim=0).to(index_dtype)
    num_token_to_ep_rank = num_tokens_to_expert.view(ep, -1).sum(dim=1)
    avg_token_to_ep_rank = num_token_to_ep_rank.sum() // ep
    deviation = num_token_to_ep_rank - avg_token_to_ep_rank
    spare_space = torch.relu(-deviation)

    # sort the local experts by token count and place smaller token chunk first
    local_exp_sorted_token_count, local_exp_sorted_idx = num_tokens_to_expert.view(ep, -1).sort(dim=1)
    spillover_tokens_per_exp_sorted_cumsum = (local_exp_sorted_token_count.cumsum(dim=1) - avg_token_to_ep_rank).clamp(min=0)
    spillover_tokens_per_exp_sorted = torch.cat([spillover_tokens_per_exp_sorted_cumsum[:, :1], torch.diff(spillover_tokens_per_exp_sorted_cumsum, dim=1)], dim=1)
    spillover_tokens_per_exp = torch.scatter(torch.empty_like(spillover_tokens_per_exp_sorted), 1, local_exp_sorted_idx, spillover_tokens_per_exp_sorted).view(-1)
    
    # [num_home_experts, num_spare_experts]
    # assignment[i][j] indicates how many tokens are offloaded from home expert i to spare expert j
    assignment = one_shot_greedy_assignment(spillover_tokens_per_exp, spare_space)

    # Find top spare_expert_per_ep_rank token chunks for each EP rank
    spare_bucket_max, spare_bucket_max_index = torch.topk(assignment, k=spare_expert_per_ep_rank, dim=0)
    num_columns = assignment.shape[1]
    matched_assignment = torch.zeros(assignment.shape[0], num_columns * spare_expert_per_ep_rank, device=device, dtype=assignment.dtype)
    row_indices = spare_bucket_max_index.transpose(0, 1).flatten()
    col_indices = torch.arange(num_columns, device=device).repeat_interleave(spare_expert_per_ep_rank) \
                  * spare_expert_per_ep_rank \
                  + torch.arange(spare_expert_per_ep_rank, device=device).repeat(num_columns)
    values = spare_bucket_max.transpose(0, 1).flatten()
    # [num_home_experts, num_spare_experts]
    # matched_assignment[i][j] indicates how many tokens are offloaded from home expert i to spare expert j
    matched_assignment[row_indices, col_indices] = values
    # Apply threshold-based offloading expert selection
    if threshold_multiplier > 0:
        matched_assignment = reclaim_spare_experts(
            num_token_to_ep_rank, avg_token_to_ep_rank, matched_assignment, threshold_multiplier
        )
    
    # Create expert_offloading_map with shape (num_home_experts, num_spare_experts)
    # expert_offloading_map[i][j] indicates if home expert i is offloaded to spare expert j
    export_offloading_map = matched_assignment > 0
    
    offloaded_tokens, token_dist_after_offloading, leftover_spare_space = breadth_first_allocation(tokens_per_expert_from_ep_rank, matched_assignment)
    offloaded_tokens2, token_dist_after_offloading = depth_first_allocation(token_dist_after_offloading, leftover_spare_space)
    rerouting_map, rerouted_probs = reroute_tokens_triton(routing_map, 
                                                         probs,
                                                         (tokens_per_expert_from_ep_rank-token_dist_after_offloading)[ep_rank].int(), 
                                                         (offloaded_tokens+offloaded_tokens2)[ep_rank].int().squeeze(), 
                                                         export_offloading_map)
    return rerouting_map, rerouted_probs, export_offloading_map