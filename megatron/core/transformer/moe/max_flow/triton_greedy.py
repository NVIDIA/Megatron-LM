import torch
import triton
import triton.language as tl
from typing import Tuple, List

@triton.jit
def greedy_max_flow_triton_kernel(
    spillover_ptr,           # Pointer to spillover tensor
    spare_space_ptr,         # Pointer to spare space tensor
    compatibility_ptr,       # Pointer to compatibility matrix
    assignment_flows_ptr,    # Pointer to output flow amounts (same shape as compatibility)
    num_spillover: tl.constexpr,
    num_buckets: tl.constexpr,
):
    """
    Triton kernel for greedy max flow algorithm.
    
    This kernel processes spillover assignments in a greedy manner:
    1. For each spillover element, find compatible buckets
    2. Assign as much as possible to the first available bucket
    3. Move to next bucket if there's overflow
    4. Continue until all spillover is assigned or no more compatible buckets
    """
    
    # Get program ID
    pid = tl.program_id(axis=0)
    
    # Only the first program processes the algorithm
    if pid != 0:
        return
    
    # Note: Using global memory for spare_space due to Triton limitations with local arrays
    
    # Load all capacities at the beginning and find first zero bucket
    first_zero_bucket = num_buckets  # Initialize to num_buckets (no zero bucket found yet)
    
    for bucket_idx in range(num_buckets):
        current_capacity = tl.load(spare_space_ptr + bucket_idx)
        if current_capacity == 0 and first_zero_bucket == num_buckets:
            first_zero_bucket = bucket_idx
    
    # Greedy algorithm: process each spillover element
    encountered_zero_spillover = False
    
    for spillover_idx in range(num_spillover):
        remaining_spillover = tl.load(spillover_ptr + spillover_idx)
        
        # Check if we've encountered a zero spillover
        if remaining_spillover == 0:
            encountered_zero_spillover = True
        
        # Only process if we haven't encountered zero spillover yet
        if not encountered_zero_spillover:
            # Try to assign this spillover to all compatible buckets
            for bucket_idx in range(num_buckets):
                # Skip buckets from the first zero bucket onwards
                if bucket_idx < first_zero_bucket:
                    # Check if we still have spillover to assign and it's non-zero
                    has_spillover = remaining_spillover > 0
                    
                    if has_spillover:
                        # Check compatibility
                        compatibility_offset = spillover_idx * num_buckets + bucket_idx
                        compatible = tl.load(compatibility_ptr + compatibility_offset)
                        
                        if compatible:
                            # Get current spare space capacity from global memory
                            current_capacity = tl.load(spare_space_ptr + bucket_idx)
                            
                            # Calculate assignment amount
                            assign_amount = tl.minimum(remaining_spillover, current_capacity)
                            
                            # Store the assignment flow in the matrix
                            tl.store(assignment_flows_ptr + compatibility_offset, assign_amount)
                            
                            # Update remaining spillover
                            remaining_spillover -= assign_amount
                            
                            # Update spare space in global memory (subtract the assigned amount)
                            new_capacity = current_capacity - assign_amount
                            tl.store(spare_space_ptr + bucket_idx, new_capacity)


def sort_and_permute_data(spillover: torch.Tensor, spare_space: torch.Tensor, compatibility_map: torch.Tensor):
    """
    Sort spillover and spare_space in descending order and permute compatibility_map accordingly.
    Fully CUDA graphable - uses only sorting and indexing operations.
    
    Args:
        spillover: 1D tensor of spillover amounts
        spare_space: 1D tensor of spare space amounts  
        compatibility_map: 2D boolean tensor of compatibility
        
    Returns:
        tuple: (sorted_spillover, sorted_spare_space, permuted_compatibility_map, spillover_perm, spare_space_perm)
    """
    device = spillover.device
    
    # Sort spillover in descending order and get permutation indices
    spillover_sorted, spillover_perm = torch.sort(spillover, descending=True)
    
    # Sort spare_space in descending order and get permutation indices
    spare_space_sorted, spare_space_perm = torch.sort(spare_space, descending=True)
    
    # Permute compatibility_map according to the sorting
    # First permute rows (spillover dimension), then columns (spare_space dimension)
    permuted_compatibility_map = compatibility_map[spillover_perm][:, spare_space_perm]
    
    return (spillover_sorted, spare_space_sorted, permuted_compatibility_map, spillover_perm, spare_space_perm)


def solve_spillover_spare_matching_triton_greedy(
    spillover: torch.Tensor,
    spare_space: torch.Tensor,
    compatibility_map: torch.Tensor
) -> torch.Tensor:
    """
    Triton-based greedy max flow algorithm.
    
    Args:
        spillover: 1D tensor of spillover amounts for each expert
        spare_space: 1D tensor of spare capacity for each bucket
        compatibility_map: 2D boolean tensor indicating which experts can use which buckets
    
    Returns:
        torch.Tensor: Assignment flows tensor with same shape as compatibility_map
                     containing flow amounts for each spillover-bucket pair
    """
    # Input validation
    assert spillover.dim() == 1, "spillover must be 1D tensor"
    assert spare_space.dim() == 1, "spare_space must be 1D tensor"
    assert compatibility_map.dim() == 2, "compatibility_map must be 2D tensor"
    assert spillover.size(0) == compatibility_map.size(0), "spillover and compatibility_map must have same first dimension"
    assert spare_space.size(0) == compatibility_map.size(1), "spare_space and compatibility_map must have same second dimension"
    
    num_spillover = spillover.size(0)
    num_buckets = spare_space.size(0)
    
    # Ensure tensors are on GPU and have correct dtypes
    device = spillover.device
    spillover = spillover.to(device, dtype=torch.int32)
    spare_space = spare_space.to(device, dtype=torch.int32)
    compatibility_map = compatibility_map.to(device, dtype=torch.bool)
    
    # Sort data in descending order and get permutation indices
    (sorted_spillover, sorted_spare_space, permuted_compatibility_map, 
     spillover_perm, spare_space_perm) = sort_and_permute_data(
        spillover, spare_space, compatibility_map
    )
    
    # Allocate and initialize output tensor with same shape as compatibility_map
    assignment_flows_sorted = torch.zeros_like(compatibility_map, dtype=torch.int32, device=device)
    
    # Launch kernel with sorted data
    grid = (1,)  # Single block since we're using sequential processing
    greedy_max_flow_triton_kernel[grid](
        sorted_spillover,
        sorted_spare_space,
        permuted_compatibility_map,
        assignment_flows_sorted,
        num_spillover=num_spillover,
        num_buckets=num_buckets,
    )
    
    # Restore original order by applying inverse permutation
    # Create inverse permutation indices
    spillover_inv_perm = torch.argsort(spillover_perm)
    spare_space_inv_perm = torch.argsort(spare_space_perm)
    
    # Apply inverse permutation to restore original order
    assignment_flows = assignment_flows_sorted[spillover_inv_perm][:, spare_space_inv_perm]
    
    return assignment_flows


def solve_spillover_spare_matching_triton_greedy_wrapper(
    spillover: torch.Tensor,
    spare_space: torch.Tensor,
    compatibility_map: torch.Tensor
) -> dict:
    """
    Wrapper function for Triton greedy max flow algorithm.
    
    Returns the same format as the CUDA version for compatibility.
    """
    assignment_flows = solve_spillover_spare_matching_triton_greedy(
        spillover, spare_space, compatibility_map
    )
    
    # Convert assignment flows to list format
    assignments = []
    total_flow = 0
    
    for i in range(assignment_flows.size(0)):
        for j in range(assignment_flows.size(1)):
            flow_amount = assignment_flows[i, j].item()
            if flow_amount > 0:
                assignments.append((i, j))
                total_flow += flow_amount
    
    return {
        'max_matches': total_flow,
        'total_spillover': spillover.sum().item(),
        'total_spare_capacity': spare_space.sum().item(),
        'assignments': assignments,
        'assignment_flows': assignment_flows
    }


def solve_spillover_spare_matching_triton_greedy_detailed(
    spillover: torch.Tensor,
    spare_space: torch.Tensor,
    compatibility_map: torch.Tensor
) -> dict:
    """
    Detailed version that provides more information about the Triton greedy algorithm results.
    """
    result = solve_spillover_spare_matching_triton_greedy_wrapper(
        spillover, spare_space, compatibility_map
    )
    
    # Calculate additional metrics
    total_spillover = spillover.sum().item()
    total_spare_capacity = spare_space.sum().item()
    spillover_utilization = result['max_matches'] / total_spillover if total_spillover > 0 else 0
    spare_utilization = result['max_matches'] / total_spare_capacity if total_spare_capacity > 0 else 0
    
    # Count how many spillover elements were fully satisfied
    spillover_satisfied = {}
    assignment_flows = result['assignment_flows']
    for i in range(assignment_flows.size(0)):
        total_assigned = assignment_flows[i, :].sum().item()
        spillover_satisfied[i] = total_assigned
    
    fully_satisfied = sum(1 for idx, assigned in spillover_satisfied.items() 
                         if assigned >= spillover[idx].item())
    
    result.update({
        'spillover_utilization': spillover_utilization,
        'spare_utilization': spare_utilization,
        'fully_satisfied_spillovers': fully_satisfied,
        'total_spillover_elements': len(spillover),
        'algorithm': 'triton_greedy'
    })
    
    return result


# Test function
def test_triton_greedy():
    """Test the Triton greedy implementation with a simple example."""
    print("Testing Triton Greedy Max Flow Algorithm...")
    
    # Create test data
    spillover = torch.tensor([10, 5, 8, 3], dtype=torch.int32, device='cuda')
    spare_space = torch.tensor([6, 4, 7], dtype=torch.int32, device='cuda')
    compatibility_map = torch.tensor([
        [True, True, False],   # spillover[0] can use buckets 0,1
        [False, True, True],   # spillover[1] can use buckets 1,2
        [True, False, True],   # spillover[2] can use buckets 0,2
        [True, True, True]     # spillover[3] can use all buckets
    ], dtype=torch.bool, device='cuda')
    
    print(f"Spillover: {spillover}")
    print(f"Spare space: {spare_space}")
    print(f"Compatibility map:\n{compatibility_map}")
    
    # Run the algorithm
    result = solve_spillover_spare_matching_triton_greedy_detailed(
        spillover, spare_space, compatibility_map
    )
    
    print(f"\nResults:")
    print(f"Max matches: {result['max_matches']}")
    print(f"Total spillover: {result['total_spillover']}")
    print(f"Total spare capacity: {result['total_spare_capacity']}")
    print(f"Spillover utilization: {result['spillover_utilization']:.2%}")
    print(f"Spare utilization: {result['spare_utilization']:.2%}")
    print(f"Fully satisfied spillovers: {result['fully_satisfied_spillovers']}/{result['total_spillover_elements']}")
    
    print(f"\nAssignments:")
    assignment_flows = result['assignment_flows']
    for i in range(assignment_flows.size(0)):
        for j in range(assignment_flows.size(1)):
            flow = assignment_flows[i, j].item()
            if flow > 0:
                print(f"  Spillover[{i}] -> Bucket[{j}]: {flow}")


if __name__ == "__main__":
    test_triton_greedy()
