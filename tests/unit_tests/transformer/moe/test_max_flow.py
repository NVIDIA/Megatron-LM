from megatron.core.transformer.moe.offloading_planner import gen_offloading_plan, gen_assignment, gen_intermediate
from megatron.core.transformer.moe.max_flow.triton_greedy import solve_spillover_spare_matching_triton_greedy
import torch
import numpy as np
import time

import maxflow, time
import networkx as nx  # ensure networkx is installed
from scipy.sparse.csgraph import maximum_flow
from scipy.sparse import csr_matrix

def solve_spillover_spare_matching_pymaxflow_simple(spillover, spare_space, compat_map):
    # Move tensors to numpy
    if hasattr(spillover, 'cpu'):
        spillover = spillover.cpu().numpy()
    if hasattr(spare_space, 'cpu'):
        spare_space = spare_space.cpu().numpy()
    if hasattr(compat_map, 'cpu'):
        compat_map = compat_map.cpu().numpy()

    C, B = len(spillover), len(spare_space)
    g = maxflow.Graph[int]()
    color_nodes = g.add_nodes(C)
    bucket_nodes = g.add_nodes(B)

    # Source → Colors
    for i in range(C):
        cap = int(spillover[i])
        if cap > 0:
            g.add_tedge(color_nodes[i], cap, 0)

    # Colors → Buckets, record original capacities
    orig_cap = {}
    for i in range(C):
        for j in range(B):
            if compat_map[i, j] and spillover[i] > 0:
                cap = int(spillover[i])
                e_id = g.add_edge(color_nodes[i], bucket_nodes[j], cap, 0)
                orig_cap[(i, j)] = cap

    # Buckets → Sink
    for j in range(B):
        cap = int(spare_space[j])
        if cap > 0:
            g.add_tedge(bucket_nodes[j], 0, cap)

    t0 = time.time()
    maxflow_value = g.maxflow()
    t1 = time.time()
    print(f"PyMaxflow time: {t1-t0:.6f}s, max flow = {maxflow_value}")

    # Build residual-capacity graph via NetworkX
    RG = g.get_nx_graph()  # RG is a DiGraph with .weight = residual capacity

    assignments = []
    bucket_usage = [0]*B

    # For each color→bucket edge, compute flow = orig_cap - residual
    for (i, j), cap in orig_cap.items():
        u = color_nodes[i]
        v = bucket_nodes[j]
        # NetworkX edge keys are (u->v); if missing, residual is 0
        res_cap = RG[u][v]['weight'] if RG.has_edge(u, v) else 0
        flow = cap - res_cap
        if flow > 0:
            assignments.extend([(i, j)]*flow)
            bucket_usage[j] += flow

    return {
        'max_matches': maxflow_value,
        'assignments': assignments,
        'bucket_usage': bucket_usage,
        'total_spillover': int(spillover.sum()),
        'total_spare_capacity': int(spare_space.sum())
    }


def solve_spillover_spare_matching_scipy(spillover, spare_space, compat_map):
    """
    SciPy-based max flow implementation using Dinic's algorithm.
    
    This is a reliable, well-tested implementation that should give correct results.
    
    Args:
        spillover: 1D array of spillover amounts for each color
        spare_space: 1D array of spare space for each bucket  
        compat_map: 2D boolean array indicating color-bucket compatibility
        
    Returns:
        dict with 'max_matches', 'assignments', 'bucket_usage', 'total_spillover', 'total_spare_capacity'
    """
    # Convert inputs to numpy
    if hasattr(spillover, 'cpu'):
        spillover = spillover.cpu().numpy()
    if hasattr(spare_space, 'cpu'):
        spare_space = spare_space.cpu().numpy()
    if hasattr(compat_map, 'cpu'):
        compat_map = compat_map.cpu().numpy()
    
    num_colors, num_buckets = len(spillover), len(spare_space)
    
    # Graph structure: source=0, colors=1..num_colors, buckets=num_colors+1..num_colors+num_buckets, sink=num_colors+num_buckets+1
    source = 0
    sink = num_colors + num_buckets + 1
    n = sink + 1
    
    # Build sparse capacity matrix
    rows, cols, data = [], [], []
    
    # Source → Colors
    for i in range(num_colors):
        if spillover[i] > 0:
            rows.append(source)
            cols.append(1 + i)
            data.append(int(spillover[i]))
    
    # Colors → Buckets (only compatible pairs)
    for i in range(num_colors):
        for j in range(num_buckets):
            if compat_map[i, j] and spillover[i] > 0:
                color_node = 1 + i
                bucket_node = 1 + num_colors + j
                rows.append(color_node)
                cols.append(bucket_node)
                data.append(int(spillover[i]))
    
    # Buckets → Sink
    for j in range(num_buckets):
        if spare_space[j] > 0:
            bucket_node = 1 + num_colors + j
            rows.append(bucket_node)
            cols.append(sink)
            data.append(int(spare_space[j]))
    
    # Create sparse matrix
    capacity_matrix = csr_matrix((data, (rows, cols)), shape=(n, n))
    
    # Solve using Dinic's algorithm
    t0 = time.time()
    result = maximum_flow(capacity_matrix, source, sink, method='dinic')
    t1 = time.time()
    print(f"SciPy Dinic time: {t1-t0:.6f}s, max flow = {result.flow_value}")
    
    # Extract assignments from flow matrix
    assignments = []
    bucket_usage = [0] * num_buckets
    
    flow_matrix = result.flow.toarray()
    for i in range(num_colors):
        for j in range(num_buckets):
            if compat_map[i, j]:
                color_node = 1 + i
                bucket_node = 1 + num_colors + j
                flow = int(flow_matrix[color_node, bucket_node])
                if flow > 0:
                    for _ in range(flow):
                        assignments.append((i, j))
                    bucket_usage[j] += flow
    
    return {
        'max_matches': result.flow_value,
        'assignments': assignments,
        'bucket_usage': bucket_usage,
        'total_spillover': int(spillover.sum()),
        'total_spare_capacity': int(spare_space.sum())
    }


# Main test execution
EP = 64
num_expert = 128
expert_per_ep_rank = num_expert // EP
spare_expert_per_ep_rank = 2
tokens_to_ep_ranks_before_offloading = torch.load("/lustre/fsw/coreai_mlperf_training/users/nanz/moe/megatron-lm_echo/token_dist/token_dist_layer2_dprank0_tprank0.pt")
tokens_to_ep_ranks_before_offloading = tokens_to_ep_ranks_before_offloading.squeeze()
batches = tokens_to_ep_ranks_before_offloading.view(-1, EP, num_expert)
spillover, spare_space = gen_intermediate(batches[0], 0, EP, spare_expert_per_ep_rank, 0)
assignment, _, _ =  gen_assignment(batches[0], 0, EP, spare_expert_per_ep_rank, 0)
export_offloading_map = assignment > 0
export_offloading_map_ep_wise = export_offloading_map.view(num_expert, -1, spare_expert_per_ep_rank).sum(dim=-1) > 0

for i in range(batches.shape[0]):
    batch = batches[i]
    spillover, spare_space = gen_intermediate(batches[i], 0, EP, spare_expert_per_ep_rank, 0)
    print(f"\n=== Iteration {i} ===")
    
    # Filter out non-zero items to reduce problem size
    spillover_nonzero_mask = spillover > 0
    spare_space_nonzero_mask = spare_space > 0
    
    # Get indices of non-zero elements
    spillover_nonzero_indices = torch.where(spillover_nonzero_mask)[0]
    spare_space_nonzero_indices = torch.where(spare_space_nonzero_mask)[0]
    
    # Filter the tensors to only include non-zero elements
    spillover_filtered = spillover[spillover_nonzero_mask]
    spare_space_filtered = spare_space[spare_space_nonzero_mask]
    
    # Filter the compatibility map to only include non-zero elements
    export_offloading_map_ep_wise_filtered = export_offloading_map_ep_wise[spillover_nonzero_indices][:, spare_space_nonzero_indices]
    
    # ===== SCIPY SOLUTION =====
    print("\n" + "="*80)
    print("SCIPY DINIC'S ALGORITHM SOLUTION")
    print("="*80)
    result_scipy = solve_spillover_spare_matching_scipy(spillover_filtered, spare_space_filtered, export_offloading_map_ep_wise_filtered)
    print(f"Max flow: {result_scipy['max_matches']}")
    print(f"Total spillover: {result_scipy['total_spillover']}")
    print(f"Total spare capacity: {result_scipy['total_spare_capacity']}")
    print(f"Number of assignments: {len(result_scipy['assignments'])}")
    print(f"Utilization: {result_scipy['max_matches'] / result_scipy['total_spillover'] * 100:.2f}%")
    
    # ===== TRITON GREEDY SOLUTION =====
    print("\n" + "="*80)
    print("TRITON GREEDY SOLUTION")
    print("="*80)
    assignment_flows = solve_spillover_spare_matching_triton_greedy(spillover, spare_space, export_offloading_map_ep_wise)
    
    # Restore token count information from expert to spare expert level
    # assignment_flows shape: [num_expert, num_ep_rank] (EP-wise)
    # export_offloading_map shape: [num_expert, num_ep_rank, spare_expert_per_ep_rank] (spare expert-wise)
    
    # Expand assignment_flows to match spare expert dimension and reshape to 2D
    assignment_flows_expanded = assignment_flows.unsqueeze(-1).expand(-1, -1, spare_expert_per_ep_rank)  # [num_expert, num_ep_rank, spare_expert_per_ep_rank]
    assignment_flows_2d = assignment_flows_expanded.reshape(assignment_flows.size(0), -1)  # [num_expert, num_ep_rank * spare_expert_per_ep_rank]
    
    # export_offloading_map is already 2D, so no need to reshape
    # Multiply to get token counts per spare expert
    token_flows_to_spare_experts = assignment_flows_2d * export_offloading_map
    
    print(f"Assignment flows shape: {assignment_flows.shape}")
    print(f"Export offloading map shape: {export_offloading_map.shape}")
    print(f"Token flows to spare experts shape: {token_flows_to_spare_experts.shape}")
    print(f"Total matches (EP-wise): {assignment_flows.sum()}")
    print(f"Total matches (spare expert-wise): {token_flows_to_spare_experts.sum()}")
    print(f"Total spillover: {spillover.sum()}")
    print(f"Total spare capacity: {spare_space.sum()}")
    
    triton_total = assignment_flows.sum().item()
    triton_spillover = spillover.sum().item()
    print(f"Utilization: {triton_total / triton_spillover * 100:.2f}%")
    
    # ===== COMPARISON SUMMARY =====
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    print(f"{'Method':<25} {'Total Flow':<15} {'Spillover':<15} {'Capacity':<15} {'Utilization':<15}")
    print("-"*80)
    print(f"{'SciPy Dinic (optimal)':<25} {result_scipy['max_matches']:<15} {result_scipy['total_spillover']:<15} {result_scipy['total_spare_capacity']:<15} {result_scipy['max_matches'] / result_scipy['total_spillover'] * 100:<15.2f}%")
    print(f"{'Triton Greedy':<25} {triton_total:<15.0f} {triton_spillover:<15.0f} {spare_space.sum().item():<15.0f} {triton_total / triton_spillover * 100:<15.2f}%")
    print(f"\nDifference (Greedy - Optimal): {triton_total - result_scipy['max_matches']:.0f}")
    if result_scipy['max_matches'] > 0:
        print(f"Greedy optimality gap: {(1 - triton_total / result_scipy['max_matches']) * 100:.2f}%")
    
    if i >= 5:  # Test first few iterations
        break

exit()