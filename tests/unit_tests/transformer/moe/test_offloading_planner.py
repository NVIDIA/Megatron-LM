import torch
import argparse
import megatron

from megatron.core.transformer.moe.offloading_planner import gen_offloading_plan

def baseline_routing(scores, num_experts, EP, topk):
    # Pick top-k largest along columns (dim=1)
    topk_values, topk_indices = torch.topk(scores, k=topk, dim=1)
    
    # Create assignment matrix: set top-k positions to 1, others to 0
    assignment = torch.zeros(scores.shape[0], num_experts, device=scores.device, dtype=torch.bool)
    
    # Use scatter_ to set the top-k positions to 1
    # scatter_(dim, index, src) where src is the values to scatter
    assignment.scatter_(1, topk_indices, 1)
    
    assignment_reshaped = assignment.view(EP, -1, num_experts)
    tokens_per_expert_from_ep_rank = assignment_reshaped.sum(dim=1)  # Sum along dim=1

    scores_reshaped = scores.view(EP, -1, num_experts)
    probs = scores_reshaped * assignment_reshaped

    return assignment_reshaped, tokens_per_expert_from_ep_rank, probs

def result_sanity_check(routing_map_all_rank, rerouting_map_all_rank, rerouted_probs_all_rank, original_probs_all_rank, expert_offloading_map, topk, ep):
    num_home_experts = routing_map_all_rank.shape[-1]
    num_spare_experts = expert_offloading_map.shape[-1] - num_home_experts
    routing_map = routing_map_all_rank.view(-1, routing_map_all_rank.shape[-1])
    rerouting_map = rerouting_map_all_rank.view(-1, rerouting_map_all_rank.shape[-1])
    num_home_expert_per_ep_rank = routing_map.shape[-1] // ep
    num_spare_expert_per_ep_rank = expert_offloading_map.shape[-1] // ep
    # topk check
    topk_checked = (rerouting_map.sum(dim=1)==topk).all()
    if not topk_checked:
        print("❌ Top-k check failed")
        import pdb; pdb.set_trace()
        raise ValueError("Top-k check failed")

    # expert routing check
    home_expert_idx, spare_expert_idx = torch.where(expert_offloading_map)
    # OR operation: rerouting_map[:, home_expert_idx] |= rerouting_map[:, spare_expert_idx]
    # Use scatter_add_ to perform the OR operation - need to copy source to avoid in-place issues
    rerouting_map_undo_reroute = rerouting_map[:, :num_home_experts].clone()
    rerouting_map_undo_reroute.scatter_add_(1, home_expert_idx.unsqueeze(0).expand(rerouting_map.shape[0], -1), rerouting_map[:, num_home_experts + spare_expert_idx])
    # Convert back to boolean (since scatter_add_ adds values, we need to convert >0 to True)
    rerouting_map_undo_reroute = rerouting_map_undo_reroute > 0
    routing_equivalence_checked = (rerouting_map_undo_reroute[:, :num_home_experts] == routing_map).all()
    if not routing_equivalence_checked:
        print("❌ Routing equivalence check failed")
        import pdb; pdb.set_trace()
        raise ValueError("Routing equivalence check failed")

    # probability routing check
    rerouted_probs = rerouted_probs_all_rank.view(-1, rerouted_probs_all_rank.shape[-1])
    original_probs = original_probs_all_rank.view(-1, original_probs_all_rank.shape[-1])
    # Add probabilities from spare experts back to home experts
    rerouted_probs_undo_reroute = rerouted_probs[:, :num_home_experts].clone()
    rerouted_probs_undo_reroute.scatter_add_(1, home_expert_idx.unsqueeze(0).expand(rerouted_probs.shape[0], -1), rerouted_probs[:, num_home_experts + spare_expert_idx])
    # should be bitwise accurate
    prob_sum_equivalence_checked = (original_probs == rerouted_probs_undo_reroute).all()
    if not prob_sum_equivalence_checked:
        print("❌ Probability equivalence check failed")
        import pdb; pdb.set_trace()
        raise ValueError("Probability sum equivalence check failed")

    # count max tokens per ep rank
    tokens_to_ep_ranks_before_offloading = routing_map.sum(dim=0).reshape(ep, -1).sum(dim=1)
    tokens_to_ep_ranks_after_offloading_home = rerouting_map[:, :num_home_experts].sum(dim=0).reshape(ep, -1).sum(dim=1)
    tokens_to_ep_ranks_after_offloading_spare = rerouting_map[:, num_home_experts:].sum(dim=0).reshape(ep, -1).sum(dim=1)
    tokens_to_ep_ranks_after_offloading = tokens_to_ep_ranks_after_offloading_home + tokens_to_ep_ranks_after_offloading_spare

    max_tokens_to_ep_ranks_before_offloading = tokens_to_ep_ranks_before_offloading.max()
    max_tokens_to_ep_ranks_after_offloading = tokens_to_ep_ranks_after_offloading.max()
    max_num_offloaded_expert = expert_offloading_map.sum(dim=1).max()
    return max_tokens_to_ep_ranks_before_offloading, max_tokens_to_ep_ranks_after_offloading, max_num_offloaded_expert

from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpInteger, LpBinary, PULP_CBC_CMD

def solve_ball_bin_placement_with_timeout(x, y, k, A=1, B=1, max_seconds=60):
    """
    Solve with time limit using CBC solver
    """
    num_colors = len(x)
    num_bins = len(y)
    M = max(x)
    
    # Create the problem
    prob = LpProblem("Ball_Bin_Placement", LpMinimize)
    
    # Variables (same as before)
    s = [[LpVariable(f"s_{i}_{j}", lowBound=0, cat=LpInteger) 
          for j in range(num_colors)] for i in range(num_bins)]
    r = [[LpVariable(f"r_{i}_{j}", cat=LpBinary) 
          for j in range(num_colors)] for i in range(num_bins)]
    z = [LpVariable(f"z_{j}", lowBound=0, cat=LpInteger) 
         for j in range(num_colors)]
    
    z_max = LpVariable("z_max", lowBound=0, cat=LpInteger)
    r_max = LpVariable("r_max", lowBound=0, cat=LpInteger)
    
    # Objective and constraints (same as before)
    prob += A * z_max + B * r_max
    
    # Add all constraints here...
    for j in range(num_colors):
        prob += lpSum(s[i][j] for i in range(num_bins)) + z[j] == x[j]
        prob += z_max >= z[j]
    
    for i in range(num_bins):
        prob += lpSum(s[i][j] for j in range(num_colors)) <= y[i]
        prob += lpSum(r[i][j] for j in range(num_colors)) <= k
        prob += r_max >= lpSum(r[i][j] for j in range(num_colors))
    
    for i in range(num_bins):
        for j in range(num_colors):
            prob += s[i][j] <= M * r[i][j]
    
    # Solve with time limit and optimality gap
    solver = PULP_CBC_CMD(
        maxSeconds=max_seconds,    # Time limit in seconds
        fracGap=0.05,             # Stop when within 5% of optimal
        msg=1                     # Show solver progress
    )
    
    status = prob.solve(solver)
    
    # Check if a feasible solution was found (even if not optimal)
    if status in [1, -1]:  # Optimal or feasible solution found
        result = {
            'status': status,
            'status_name': prob.status,
            'objective_value': prob.objective.value(),
            'z_max': z_max.value(),
            'r_max': r_max.value(),
            'residual_balls': [z[j].value() for j in range(num_colors)],
            'ball_placement': [[s[i][j].value() for j in range(num_colors)] 
                              for i in range(num_bins)],
            'color_indicators': [[r[i][j].value() for j in range(num_colors)] 
                               for i in range(num_bins)],
            'time_limit_hit': status == -1
        }
    else:
        result = {'status': status, 'message': 'No feasible solution found'}
    
    return result

def main():
    parser = argparse.ArgumentParser(description='Generate tensor and perform top-k selection')
    parser.add_argument('--num-experts', type=int, default=8, help='Number of experts')
    parser.add_argument('--EP', type=int, default=2, help='Experts per token')
    parser.add_argument('--topk', type=int, default=2, help='Top-k selection')
    parser.add_argument('--batch', type=int, default=4, help='Batch size')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--threshold-multiplier', type=float, default=0.0, help='Threshold multiplier for average tokens per expert')
    parser.add_argument('--spare-expert-per-ep-rank', type=int, default=1, help='Number of spare experts per EP rank')
    parser.add_argument('--batch-test', action='store_true', help='Enable batch test mode with CUDA graph capture')
    parser.add_argument('--num-iterations', type=int, default=100, help='Number of iterations for batch test mode')
    parser.add_argument('--assignment-algorithm', type=str, default='approx_bin_packing', 
                        choices=['one_shot_greedy', 'approx_bin_packing'],
                        help='Assignment algorithm to use (default: one_shot_greedy)')

    
    args = parser.parse_args()
    

    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    # Generate random tensor on GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Generate skewed distribution for scores
    # First, sample probabilities for each expert from uniform distribution
    expert_probs = torch.rand(args.num_experts, device=device)  # Shape: [num_experts]
    
    # Expand expert_probs to match batch size for broadcasting
    expert_probs_expanded = expert_probs.unsqueeze(0).expand(args.batch * args.EP, args.num_experts)  # Shape: [batch*EP, num_experts]
    
    # Generate scores using normal distribution with mean=expert_probs_expanded and std=0.5*expert_probs_expanded
    scores = torch.normal(expert_probs_expanded, 1 * expert_probs_expanded)
    
    # Call the router function
    routing_map_all_rank, tokens_per_expert_from_ep_rank, probs = baseline_routing(scores, args.num_experts, args.EP, args.topk)

    # Compile the balanced_routing function
    compiled_balanced_routing = torch.compile(gen_offloading_plan)
    # compiled_balanced_routing = gen_offloading_plan
    
    if args.batch_test:
        # Use the same static-shape CUDA graph flow as in the else-branch
        # Warm up run
        ep_rank_static = torch.zeros(1, device=device, dtype=torch.int32)
        rerouting_map, rerouted_probs, expert_offloading_map = compiled_balanced_routing(routing_map_all_rank[0], probs[0], tokens_per_expert_from_ep_rank, ep_rank_static, args.EP, args.spare_expert_per_ep_rank, args.threshold_multiplier, assignment_algorithm=args.assignment_algorithm)
        
        routing_map_static = torch.empty_like(routing_map_all_rank[0])
        routing_map_static.copy_(routing_map_all_rank[0])
        probs_static = torch.empty_like(probs[0])
        probs_static.copy_(probs[0])
        rerouting_map_all_rank = torch.empty((routing_map_all_rank.shape[0], *rerouting_map.shape), dtype=rerouting_map.dtype, device=rerouting_map.device)
        rerouted_probs_all_rank = torch.empty((routing_map_all_rank.shape[0], *rerouted_probs.shape), dtype=rerouted_probs.dtype, device=rerouted_probs.device)
        # Capture CUDA graph
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            rerouting_map_static, rerouted_probs_static, expert_offloading_map_static = compiled_balanced_routing(routing_map_static, probs_static, tokens_per_expert_from_ep_rank, ep_rank_static, args.EP, args.spare_expert_per_ep_rank, args.threshold_multiplier, assignment_algorithm=args.assignment_algorithm)
        # Arrays to store statistics across iterations
        max_tokens_before_offloading = []
        max_tokens_after_offloading = []
        max_num_offloaded_expert = []
        
        # Run multiple iterations: regenerate scores and routing each time
        for i in range(args.num_iterations):
            torch.manual_seed(args.seed + i)
            expert_probs = torch.rand(args.num_experts, device=device)
            expert_probs_expanded = expert_probs.unsqueeze(0).expand(
                args.batch * args.EP, args.num_experts
            )
            scores = torch.normal(expert_probs_expanded, 1 * expert_probs_expanded)

            routing_map_all_rank, tokens_per_expert_from_ep_rank_iter, probs_iter = baseline_routing(
                scores, args.num_experts, args.EP, args.topk
            )

            tokens_per_expert_from_ep_rank.copy_(tokens_per_expert_from_ep_rank_iter)
            for ep in range(routing_map_all_rank.shape[0]):
                ep_rank = torch.tensor([ep], device=device)
                ep_rank_static.copy_(ep_rank)
                routing_map_static.copy_(routing_map_all_rank[ep])
                probs_static.copy_(probs_iter[ep])
                graph.replay()
                # rerouting_map_static, expert_offloading_map_static = compiled_balanced_routing(routing_map_static, tokens_per_expert_from_ep_rank, ep_rank_static, args.EP, args.spare_expert_per_ep_rank, args.threshold_multiplier)
                rerouting_map_all_rank[ep].copy_(rerouting_map_static)
                rerouted_probs_all_rank[ep].copy_(rerouted_probs_static)

            # Per-iteration sanity check and collect statistics
            max_tokens_before, max_tokens_after, max_offloaded = result_sanity_check(
                routing_map_all_rank,
                rerouting_map_all_rank,
                rerouted_probs_all_rank,
                probs_iter,
                expert_offloading_map_static,
                args.topk,
                args.EP,
            )
            
            # Store statistics for this iteration
            max_tokens_before_offloading.append(max_tokens_before.item())
            max_tokens_after_offloading.append(max_tokens_after.item())
            max_num_offloaded_expert.append(max_offloaded.item())
        
        # Calculate and print summary statistics
        print(f"\n=== Batch Test Results ({args.num_iterations} iterations) ===")
        print(f"Max tokens before offloading - Mean: {sum(max_tokens_before_offloading)/len(max_tokens_before_offloading):.2f}, Max: {max(max_tokens_before_offloading)}, Min: {min(max_tokens_before_offloading)}")
        print(f"Max tokens after offloading - Mean: {sum(max_tokens_after_offloading)/len(max_tokens_after_offloading):.2f}, Max: {max(max_tokens_after_offloading)}, Min: {min(max_tokens_after_offloading)}")
        print(f"Max number of offloaded experts - Mean: {sum(max_num_offloaded_expert)/len(max_num_offloaded_expert):.2f}, Max: {max(max_num_offloaded_expert)}, Min: {min(max_num_offloaded_expert)}")
    
    else:
        # Normal mode: single run with sanity check
        # Warm up run
        ep_rank_static = torch.zeros(1, device=device, dtype=torch.int32)
        # torch.cuda.profiler.start()
        for i in range(10):
            torch.cuda.nvtx.range_push(str(i))
            rerouting_map, rerouted_probs, expert_offloading_map = compiled_balanced_routing(routing_map_all_rank[0], probs[0], tokens_per_expert_from_ep_rank, ep_rank_static, args.EP, args.spare_expert_per_ep_rank, args.threshold_multiplier, assignment_algorithm=args.assignment_algorithm)
            torch.cuda.nvtx.range_pop()
        # torch.cuda.profiler.stop()
        # torch.cuda.profiler.start()
        routing_map_static = torch.empty_like(routing_map_all_rank[0])
        routing_map_static.copy_(routing_map_all_rank[0])
        probs_static = torch.empty_like(probs[0])
        probs_static.copy_(probs[0])
        rerouting_map_all_rank = torch.empty((routing_map_all_rank.shape[0], *rerouting_map.shape), dtype=rerouting_map.dtype, device=rerouting_map.device)
        rerouted_probs_all_rank = torch.empty((routing_map_all_rank.shape[0], *rerouted_probs.shape), dtype=rerouted_probs.dtype, device=rerouted_probs.device)
        # Capture CUDA graph
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            rerouting_map_static, rerouted_probs_static, expert_offloading_map_static = compiled_balanced_routing(routing_map_static, probs_static, tokens_per_expert_from_ep_rank, ep_rank_static, args.EP, args.spare_expert_per_ep_rank, args.threshold_multiplier, assignment_algorithm=args.assignment_algorithm)
        
        torch.cuda.profiler.start()
        # Replay the graph
        for ep in range(routing_map_all_rank.shape[0]):
            ep_rank = torch.tensor([ep], device=device)
            ep_rank_static.copy_(ep_rank)
            routing_map_static.copy_(routing_map_all_rank[ep])
            probs_static.copy_(probs[ep])
            graph.replay()
            # rerouting_map_static, rerouted_probs_static, expert_offloading_map_static = compiled_balanced_routing(routing_map_static, probs_static, tokens_per_expert_from_ep_rank, ep_rank_static, args.EP, args.spare_expert_per_ep_rank, args.threshold_multiplier, assignment_algorithm="approx_bin_packing")
            rerouting_map_all_rank[ep].copy_(rerouting_map_static)
            rerouted_probs_all_rank[ep].copy_(rerouted_probs_static)
        torch.cuda.profiler.stop()
        # Run sanity check
        max_tokens_to_ep_ranks_before_offloading, max_tokens_to_ep_ranks_after_offloading, max_num_offloaded_expert = result_sanity_check(routing_map_all_rank, rerouting_map_all_rank, rerouted_probs_all_rank, probs, expert_offloading_map_static, args.topk, args.EP)
        print(f"Max tokens to EP ranks before offloading: {max_tokens_to_ep_ranks_before_offloading}")
        print(f"Max tokens to EP ranks after offloading: {max_tokens_to_ep_ranks_after_offloading}")
        print(f"Max number of offloaded expert: {max_num_offloaded_expert}")

if __name__ == "__main__":
    main()
