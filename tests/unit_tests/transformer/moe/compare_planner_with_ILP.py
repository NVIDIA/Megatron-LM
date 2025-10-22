from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpInteger, LpBinary, PULP_CBC_CMD
import torch
import argparse
import megatron

from megatron.core.transformer.moe.offloading_planner import gen_offloading_plan, gen_assignment

def solve_ball_bin_placement_with_timeout(l, s, k, A=1, B=1, max_seconds=60, colors_per_group=None, max_overflow_cap=None):
    """
    Solve with time limit using CBC solver
    
    Args:
        l: torch.Tensor or list - load per color (num_colors,)
        s: torch.Tensor or list - spare capacities (num_bins,)
        k: int - maximum number of different colors allowed per bin
        A: float - weight for r_max in objective
        B: float - weight for g_max (or m_max if colors_per_group=None) in objective
        max_seconds: int - time limit for solver
        colors_per_group: int or None - number of consecutive colors per group
                         If None, uses m_max (max bins per color) instead of g_max
        max_overflow_cap: float or None - maximum allowed value for r_max (residual cap)
                         If None, no constraint on r_max
    """
    # Convert inputs to lists if they are tensors
    if isinstance(l, torch.Tensor):
        l = l.tolist()
    if isinstance(s, torch.Tensor):
        s = s.tolist()
    
    num_colors = len(l)
    num_bins = len(s)
    M = max(l)
    
    # Create the problem
    prob = LpProblem("Ball_Bin_Placement", LpMinimize)
    
    # Variables: p[i][j] where i=color/expert, j=bin/EP_rank (placement amount)
    p = [[LpVariable(f"p_{i}_{j}", lowBound=0, cat=LpInteger) 
          for j in range(num_bins)] for i in range(num_colors)]
    # m[i][j] = binary indicator: 1 if color i uses bin j
    m = [[LpVariable(f"m_{i}_{j}", cat=LpBinary) 
          for j in range(num_bins)] for i in range(num_colors)]
    # r[i] = residual (unplaced) load for color i
    r = [LpVariable(f"r_{i}", lowBound=0, cat=LpInteger) 
         for i in range(num_colors)]
    
    r_max = LpVariable("r_max", lowBound=0, cat=LpInteger)
    
    # Choose between g_max (group-based) or m_max (color-based) objective
    if colors_per_group is not None:
        # Calculate number of groups
        num_groups = (num_colors + colors_per_group - 1) // colors_per_group
        
        # g[group_idx][j] = number of colors from group_idx present in bin j
        g = [[LpVariable(f"g_{group_idx}_{j}", lowBound=0, cat=LpInteger) 
              for j in range(num_bins)] for group_idx in range(num_groups)]
        
        # g_total[group_idx] = total number of sends for group_idx (sum over all bins)
        g_total = [LpVariable(f"g_total_{group_idx}", lowBound=0, cat=LpInteger) 
                   for group_idx in range(num_groups)]
        g_max = LpVariable("g_max", lowBound=0, cat=LpInteger)
        
        # Objective: minimize r_max and g_max
        prob += A * r_max + B * g_max
    else:
        # Original objective with m_max
        m_max = LpVariable("m_max", lowBound=0, cat=LpInteger)
        prob += A * r_max + B * m_max
    
    # Add all constraints here...
    # For each color/expert: total placed + residual = total load
    for i in range(num_colors):
        prob += lpSum(p[i][j] for j in range(num_bins)) + r[i] == l[i]
        prob += r_max >= r[i]
    
    # Optional: constrain maximum overflow/residual
    if max_overflow_cap is not None:
        prob += r_max <= max_overflow_cap
    
    # For each bin/EP_rank: capacity and color/expert limit constraints
    for j in range(num_bins):
        prob += lpSum(p[i][j] for i in range(num_colors)) <= s[j]
        prob += lpSum(m[i][j] for i in range(num_colors)) <= k
    
    # Group-based or color-based spreading constraints
    if colors_per_group is not None:
        # For each group and bin, g[group_idx][j] = sum of m[i][j] for colors in that group
        for group_idx in range(num_groups):
            start_color = group_idx * colors_per_group
            end_color = min(start_color + colors_per_group, num_colors)
            
            for j in range(num_bins):
                # g[group_idx][j] = number of colors from group_idx in bin j
                prob += g[group_idx][j] == lpSum(m[i][j] for i in range(start_color, end_color))
            
            # g_total[group_idx] = total number of sends (color-bin pairs) for this group
            # This is just the sum of g[group_idx][j] over all bins j
            prob += g_total[group_idx] == lpSum(g[group_idx][j] for j in range(num_bins))
            
            # g_max tracks the maximum total sends by any group
            prob += g_max >= g_total[group_idx]
    else:
        # m_max tracks the max occurrence of any color across all bins
        for i in range(num_colors):
            prob += m_max >= lpSum(m[i][j] for j in range(num_bins))
    
    # Link placement to indicators
    for i in range(num_colors):
        for j in range(num_bins):
            prob += p[i][j] <= M * m[i][j]
    
    # Solve with time limit and optimality gap
    solver = PULP_CBC_CMD(
        timeLimit=max_seconds,    # Time limit in seconds
        gapRel=0.05,             # Stop when within 5% of optimal
        msg=0                     # Mute solver output (set to 1 to show progress)
    )
    
    status = prob.solve(solver)
    
    # Check if a feasible solution was found (even if not optimal)
    if status in [1, -1]:  # Optimal or feasible solution found
        result = {
            'status': status,
            'status_name': prob.status,
            'objective_value': prob.objective.value(),
            'r_max': r_max.value(),
            'residual': torch.tensor([r[i].value() for i in range(num_colors)], dtype=torch.float32),
            'placement': torch.tensor([[p[i][j].value() for j in range(num_bins)] 
                              for i in range(num_colors)], dtype=torch.float32),
            'indicators': torch.tensor([[m[i][j].value() for j in range(num_bins)] 
                               for i in range(num_colors)], dtype=torch.bool),
            'time_limit_hit': status == -1
        }
        
        # Add group-specific or color-specific metrics
        if colors_per_group is not None:
            result['g_max'] = g_max.value()
            result['group_total_sends'] = torch.tensor([g_total[group_idx].value() for group_idx in range(num_groups)], dtype=torch.float32)
            result['group_colors_per_bin'] = torch.tensor([[g[group_idx][j].value() for j in range(num_bins)] 
                                                           for group_idx in range(num_groups)], dtype=torch.float32)
            result['colors_per_group'] = colors_per_group
        else:
            result['m_max'] = m_max.value()
    else:
        result = {'status': status, 'message': 'No feasible solution found'}
    
    return result

def get_greedy_cost(batch, assignment):
    num_tokens_tokens_to_ep_rank = batch.sum(dim=0)
    exceeding_capacity = assignment.sum(dim=1).view(-1, 2).sum(dim=-1) - num_tokens_tokens_to_ep_rank
    num_echo_experts = (assignment>0).sum(dim=1)
    num_total_echo_experts = num_echo_experts.sum()
    num_max_echo_experts = num_echo_experts.max()
    return exceeding_capacity, num_max_echo_experts, num_total_echo_experts

def get_cost_coef():
    hidden_size = 7168
    moe_hidden_size = 2048
    flops = 5e15
    nvlink_bw = 900e9
    A = 6 * hidden_size * moe_hidden_size / flops + 2*hidden_size / nvlink_bw
    B = 3 * hidden_size * moe_hidden_size / nvlink_bw
    return A*1e6, B*1e6

def get_baseline_cost(batch, A, B, num_exp_per_ep_rank):
    max_token_count = batch.sum(dim=0).view(-1, num_exp_per_ep_rank).sum(dim=-1).max()
    return max_token_count, 0


def get_greedy_cost(spillover_tokens_per_exp, assignment, avg_token_count, num_exp_per_ep_rank):
    spillover_after_assigment = spillover_tokens_per_exp - assignment.sum(dim=1)
    max_token_count = (spillover_after_assigment.view(-1,num_exp_per_ep_rank).sum(dim=-1)).max() + avg_token_count*num_exp_per_ep_rank
    g_max = (assignment > 0).sum(dim=1).reshape(-1,num_exp_per_ep_rank).sum(dim=1).max()
    return max_token_count, g_max

def get_ilp_cost(avg_token_count, num_exp_per_ep_rank):
    spillover_after_assigment = result['residual'].cuda()
    max_token_count = (spillover_after_assigment.view(-1,num_exp_per_ep_rank).sum(dim=-1)).max() + avg_token_count*num_exp_per_ep_rank
    g_max = result['g_max']
    return max_token_count, g_max

EP = 128
spare_expert_per_ep_rank = 1
threshold_multiplier = 0.2

tokens_to_ep_ranks_before_offloading = torch.load("/lustre/fsw/coreai_mlperf_training/users/nanz/moe/megatron-lm_echo/token_dist/token_dist_layer2_dprank0_tprank0.pt")
tokens_to_ep_ranks_before_offloading = tokens_to_ep_ranks_before_offloading.squeeze()

num_expert = tokens_to_ep_ranks_before_offloading.shape[-1]
num_exp_per_ep_rank = num_expert // EP
batches = tokens_to_ep_ranks_before_offloading.view(-1, EP, num_expert)
A, B = get_cost_coef()

# Storage for statistics across iterations
num_iterations = 10
baseline_totals = []
greedy_totals = []
ilp_totals = []

for iter_idx in range(min(num_iterations, batches.shape[0])):
    batch = batches[iter_idx]
    
    print(f"\n{'='*90}")
    print(f"ITERATION {iter_idx}")
    print(f"{'='*90}")
    
    assignment, spillover_tokens_per_exp, spare_space = gen_assignment(batch, 0, EP, spare_expert_per_ep_rank, threshold_multiplier)
    avg_token_count = (batch.sum() // num_expert).item()
    x = spillover_tokens_per_exp.cpu()
    y = spare_space.cpu()
    result = solve_ball_bin_placement_with_timeout(x, y, spare_expert_per_ep_rank, A=A, B=B, max_seconds=2, colors_per_group=num_exp_per_ep_rank, max_overflow_cap=avg_token_count)
    tk_baseline, g_baseline = get_baseline_cost(batch, A, B, num_exp_per_ep_rank)
    tk_greedy, g_greedy = get_greedy_cost(spillover_tokens_per_exp, assignment, avg_token_count, num_exp_per_ep_rank)
    tk_ilp, g_ilp = get_ilp_cost(avg_token_count, num_exp_per_ep_rank)
    
    # Calculate costs
    baseline_tk_cost = tk_baseline * A
    baseline_g_cost = g_baseline * B
    baseline_total = baseline_tk_cost + baseline_g_cost
    
    greedy_tk_cost = tk_greedy * A
    greedy_g_cost = g_greedy * B
    greedy_total = greedy_tk_cost + greedy_g_cost
    
    ilp_tk_cost = tk_ilp * A
    ilp_g_cost = g_ilp * B
    ilp_total = ilp_tk_cost + ilp_g_cost
    
    # Store for summary
    baseline_totals.append(baseline_total)
    greedy_totals.append(greedy_total)
    ilp_totals.append(ilp_total)
    
    # Calculate baseline capacity
    baseline_capacity = avg_token_count * num_exp_per_ep_rank
    
    # Print comparison table for this iteration
    print(f"{'Method':<15} {'Token Cost':<20} {'Echo Cost':<20} {'Total Cost':<20} {'tk/capacity':<15}")
    print("-"*90)
    print(f"{'Baseline':<15} {baseline_tk_cost:<20.6f} {baseline_g_cost:<20.6f} {baseline_total:<20.6f} {tk_baseline/baseline_capacity:<15.4f}")
    print(f"{'Greedy':<15} {greedy_tk_cost:<20.6f} {greedy_g_cost:<20.6f} {greedy_total:<20.6f} {tk_greedy/baseline_capacity:<15.4f}")
    print(f"{'ILP':<15} {ilp_tk_cost:<20.6f} {ilp_g_cost:<20.6f} {ilp_total:<20.6f} {tk_ilp/baseline_capacity:<15.4f}")
    print("-"*90)
    print(f"{'Speedup vs Baseline':<20}")
    print(f"  {'Greedy':<18} {baseline_total / greedy_total:>23.2f}x")
    print(f"  {'ILP':<18} {baseline_total / ilp_total:>23.2f}x")
    print(f"{'ILP vs Greedy Speedup':<20} {greedy_total / ilp_total:>23.2f}x")

# Print summary statistics
print(f"\n{'='*90}")
print(f"SUMMARY STATISTICS OVER {len(baseline_totals)} ITERATIONS")
print(f"{'='*90}")
print(f"{'Method':<20} {'Avg Speedup':<20} {'Min Speedup':<20} {'Max Speedup':<20}")
print("-"*90)

avg_greedy_speedup = sum(b/g for b, g in zip(baseline_totals, greedy_totals)) / len(baseline_totals)
min_greedy_speedup = min(b/g for b, g in zip(baseline_totals, greedy_totals))
max_greedy_speedup = max(b/g for b, g in zip(baseline_totals, greedy_totals))
print(f"{'Greedy vs Baseline':<20} {avg_greedy_speedup:<20.2f}x {min_greedy_speedup:<20.2f}x {max_greedy_speedup:<20.2f}x")

avg_ilp_speedup = sum(b/i for b, i in zip(baseline_totals, ilp_totals)) / len(baseline_totals)
min_ilp_speedup = min(b/i for b, i in zip(baseline_totals, ilp_totals))
max_ilp_speedup = max(b/i for b, i in zip(baseline_totals, ilp_totals))
print(f"{'ILP vs Baseline':<20} {avg_ilp_speedup:<20.2f}x {min_ilp_speedup:<20.2f}x {max_ilp_speedup:<20.2f}x")

avg_ilp_vs_greedy = sum(g/i for g, i in zip(greedy_totals, ilp_totals)) / len(greedy_totals)
min_ilp_vs_greedy = min(g/i for g, i in zip(greedy_totals, ilp_totals))
max_ilp_vs_greedy = max(g/i for g, i in zip(greedy_totals, ilp_totals))
print(f"{'ILP vs Greedy':<20} {avg_ilp_vs_greedy:<20.2f}x {min_ilp_vs_greedy:<20.2f}x {max_ilp_vs_greedy:<20.2f}x")
print("="*90)

# import pdb; pdb.set_trace()