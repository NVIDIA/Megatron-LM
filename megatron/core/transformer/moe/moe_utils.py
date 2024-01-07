import torch


def switch_load_balancing_loss_func(gates, mask, moe_aux_loss_coeff):
    """Calculate the auxiliary loss for better load balacing. 
    Please refer to the Switch Transformer paper (https://arxiv.org/abs/2101.03961) for details.

    Args:
        gates (torch.Tensor): The gates tensor representing the routing probabilities for each expert.
        mask (torch.Tensor): The 2D mask tensor indicating which experts are selected.

    Returns:
        torch.Tensor: The auxiliary loss for load balancing.
    """
    num_experts = mask.size(-1)
    gates_mean = gates.mean(dim=0)
    selection_mean = mask.float().mean(dim=0)
    aux_loss = torch.sum(gates_mean * selection_mean) * num_experts
    aux_loss *= moe_aux_loss_coeff
    return aux_loss


def z_loss_func(logits):
    """Encourages the router's logits to remain small to enhance stability.
    Please refer to the ST-MoE paper (https://arxiv.org/pdf/2202.08906.pdf) for details.
    
    Args:
        logits (torch.Tensor): The logits of the router.
    
    Returns:
        torch.Tensor: The logits after applying the z-loss.
    """

    z_loss = torch.mean(torch.square(torch.logsumexp(logits, dim=-1)))
    return z_loss


def sinkhorn(cost: torch.Tensor, tol: float = 0.0001):
    """Sinkhorn based MoE routing function"""
    cost = torch.exp(cost)
    d0 = torch.ones(cost.size(0), device=cost.device, dtype=cost.dtype)
    d1 = torch.ones(cost.size(1), device=cost.device, dtype=cost.dtype)

    eps = 0.00000001
    error = 1e9
    d1_old = d1
    while error > tol:
        d0 = (1 / d0.size(0)) * 1 / (torch.sum(d1 * cost, 1) + eps)
        d1 = (1 / d1.size(0)) * 1 / (torch.sum(d0.unsqueeze(1) * cost, 0) + eps)
        error = torch.mean(torch.abs(d1_old - d1))
        d1_old = d1
    return d1 * cost * d0.unsqueeze(1)
