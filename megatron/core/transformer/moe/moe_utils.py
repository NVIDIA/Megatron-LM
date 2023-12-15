import torch


def switch_load_balancing_loss_func(config, gates, mask):
    """Calculate the auxiliary loss for better load balacing. 
    Please refer to the Switch Transformer paper (https://arxiv.org/abs/2101.03961) for details.

    Args:
        gates (torch.Tensor): The gates tensor representing the routing probabilities for each expert.
        mask (torch.Tensor): The 2D mask tensor indicating which experts are selected.

    Returns:
        torch.Tensor: The auxiliary loss for load balancing.
    """
    num_experts = mask.size(1)
    assert num_experts == config.num_moe_experts
    gates_mean = gates.mean(dim=0)
    selection_mean = mask.float().mean(dim=0)
    aux_loss = torch.sum(gates_mean * selection_mean) * num_experts
    aux_loss *= config.aux_loss_coeff
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
