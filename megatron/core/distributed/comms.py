import torch.distributed


def all_gather_into_tensor(output_tensor, input_tensor, group=None, async_op=False) -> None:
    if hasattr(torch.distributed, "all_gather_into_tensor"):
        handle = torch.distributed.all_gather_into_tensor(output_tensor, input_tensor, group=group, async_op=async_op)
    else:
        handle = torch.distributed._all_gather_base(output_tensor, input_tensor, group=group, async_op=async_op)

    return handle


def reduce_scatter_tensor(output_tensor, input_tensor, op=torch.distributed.ReduceOp.SUM, group=None, async_op=False) -> None:
    if hasattr(torch.distributed, "reduce_scatter_tensor"):
        handle = torch.distributed.reduce_scatter_tensor(output_tensor, input_tensor, op=op, group=group, async_op=async_op)
    else:
        handle = torch.distributed._reduce_scatter_base(output_tensor, input_tensor, op=op, group=group, async_op=async_op)

    return handle
