# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Training-wide sparse-attention loss tracking and gradient-scale state.

Operation modules attach indexer losses here; training and pipeline scheduling
own scale updates and reductions across the model's layers and ranks.
"""

from typing import Optional

import torch

from megatron.core.process_groups_config import ProcessGroupCollection


class DSAIndexerLossLoggingHelper:
    """Helper class for logging sparse attention indexer losses."""

    tracker = {}

    @staticmethod
    def save_loss_to_tracker(
        loss: torch.Tensor,
        layer_number: int,
        num_layers: int,
        reduce_group: torch.distributed.ProcessGroup = None,
        avg_group: torch.distributed.ProcessGroup = None,
    ):
        """Save the indexer loss for logging.

        Args:
            loss: The loss tensor.
            layer_number: Layer index of the loss, 1-indexed.
            num_layers: The number of total layers.
            reduce_group: The group for reducing the loss.
            avg_group: The group for averaging the loss.
        """
        # Skip indexer loss logging if layer_number is None.
        if layer_number is None:
            return

        tracker = DSAIndexerLossLoggingHelper.tracker
        # Hybrid MTP layer numbers can exceed ``num_layers + mtp_num_layers``
        # because every prediction depth can contain multiple hybrid layers.
        needed = max(num_layers, layer_number)
        if "values" not in tracker:
            tracker["values"] = torch.zeros(needed, device=torch.cuda.current_device())
        elif tracker["values"].shape[0] < needed:
            grown = torch.zeros(
                needed, device=tracker["values"].device, dtype=tracker["values"].dtype
            )
            grown[: tracker["values"].shape[0]] = tracker["values"]
            tracker["values"] = grown
        tracker["values"][layer_number - 1] += loss.detach()
        tracker["reduce_group"] = reduce_group
        tracker["avg_group"] = avg_group

    @staticmethod
    def clean_loss_in_tracker(preserve_groups: bool = False):
        """Clear the indexer losses."""
        tracker = DSAIndexerLossLoggingHelper.tracker
        reduce_group = tracker.get("reduce_group") if preserve_groups else None
        avg_group = tracker.get("avg_group") if preserve_groups else None
        if "values" in tracker:
            tracker["values"].zero_()
        tracker["reduce_group"] = reduce_group
        tracker["avg_group"] = avg_group

    @staticmethod
    def reduce_loss_in_tracker(
        pg_collection: ProcessGroupCollection, num_layers: Optional[int] = None
    ):
        """Collect and reduce indexer losses across every pipeline rank.

        Args:
            pg_collection: Process groups used for pipeline and data-parallel reductions.
            num_layers: Total number of decoder and MTP layers. When provided, ranks without
                local indexer losses contribute zeros to the pipeline-wide reduction.
        """
        tracker = DSAIndexerLossLoggingHelper.tracker
        pp_group = pg_collection.pp

        # Pipeline ranks can own different attention variants, so first agree on
        # a common tracker size. Cache the result because layer allocation is
        # static and the negotiation requires a device-to-host synchronization.
        if tracker.get("agreed_size") is not None:
            size = tracker["agreed_size"]
        else:
            local_size = tracker["values"].shape[0] if "values" in tracker else (num_layers or 0)
            size_t = torch.tensor(
                [local_size], device=torch.cuda.current_device(), dtype=torch.long
            )
            torch.distributed.all_reduce(size_t, op=torch.distributed.ReduceOp.MAX, group=pp_group)
            size = int(size_t.item())
            tracker["agreed_size"] = size
        if size == 0:
            return
        if "values" not in tracker:
            tracker["values"] = torch.zeros(size, device=torch.cuda.current_device())
        elif tracker["values"].shape[0] < size:
            grown = torch.zeros(
                size, device=tracker["values"].device, dtype=tracker["values"].dtype
            )
            grown[: tracker["values"].shape[0]] = tracker["values"]
            tracker["values"] = grown
        values = tracker["values"]

        torch.distributed.all_reduce(values, group=pp_group)
        # Reduce indexer losses across ranks.
        if tracker.get('reduce_group') is not None:
            torch.distributed.all_reduce(values, group=tracker.get('reduce_group'))
        if tracker.get('avg_group') is not None:
            torch.distributed.all_reduce(
                values, group=tracker['avg_group'], op=torch.distributed.ReduceOp.AVG
            )
        torch.distributed.all_reduce(
            values, group=pg_collection.dp, op=torch.distributed.ReduceOp.AVG
        )

    @staticmethod
    def track_indexer_metrics(
        loss_scale: float,
        iteration: int,
        writer,
        pg_collection: ProcessGroupCollection,
        wandb_writer=None,
        total_loss_dict=None,
        per_layer_logging: bool = False,
        num_layers: Optional[int] = None,
        num_indexer_layers: Optional[int] = None,
        preserve_groups: bool = False,
    ):
        """Track the sparse attention indexer metrics for logging.

        Args:
            loss_scale: Scale factor for the loss.
            iteration: Current training iteration.
            writer: TensorBoard writer.
            pg_collection: Process groups used for pipeline and data-parallel reductions.
            wandb_writer: Weights & Biases writer.
            total_loss_dict: Dictionary to accumulate total losses.
            per_layer_logging: Whether to log per-layer losses.
            num_layers: Total number of decoder and MTP layers. Passing it makes ranks
                without a local indexer participate in the pipeline reduction.
            num_indexer_layers: Number of layers that own an indexer. Defaults to the
                tracker size when every tracked layer owns one.
            preserve_groups: Keep the saved reduction groups for CUDA Graph replays.
        """
        DSAIndexerLossLoggingHelper.reduce_loss_in_tracker(
            pg_collection=pg_collection, num_layers=num_layers
        )
        tracker = DSAIndexerLossLoggingHelper.tracker
        if "values" not in tracker:
            return

        indexer_loss_values = tracker["values"] * loss_scale
        if num_indexer_layers is None:
            num_indexer_layers = indexer_loss_values.shape[0]
        avg_indexer_loss = indexer_loss_values.sum() / max(num_indexer_layers, 1)

        # Log average loss
        if total_loss_dict is not None:
            if "indexer loss" in total_loss_dict:
                total_loss_dict["indexer loss"] += avg_indexer_loss
            else:
                total_loss_dict["indexer loss"] = avg_indexer_loss

        if writer is not None:
            writer.add_scalar("indexer loss", avg_indexer_loss, iteration)

        if wandb_writer is not None:
            wandb_writer.log({"indexer loss": avg_indexer_loss}, iteration)

        DSAIndexerLossLoggingHelper.clean_loss_in_tracker(preserve_groups=preserve_groups)


class DSAIndexerLossAutoScaler(torch.autograd.Function):
    """An AutoScaler that triggers the backward pass and scales the grad for indexer loss.

    This custom autograd function attaches a KL divergence loss to the activation
    to train the indexer to predict attention scores without affecting the forward pass.
    """

    main_loss_backward_scale: Optional[torch.Tensor] = None

    @staticmethod
    def forward(ctx, output: torch.Tensor, indexer_loss: torch.Tensor):
        """Preserve the indexer_loss by storing it in the context to avoid garbage collection.

        Args:
            output: The output tensor (activation).
            indexer_loss: The indexer KL divergence loss tensor.

        Returns:
            torch.Tensor: The output tensor unchanged.
        """
        ctx.save_for_backward(indexer_loss)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """Compute and scale the gradient for indexer loss.

        Args:
            grad_output: The gradient of the output.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The gradient of the output, scaled indexer loss
                gradient.
        """
        (indexer_loss,) = ctx.saved_tensors
        if DSAIndexerLossAutoScaler.main_loss_backward_scale is None:
            DSAIndexerLossAutoScaler.main_loss_backward_scale = torch.tensor(
                1.0, device=indexer_loss.device
            )
        indexer_loss_backward_scale = DSAIndexerLossAutoScaler.main_loss_backward_scale.to(
            device=indexer_loss.device
        )
        scaled_indexer_loss_grad = torch.ones_like(indexer_loss) * indexer_loss_backward_scale
        return grad_output, scaled_indexer_loss_grad

    @staticmethod
    def set_loss_scale(scale: torch.Tensor):
        """Set the scale of the indexer loss.

        Args:
            scale: The scale value to set.
        """
        if not isinstance(scale, torch.Tensor):
            raise TypeError("DSAIndexerLossAutoScaler.set_loss_scale requires a torch.Tensor.")
        scale = scale.detach()

        if DSAIndexerLossAutoScaler.main_loss_backward_scale is None:
            DSAIndexerLossAutoScaler.main_loss_backward_scale = scale
        else:
            DSAIndexerLossAutoScaler.main_loss_backward_scale.copy_(scale)
