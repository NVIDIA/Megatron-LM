from typing import Iterable, List, Callable
import torch
from megatron.core.utils import get_attr_wrapped_model

# for Modality-Decoupled Parallelism(MDP)
class ModalityDecoupledParallelism:
    """
    ModalityDecoupledParallelism is a class that handles the modality-decoupled parallelism.
    """
    def __init__(self,forward_step_func: Callable, data_iterator: Iterable, model: torch.nn.Module, pp_rank: int):
        """
        Initialize the ModalityBridge.
        """
        self.forward_step_func = forward_step_func
        self.model = model
        self.data_iterator = data_iterator
        self.pp_rank = pp_rank
        self.set_execute_mode = get_attr_wrapped_model(model, "set_execute_mode")
        self.set_batch = get_attr_wrapped_model(model, "set_batch")
        self.set_vision_model_output = get_attr_wrapped_model(model, "set_vision_model_output")
    
    def get_global_batches(self, num_microbatches: int):
        execute_mode_only_get_batch = {
            "execute_get_batch": True,
            "execute_vision_model_forward": False,
            "execute_language_model_forward": False,
        }
        global_batches = []
        self.set_execute_mode(execute_mode_only_get_batch)
        for i in range(num_microbatches):
            with torch.no_grad():
                batch, loss_function = self.forward_step_func(self.data_iterator, self.model)
            global_batches.append(batch)
        return global_batches

    def balance_data(self):
        print(f"for debug, in balance_data, TODO: implement balance_data")
        return None # TODO: Implement this

    def vision_model_forward(self, global_batches: List[dict[str, torch.Tensor]], num_microbatches: int):
        execute_mode_only_vision_model_forward = {
            "execute_get_batch": False,
            "execute_vision_model_forward": True,
            "execute_language_model_forward": False,
        }
        vision_model_outputs = []

        self.set_execute_mode(execute_mode_only_vision_model_forward)
        for i in range(num_microbatches):
            self.set_batch(global_batches[i])
            with torch.no_grad():
                output_tensor, loss_function = self.forward_step_func(self.data_iterator, self.model)
            vision_model_outputs.append(output_tensor)
        return vision_model_outputs

    def modality_bridge(self):
        print(f"for debug, in modality_bridge, TODO: implement modality_bridge")
        return None # TODO: Implement this

    def before_language_model_forward_step(self, global_batches: List[dict[str, torch.Tensor]], vision_model_outputs: List[torch.Tensor], microbatch_id: int):
        execute_mode_only_language_model_forward = {
            "execute_get_batch": False,
            "execute_vision_model_forward": False,
            "execute_language_model_forward": True,
        }
        self.set_execute_mode(execute_mode_only_language_model_forward)
        if self.pp_rank == 0:
            batch = global_batches[microbatch_id]
            self.set_batch(batch)
            vision_model_output = vision_model_outputs[microbatch_id]
            self.set_vision_model_output(vision_model_output)
            # TODO(shifang): Need to check whether the grad calculated from vision_model_output is correct.
            visual_embeds = vision_model_output["visual_embeds"]
            visual_embeds.requires_grad = True
            visual_embeds.retain_grad()
            deepstack_visual_embeds_list = vision_model_output["deepstack_visual_embeds"]
            for deepstack_visual_embeds in deepstack_visual_embeds_list:
                deepstack_visual_embeds.requires_grad = True
                deepstack_visual_embeds.retain_grad()

    def vision_model_backward(self, global_batches: List[dict[str, torch.Tensor]], vision_model_outputs: List[torch.Tensor]):
        print(f"for debug, in vision_model_backward, TODO: implement vision_model_backward")
        # print(f"for debug, in vision_model_backward, vision_embeds.grad: {vision_model_outputs[0]['visual_embeds'].grad}")
        return None # TODO: Implement this
