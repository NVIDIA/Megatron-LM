from argparse import ArgumentParser

from megatron.bridge import AutoBridge
from megatron.bridge.utils.common_utils import print_rank_0

try:
    from megatron.bridge.utils.common_utils import get_last_rank
except:
    def get_last_rank() -> int:
        """Get the last rank in the distributed group"""
        if not torch.distributed.is_initialized():
            return 0
        return torch.distributed.get_world_size() - 1

from megatron.bridge.training.model_load_save import load_megatron_model, save_megatron_model, load_tokenizer

from megatron.bridge.training.tokenizers.tokenizer import _HuggingFaceTokenizer

from megatron.core.pipeline_parallel.schedules import get_forward_backward_func

import os

from megatron.core import parallel_state
from megatron.core import parallel_state as mpu
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer

# load pretrain/SFT model info, only bf16 supported for the moment
MODEL="gpt-oss-20b-BF16"

# create soft links to /workspace/models
MODEL_DIR="/workspace/models"

HF_MODEL_DIR=f"{MODEL_DIR}/{MODEL}"

# Specify model partitions, we use parallel folding strategy to separate EP for MLP from pp-tp-cp-dp for Attention
TP=int(os.environ.get("TP", 8))
PP=int(os.environ.get("PP", 1))
CP=int(os.environ.get("CP", 1))

# Assume a single node setup in this script
EP=int(os.environ.get("EDP", 8 // PP)) # distributed evenly among all gpu cards
# ETP can only be 1 for GptOSS for the moment with Mcore backend
ETP=1

SAVER="mcore_bridge"

SEED=42

# adpated from megatron bridge examples/
class SingleBatchIterator:
    """Iterator that yields a single batch of data for text generation.
    Required by the forward_backward_func function.

    This class creates an iterator that yields exactly one batch containing
    input tokens, position IDs, and attention mask, then raises StopIteration.
    Used for single-step inference in the forward pass.
    """

    def __init__(self, input_ids, position_ids, attention_mask):
        self.batch = dict(
            tokens=input_ids,
            position_ids=position_ids,
            # attention_mask=attention_mask,
        )
        self._yielded = False

    def __iter__(self):
        return self

    def __next__(self):
        if self._yielded:
            raise StopIteration
        self._yielded = True
        return self.batch


def text_forward_step(data_iterator, model, **kwargs) -> torch.Tensor:
    """Forward step function for text generation.
    Required by the forward_backward_func function.

    Extracts a batch from the data iterator and runs the model forward pass
    with the provided input tokens, position IDs, and attention mask.

    Args:
        data_iterator: Iterator providing batches of input data
        model: The Megatron model to run forward pass on
        **kwargs: Additional keyword arguments (unused)

    Returns:
        Tuple of (model_output, loss_function)
    """
    batch = next(data_iterator)
    forward_args = {
        "input_ids": batch["tokens"],
        "position_ids": batch["position_ids"],
        "attention_mask": batch.get("attention_mask", None),
    }

    def loss_func(x, **kwargs):
        return x

    return model(**forward_args), loss_func


def export(checkpoint=True):
    # gptoss bf16 recipe for post training
    dtype="bf16"

    # using Megatron Bridge provider API
    bridge = AutoBridge.from_hf_pretrained(f"{HF_MODEL_DIR}", trust_remote_code=True)

    provider = bridge.to_megatron_provider()

    provider.tensor_model_parallel_size = TP
    provider.pipeline_model_parallel_size = PP
    provider.context_parallel_size = CP    

    # sparse model
    provider.expert_model_parallel_size = EP
    provider.expert_tensor_parallel_size = ETP

    provider.finalize()

    model = provider.provide_distributed_model(wrap_with_ddp=False)

    # output info
    OUTPUT=f"{MODEL_DIR}/{MODEL}-to-{SAVER}-tp{TP}-pp{PP}-cp{CP}-ep{EP}-{dtype}"

    if not checkpoint:
        # to huggingface
        bridge.save_hf_pretrained(model, f"{OUTPUT}")
    else:
        # to megatron checkpoint
        save_megatron_model(model, f"{OUTPUT}", hf_tokenizer_path=f"{HF_MODEL_DIR}")
        OUTPUT = f"{OUTPUT}/iter_0000000"

    return model, OUTPUT


def _verify_tokenizer_and_hfmodel(hf_tokenizer, model):
    texts = ["Once upon the time",]
    messages = [
        {"role": "user", "content": text} for text in texts
    ]

    prompts = hf_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True)

    model_inputs = hf_tokenizer([prompts], return_tensors="pt").to(model.device)

    outputs_ids = model.generate(**model_inputs, max_new_tokens=16)

    outputs_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, outputs_ids)
    ]

    response = hf_tokenizer.batch_decode(outputs_ids, skip_special_tokens=True)[0]
    print(f"[Rank#{torch.distributed.get_rank()}] response : {response}")

def verify_tokenizer_and_hfmodel(hf_tokenizer_path, model):
    hf_tokenizer = AutoTokenizer.from_pretrained(hf_tokenizer_path)

    _verify_tokenizer_and_hfmodel(hf_tokenizer, model)

def verify_megatron_fwd(tokenizer_path, model, max_length=16):
    tokenizer = load_tokenizer(tokenizer_path)

    assert isinstance(tokenizer, _HuggingFaceTokenizer), "update script to adapt to mcore tokenizer (I am using legacy huggingface tokenizer)"

    model = [m.cuda() for m in model]
    for m in model:
        m.eval()

    prompt = "Once upon the time"
    token_ids = tokenizer.tokenize(prompt)

    with torch.no_grad():
        input_batch = torch.tensor([token_ids]).cuda()

        output_ids = input_batch.clone()

        fwd_bwd_function = get_forward_backward_func()
        
        for i in range(max_length - len(token_ids)):
            position_ids = torch.arange(output_ids.size(1), dtype=torch.long, device=output_ids.device)   
            attention_mask = torch.ones_like(output_ids, dtype=torch.bool)         

            data_iterator = SingleBatchIterator(output_ids, position_ids, attention_mask)

            output = fwd_bwd_function(
                forward_step_func=text_forward_step,
                data_iterator=data_iterator,
                model=model,
                num_microbatches=1,
                forward_only=True,
                seq_length=input_batch.size(1),
                micro_batch_size=1,
                collect_non_loss_data=True,
            )

            if isinstance(output, list) and len(output) > 0:
                output = output[0]

            if parallel_state.is_pipeline_last_stage():
                world_size = parallel_state.get_tensor_model_parallel_world_size()
                gathered_tensors = [torch.zeros_like(output) for _ in range(world_size)]

                dist.all_gather(gathered_tensors, output, group=parallel_state.get_tensor_model_parallel_group())
                
                logits = torch.cat(gathered_tensors, dim=2)
                next_token_id = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(-1)
            else:
                next_token_id = torch.ones((1, 1), device=output_ids.device, dtype=output_ids.dtype)

            torch.distributed.broadcast(next_token_id, get_last_rank())
            output_ids = torch.cat([output_ids, next_token_id], dim=1)

            if next_token_id.item() == tokenizer._tokenizer.eos_token_id:
                break

    response = tokenizer._tokenizer.decode(output_ids[0].cpu().numpy(), skip_special_tokens=True)
    print_rank_0(f"Rank#{torch.distributed.get_rank()} Response : {response}")


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
        "--source_model", default=None, type=str, required=False, help="source model."
    )
    parser.add_argument(
        "--output_hf_dir", default=None, type=str, required=False, help="Where to save the converted model."
    )
    parser.add_argument(
        "--output_ckpt_dir", default=None, type=str, required=False, help="Where to save the converted model."
    )
    args = parser.parse_args()

    if args.source_model:
        MODEL_DIR = args.source_model
        HF_MODEL_DIR=f"{MODEL_DIR}/{MODEL}"

    if args.output_hf_dir:
        OUTPUT_DIR = args.output_hf_dir

        model = AutoModelForCausalLM.from_pretrained(OUTPUT_DIR, 
                                                     torch_dtype="auto",
                                                     trust_remote_code=True)
        
        verify_tokenizer_and_hfmodel(OUTPUT_DIR, model)
    elif args.output_ckpt_dir:
        OUTPUT_DIR = f"{args.output_ckpt_dir}/iter_0000000"

        bridge = AutoBridge.from_hf_pretrained(f"{HF_MODEL_DIR}", trust_remote_code=True)

        provider = bridge.to_megatron_provider()

        provider.tensor_model_parallel_size = TP
        provider.pipeline_model_parallel_size = PP
        provider.context_parallel_size = CP    

        # sparse model
        provider.expert_model_parallel_size = EP
        provider.expert_tensor_parallel_size = ETP
        
        # provider.sequence_parallel = True

        provider.finalize()
        provider.initialize_model_parallel(seed=SEED)

        model = load_megatron_model(OUTPUT_DIR)

        verify_megatron_fwd(OUTPUT_DIR, model)

        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
    else:
        model, OUTPUT_DIR = export()

        verify_megatron_fwd(OUTPUT_DIR, model)

        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
