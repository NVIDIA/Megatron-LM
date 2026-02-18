import argparse
import os
from pathlib import Path
from typing import Union

# hf path
import requests
import torch
from PIL import Image
from transformers import AutoProcessor
from transformers import AutoTokenizer
import soundfile as sf
import io
import numpy as np
import scipy.signal as signal

from examples.mimo.model_providers.llava_avlm import model_provider_llava_avlm
from megatron.core import dist_checkpointing, parallel_state, tensor_parallel
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.training import print_rank_0
from examples.mimo.data.utils.calculate_audio_tokens import calculate_num_audio_tokens

def init_distributed(tp_size: int = 1, pp_size: int = 1):
    if torch.distributed.is_initialized():
        return
    rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    torch.cuda.set_device(rank % torch.cuda.device_count())
    torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)
    parallel_state.initialize_model_parallel(tp_size, pp_size)

def get_input_data(
    processor: AutoProcessor,
    image_processor: AutoProcessor,
    audio_processor: AutoProcessor,
    audio_path: str,
    image_path: str,
    prompt: str,
    device: Union[int, str] = 0):
    """
    Prepare inputs for the MIMO model forward pass.
    """

    def read_audio(audio_path):
        """Process audio file and return tensor."""
        with open(audio_path, 'rb') as f:
            audio_bytes = f.read()
        audio_io = io.BytesIO(audio_bytes)
        waveform, sample_rate = sf.read(audio_io)
        
        # Resample if needed
        fixed_sample_rate = 16000
        if sample_rate != fixed_sample_rate:
            num_samples = int(len(waveform) * fixed_sample_rate / sample_rate)
            waveform = signal.resample(waveform, num_samples)
        
        # Convert to tensor
        audio_tensor = torch.from_numpy(waveform).float()
        return audio_tensor

    def read_image(image_path):
        """Process image file and return tensor."""
        with open(image_path, 'rb') as f:
            image_bytes = f.read()
        image_io = io.BytesIO(image_bytes)
        image = Image.open(image_io)
        image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1)  # Convert to CxHxW format
        image_tensor = image_tensor.float() / 255.0  # rescale to [0,1] range
        return image_tensor


    # read audio and image
    audio_tensor = read_audio(audio_path)
    image_tensor = read_image(image_path)

    # set up prompt
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
            ],
        }
    ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

    # process audio
    processed_audios = audio_processor(audio_tensor, sampling_rate=16000)
    processed_audios = torch.tensor(processed_audios["input_features"])
    processed_audios = processed_audios.squeeze(0) # remove batch dim
    num_audio_tokens = calculate_num_audio_tokens(audio_tensor.unsqueeze(0), "openai/whisper-base")
    audios_seq_lengths = torch.tensor(num_audio_tokens)
    prompt = prompt.replace("<audio>", "<audio>" * num_audio_tokens)

    # process image
    processed_images = image_processor(
        images=image_tensor,
        return_tensors="pt",
        do_rescale=False,
    )["pixel_values"]
    processed_images = processed_images.squeeze(0) # remove batch dim

    # process prompt
    processed_prompt_inputs = processor(
        images=image_tensor,
        text=prompt,
        add_special_tokens=False,
        return_tensors="pt",
        do_rescale=False,
    )

    # set batch data
    processed_images = processed_images.unsqueeze(0).to(device)
    processed_audios = processed_audios.unsqueeze(0).to(device)
    audios_seq_lengths = audios_seq_lengths.unsqueeze(0).to(device)
    tokens = processed_prompt_inputs["input_ids"].to(device)
    modality_inputs = {
        "images": {"clip_encoder": {"pixel_values": processed_images}},
        "audios": {"whisper_encoder": {"input_features": processed_audios, "seq_lengths": audios_seq_lengths}}
    }
    batch_data = {
        "tokens": tokens,
        "modality_inputs": modality_inputs,
    }

    return batch_data


def main():
    parser = argparse.ArgumentParser("Test loading a distributed LLaVA checkpoint")
    parser.add_argument("--ckpt", required=False, help="Path to checkpoint optional")
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--pp", type=int, default=1, help="Pipeline parallel size")
    parser.add_argument("--audio-path", type=str,required=True, help="Path to audio file")
    parser.add_argument("--image-path", type=str,required=True, help="Path to image file")
    parser.add_argument("--prompt", type=str,required=True, help="Prompt")
    args = parser.parse_args()


    init_distributed(args.tp, args.pp)
    model_parallel_cuda_manual_seed(123)

    device = torch.device("cuda")

    model = model_provider_llava_avlm().to(device)

    # Load distributed checkpoint if provided.
    if args.ckpt:
        load_distributed_checkpoint(model, args.ckpt)

    # set tokenizer
    tokenizer = AutoTokenizer.from_pretrained("llava-hf/llava-1.5-7b-hf")
    tokenizer.add_special_tokens({'additional_special_tokens': ["<audio>"]})
    tokenizer.vocab["<audio>"] = 32002
    tokenizer.added_tokens_encoder["<audio>"] = 32002
    tokenizer.added_tokens_decoder[32002] = "<audio>"

    # set processors
    processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
    processor.tokenizer = tokenizer
    image_processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf").image_processor
    audio_processor = AutoProcessor.from_pretrained("openai/whisper-base")
    
    
    data = get_input_data(
        processor,
        image_processor,
        audio_processor,
        args.audio_path,
        args.image_path,
        args.prompt,
        device=device)

    # ------------------------------------------------------------------
    # Greedy generation
    # ------------------------------------------------------------------
    max_new_tokens = 128
    model.eval()    

    tokens = data["tokens"]

    with torch.no_grad():
        for _ in range(max_new_tokens):
            seq_len = tokens.size(1)
            position_ids = torch.arange(seq_len, device=tokens.device).unsqueeze(0)
            logits, _ = model(
                input_ids=tokens,
                position_ids=position_ids,
                attention_mask=None,
                modality_inputs=data["modality_inputs"],
            )
            
            # All-gather logits across tensor parallel ranks
            # logits shape: [batch, seq, vocab_parallel_size]
            gathered_logits = tensor_parallel.gather_from_tensor_model_parallel_region(logits)

            # The language model returns logits in [batch, seq, vocab] format.
            next_token_logits = gathered_logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            tokens = torch.cat([tokens, next_token], dim=1)

            if processor.tokenizer.eos_token_id is not None and next_token.item() == processor.tokenizer.eos_token_id:
                break

    # Only decode and print on rank 0
    if torch.distributed.get_rank() == 0:
        generated_text = processor.tokenizer.decode(tokens[0], skip_special_tokens=True)
        print("\n=== Generated text ===\n")
        print(generated_text)


def load_distributed_checkpoint(model: torch.nn.Module, ckpt_dir: str):
    """Load a MIMO model from a Megatron distributed checkpoint directory"""

    if not os.path.isdir(ckpt_dir):
        raise FileNotFoundError(f"Checkpoint directory does not exist: {ckpt_dir}")


    template_sd = {"model": model.sharded_state_dict()}

    loaded_sd = dist_checkpointing.load(template_sd, ckpt_dir)

    model_state_dict = loaded_sd["model"]
    incompat = model.load_state_dict(model_state_dict, strict=False)

    missing = [k for k in incompat.missing_keys if "extra_state" not in k]
    unexpected = [k for k in incompat.unexpected_keys if "extra_state" not in k]
    if missing or unexpected:
        print_rank_0(
            f"[Rank {torch.distributed.get_rank() if torch.distributed.is_initialized() else 0}] "
            f"Checkpoint loaded with mismatches. Missing: {missing}, Unexpected: {unexpected}"
        )

    print_rank_0(
        f"[Rank {torch.distributed.get_rank() if torch.distributed.is_initialized() else 0}] "
        f"Successfully loaded checkpoint from {ckpt_dir}"
    )

    return model


if __name__ == "__main__":
    main()