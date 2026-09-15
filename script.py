from megatron.core.tokenizers import MegatronTokenizer

tok_path = "/lustre/fsw/coreai_dlalgo_llm/dpykhtar/models/multimodal"
special_tokens=["<image>", "<img>", "</img>", "<quad>", "</quad>", "<ref>", "</ref>", "<box>", "</box>"]
image_tag_type="internvl"
force_system_message=False
prompt_format="nemotron-h-aligned"
tokenizer_fast = MegatronTokenizer.from_pretrained(
    tokenizer_path=tok_path,
    metadata_path={"library": "sft"},
    #special_tokens=special_tokens,
    #image_tag_type=image_tag_type,
    prompt_format=prompt_format,
    #force_system_message=force_system_message,
    use_gigatoken=True,
)

tokenizer_slow = MegatronTokenizer.from_pretrained(
    tokenizer_path=tok_path,
    metadata_path={"library": "sft"},
    #special_tokens=special_tokens,
    #image_tag_type=image_tag_type,
    prompt_format=prompt_format,
    #force_system_message=force_system_message,
    use_gigatoken=False,
)

import json

data_path = "/lustre/fsw/coreai_dlalgo_llm/dpykhtar/data/conv.json"

with open(data_path, "r") as f:
    conversation = json.load(f)

conversation = conversation * 10

print(len(conversation))

import time
start_time = time.perf_counter()
result1 = tokenizer_fast.tokenize_conversation(conversation, return_target=False, add_generation_prompt=True)
# End the timer
end_time = time.perf_counter()
# Calculate elapsed time
execution_time = end_time - start_time
print(f"Execution time Giga: {execution_time:.6f} seconds")

start_time = time.perf_counter()
result2 = tokenizer_slow.tokenize_conversation(conversation, return_target=False, add_generation_prompt=True)
end_time = time.perf_counter()
# Calculate elapsed time
execution_time = end_time - start_time
print(f"Execution time HF Default: {execution_time:.6f} seconds")

print(len(result1))
print(len(result2))

#print(result1[-10:])
#print(result2[-10:])

assert result1.tolist() == result2.tolist()
#print(tokenizer.detokenize([1]))
