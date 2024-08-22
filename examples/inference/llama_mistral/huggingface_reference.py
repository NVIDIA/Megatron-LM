import argparse
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

# Set up argument parsing
parser = argparse.ArgumentParser(description="Script for text generation with a specific model and prompt.")
parser.add_argument('--prompt', type=str, required=True, help="Prompt text to use for text generation")
parser.add_argument('--model-path', type=str, required=True, help="Path to the Huggingface model checkpoint")

# Parse command-line arguments
args = parser.parse_args()

model_path = args.model_path
prompt = args.prompt

config = AutoConfig.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path, config=config)
model = AutoModelForCausalLM.from_pretrained(model_path, config=config).cuda()

inputs = tokenizer(prompt, return_tensors="pt")
for key in inputs:
    inputs[key] = inputs[key].cuda()
# top_k, top_p and do_sample are set for greedy argmax based sampling
outputs = model.generate(**inputs, max_length=100, do_sample=False, top_p=0, top_k=0, temperature=1.0)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))