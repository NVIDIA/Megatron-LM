# GSM8K prompt subset

This directory contains a 256-example subset of the GSM8K dataset from
OpenAI's grade-school-math repository, used as realistic prompt input for
inference performance benchmarks.

- Original dataset: https://github.com/openai/grade-school-math
- License: MIT

`gsm8k_prompts.jsonl` holds one prompt per line as `{"prompt": "..."}`,
drawn from the dataset's test split. Only the question text is retained;
answers and chain-of-thought solutions are stripped because the benchmark
generates a fixed `NUM_OUTPUT_TOKENS` worth of tokens and discards the
content.
