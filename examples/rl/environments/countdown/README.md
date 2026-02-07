# Countdown Agentic Environment
The `CountdownAgenticEnv` is based off of the countdown task introduced in https://github.com/Jiayi-Pan/TinyZero. The objective is for the LLM to provide an algebraic expression combining a set of numbers in order to produce a provided "target" number.

The data is loaded from the below HF dataset and most of the evaluation code (in `countdown.py`) is inherited from the above GitHub repository.

https://huggingface.co/datasets/Jiayi-Pan/Countdown-Tasks-3to4

It is an example of a `megatron.rl.agent.reward_only_agent` so many tasks that have only a reward calcuation can use this as a prototype.
