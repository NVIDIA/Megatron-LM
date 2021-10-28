This is a short tutorial of how to use/tune the curriculum learning (CL) integration. Currently it is only integrated for GPT pre-training. For technical details please refer to our [paper](https://arxiv.org/abs/2108.06084).

# Disable batch size warmup (--rampup-batch-size)
In our [paper](https://arxiv.org/abs/2108.06084) section 5.4 we demonstrate that curriculum learning (seqlen-based) provides much better training stability than the batch size warmup technique. So when using CL you need to remove the `--rampup-batch-size` config in your training script. It's not recommended to use both CL and batch size warmup, because both of them will reduce the number of tokens in a batch. Another related change you might want is to increase your micro batch size, since without batch size warmup your batch size will be fixed now.

# Token-based training termination

Because CL changes length of each sequence/sample during training, it is very hard/impossible to use number of steps/samples to terminate the training exactly at the desired number of tokens. Thus we add a `--train-tokens` config as an alternative accurate token-based termination. We recommend increase your original `--train-samples` or `--train-iters` to a large enough number (e.g., 2X of what you used for baseline), and set `--train-tokens` at the exact desired number of training tokens (e.g., 300B for GPT-3 like training).

# Token-based LR decay

Again because CL changes the number of tokens per batch, in our [paper](https://arxiv.org/abs/2108.06084) Appendix A.2 we show that it is also necessary to change the LR decay to token-based (to avoid decaying LR too fast). Thus we add a `--lr-decay-tokens` which will be the number of LR decay tokens. If previously you were using `--lr-decay-samples`, you can calculate your `--lr-decay-tokens` simply by multiplying the former by full seqlen (e.g. 2K for GPT-3). Then you need to replace `--lr-decay-samples` with `--lr-decay-tokens` in your script.

# LR warmup adjustment

For LR warmup we don't change it to token-based, because doing so for CL means slowing down the LR warmup, which is both unnecessary and harmful. However, you may need to adjust your `--lr-warmup-samples` or `--lr-warmup-iters` from non-CL cases for various reasons (e.g., if you used `--rampup-batch-size` in non-CL case, for CL we don't use it so the number of samples per batch will be different at beginning). Assuming you want to use `X` tokens to warmup the LR (for OpenAI GPT-3 this was 375M tokens), then for CL case you shall set `--lr-warmup-samples` as `X` divided by the `min_difficulty` below, or set `--lr-warmup-iters` as `X` divided by `min_difficulty * --global-batch-size`. This is a rough estimation based on that CL starts from seqlen `min_difficulty` and it won't increase too much during LR warmup.

# Token-based tensorboard

Because of the above changes, we also add token-based tensorboard scalars. We also add scalars that plot the seqlen at each step.

# Curriculum learning hyperparameters tuning strategy

The curriculum learning hyperparameters are all located in the deepspeed config json file (see the example `ds_config_cl.json` in this dir). There are a few config entries that you may need to adjust to your circumstances, and two of which require some tuning. In our [paper](https://arxiv.org/abs/2108.06084) Appendix A.1 we have a more detailed tuning strategy description.

1. `max_difficulty` should be set as the full seqlen (i.e., your `--seq-length`). No need to tune this.

2. `min_difficulty` is the beginning seqlen used by CL. In general smaller `min_difficulty` could provide better stability/convergence speed benefit. However we observe that for a larger model or for different training data, starting from a very small seqlen could lead to significant validation PPL fluctuation (or even divergence) at the very beginning. We recommend to start with `min_difficulty` at 64, and then increase it if you observe problems at the very beginning. Note that to enable Tensor Core acceleration you should always use a multiple of 8.

3. `total_curriculum_step` is the total number of steps used by CL. In general larger `total_curriculum_step` could provide better stability/convergence speed benefit. However we observe that a too large `total_curriculum_step` could lead to overfitting and significant validation PPL fluctuation (or even divergence) at the first few multiple of LR warmup steps. In our paper we have a detailed tuning strategy based on binary search. However, if you want to reduce the tuning effort we recommend directly setting `total_curriculum_step` as half of baseline's total number of steps. This may not provide the highest convergence speed benefit, but should provide enough training stability gains.

4. `difficulty_step` is the change in seq length per CL step. A smaller value is preferable since it gives more smooth CL and better stability. Like `min_difficulty` it too needs to be multiple of 8 for Tensor core acceleration, thus 8 is a good default.
