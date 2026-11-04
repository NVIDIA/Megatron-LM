# Decoder-entry RMSNorm

Hybrid models can normalize the decoder input once, before the first layer adds
to the residual stream:

```bash
--normalization RMSNorm --post-embedding-rmsnorm
```

The option is disabled by default. It creates a learned RMSNorm at
`decoder.input_norm`, using the configured hidden size, epsilon and RMSNorm
initialization. The default effective gamma is one.

The norm runs inside the first main decoder stage, after embeddings have been
assembled and before wide-residual or mHC stream expansion. Standalone text embeddings
and externally assembled multimodal embeddings therefore use the same boundary.
Later pipeline stages and auxiliary MTP stacks do not create another input norm.
The incoming residual dtype is preserved, including FP32 residual connections.
This option requires `--normalization RMSNorm` and applies to hybrid decoders.

## Norm weight decay

The input norm follows the existing optimizer policy for one-dimensional norm
weights: the standard parameter-group overrides exclude it from weight decay.
Enabling decoder-entry RMSNorm does not change that policy.

## Checkpoints

The learned weight is saved as `decoder.input_norm.weight`, including in sharded
checkpoints. The architecture flag is also saved and checked when resuming. A
checkpoint and command line that disagree about the flag are rejected.

With `--use-checkpoint-args`, the saved flag takes precedence over the command
line. Older checkpoints that lack the flag are interpreted as having the option
disabled, so existing models keep their original architecture.
