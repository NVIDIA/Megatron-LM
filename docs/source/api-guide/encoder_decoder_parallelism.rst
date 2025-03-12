encoder-decoder-parallelism package
===================================

Mcore (as of 0.9) supports heterogeneous parallelism for encoder-decoder models.
In particular, the user is now able to specify the amount of tensor and pipeline parallelism and have it be
distinct from that in the decoder.

Submodules
----------

Encoder Pipeline Parallelism
----------------------------

Supported in: T5, LLaVa.

The new argument for encoder parallelism is `--encoder-pipeline-model-parallel-size`. This argument is completely distinct
from the usual argument that controls pipelining: `--pipeline-model-parallel-size`, which controls the amount of pipelining in the decoder
in the context of encoder-decoder models.

The total amount of pipelining in an encoder-decoder model is the sum of these two arguments. By default, the amount of
encoder pipelining is 0, and the amount of decoder pipelining is 1, meaning that the encoder & decoder share the single pipeline rank.
If `--pipeline-model-parallel-size` > 1,then the amount of encoder parallelism has to be specified and has to be greater than 0.
This is because we are not able to share pipeline ranks between the encoder and decoder anymore.

Encoder Tensor Parallelism
--------------------------

Supported in: LLaVa.

Since we expect encoders to be much smaller than decoders, we also give users the ability to set a different amount of tensor
parallelism than the decoder. This is achieved with the argument `--encoder-tensor-model-parallel-size`. To use this option, you must
be using encoder pipeline parallelism (ie, `--encoder-pipeline-model-parallel-size` > 0).

Unlike with encoder pipeline parallelism, which was unrestricted by the amount of decoder pipeline parallelism, we only allow encoders to have
less than or the same amount of tensor parallelism as the decoder. The summary of how we do this is that within p2p_communication.py, we have
to send the activations of one encoder rank to several decoder ranks; correspondingly, we have to add support for summing gradients from several
(downstream) decoder ranks for the encoder rank. We have not seen a quantization-related degradation from summing these gradient tensors
together yet; it could happen in very large models.


Number of GPUs Required
-----------------------

The total amount of GPUs required to train a model when these options enabled is:

dp * etp * epp * cp + dp * tp * pp * cp

where:
dp: amount of data parallelism (this is the same for the encoder & decoder)
[e]tp: amount of tensor parallelism
[e]pp: amount of pipeline parallelism
cp: amount of context parallelism (as with dp, this is the same for the encoder & decoder)

The default value of this argument is 0; in practice, we will use the amount of tensor parallelism in the decoder to construct the encoder.
