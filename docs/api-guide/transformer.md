# transformer package

The `transformer` package provides a customizable and configurable
implementation of the transformer model architecture. Each component
of a transformer stack, from entire layers down to individual linear
layers, can be customized by swapping in different PyTorch modules
using the "spec" parameters (see [here](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/nlp/nemo_megatron/mcore_customization.html)). The
configuration of the transformer (hidden size, number of layers,
number of attention heads, etc.) is provided via a `TransformerConfig`
object.

