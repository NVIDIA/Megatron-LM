Multi-Latent Attention
======================

Multi-Latent Attention overview 
-------------------------------

Multi-Latent Attention ("MLA") is an innovative attention mechanism introduced by Deepseek team that enhances the efficiency of attention computation by leveraging multiple latent spaces. This approach is particularly beneficial for large language models (LLMs), as it reduces the computational burden associated with traditional attention mechanisms. According to Deepseek-V2 technical report, MLA achieves better performance compared to Multi-Head Attention (MHA) and requires smaller KV cache.

Enabling Multi-Latent Attention
-------------------------------

To enable MLA in Megatron-LM, set the following flags in command line:
- `--multi-latent-attention` to enable MLA in MLP.
- Set `MLATransformerConfig` to configure MLA.
