# Megatron Discussions

This directory contains in-depth guides, tutorials, and discussions about optimizing and using Megatron for various use cases.

## Available Guides

### Training Guides

- **[Megatron-FSDP User Guide](megatron-fsdp-user-guide/megatron-fsdp-user-guide.md)**

  A practical guide to enable Megatron-FSDP training, including a quick-start example for DeepSeek-V3, required and recommended configurations, and instructions for checkpoint conversion from torch_dist to fsdp_dtensor.

## Contributing

If you'd like to contribute a guide or tutorial, please follow this structure:

1. Create a new directory: `docs/discussions/your-guide-name/`
2. Add your main guide: `docs/discussions/your-guide-name/your-guide-name.md`
3. Create an images directory: `docs/discussions/your-guide-name/images/`
4. Update this README.md with a link to your guide

Each guide should be self-contained with its own images and supporting files.