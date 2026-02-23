# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Utility functions for logging and printing MIMO model structure."""

# Use Megatron utility if available – covers both distributed and non-distributed cases.
from megatron.training.utils import print_rank_0


def print_mimo_structure(model):
    """Print a clean summary of MIMO model structure showing components and their types."""
    print_rank_0("MIMO Model Structure:")
    
    # Print modality submodules and their components
    print_rank_0("├── Modalities:")
    if hasattr(model, 'modality_submodules'):
        for modality_name, submodule in model.modality_submodules.items():
            print_rank_0(f"│   ├── {modality_name}")
            
            # Print encoders
            if hasattr(submodule, 'encoders') and submodule.encoders:
                print_rank_0("│   │   ├── Encoders:")
                for encoder_name, encoder in submodule.encoders.items():
                    encoder_type = encoder.__class__.__name__
                    print_rank_0(f"│   │   │   ├── {encoder_name}: {encoder_type}")
            
            # Print input projections
            if hasattr(submodule, 'input_projections') and submodule.input_projections:
                print_rank_0("│   │   ├── Input Projections:")
                for i, proj in enumerate(submodule.input_projections):
                    proj_type = proj.__class__.__name__
                    print_rank_0(f"│   │   │   ├── {i}: {proj_type}")
            
            # Print decoders
            if hasattr(submodule, 'decoders') and submodule.decoders:
                print_rank_0("│   │   ├── Decoders:")
                for decoder_name, decoder in submodule.decoders.items():
                    decoder_type = decoder.__class__.__name__
                    print_rank_0(f"│   │   │   ├── {decoder_name}: {decoder_type}")
            
            # Print output projections
            if hasattr(submodule, 'output_projections') and submodule.output_projections:
                print_rank_0("│   │   ├── Output Projections:")
                for i, proj in enumerate(submodule.output_projections):
                    proj_type = proj.__class__.__name__
                    print_rank_0("│   │   │   ├── {i}: {proj_type}")
    
    # Print language model
    if hasattr(model, 'language_model'):
        lm_type = model.language_model.__class__.__name__
        print_rank_0(f"├── Language Model: {lm_type}")