# MIMO: Multimodal In/Out Model

## What is MIMO?

MIMO is a model architecture that enables language models to understand and generate multiple modalities (text, images, audio, etc.). It achieves this through:

- A core language model that processes unified embeddings
- Modality-specific submodules that:
  - Encode inputs into embeddings (e.g. image->embeddings)
  - Decode embeddings back to outputs (e.g. embeddings->image)
  - Project between modality and language model spaces
- The MimoModel handles:
  - Aligning modality embeddings at special token positions in the sequence
  - Processing the combined embeddings through the language model

MIMO provides a flexible and canonical architecture that can be configured into various multimodal models, for example

- Vision-Language Models (VLMs)
- Audio-Visual Language Models  
- Multimodal understanding and generation

## How It Works

The model architecture consists of 2 main components:

1) Language model
2) Modality submodules

The complete data flow:

```
Input → Encoder → Projection → Align input embeddings → Language Model → Hidden states for special generation tokens ->  Output Projection → Decoder → Output
```

1. **Encoding**:
   - Modality submodules convert inputs to embeddings (e.g., images → embeddings).
   - The MimoModel aligns all modality embeddings along with text embeddings by token positions.
   - The language model processes the unified embeddings.

2. **Decoding**:
   - We select hidden states that correspond to special modality generation tokens.
   - Modality submodules convert embeddings back to outputs (e.g., embeddings → images).

## Components in Detail

### Language Model

The language model is the core component that processes all modality information in a unified embedding space:

- Acts as the central processor for all modalities through a shared vocabulary
- Processes the combined sequence containing both text and modality tokens

### Modality Submodules

`ModalitySubmodules` connect raw modality data with the language model:

- Each submodule handles **encoding** (modality → embeddings) and **decoding** (embeddings → modality) 
- Manages the **projection** between modality space and language model dimensions

```python
# Base class constructor with named encoders and decoders
class ModalitySubmodules(ABC, nn.Module):
    def __init__(
        self,
        encoders: Optional[Dict[str, nn.Module]] = None,
        decoders: Optional[Dict[str, nn.Module]] = None,
        input_projections: Optional[List[nn.Module]] = None,
        output_projections: Optional[List[nn.Module]] = None,
    ):
```

MIMO provides default implementations (`VisionModalitySubmodules`, `AudioModalitySubmodules`), but you can create custom submodules for specialized processing:

```python
# Custom implementation
class CustomVisionSubmodules(ModalitySubmodules):
    def encode(self, inputs):
        # Specialized encoding logic
        return projected_embeddings

# Use custom submodules when creating the model
model = MimoModel(
    mimo_config,
    modality_submodules={"images": ModuleSpec(module=CustomVisionSubmodules, params={...})}
)
```

### Embedding Alignment

The `MimoModel` handles the integration of different modality embeddings through its `align_embeddings_by_token_positions` method:

- Places modality embeddings at their special token positions in the input sequence
- Handles dimension matching and position tracking for proper embedding placement

Example of what happens internally:
```python
# Inside MimoModel's forward method
aligned_embeddings = self.align_embeddings_by_token_positions(
    modality_embeddings={"text": text_emb, "images": image_emb},
    input_ids=tokens,
    special_token_ids={"images": 32000}
)
```

## Configuration and Usage

### MimoModel Parameters

```python
MimoModel(
    config: MimoModelConfig,    # Required: Configuration for the model
)
```

### Configuration Details

MIMO models are instantiated with a `MimoModelConfig`, which contains:
1. A specification for the language model
2. A dictionary mapping modality names to their submodule specifications

```python
MimoModelConfig(
    language_model: ModuleSpec,                         # Specification for the language model
    modality_submodules: Dict[str, ModuleSpec],         # Dictionary mapping modality names to their submodule specifications
    special_token_ids: Dict[str, int] = {}              # Dictionary mapping modality names to their special token IDs
)
```

### Example: Creating a Vision-Language Model (VLM)

```python
# Language model specification
lm_spec = ModuleSpec(
    module=GPTModel,
    params={
        "config": language_config,
        "transformer_layer_spec": get_mock_language_layer_spec(),
        "vocab_size": 50304,
    }
)

# Vision modality specification
vision_submodule_spec = ModuleSpec(
    module=VisionModalitySubmodules,
    params={
        # Any general parameters for the submodule can go here
    },
    submodules={
        "encoders": {
            "clip_encoder": ModuleSpec(
                module=CLIPViTModel,
                params={
                    "transformer_config": vision_config,
                    "transformer_layer_spec": get_mock_vision_layer_spec(),
                    "patch_dim": 16,
                    "img_h": 224,
                    "img_w": 224,
                }
            ),
        },
        "input_projections": [
            ModuleSpec(
                module=MultimodalProjector,
                params={
                    "config": get_mock_projection_config(),
                    "submodules": get_mock_projection_layer_spec().submodules,
                    "projector_type": "mlp",
                    "input_size": 128
                }
            ),
        ],
    }
)

# Instantiate the model
vlm = MimoModel(
    MimoModelConfig(
        language_model=lm_spec,
        modality_submodules={"images": vision_submodule_spec},
        special_token_ids={"images": 32000}
    )
)
```

### MIMO Forward Method Usage

```python
# Prepare inputs for multiple modalities and encoders
modality_inputs = {
    # modality names and encoder names should match the keys used in mimo config during initialization.
    "images": {
        "clip_encoder": {"pixel_values": images},  # Encoder-specific inputs
        "vit_encoder": {"images": vit_images}
    },
    "audio": {
        "whisper_encoder": {"input_features": audio_features}
    }
}

# Call forward method
outputs, _ = mimo_model(
    input_ids=input_ids,
    position_ids=position_ids,
    modality_inputs=modality_inputs,
)
```
