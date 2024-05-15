# Multimodal Example

NOTE: This is work in progress and not fully functional yet.

## Setup

### Vision model

This example uses the OpenAI CLIP `ViT-L/14@336px` Vision model. To download the weights from OpenAI and convert them to a format that can be loaded in megatron, please run the following:

```
python examples/multimodal/clip_converter.py --download-root /some/download/folder --output /some/output/folder --tensor-parallel-size 4
```

## Training

### Pretraining

Run the following script:
```
examples/multimodal/pretrain_8b.sh
```

### SFT

Run the following script:
```
examples/multimodal/sft_8b.sh
```
