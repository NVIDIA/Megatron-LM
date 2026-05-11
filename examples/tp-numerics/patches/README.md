# TE patches

Snapshots of TE files the submit scripts copy into the container's TE
site-packages before training. Squashfs containers are read-only per-srun, so
patches must re-apply on every launch.

Submit scripts detect the container's TE version and pick `v<version>/`. No
matching subdirectory → fail-fast.

## Layout

```
v<version>/transformer_engine/pytorch/  # mirrors TE source tree
  module/{layernorm_linear,linear}.py
  attention/dot_product_attention/{backends,dot_product_attention}.py
```

## Supported versions

| TE version | Base commit | Source |
|---|---|---|
| `v2.9.0` | `70f53666` | [jinzex/TransformerEngine@d1bcd9a6](https://github.com/jinzex/TransformerEngine/tree/jinzex/tp-invariant-numerics) |

Gated on `NVTE_TP_INVARIANT_MODE=1` — env-var unset = stock TE behavior.

## Refresh

```bash
TE=<TransformerEngine checkout>; VER=2.9.0
DEST=v${VER}/transformer_engine/pytorch
mkdir -p $DEST/module $DEST/attention/dot_product_attention
cp $TE/transformer_engine/pytorch/module/{layernorm_linear,linear}.py $DEST/module/
cp $TE/transformer_engine/pytorch/attention/dot_product_attention/{backends,dot_product_attention}.py \
   $DEST/attention/dot_product_attention/
```
