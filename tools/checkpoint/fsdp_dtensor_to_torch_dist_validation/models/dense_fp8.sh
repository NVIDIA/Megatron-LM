# (Extra archetype) Dense + SwiGLU + FP8 (llama3-like). Exercises the converter's
# handling of FP8: the _extra_state (amax history / scaling) is intentionally
# dropped — the fsdp_dtensor checkpoint already discarded it. FP8 resume
# re-initializes amax, so the first post-load steps track ~1% looser than bf16
# tolerance. This is expected, not a bug, and is called out in the README.
MODEL_LABEL="Dense + FP8 (llama3-like)"
MODEL_TRANSFORM="FP8 _extra_state drop (loose ~1% resume, by design)"
NUM_LAYERS=12
ARCH="--swiglu --fp8-format hybrid --fp8-amax-history-len 32 --fp8-param-gather"
# Load-side reshard not part of the validated coverage for this extra archetype.
RESHARD_LAYOUTS=""
