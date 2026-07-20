#!/usr/bin/env python3
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CUDA-only repro for why autocast does not replace MFSDP optimizer wrapping.

Run with:

    python docs/discussions/mfsdp-optimizer-design/autocast_optimizer_repro.py

This is an experiment accompanying the MFSDP optimizer design discussion, not
a pytest test. It intentionally uses independent ``main_weight`` and
``model_weight`` tensors to model the MFSDP storage contract.
"""

import torch


def case_autocast_does_not_sync_model_weight() -> None:
    """An optimizer step updates only the tensor it owns."""
    main_weight = torch.nn.Parameter(torch.tensor([1.0], device="cuda"))
    model_weight = torch.tensor([1.0], device="cuda", dtype=torch.bfloat16)
    optimizer = torch.optim.SGD([main_weight], lr=0.25)
    main_weight.grad = torch.ones_like(main_weight)

    with torch.autocast("cuda", dtype=torch.bfloat16):
        optimizer.step()

    assert main_weight.item() == 0.75
    assert model_weight.item() == 1.0
    print(
        "case 1: autocast optimizer.step() updated main_weight but did not "
        "sync model_weight (0.75 != 1.0)"
    )


def case_bf16_params_remain_bf16() -> None:
    """Autocast does not introduce a separate FP32 main-weight parameter."""
    parameter = torch.nn.Parameter(torch.ones(4, device="cuda", dtype=torch.bfloat16))
    optimizer = torch.optim.SGD([parameter], lr=0.25)

    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = (parameter * parameter).sum()
    loss.backward()
    optimizer.step()

    assert parameter.dtype is torch.bfloat16
    assert parameter.grad is not None
    assert parameter.grad.dtype is torch.bfloat16
    assert optimizer.param_groups[0]["params"][0] is parameter
    print(
        "case 2: bf16 parameter and grad remain bf16; no fp32 main-weight path exists"
    )


def case_fp32_params_with_bf16_grads_still_fail() -> None:
    """Autocast does not make an optimizer accept a mismatched grad dtype."""
    parameter = torch.nn.Parameter(torch.ones(4, device="cuda", dtype=torch.float32))
    optimizer = torch.optim.Adam([parameter], lr=0.25, foreach=False)

    # Assign a normal grad first, then replace its storage to model an FSDP
    # main_grad whose dtype differs from its FP32 main_weight.
    parameter.grad = torch.zeros_like(parameter)
    parameter.grad.data = torch.ones_like(parameter, dtype=torch.bfloat16)

    try:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            optimizer.step()
    except RuntimeError as error:
        print(
            f"case 3: fp32 parameter with bf16 grad still fails under autocast: {error}"
        )
        return

    raise AssertionError("Adam unexpectedly accepted a bf16 grad for an fp32 parameter")


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("This repro requires a CUDA-enabled PyTorch installation.")

    print(f"PyTorch {torch.__version__}; CUDA device: {torch.cuda.get_device_name()}")
    case_autocast_does_not_sync_model_weight()
    case_bf16_params_remain_bf16()
    case_fp32_params_with_bf16_grads_still_fail()
    print("All three autocast limitations reproduced.")


if __name__ == "__main__":
    main()
