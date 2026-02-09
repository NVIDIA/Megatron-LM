import re
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple, Union

import torch
from torch.distributed.checkpoint.filesystem import FileSystemReader
from torch.distributed.checkpoint.state_dict_loader import load

TensorLike = Union[torch.Tensor, Iterable[torch.Tensor]]


def _as_iter(x: TensorLike):
    return x if (isinstance(x, Iterable) and not isinstance(x, torch.Tensor)) else [x]


def _fro_norm(x: TensorLike) -> torch.Tensor:
    """Frobenius norm; supports sharded tensors (sum of shard ||·||_F^2)."""
    it = _as_iter(x)
    s = torch.tensor(0.0, device=next(iter(it)).device if it else "cpu")
    for t in it:
        s = s + t.float().pow(2).sum()
    return torch.sqrt(s)


def machine_epsilon_for_dtype(dtype: torch.dtype) -> float:
    """Return machine epsilon for dtype. For FP8, use BF16 epsilon per paper."""
    # Standard types
    if dtype in (torch.float32, torch.float16, torch.bfloat16):
        return float(torch.finfo(dtype).eps)
    # FP8 recipes: accum/store typically BF16/FP32; use BF16 epsilon
    if hasattr(torch, "float8_e4m3fn") and dtype in (
        torch.float8_e4m3fn,
        getattr(torch, "float8_e5m2fn", None),
    ):
        return float(torch.finfo(torch.bfloat16).eps)
    # Fallback
    return float(torch.finfo(torch.float32).eps)


def relative_grad_diff(g_hat: TensorLike, g_ref: TensorLike, eps_den: float = 1e-30) -> float:
    """
    Relative difference ||g_hat - g_ref||_F / ||g_ref||_F.
    Accepts a single tensor or an iterable of shards for each argument.
    """
    # If sharded, assume shards align 1:1; otherwise pass the merged tensors.
    gh_iter, gr_iter = _as_iter(g_hat), _as_iter(g_ref)
    if len(list(gh_iter)) != len(list(gr_iter)):
        # Re-materialize since we consumed generators above:
        gh_iter, gr_iter = _as_iter(g_hat), _as_iter(g_ref)
    num_sq = torch.tensor(0.0, device=next(iter(gh_iter)).device)
    for a, b in zip(_as_iter(g_hat), _as_iter(g_ref)):
        num_sq = num_sq + (a.float() - b.float()).pow(2).sum()
    num = torch.sqrt(num_sq)
    den = _fro_norm(g_ref)
    return float(num / (den + eps_den))


def expected_rel_bound(
    l: int,
    *,
    L: int = 32,
    C: float = 1.03,
    dtype: Optional[torch.dtype] = torch.bfloat16,
    k: float = 4.0,
) -> float:
    """
    Bound ~ k * (C ** (L + 1 - l)) * eps_mch, with 1-based layer index l.
    - L is hard-coded default to 32 per your request.
    - C is 'close to 1'; 1.01–1.05 are reasonable defaults.
    - k absorbs the hidden constant in big-O; 2–8 are common choices.
    - dtype controls eps_mch; for FP8 use BF16 epsilon (see https://www.arxiv.org/pdf/2506.09280 theorem 5.3).
    """
    eps_mch = machine_epsilon_for_dtype(dtype or torch.bfloat16)
    depth = L + 1 - l  # 1-based depth from the top (as in the theorem)
    depth = max(depth, 0)
    return float(k * (C**depth) * eps_mch)


def check_gradient(
    g_hat: TensorLike,
    g_ref: TensorLike,
    l: int,
    *,
    L: int = 32,
    C: float = 1.03,
    dtype: Optional[torch.dtype] = None,
    k: float = 4.0,
) -> Tuple[float, float, bool]:
    """
    Compute (rel_error, bound, ok) for layer l.
    - If dtype is None, infer from g_ref (or g_hat if needed).
    # See https://www.arxiv.org/pdf/2506.09280 theorem 5.3
    """
    # Infer dtype if not provided
    if dtype is None:
        t0 = next(iter(_as_iter(g_ref)))
        dtype = t0.dtype
    rel = relative_grad_diff(g_hat, g_ref)
    bnd = expected_rel_bound(l, L=L, C=C, dtype=dtype, k=k)
    return rel, bnd, (rel <= bnd)


def _filter_optimizer_tensors(plain_tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Return only optimizer-related tensors from a flat checkpoint tensor dict."""
    return {
        k: v for k, v in plain_tensors.items() if k.startswith("optimizer.") and ".exp_avg." in k
    }


def assert_grads_close(left: torch.Tensor, right: torch.Tensor):
    # Implement theorem 5.3 of https://www.arxiv.org/pdf/2506.09280

    # This is the real test:
    rel, bnd, ok = check_gradient(
        left, right, l=0, dtype=torch.bfloat16
    )  # hard code to layer 0 since that's the most permissive

    # If the real test above fails, run an assert close for the useful diagnostics and raise either way.
    if not ok:
        rel_shuff, _, ok_shuff = check_gradient(
            left, torch.roll(right, shifts=-1, dims=-1), l=0, dtype=torch.bfloat16
        )

        try:
            torch.testing.assert_close(left, right)
            msg = (
                "AssertionError on relative norm magnitude "
                f"(rel={rel}, bnd={bnd}, ok={ok}, rel_shuff={rel_shuff}, ok_shuff={ok_shuff}) "
                "but torch.testing.assert_close(left, right) passes. \n"
                f"Left: {left.shape}/{left.dtype} {left}\n"
                f"Right: {right.shape}/{right.dtype} {right}"
            )
        except AssertionError as e:
            msg = (
                "AssertionError on relative norm magnitude "
                f"(rel={rel}, bnd={bnd}, ok={ok}, rel_shuff={rel_shuff}, ok_shuff={ok_shuff}): {e}\n"
                f"Left: {left.shape}/{left.dtype} {left}\n"
                f"Right: {right.shape}/{right.dtype} {right}"
            )
        raise AssertionError(msg)


def unshard_row_parallel_state(saved_state, out_features, in_features, tp):
    # saved_state: [..., tp, out_features * (in_features // tp)]
    prefix = saved_state.shape[:-2]
    per = in_features // tp
    x = saved_state.view(*prefix, tp, out_features, per)  # [..., tp, O, I_shard]
    x = x.permute(*range(len(prefix)), -2, -3, -1)  # [..., O, tp, I_shard]
    x = x.reshape(*prefix, out_features, in_features)  # [..., O, I]
    return x


def _assert_optimizer_tensors_equal(
    left: Dict[str, torch.Tensor],
    right: Dict[str, torch.Tensor],
    left_empty: Dict[str, torch.Tensor],
    right_empty: Dict[str, torch.Tensor],
    eps=1e-4,
):
    left_keys = set(left.keys())
    right_keys = set(right.keys())

    only_left = sorted(left_keys - right_keys)
    only_right = sorted(right_keys - left_keys)
    assert (
        not only_left and not only_right
    ), f"Optimizer tensor keys mismatch.\nOnly in left: {only_left}\nOnly in right: {only_right}"
    some_non_zero = False
    assertions = []
    for key in sorted(left_keys):
        lt, rt = left[key], right[key]
        rt_colpar, rt_rowpar = None, None
        if lt.shape != rt.shape:
            # "Tensor shape mismatch for {key}: {lt.shape} vs {rt.shape}, trying simple reshape
            original_key = key.replace("optimizer.state.exp_avg.", "")
            # Unsharded shape
            # {'decoder.layers.self_attention.linear_proj.weight': torch.Size([32, 3072, 4096]), 'optimizer.state.exp_avg.decoder.layers.self_attention.linear_proj.weight': torch.Size([32, 1, 1, 12582912]), 'optimizer.state.exp_avg_sq.decoder.layers.self_attention.linear_proj.weight': torch.Size([32, 1, 1, 12582912]), 'optimizer.state.fp32_param.decoder.layers.self_attention.linear_proj.weight': torch.Size([32, 1, 1, 12582912])}
            # Sharded shape
            # {'decoder.layers.self_attention.linear_proj.weight': torch.Size([32, 3072, 4096]), 'optimizer.state.exp_avg.decoder.layers.self_attention.linear_proj.weight': torch.Size([32, 1, 2, 6291456]), 'optimizer.state.exp_avg_sq.decoder.layers.self_attention.linear_proj.weight': torch.Size([32, 1, 2, 6291456]), 'optimizer.state.fp32_param.decoder.layers.self_attention.linear_proj.weight': torch.Size([32, 1, 2, 6291456])}
            left_shape = left_empty[original_key].shape
            right_shape = right_empty[original_key].shape
            skip_tp_check = False

            if left_shape != right_shape:
                if "embedding.word_embeddings.weight" in key or ".output_layer.weight" in key:
                    # First handle different padding on the input/output dimensions.
                    lt = lt.reshape(left_shape)
                    rt = rt.reshape(right_shape)
                    min_dim = min(left_shape[0], right_shape[0])
                    lt = lt[:min_dim, ...]
                    rt = rt[:min_dim, ...]
                    left_shape = lt.shape
                    right_shape = rt.shape
                    skip_tp_check = True
                else:
                    raise AssertionError(
                        f"Tensor shape mismatch for {key}: {left_shape} vs {right_shape}"
                    )
            # problem: we do not know the TP axis for this tensor. We can guess though.
            if len(left_shape) == 3 and not skip_tp_check:
                # TP axis is 1
                lt = lt.reshape(left_shape[0], 1, left_shape[1], left_shape[2])
            elif len(left_shape) == 2 and not skip_tp_check:
                # TP axis is 2
                lt = lt.reshape(left_shape[0], 1, left_shape[1])

            if (
                key.endswith("mlp.linear_fc2.weight")
                or key.endswith("self_attention.linear_proj.weight")
            ) and not skip_tp_check:
                # Handle row parallel linear layers.
                # TODO come up with a better way to determine row parallel linear layers.
                rt = unshard_row_parallel_state(
                    rt, out_features=left_shape[1], in_features=left_shape[2], tp=rt.shape[2]
                )
            else:
                try:
                    rt = rt.reshape(lt.shape)
                except Exception as e:
                    msg = f"Tensor shape mismatch for {key}: {lt.shape} vs {rt.shape}, simple reshape failed: {e}"
                    if "embedding.word_embeddings.weight" in key or ".output_layer.weight" in key:
                        print(
                            f"FIXME: Skipping {key} because it's a word embedding or output layer,"
                            "and something about padding changes under TP."
                        )
                        continue
                    raise AssertionError(msg)

        assert (
            lt.shape == rt.shape and lt.dtype == rt.dtype
        ), f"Tensor meta mismatch for {key}: {lt.shape}/{lt.dtype} vs {rt.shape}/{rt.dtype}"
        # Reduce the rate of 0 vs near 0 rtol failures by adding a small epsilon
        left_scale = torch.max(torch.abs(lt))
        right_scale = torch.max(torch.abs(rt))
        if left_scale <= eps and right_scale <= eps:
            print(
                f"WARNING: zero-ish scale tensors ({left_scale=} vs {right_scale=}) "
                f"so they will trivially pass comparing {key=}"
            )
        else:
            some_non_zero = True
        try:
            assert_grads_close(lt, rt)
            print(f"Optimizer tensors match for {key}")
        except AssertionError as e:
            assertions.append(AssertionError(f"AssertionError for {key}: {e}"))
    assert not assertions, f"Assertion Errors found comparing keys: {assertions}"
    assert some_non_zero, "No non-zero tensors found in this comparison"


def load_dist_checkpoint_pt(
    ckpt_dir,
    metadata_ckpt_dir=None,
    pattern=r"optimizer",
    device="cpu",
    return_full_empty: bool = False,
):
    """Return {full_key: tensor} for every tensor whose key matches *pattern*."""
    meta_ckpt_dir = Path(metadata_ckpt_dir or ckpt_dir)
    meta_reader = FileSystemReader(str(meta_ckpt_dir))

    # --- fast metadata pass (no tensor data yet) -----------------------------
    meta = meta_reader.read_metadata()  # tiny JSON read
    tmeta = meta.state_dict_metadata  # key ➜ TensorMetadata
    if return_full_empty:
        wanted = [k for k in tmeta if hasattr(tmeta[k], "size")]
    else:
        wanted = [k for k in tmeta if re.search(pattern, k) and hasattr(tmeta[k], "size")]
    if not wanted:
        raise ValueError(f"No keys matching /{pattern}/ in {ckpt_dir}")

    # --- build "empty" placeholders -----------------------------------------
    placeholders = {
        k: torch.empty(tuple(tmeta[k].size), dtype=tmeta[k].properties.dtype, device=device)
        for k in wanted
    }
    if return_full_empty:
        return placeholders
    # --- stream just those tensors (no process-group needed) -----------------
    data_reader = FileSystemReader(str(ckpt_dir))

    load(
        state_dict=placeholders,
        storage_reader=data_reader,
        no_dist=True,  # switches off all collectives
    )
    return placeholders  # dict[str, Tensor]


def test_optimizer_states_match(checkpoint_dirs):
    """
    Compare optimizer state across provided torch_dist checkpoints:
    - Keys: ensure the set of optimizer tensor keys match across checkpoints
    - Values: ensure corresponding tensors are equal (allclose)
    - Structure (non-tensor common state): ensure common optimizer structures match
    """
    assert len(checkpoint_dirs) > 1, "This test requires 2 or more checkpoints <dir1> [<dir2> ...]."

    base_dir = checkpoint_dirs[0]

    # Compare optimizer tensors
    base_plain = load_dist_checkpoint_pt(base_dir)
    base_empty = load_dist_checkpoint_pt(base_dir, return_full_empty=True, device="meta")
    base_opt_tensors = _filter_optimizer_tensors(base_plain)
    assert base_opt_tensors, f"No optimizer tensors found in checkpoint: {base_dir}"
    assertions = []
    for other_dir in checkpoint_dirs[1:]:
        try:
            other_plain = load_dist_checkpoint_pt(other_dir)
            other_empty = load_dist_checkpoint_pt(other_dir, return_full_empty=True, device="meta")
            other_opt_tensors = _filter_optimizer_tensors(other_plain)
            assert other_opt_tensors, f"No optimizer tensors found in checkpoint: {other_dir}"
            _assert_optimizer_tensors_equal(
                base_opt_tensors, other_opt_tensors, base_empty, other_empty
            )
            print(f"Optimizer tensors match for {base_dir} and {other_dir}")
            del other_plain
            del other_opt_tensors
        except AssertionError as e:
            msg = f"AssertionError comparing {base_dir} to {other_dir}:\n{e}"
            print(f"Optimizer tensors mismatch for {base_dir} and {other_dir}:\n{msg}")
            assertions.append(AssertionError(msg))
    assert not assertions, f"AssertionErrors comparing {checkpoint_dirs}:\n{assertions}"


def main():
    parser = ArgumentParser(
        description="Given checkpoints saved with adam b1,b2=0 trained for one step, "
        "we can check that the gradients match under different training configurations. "
        "Currently this test script has some hard-coded assumptions for GPT style models, "
        "namely which layers are RowParallel and require different unsharding logic."
    )
    parser.add_argument(
        "checkpoints", nargs="+", type=Path, help="Path to the checkpoints to compare"
    )
    args = parser.parse_args()
    test_optimizer_states_match(args.checkpoints)


if __name__ == "__main__":
    main()
