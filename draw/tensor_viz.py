"""
Tensor visualization utilities: visualize an N-D tensor as a thin bar chart
and print summary statistics (min/max/mean/std, indices, etc.).

Designed for numpy arrays, but will accept PyTorch tensors if available
without requiring torch as a dependency.
"""

from __future__ import annotations

import math
import sys
from typing import Any, Iterable, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt


def _to_numpy(array_like: Any) -> np.ndarray:
    """
    Convert a variety of array-like inputs to a numpy.ndarray.
    Supports:
    - numpy.ndarray
    - list/tuple (converted via np.asarray)
    - torch.Tensor (if torch is installed), moved to CPU and detached
    """
    if isinstance(array_like, np.ndarray):
        return array_like

    # Try PyTorch without introducing a hard dependency
    try:
        import torch  # type: ignore

        if isinstance(array_like, torch.Tensor):
            return array_like.detach().cpu().numpy()
    except Exception:
        # Either torch isn't installed or conversion failed; fall back below
        pass

    return np.asarray(array_like)


def _nan_safe_stats(values: np.ndarray) -> Tuple[float, float, float, float, float]:
    """Return (min, max, mean, std, median) ignoring NaN where possible."""
    # Use nan* variants to ignore NaNs. If all values are NaN, results will be NaN.
    vmin = float(np.nanmin(values)) if np.any(np.isfinite(values)) else float("nan")
    vmax = float(np.nanmax(values)) if np.any(np.isfinite(values)) else float("nan")
    vmean = float(np.nanmean(values)) if np.any(np.isfinite(values)) else float("nan")
    vstd = float(np.nanstd(values)) if np.any(np.isfinite(values)) else float("nan")
    vmed = float(np.nanmedian(values)) if np.any(np.isfinite(values)) else float("nan")
    return vmin, vmax, vmean, vstd, vmed


def _finite_arg_indices(values: np.ndarray, shape: Tuple[int, ...]) -> Tuple[Optional[Tuple[int, ...]], Optional[Tuple[int, ...]]]:
    """
    Return unravelled indices for finite min and max. If no finite values exist,
    return (None, None).
    """
    flat = values.reshape(-1)
    finite_mask = np.isfinite(flat)
    if not np.any(finite_mask):
        return None, None

    finite_vals = flat[finite_mask]
    # Positions within finite subset
    min_pos = int(np.argmin(finite_vals))
    max_pos = int(np.argmax(finite_vals))
    # Map back to original flat indices
    finite_indices = np.nonzero(finite_mask)[0]
    flat_min_idx = int(finite_indices[min_pos])
    flat_max_idx = int(finite_indices[max_pos])
    return np.unravel_index(flat_min_idx, shape), np.unravel_index(flat_max_idx, shape)


def visualize_tensor_bars(
    tensor: Any,
    title: Optional[str] = None,
    color_mode: str = "by_first_dim",
    figsize: Tuple[float, float] = (14, 6),
    show: bool = True,
    save_path: Optional[str] = None,
    max_bars_to_plot: int = 50000,
) -> None:
    """
    Visualize an N-D tensor as a thin bar chart, and print stats.

    Parameters
    ----------
    tensor : Any
        Input array-like. Supports numpy arrays, lists/tuples, and torch Tensors.
    title : Optional[str]
        Figure title.
    color_mode : str
        Currently supports:
        - "by_first_dim": color bars by the first dimension index (groups)
        - "single": single color for all bars
    figsize : Tuple[float, float]
        Matplotlib figure size.
    show : bool
        Whether to call plt.show(). If False, the figure is created but not shown.
    save_path : Optional[str]
        If provided, save the figure to this path.
    max_bars_to_plot : int
        Safety cap to avoid rendering too many bars.
    """
    arr = _to_numpy(tensor)

    # Ensure numeric dtype for plotting; attempt best-effort conversion
    if not np.issubdtype(arr.dtype, np.number):
        arr = arr.astype(np.float64)

    print("=== Tensor Stats ===")
    print(f"形状 (shape): {arr.shape}")
    print(f"元素总数 (size): {arr.size}")
    print(f"数据类型 (dtype): {arr.dtype}")

    flat = arr.reshape(-1)
    finite_mask = np.isfinite(flat)
    num_finite = int(np.count_nonzero(finite_mask))
    num_nan = int(np.count_nonzero(np.isnan(flat)))
    num_pos_inf = int(np.count_nonzero(np.isposinf(flat)))
    num_neg_inf = int(np.count_nonzero(np.isneginf(flat)))
    print(f"有限值数量 (finite count): {num_finite}")
    print(f"NaN 数量: {num_nan}, +Inf 数量: {num_pos_inf}, -Inf 数量: {num_neg_inf}")

    vmin, vmax, vmean, vstd, vmed = _nan_safe_stats(flat.astype(np.float64))
    min_idx, max_idx = _finite_arg_indices(arr.astype(np.float64), arr.shape)

    print(f"最小值 (min): {vmin}")
    print(f"最大值 (max): {vmax}")
    print(f"平均值 (mean): {vmean}")
    print(f"标准差 (std): {vstd}")
    print(f"中位数 (median): {vmed}")
    print(f"最小值位置 (argmin multi-index): {min_idx}")
    print(f"最大值位置 (argmax multi-index): {max_idx}")

    # If empty, nothing to plot
    if arr.size == 0:
        print("空张量，无需绘图。")
        return

    # Prepare data for plotting
    flat_values = flat
    total = flat_values.size
    if total > max_bars_to_plot:
        step = int(math.ceil(total / float(max_bars_to_plot)))
        selected_indices = np.arange(0, total, step)
        values_to_plot = flat_values[selected_indices]
        print(f"数据过大，已按步长 {step} 下采样，可视化 {values_to_plot.size}/{total} 个柱子。")
    else:
        selected_indices = np.arange(total)
        values_to_plot = flat_values

    # Compute group ids for coloring when using by_first_dim
    shape = arr.shape
    ndim = arr.ndim
    if color_mode == "by_first_dim" and ndim >= 1 and shape[0] > 0:
        group_size = int(np.prod(shape[1:])) if ndim > 1 else 1
        if group_size == 0:
            group_ids = np.zeros_like(selected_indices)
        else:
            group_ids = (selected_indices // group_size) % shape[0]
        num_groups = int(shape[0])
        # Choose a colormap and map each group id to a color
        if num_groups <= 20:
            cmap = plt.get_cmap("tab20")
        else:
            cmap = plt.get_cmap("hsv")
        colors = np.array([cmap(i / max(1, num_groups - 1)) for i in range(num_groups)])
        bar_colors = colors[group_ids]
    else:
        bar_colors = "C0"

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(values_to_plot.size)
    ax.bar(x, values_to_plot, width=1.0, color=bar_colors, edgecolor="none")
    ax.set_xlim(-1, values_to_plot.size)
    ax.set_xlabel("Flattened index")
    ax.set_ylabel("Value")

    if title is None:
        title = f"Tensor bars: shape={shape}, size={arr.size}"
    ax.set_title(title)
    ax.grid(axis="y", linestyle=":", alpha=0.5)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150)
        print(f"图像已保存到: {save_path}")
    if show:
        plt.show()


__all__ = ["visualize_tensor_bars"]

