import numpy as np

from tensor_viz import visualize_tensor_bars


def main() -> None:
    # Create a 3D tensor with a clear structure to demonstrate coloring by first dim
    rng = np.random.default_rng(42)
    tensor = rng.normal(loc=0.0, scale=1.0, size=(4, 16, 64)).astype(np.float64)

    # Inject a few special values to demonstrate reporting
    tensor[0, 0, 0] = -10.0
    tensor[-1, -1, -1] = 15.0
    tensor[1, 2, 3] = np.nan
    tensor[2, 3, 4] = np.inf

    visualize_tensor_bars(
        tensor,
        title="多维细柱状图可视化 (按第一维分组着色)",
        color_mode="by_first_dim",
        figsize=(16, 6),
        show=True,
        save_path='./draw/res',
        max_bars_to_plot=60000,
    )


if __name__ == "__main__":
    main()

