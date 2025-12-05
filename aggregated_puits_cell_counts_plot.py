import h5py
import numpy as np
import matplotlib.pyplot as plt

# ---- 可调参数放在一起，方便修改 ----
# 只需改一次标签，输入HDF5和输出图片名都会同步
RUN_TAG = "1027_0.0375"
HDF5_PATH = f"results 141125/output_file_{RUN_TAG}.hdf5"
PUITS_IMAGE_SUFFIX = [str(s) for s in range(1, 10)]
# 如需包含A1/B1等组，可参考下方格式添加：
# PUITS_GROUPS = {
#     'A1': [f'ImageA1-{s}-C2' for s in PUITS_IMAGE_SUFFIX],
#     ...
# }
PUITS_GROUPS = {
    'A2': [f'ImageA2-{s}-C2' for s in PUITS_IMAGE_SUFFIX],
    'B2': [f'ImageB2-{s}-C2' for s in PUITS_IMAGE_SUFFIX],
    'C2': [f'ImageC2-{s}-C2' for s in PUITS_IMAGE_SUFFIX],
    'A3': [f'ImageA3-{s}-C2' for s in PUITS_IMAGE_SUFFIX],
    'B3': [f'ImageB3-{s}-C2' for s in PUITS_IMAGE_SUFFIX],
    'C3': [f'ImageC3-{s}-C2' for s in PUITS_IMAGE_SUFFIX],
}
# 从A2到C3的初始浓度（0.11 x10^5开始，每次*2到3.52 x10^5）
CONCENTRATION_VALUES = [0.043, 0.087, 0.174, 0.348, 0.696, 1.39]
PUITS_CONCENTRATIONS = {name: conc for name, conc in zip(PUITS_GROUPS.keys(), CONCENTRATION_VALUES)}
OUTPUT_PATH = f"puits_cell_counts_{RUN_TAG}_aggregated.png"
PIXEL_SIZE_UM = 1.24
IMAGE_WIDTH_PX = 1408
IMAGE_HEIGHT_PX = 1040
FIELD_AREA_MICRONS2 = (IMAGE_WIDTH_PX * PIXEL_SIZE_UM) * (IMAGE_HEIGHT_PX * PIXEL_SIZE_UM)
# 每帧间隔（小时）；改这里即可调整时间轴单位
FRAME_INTERVAL_HOURS = 3
# y 轴模式："count"（默认，细胞计数）或 "density"（细胞密度）
Y_MODE = "density"
# 根据开关只画到 y 第一次达到 1.5e-3 的功能
CUT_AT_THRESHOLD = False
THRESHOLD_VALUE = 0.8e-3


def read_all_cell_counts(hdf5_path):
    """读取HDF5文件中每个图像组各帧的细胞数量。"""
    cell_counts = {}
    with h5py.File(hdf5_path, 'r') as f:
        image_groups = list(f.keys())
        for image_group in image_groups:
            frame_counts = {}
            for frame_num in range(161):
                frame_name = f"frame{frame_num}"
                if frame_name in f[image_group]:
                    cell_count = f[image_group][frame_name]['block0_values'].shape[0]
                    frame_counts[frame_num] = cell_count
            cell_counts[image_group] = frame_counts
    return cell_counts


def calculate_puits_stats(cell_counts, puits_groups):
    """按puits分组计算每一帧的平均值与方差。"""
    puits_stats = {}
    for puits_name, image_groups in puits_groups.items():
        time_series = []
        variances = []
        for frame in range(161):
            counts = []
            for group in image_groups:
                if frame in cell_counts[group]:
                    counts.append(cell_counts[group][frame])
            if counts:
                mean_count = np.mean(counts)
                variance = np.var(counts, ddof=1) if len(counts) > 1 else 0.0
                time_series.append(mean_count)
                variances.append(variance)
        puits_stats[puits_name] = {
            'mean_counts': time_series,
            'variances': variances
        }
    return puits_stats


def prepare_y_values(mean_counts, variances, y_mode):
    """根据y轴模式返回对应的y值与标准差。"""
    y = np.array(mean_counts, dtype=float)
    std_dev = np.sqrt(np.array(variances, dtype=float))
    if y_mode == "density":
        y = y / FIELD_AREA_MICRONS2
        std_dev = std_dev / FIELD_AREA_MICRONS2
    return y, std_dev


def maybe_truncate_curve(x_values, y_values, std_values=None):
    """根据开关只画到 y 第一次达到 1.5e-3 的功能。"""
    if not CUT_AT_THRESHOLD:
        return x_values, y_values, std_values
    hit_indices = np.where(y_values >= THRESHOLD_VALUE)[0]
    if hit_indices.size == 0:
        return x_values, y_values, std_values
    end_idx = int(hit_indices[0])
    if std_values is None:
        return x_values[: end_idx + 1], y_values[: end_idx + 1], None
    return (
        x_values[: end_idx + 1],
        y_values[: end_idx + 1],
        std_values[: end_idx + 1],
    )


def plot_puits_cell_counts(puits_stats, output_path, puits_concentrations=None, y_mode="count"):
    plt.figure(figsize=(15, 8))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    y_label = "Cell count" if y_mode == "count" else "Cell density (cells/µm²)"
    for i, (puits_name, stats) in enumerate(puits_stats.items()):
        n_points = len(stats['mean_counts'])
        x = np.arange(0, n_points * FRAME_INTERVAL_HOURS, FRAME_INTERVAL_HOURS)
        y, std_dev = prepare_y_values(stats['mean_counts'], stats['variances'], y_mode)
        label_text = puits_name
        if puits_concentrations and puits_name in puits_concentrations:
            label_text = f" ({puits_concentrations[puits_name]:.2f} x10^5/ml)"
        x_plot, y_plot, std_plot = maybe_truncate_curve(x, y, std_dev)
        plt.scatter(x_plot, y_plot, label=label_text, color=colors[i], marker='o', s=50, alpha=0.7)
        if std_plot is not None:
            plt.fill_between(x_plot, y_plot - std_plot, y_plot + std_plot, color=colors[i], alpha=0.1)
    plt.xlabel('Time (h)', fontsize=24, fontweight='bold')
    plt.ylabel(y_label, fontsize=24, fontweight='bold')
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.legend(fontsize=20, loc='upper right', frameon=True, framealpha=0.95)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()


def main():
    try:
        cell_counts = read_all_cell_counts(HDF5_PATH)
        puits_stats = calculate_puits_stats(cell_counts, PUITS_GROUPS)
        plot_puits_cell_counts(puits_stats, OUTPUT_PATH, PUITS_CONCENTRATIONS, y_mode=Y_MODE)
        print(f"Aggregated plot has been saved to {OUTPUT_PATH}.")
    except Exception as e:
        print(f"Error occurred: {str(e)}")


if __name__ == "__main__":
    main()
