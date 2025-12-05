import h5py
import numpy as np
import matplotlib.pyplot as plt

# ---- 可调参数集中 ----
# 只改一次标签，输入HDF5路径和输出文件名都会同步
RUN_TAG = "1027_0.0375"
HDF5_PATH = f"results 141125/output_file_{RUN_TAG}.hdf5"
SUFFIX_RANGE = range(1, 10)  # 生成1到9
PUITS_GROUP_TEMPLATES = {
    # 顺序与 aggregated 保持一致，颜色/图例一致
    'A2': 'ImageA2-{suffix}-C2',
    'B2': 'ImageB2-{suffix}-C2',
    'C2': 'ImageC2-{suffix}-C2',
    'A3': 'ImageA3-{suffix}-C2',
    'B3': 'ImageB3-{suffix}-C2',
    'C3': 'ImageC3-{suffix}-C2',
    # 如需加入A1/B1等，按同样格式扩展即可
}
CONCENTRATION_VALUES = [0.11, 0.22, 0.44, 0.88, 1.76, 3.52]
CONCENTRATION_VALUES = [0.043, 0.087, 0.174, 0.348, 0.696, 1.39]
PUITS_CONCENTRATIONS = {name: conc for name, conc in zip(PUITS_GROUP_TEMPLATES.keys(), CONCENTRATION_VALUES)}
OUTPUT_TEMPLATE = "results 141125/initial curve/puits_cell_counts_{run_tag}_{suffix}.pdf"
PIXEL_SIZE_UM = 1.24
IMAGE_WIDTH_PX = 1408
IMAGE_HEIGHT_PX = 1040
FIELD_AREA_MICRONS2 = (IMAGE_WIDTH_PX * PIXEL_SIZE_UM) * (IMAGE_HEIGHT_PX * PIXEL_SIZE_UM)
# 每帧间隔（小时）；改这里即可调整时间轴单位
FRAME_INTERVAL_HOURS = 3
# y 轴模式："count"（默认，细胞计数）或 "density"（细胞密度）
Y_MODE = "density"
# y 轴上限；None 时按 Matplotlib 自动缩放
Y_MAX = 0.002


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
                variance = np.var(counts)
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


def plot_puits_cell_counts(puits_stats, output_path, puits_concentrations=None, y_mode="count", y_max=None):
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
        plt.scatter(x, y, label=label_text, color=colors[i], marker='o', s=50, alpha=0.7)
        plt.fill_between(x, y - std_dev, y + std_dev, color=colors[i], alpha=0.1)
    plt.xlabel('Time (h)', fontsize=24, fontweight='bold')
    plt.ylabel(y_label, fontsize=24, fontweight='bold')
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    if y_max is not None:
        plt.ylim(top=y_max)
    plt.legend(fontsize=20, loc='upper right', frameon=True, framealpha=0.95)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()


def main():
    try:
        cell_counts = read_all_cell_counts(HDF5_PATH)

        for suffix in SUFFIX_RANGE:
            puits_groups = {
                name: [template.format(suffix=suffix)]
                for name, template in PUITS_GROUP_TEMPLATES.items()
            }
            puits_stats = calculate_puits_stats(cell_counts, puits_groups)
            output_path = OUTPUT_TEMPLATE.format(run_tag=RUN_TAG, suffix=suffix)
            plot_puits_cell_counts(puits_stats, output_path, PUITS_CONCENTRATIONS, y_mode=Y_MODE, y_max=Y_MAX)
        print("All plots have been saved.")
    except Exception as e:
        print(f"Error occurred: {str(e)}")


if __name__ == "__main__":
    main()
