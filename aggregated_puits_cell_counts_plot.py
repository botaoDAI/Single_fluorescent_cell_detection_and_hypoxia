import h5py
import numpy as np
import matplotlib.pyplot as plt

# ---- 可调参数放在一起，方便修改 ----
# 只需改一次标签，输入HDF5和输出图片名都会同步
RUN_TAG = "1022_0.0375"
HDF5_PATH = f"results 151025/output_file_{RUN_TAG}.hdf5"
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
CONCENTRATION_VALUES = [0.11, 0.22, 0.44, 0.88, 1.76, 3.52]
PUITS_CONCENTRATIONS = {name: conc for name, conc in zip(PUITS_GROUPS.keys(), CONCENTRATION_VALUES)}
OUTPUT_PATH = f"puits_cell_counts_{RUN_TAG}_aggregated.png"


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


def plot_puits_cell_counts(puits_stats, output_path, puits_concentrations=None):
    plt.figure(figsize=(15, 8))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    for i, (puits_name, stats) in enumerate(puits_stats.items()):
        n_points = len(stats['mean_counts'])
        x = np.arange(0, n_points * 1.5, 1.5)
        y = stats['mean_counts']
        variance = stats['variances']
        label_text = puits_name
        if puits_concentrations and puits_name in puits_concentrations:
            label_text = f" ({puits_concentrations[puits_name]:.2f} x10^5/ml)"
        plt.scatter(x, y, label=label_text, color=colors[i], marker='o', s=50, alpha=0.7)
        plt.fill_between(x, np.array(y) - np.sqrt(np.array(variance)), np.array(y) + np.sqrt(np.array(variance)), color=colors[i], alpha=0.1)
    plt.xlabel('Time (h)', fontsize=24, fontweight='bold')
    plt.ylabel('Average Cell Count', fontsize=24, fontweight='bold')
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
        plot_puits_cell_counts(puits_stats, OUTPUT_PATH, PUITS_CONCENTRATIONS)
        print(f"Aggregated plot has been saved to {OUTPUT_PATH}.")
    except Exception as e:
        print(f"Error occurred: {str(e)}")


if __name__ == "__main__":
    main()
