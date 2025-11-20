import h5py
import numpy as np
import matplotlib.pyplot as plt

def read_group_average_counts(hdf5_path, group_names):
    # group_names: list of group name strings
    all_counts = []  # shape: (len(group_names), 161)
    with h5py.File(hdf5_path, 'r') as f:
        for group_name in group_names:
            group_counts = []
            for frame_num in range(161):
                frame_name = f"frame{frame_num}"
                if group_name in f and frame_name in f[group_name]:
                    count = f[group_name][frame_name]['block0_values'].shape[0]
                    group_counts.append(count)
                else:
                    group_counts.append(np.nan)
            all_counts.append(group_counts)
    # 按列取平均（忽略nan）
    mean_counts = np.nanmean(np.array(all_counts), axis=0)
    return mean_counts

def plot_two_groups(normoxie_counts, hypoxie_counts):
    plt.figure(figsize=(12, 7))
    x = range(len(normoxie_counts))
    plt.plot(x, normoxie_counts, label='Normoxie (B3)', color='#1f77b4', marker='o', alpha=0.8)
    plt.plot(x, hypoxie_counts, label='Hypoxie (B1)', color='#d62728', marker='s', alpha=0.8)
    plt.xlabel('Time (Frame)', fontsize=18, fontweight='bold')
    plt.ylabel('Cell Count', fontsize=18, fontweight='bold')
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=16)
    plt.tight_layout()
    plt.savefig('cell_counts_two_groups.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("图像已保存为 cell_counts_two_groups.png")

def main():
    normoxie_path = "results 120724/output_file_870_0.0375.hdf5"
    hypoxie_path = "results 020425/output_file_958_0.0375.hdf5"
    normoxie_groups = [f"ImageB3-{i}-C2" for i in range(1, 10)]
    hypoxie_groups = [f"ImageB1-{i}-C2" for i in range(1, 10)]
    normoxie_counts = read_group_average_counts(normoxie_path, normoxie_groups)
    hypoxie_counts = read_group_average_counts(hypoxie_path, hypoxie_groups)
    plot_two_groups(normoxie_counts, hypoxie_counts)

if __name__ == "__main__":
    main() 