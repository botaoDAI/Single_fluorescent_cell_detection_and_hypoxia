import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def read_all_cell_counts(hdf5_path):
    # 创建一个字典来存储结果
    cell_counts = {}
    
    # 打开HDF5文件
    with h5py.File(hdf5_path, 'r') as f:
        # 获取所有图像组的名称
        image_groups = list(f.keys())
        
        # 遍历每个图像组
        for image_group in image_groups:
            frame_counts = {}
            
            # 遍历每一帧（0-160）
            for frame_num in range(161):
                frame_name = f"frame{frame_num}"
                if frame_name in f[image_group]:
                    # 获取该帧的数据形状（第一维度就是细胞数量）
                    cell_count = f[image_group][frame_name]['block0_values'].shape[0]
                    frame_counts[frame_num] = cell_count
            
            cell_counts[image_group] = frame_counts
    
    return cell_counts

def calculate_puits_stats(cell_counts, puits_groups):
    # 存储每个puits的统计信息
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

def plot_puits_cell_counts(puits_stats, suffix):
    plt.figure(figsize=(15, 8))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    for i, (puits_name, stats) in enumerate(puits_stats.items()):
        n_points = len(stats['mean_counts'])
        x = np.arange(0, n_points * 1.5, 1.5)
        y = stats['mean_counts']
        variance = stats['variances']
        plt.scatter(x, y, label=puits_name, color=colors[i], marker='o', s=50, alpha=0.7)
        plt.fill_between(x, np.array(y) - np.sqrt(np.array(variance)), np.array(y) + np.sqrt(np.array(variance)), color=colors[i], alpha=0.1)
    plt.xlabel('Time (h)', fontsize=24, fontweight='bold')
    plt.ylabel('Average Cell Count', fontsize=24, fontweight='bold')
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.legend(fontsize=20, loc='upper right', frameon=True, framealpha=0.95)
    plt.tight_layout()
    plt.savefig(f'puits_cell_counts_1024_0.0375_{suffix}.png', dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()

def main():
    hdf5_path = "results 151025/output_file_1024_0.0375.hdf5"
    try:
        # 读取细胞数量数据
        cell_counts = read_all_cell_counts(hdf5_path)

        # 汇总9个champs后绘制单张图
        puits_image_suffix = [str(s) for s in range(1, 10)]
        puits_groups = {
#                'A1': [f'ImageA1-{s}-C2' for s in puits_image_suffix],
                'A2': [f'ImageA2-{s}-C2' for s in puits_image_suffix],
                'A3': [f'ImageA3-{s}-C2' for s in puits_image_suffix],
#                'B1': [f'ImageB1-{s}-C2' for s in puits_image_suffix],
                'B2': [f'ImageB2-{s}-C2' for s in puits_image_suffix],
                'B3': [f'ImageB3-{s}-C2' for s in puits_image_suffix],
                'C2': [f'ImageC2-{s}-C2' for s in puits_image_suffix],
                'C3': [f'ImageC3-{s}-C2' for s in puits_image_suffix],
        }
        # 计算每个puits在9个champs下的统计信息
        puits_stats = calculate_puits_stats(cell_counts, puits_groups)
        # 绘制并保存汇总图像
        plot_puits_cell_counts(puits_stats, 'aggregated')
        print("Aggregated plot has been saved.")
    except Exception as e:
        print(f"Error occurred: {str(e)}")

if __name__ == "__main__":
    main() 
