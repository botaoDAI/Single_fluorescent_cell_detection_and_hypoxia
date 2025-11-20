# ================ DESCRIPTION ==============================================================================

# This program is used to extract cell counts from an HDF5 file and calculate cell density, 
# which will then be used as input to the Marianne model.

import os
import h5py
import numpy as np


def read_frame_counts_per_group(hdf5_path: str) -> dict:
    """
    读取 HDF5 中每个图像组每一帧的细胞数量（以 block0_values 的第一维作为计数）。

    返回:
        dict[group_name -> dict[frame_index -> count]]
    """
    group_to_framecounts = {}
    with h5py.File(hdf5_path, 'r') as f:
        image_groups = list(f.keys())
        for image_group in image_groups:
            frame_counts = {}
            for frame_num in range(161):
                frame_name = f"frame{frame_num}"
                if frame_name in f[image_group]:
                    frame_counts[frame_num] = f[image_group][frame_name]['block0_values'].shape[0]
            group_to_framecounts[image_group] = frame_counts
    return group_to_framecounts


def compute_mean_std_density_for_puits(
    group_to_framecounts: dict,
    puits_name: str,
    suffixes: list,
    channel: str = "C2",
    max_frame: int = 160,
    field_area_microns2: float = None,
) -> tuple:
    """
    基于指定 puits（例如 A1）和多次重复（suffixes，如 1..9），
    计算每一帧的密度（cells/µm^2）的均值与标准差。

    返回:
        (mean_array, std_array)，长度为帧数（0..max_frame）。
    """
    image_groups = [f"Image{puits_name}-{s}-{channel}" for s in suffixes]

    means = []
    stds = []
    for frame in range(max_frame + 1):
        densities = []
        for g in image_groups:
            frames = group_to_framecounts.get(g, {})
            if frame in frames:
                count_value = frames[frame]
                if field_area_microns2 is not None and field_area_microns2 > 0:
                    densities.append(float(count_value) / float(field_area_microns2))
        if densities:
            dens_arr = np.array(densities, dtype=float)
            means.append(np.mean(dens_arr))
            stds.append(np.std(dens_arr, ddof=0))
        else:
            means.append(np.nan)
            stds.append(np.nan)
    return np.array(means), np.array(stds)


def save_mean_std_txt(output_path: str, means: np.ndarray, stds: np.ndarray) -> None:
    data = np.column_stack((means, stds))
    np.savetxt(output_path, data, fmt='%.6f')


def main():
    # ===== 可按需修改的输入路径与设置 =====
    # HDF5 路径（请指向含有辐射实验的数据文件）
    hdf5_path = "/Users/dai/Desktop/detection rouge/results 120724/output_file_869_0.0375.hdf5"
    # 输出目录
    output_dir = "/Users/dai/Desktop/detection rouge/results 120724"
    # 剂量（用于输出文件命名，形如 "{dose}_Gy_WellX_3exps.txt"）
    dose_gy = 15  # 可修改为 0, 5, 10, 15, 20 等
    # 命名所需标签（与 Model_in_Response_to_Radiation_with_logistic_in_Cd_Cr.py 的 path_exp 一致）
    experiment_tag = "Incucyte_F98"
    date_str = "2024_07_12"
    smooth_tag = "smooth=35"
    # 将 1..9 次拍摄作为“重复实验”汇总
    suffixes = list(range(1, 10))
    # 孔位顺序与 Well 编号对应关系（1..6）
    puits_order = ["A1", "A2", "A3", "B1", "B2", "B3"]

    # 视野面积（微米^2）：(宽像素×分辨率)×(高像素×分辨率)
    pixel_size_um = 1.24
    image_width_px = 1408
    image_height_px = 1040
    field_area_microns2 = (image_width_px * pixel_size_um) * (image_height_px * pixel_size_um)

    os.makedirs(output_dir, exist_ok=True)

    # 读取所有组-帧的计数
    group_to_framecounts = read_frame_counts_per_group(hdf5_path)

    # 为每个孔位生成一个两列的 txt：第一列均值，第二列标准差（单位：cells/µm^2）
    for idx, puits in enumerate(puits_order, start=1):
        means, stds = compute_mean_std_density_for_puits(
            group_to_framecounts,
            puits,
            suffixes,
            field_area_microns2=field_area_microns2,
        )
        # 文件名示例：Well{idx}_Incucyte_F98_{dose}Gy_2024_09_24_smooth=35.txt
        out_txt = os.path.join(
            output_dir,
            f"Well{idx}_{experiment_tag}_{dose_gy}Gy_{date_str}_{smooth_tag}.txt",
        )
        save_mean_std_txt(out_txt, means, stds)
        print(f"Saved: {out_txt}")


if __name__ == "__main__":
    main()


