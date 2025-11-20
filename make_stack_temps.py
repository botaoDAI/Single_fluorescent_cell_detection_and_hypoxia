#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 17:18:03 2025

@author: dai
"""

import os
from datetime import datetime
import tifffile

# 配置参数
subdir_name = '971'  # 可配置的子目录名
base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Mathilde incucyte_proton_juin2025/proton juin25/EssenFiles/ScanData/")
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"stack rouge {subdir_name}")
puits = ['A1', 'A2', 'A3', 'B1', 'B2', 'B3']
positions = range(1, 10)  # 1-9

os.makedirs(output_dir, exist_ok=True)

file_records = {(p, pos): [] for p in puits for pos in positions}

# 遍历月份文件夹
for month_dir in os.listdir(base_path):
    month_path = os.path.join(base_path, month_dir)
    if not (os.path.isdir(month_path) and len(month_dir) == 4 and month_dir.isdigit()):
        continue
    
    # 解析年份和月份
    try:
        year = 2000 + int(month_dir[:2])
        month = int(month_dir[2:])
    except ValueError:
        print(f"跳过无效的月份文件夹: {month_dir}")
        continue

    print(f"处理 {year}年{month}月 的数据...")

    # 遍历日期文件夹
    for date_dir in os.listdir(month_path):
        date_path = os.path.join(month_path, date_dir)
        if not (os.path.isdir(date_path) and date_dir.isdigit()):
            continue
        
        day = int(date_dir)

        for time_dir in os.listdir(date_path):
            time_path = os.path.join(date_path, time_dir)
            if not (os.path.isdir(time_path) and len(time_dir) == 4 and time_dir.isdigit()):
                continue

            img_subdir = os.path.join(time_path, subdir_name)
            if not os.path.isdir(img_subdir):
                continue

            try:
                hour = int(time_dir[:2])
                minute = int(time_dir[2:])
                dt = datetime(year, month, day, hour, minute)
            except ValueError:
                continue

            for puit in puits:
                for pos in positions:
                    filename = f"{puit}-{pos}-Ph.tif"
                    file_path = os.path.join(img_subdir, filename)
                    if os.path.isfile(file_path):
                        file_records[(puit, pos)].append((dt, file_path))

# 生成堆栈文件
for (puit, pos), records in file_records.items():
    if not records:
        print(f"未找到 {puit}-{pos} 的文件")
        continue
    # Sort the image records by their timestamp (each record is a (datetime, filepath) tuple)    
    sorted_records = sorted(records, key=lambda x: x[0])
    # Extract only the file paths from the sorted records to create a time-ordered list of images
    sorted_files = [path for dt, path in sorted_records]

    output_path = os.path.join(output_dir, f"{puit}-{pos}-C2_stack.tif")

    try:
        with tifffile.TiffWriter(output_path, bigtiff=True) as tif:
            for file_path in sorted_files:
                img = tifffile.imread(file_path)
                tif.write(img)
        print(f"已创建 {output_path} ({len(sorted_files)} 张图像)")
    except Exception as e:
        print(f"处理 {puit}-{pos} 时出错: {str(e)}")

print("处理完成！")