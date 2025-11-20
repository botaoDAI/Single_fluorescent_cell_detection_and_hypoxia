# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 18:12:36 2024

@author: Dai_botao
"""

import pandas as pd
import matplotlib.pyplot as plt

def plot_cell_counts_over_time(image_name, hdf5_path, num_frames, time_interval=3):
    # Define stacks for each dose level
    dose_levels = ['0Gy', '10Gy', '15Gy']
    cell_counts = {dose: [] for dose in dose_levels}

    for dose in dose_levels:
        for i in range(num_frames):
            num_frame = f'Image{dose}_{image_name}/frame{i}'
            try:
                p = pd.read_hdf(hdf5_path, key=num_frame)  # Read the coordinates DataFrame
                cell_count = p.shape[0]  # Number of cells (rows in DataFrame)
            except KeyError:
                cell_count = 0  # If no data is found, assume 0 cells
            cell_counts[dose].append(cell_count)

    # Create time points for the x-axis
    time_points = [i * time_interval for i in range(num_frames)]

    # Plot the data
    plt.figure(figsize=(10, 6))
    for dose in dose_levels:
        plt.plot(time_points, cell_counts[dose], marker='o', label=f'{dose}')

    plt.xlabel('Time (hours)')
    plt.ylabel('Cell Count (n)')
    plt.title(f'Cell Count Over Time for {image_name}')
    plt.legend()
    plt.grid(True)
    plt.show()

# Example usage
image_name = 'RFP_B3_8'  # Define your image name here
hdf5_path = './output_file.hdf5'  # Path to your HDF5 file
num_frames = 72  # Define the total number of frames in your stack
plot_cell_counts_over_time(image_name, hdf5_path, num_frames)
