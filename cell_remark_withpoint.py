# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 20:09:18 2024

@author: Dai_botao
"""

import tifffile as tiff
import pandas as pd
import numpy as np
from PIL import Image

# Define stack name
stack_name = 'B1-1-C2'  # Define your stack name here

# Define paths based on stack name
stack_path = f'./results 020425/stack rouge 957/{stack_name}_stack.tif'
hdf5_path = './results 020425/output_file_957_0.0375.hdf5'  # Updated HDF5 file path
output_path = f'./annotated_{stack_name}_stack.tif'  # Output path for the annotated stack

# Load the stack
with tiff.TiffFile(stack_path) as tif:
    print("Number of pages:", len(tif.pages))
    # Read all pages into a list and stack them
    pages = []
    for page in tif.pages:
        pages.append(page.asarray())
    stack = np.stack(pages, axis=0)
print("Final stack shape:", stack.shape)
print("Stack dtype:", stack.dtype)
# Determine the bit depth
bit_depth = stack.dtype

# List to store annotated frames
annotated_frames = []

# Number of frames in the stack
num_frames = stack.shape[0]
print(stack.shape)
# Loop through each frame in the stack
for i in range(num_frames):
    # Get the key for the current frame
    num_frame = f'Image{stack_name}/frame{i}'
    
    # Check if the key exists in HDF5 file
    try:
        p = pd.read_hdf(hdf5_path, key=num_frame)  # p is a (n, 2) dataframe
    except KeyError:
        print(f'No coordinates found for frame {i}, skipping annotation.')
        p = pd.DataFrame(columns=[0, 1])  # Create empty dataframe if no data
    
    # Get the current frame
    frame = stack[i].copy()
    
    # Convert to RGB image
    rgb_frame = np.stack([frame, frame, frame], axis=-1)
    
    # Get maximum pixel value in the frame
    max_pixel_value = frame.max()
    
    # Ensure coordinates are integers within image bounds
    coords = p.round().astype(int)
    #coords = coords[(coords[0] >= 0) & (coords[0] < frame.shape[1]) & (coords[1] >= 0) & (coords[1] < frame.shape[0])]
    
    # Define a color for the points (e.g., red [255, 0, 0] for 8-bit images)
    color_value = [max_pixel_value, 0, 0]  # You can change [R, G, B] values to other colors
    color_value = [300, 0, 0]  # You can change [R, G, B] values to other colors
    
    # Draw larger colored points (3x3 square) at each coordinate
    for idx, row in coords.iterrows():
        x, y = row[0], row[1]
        # Define the square boundaries
        x_min = max(x - 1, 0)
        x_max = min(x + 1, frame.shape[1] - 1)
        y_min = max(y - 1, 0)
        y_max = min(y + 1, frame.shape[0] - 1)
        # Set the pixel values in the square to the chosen color
        rgb_frame[y_min:y_max+1, x_min:x_max+1] = color_value
    
    # Append the annotated frame to the list
    annotated_frames.append(rgb_frame)

# Save the new stack
tiff.imwrite(output_path, np.array(annotated_frames), photometric='rgb')

print('Annotated stack with colored points has been saved successfully.')


