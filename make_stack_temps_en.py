#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script traverses a directory structure containing time-series TIFF images from multiple wells and positions (e.g., from a multi-well plate experiment). 
It searches for images from a specified camera (subdir_name) across all available dates and times, then stacks the images for each well and position in chronological order into a single multi-page TIFF file. 
The output is organized in a new directory, with one stacked file per well-position combination. 

Main steps:
1. Traverse the directory tree by month, date, and time, looking for the specified camera subdirectory.
2. For each well and position, collect all matching image files along with their timestamps.
3. Sort the images by time and stack them into a single TIFF file for each well-position.
4. Save the stacked files in a dedicated output directory.
"""

import os
from datetime import datetime
import tifffile

# Configurable parameters
subdir_name = '1031'  # We process all wells (puits) and all positions of one plate at a time. 
                     # The plate is identified by the Incucyte camera number (subdir_name). 
                     # For example, for the first plate of the last experiment, 
                     # we can find it under /Mathilde 020725 hypoxie/hypoxie/EssenFiles/ScanData/2507/02/1522

channel = 'C2'  # Channel name, e.g., 'C2' for red fluorescence, 'Ph' for bright field, etc.

#base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Mathilde hypoxie 151025/Hypoxie 151025/EssenFiles/ScanData/")
base_path = "/Volumes/Mathilde 3/Mathilde 141125 Hypoxie +- IR/141125 Hypoxie +- IR/EssenFiles/ScanData"
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"stack bf {subdir_name}")
puits = ['A1', 'A2', 'A3', 'B1', 'B2', 'B3', 'C1', 'C2', 'C3']  # List of well names
positions = range(1, 10)  # Positions 1-9 in each well

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Initialize a dictionary to record found image files for each (well, position)
file_records = {(p, pos): [] for p in puits for pos in positions}

# Traverse all month folders in the base path
for month_dir in os.listdir(base_path):
    month_path = os.path.join(base_path, month_dir)
    # Check if the current item is a valid month folder (e.g. "2507" for July 2025)
    # Make sure it's a directory (not a file)
    # The folder name must be 4 characters long
    # The folder name must contain only digits
    if not (os.path.isdir(month_path) and len(month_dir) == 4 and month_dir.isdigit()):
        continue
    
    # Parse year and month from folder name
    try:
        year = 2000 + int(month_dir[:2])
        month = int(month_dir[2:])
    except ValueError:
        print(f"Skipping invalid month folder: {month_dir}")
        continue

    print(f"Processing data for {year}-{month:02d} ...")

    # Traverse all date folders within the month
    for date_dir in os.listdir(month_path):
        date_path = os.path.join(month_path, date_dir)
        # Only process folders named as numbers (e.g., '02' for the 2nd day)
        if not (os.path.isdir(date_path) and date_dir.isdigit()):
            continue
        
        day = int(date_dir)

        # Traverse all time folders within the date
        for time_dir in os.listdir(date_path):
            time_path = os.path.join(date_path, time_dir)
            # Only process folders named as 4-digit numbers (e.g., '1522' for 15:22)
            if not (os.path.isdir(time_path) and len(time_dir) == 4 and time_dir.isdigit()):
                continue

            # Look for the camera subdirectory (e.g., '989') inside the time folder
            img_subdir = os.path.join(time_path, subdir_name)
            if not os.path.isdir(img_subdir):
                continue

            # Parse hour and minute from the time folder name
            try:
                hour = int(time_dir[:2])
                minute = int(time_dir[2:])
                dt = datetime(year, month, day, hour, minute)
            except ValueError:
                continue

            # For each well and position, check if the expected TIFF file exists
            for puit in puits:
                for pos in positions:
                    filename = f"{puit}-{pos}-{channel}.tif"
                    file_path = os.path.join(img_subdir, filename)
                    if os.path.isfile(file_path):
                        # Record the timestamp and file path
                        file_records[(puit, pos)].append((dt, file_path))

# For each well and position, stack the images in chronological order and save as a multi-page TIFF
for (puit, pos), records in file_records.items():
    if not records:
        print(f"No files found for {puit}-{pos}")
        continue

    # Sort the image records by their timestamp (each record is a (datetime, filepath) tuple) 
    sorted_records = sorted(records, key=lambda x: x[0])
    # Extract only the file paths from the sorted records to create a time-ordered list of images   
    sorted_files = [path for dt, path in sorted_records]

    output_path = os.path.join(output_dir, f"{puit}-{pos}-{channel}_stack.tif")

    try:
        # Write all images into a single TIFF stack
        with tifffile.TiffWriter(output_path, bigtiff=True) as tif:
            for file_path in sorted_files:
                img = tifffile.imread(file_path)
                tif.write(img)
        print(f"Created {output_path} ({len(sorted_files)} images)")
    except Exception as e:
        print(f"Error processing {puit}-{pos}: {str(e)}")

print("Processing complete!") 