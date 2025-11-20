
# ================ DESCRIPTION ==============================================================================

# This file applies a LoG detector on fluorescent images (tiff/vsi) to detect cells.
# The code now takes its inputs from parameters defined in the CONFIGURATION section below
# (paths, LoG settings, blob size constraints) instead of interactive CLI arguments.
# Workflow:
#   1) Load stack paths from INPUT_PATTERN and prepare OUTPUT_HDF5.
#   2) For each image, read frames with the selected CHANNEL.
#   3) On the first frame, estimate background (BACKGROUND_WINDOW) to compute SNR.
#   4) Apply Gaussian smoothing (LOG_SIGMA), Laplacian + threshold (LOG_THRESHOLD) to detect blobs,
#      constrained by BLOB_MIN_PIXELS/BLOB_MAX_PIXELS.
#   5) Find local maxima inside blobs; save positions to an HDF5 store (one group per image/frame)
#      along with background and run metadata.
# See CONFIGURATION for parameters and the COMMANDS section for the run confirmation.

# Commands dataframes:
    
#   to open with pandas, use:
#       store=pandas.HDFStore(filename,'r')
#       p=pandas.read_hdf(store,key=num_frame)
#       store.close()
#
#   to open with h5py:
#       f=h5py.File(filename,'r')
#   to get the information saved in the attributes:
#       f.attrs.keys()
#   and then:
#       f.attrs[name_atribute]
#   to get the background:
#       f['background']['im'][:] as an array -> plt.imshow() to see it
#   to get one of the dataframes:
#       
#
#

# =============== REQUIRED PACKAGES =========================================================================================

import javabridge # To use bioformats 
import bioformats # to open the images
import sep # to determine the background
import Find_Local_Maxima as findMax # python file to determine blobs and local maxima
import h5py # to have compacted dataframes
import pandas # to use dataframes
import sys,os # to access our images and use the terminal
import glob # finds all the pathnames matching a specified pattern according to the rules used by the Unix shell
from tqdm import tqdm # to get a processing bar in the terminal
from scipy.spatial import KDTree

import numpy as np

# =============== CONFIGURATION ====================================================================================

# Input/output parameters
INPUT_PATTERN = "./results 151025/stack rouge 1024/*.tif"  # Glob pattern for the stack images to process
OUTPUT_HDF5 = "./results 151025/output_file_1024_0.0375.hdf5"  # Destination file for results
CONFIRM_BEFORE_RUN = True  # Keep the confirmation prompt enabled

# Detection parameters
CHANNEL = 0  # Use 1 when processing bright-field + fluorescence stacks
BACKGROUND_WINDOW = 256  # Background window size in pixels; large enough to avoid local effects
LOG_SIGMA = 3.5  # Gaussian smoothing applied before Laplacian
LOG_THRESHOLD = 0.0375  # LoG threshold used to select blobs
BLOB_MIN_PIXELS = 30  # Minimum number of pixels to accept a blob
BLOB_MAX_PIXELS = None  # Optional maximum blob size; keep None to disable
MIN_DUPLICATE_DISTANCE = 12  # Radius for optional duplicate filtering (currently unused)
FILENAME_ID_INDEX = 0  # Which filename token to use as the HDF5 group identifier


# =============== COMMANDS =========================================================================================

# Collect images matching the input pattern and confirm output target
input_files = glob.glob(INPUT_PATTERN)
output_file = OUTPUT_HDF5
input_dir = os.path.dirname(INPUT_PATTERN)

print(input_files)
print("input dir={}".format(input_dir))
for image_path in input_files:
    base=os.path.basename(image_path)
    print(base)
#    assert base.split("_")[-2]=="Image", "wrong vsi={}".format(base)
print("output file will be ={}".format(output_file))
if CONFIRM_BEFORE_RUN:
    ans=input("is this OK? (y/n) ")
    if ans != "y":
        sys.exit()


#Then we start java to access our images
javabridge.start_vm(class_path=bioformats.JARS)
# we create the output file to write in it
f=h5py.File(output_file,'w')
 
# we specify the metadata in our output file
f.attrs['channel']=CHANNEL
f.attrs['sigma']=LOG_SIGMA
f.attrs['seuil']=LOG_THRESHOLD
f.attrs['ccmin']=BLOB_MIN_PIXELS
if BLOB_MAX_PIXELS is not None:
    f.attrs['ccmax']=BLOB_MAX_PIXELS
# for pnadas df
store=pandas.HDFStore(output_file,'a')

# Then we loop on the images 
for ifile,fin in enumerate(input_files):
    # determine the image number from file name
    b=os.path.basename(fin)
#    imnum=int(b.split(".")[0].split("Image_")[1])
#    imnum = int(b.split(".")[0].split("_")[-1].replace("d", "").replace("h", "").replace("m", ""))
#    imnum = int(b.split("_")[2])
#    imnum = f"{b.split('_')[0]}_{b.split('_')[1]}_{b.split('_')[2]}_{b.split('_')[3]}"
    imnum = b.split("_")[FILENAME_ID_INDEX]


    # open the image with bioformats
    ome=bioformats.OMEXML(bioformats.get_omexml_metadata(fin))
    print(ome.image().AcquisitionDate)
    # Retrieve properties of the image
    nt=ome.image().Pixels.SizeT
    Nx=ome.image().Pixels.SizeX
    Ny=ome.image().Pixels.SizeY
    nchan=ome.image().Pixels.channel_count
    # read the image 
    reader=bioformats.ImageReader(fin)


    # HDF output (how information will be organized in the out file)
    print("><"*100)
    print(r"{} ({}/{})".format(fin,ifile,len(input_files)))
    print(r" ->{} frames. image size=({}x{}) with {} channels: using channel={}".format(nt,Nx,Ny,nchan,CHANNEL))
    
    # we loop on the frames
    for ip in tqdm(range(nt),desc="Processing"):
        yy=reader.read(c=CHANNEL,t=ip)
        if yy.ndim == 3 and yy.shape[2] == 3:  # examiner si c'est un RGB image
            # separate the 3 channels
            yy = yy[:, :, 0]   # R channel
            data=yy.reshape(Ny,Nx)
            data = np.ascontiguousarray(data)
        
        else:
            data=yy.reshape(Ny,Nx)
        # compute background if we are at the first frame and save it in the output file
        if ip==0:
            bkg=sep.Background(data,bw=BACKGROUND_WINDOW,bh=BACKGROUND_WINDOW)
            key="Image{}/background".format(imnum)
            g = f.create_group(key)
            ds=g.create_dataset("image",data=bkg.back(),compression='gzip')
            g.attrs['rms']=bkg.globalrms
            g.attrs['bw']=BACKGROUND_WINDOW

        # work on SNR
        snr=(data-bkg.back())/bkg.globalrms
        # compute LoG and threshold to obtain blobs
        blobs=findMax.getBlobs(snr,s=LOG_THRESHOLD,ccmin=BLOB_MIN_PIXELS,sigma=LOG_SIGMA,ccmax=BLOB_MAX_PIXELS)
        # get local maxima for each blob: output is a dataframe
        pos=findMax.findMax(blobs,data.shape)
        
        
        #pos = findMax.filter_coordinates(pos, MIN_DUPLICATE_DISTANCE)
        #pos = pandas.DataFrame(pos, columns=['x', 'y'])
        # write df (careful, here floats)
        key="Image{}/frame{}".format(imnum,ip)
        pos.to_hdf(store,key=key)

# we close the files created
store.close()
f.close()
# and java
javabridge.kill_vm()
