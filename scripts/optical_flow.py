# Calculate flows using:
# calcOpticalFlowFarneback
# calcOpticalFlowPyrLK

# mayyyyybe try OpenPIV later:
# https://openpiv.readthedocs.io/en/latest/

import zarr
import numpy as np
import napari
import cv2
from tqdm import tqdm
from pathlib import Path
from skimage.exposure import equalize_adapthist

images_dir = Path("/mnt/efs/dl_jrc/student_data/S-DK/Sphere/220725_i11w-hT-M33-I76_sg1035_d10sphere")
images = list(sorted(images_dir.glob('*.zarr')))
print(f"Found {len(images)} zarr files")

img_num = 9

print(images[img_num].name)
zarr_root = zarr.open(images[img_num], mode = 'r')
arr = zarr_root['s0']
np_arr = np.array(arr)
print(np_arr.shape)

print("Removing unused channels...")
np_arr = np_arr[:, 0]
print(np_arr.shape)

print("Generating empty output zarr")
# output_zarr = zarr.zeros(np_arr.shape, dtype=np_arr.dtype, chunks=(1, 2048, 2048))

# ^ this would work find but focuses you to hold the entire zarr in RAM and just defeats the purpose of zarr (kinda)
T, Y, X = np_arr.shape
output_group = zarr.open_group('flow.zarr', mode='w')
output_zarr = output_group.create_array(
    'flow_raw',
    shape=(T, Y, X, 2),
    dtype=np.float32,
    chunks=(1, Y, X, 2)
)
print(f"The output_zarr shape is: {output_zarr.shape}")
print(f"The output_zarr type is: {type(output_zarr)}")

# Normalize img using min and max values from the img
# was i suppose to do this for each frame? I dont think so
print("Normalizing frames...")
min_int_val = np_arr.min()
max_int_val = np_arr.max()
arr_norm = ((np_arr - min_int_val) / (max_int_val - min_int_val) * 255).astype(np.uint8)

# Get first frame and enhance its contrast
#         contrast: *enhance_contrast_AHE* - write this one in source utils? 
print("Computing flow...")
prev_frame = arr_norm[0] # check in napari

# viewer = napari.Viewer()
# viewer.add_image(prev_frame, name='Raw')

# Enhance contrast
# Jen had a rationale for this kernel size (and clip_limit) before but does not 
# really remember so double check that the output image looks okay
# Jen said this looked good for my data - the orginal was 8x8 and we can change back to that if needed
kernel_size = [16,16] 
prev_frame_contrast = equalize_adapthist(prev_frame, kernel_size=kernel_size, clip_limit=0.01)
prev_frame_contrast = (prev_frame_contrast * 255).astype(np.uint8) #print me to a png as 'enhanced'
print("The first frame has enhanced contrast")
#viewer.add_image(contrast_test, name='Enhanced 16x16 0.01')
#napari.run()

# Set up for previous flow
prev_flow = None

# Open loop
#     get next frame, enhance its contrast, and start flow against previous frame 
#         flow: *calcOpticalFlowFarneback* OR *calcOpticalFlowPyrLK*

# open loop to iterate through each frame of the time series
for i in tqdm(range(1, arr_norm.shape[0]), desc = "Computing optical flow"):
    curr_frame = arr_norm[i]

    # normalize current_frame
    kernel_size = [16,16] 
    curr_frame_contrast = equalize_adapthist(curr_frame, kernel_size=kernel_size, clip_limit=0.01)
    curr_frame_contrast = (curr_frame_contrast * 255).astype(np.uint8)

    # Calculate flow
    # These are the settings that we set up prevously but it would be adventageous 
    # to take the values from a config file
    flow = cv2.calcOpticalFlowFarneback(
        prev = prev_frame_contrast,
        next = curr_frame_contrast,
        flow = None, 
        pyr_scale = 0.6,
        levels=5,
        winsize=15,
        iterations=8,
        poly_n=3,
        poly_sigma=0.7,
        flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN
        )

    # Goal is to also smooth the flow as we calculate it
    # Calculated using 30% of the flow from the prevous frame
    if prev_flow is not None: 
        temporal_smoothing_sigma = 0.7 
        flow = temporal_smoothing_sigma * flow + (1 - temporal_smoothing_sigma) * prev_flow

    prev_frame_contrast = curr_frame_contrast
    prev_flow = flow

    output_zarr[i-1] = flow.astype(np.float32)

print(output_zarr)