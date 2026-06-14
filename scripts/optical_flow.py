# Calculate flows using:
# calcOpticalFlowFarneback
# calcOpticalFlowPyrLK

# mayyyyybe try OpenPIV later:
# https://openpiv.readthedocs.io/en/latest/

import zarr
import numpy as np
import napari
from pathlib import Path
from skimage.exposure import equalize_adapthist

images_dir = Path("/mnt/efs/dl_jrc/student_data/S-DK/Sphere/220725_i11w-hT-M33-I76_sg1035_d10sphere")
images = list(sorted(images_dir.glob('*.zarr')))
print(f"Found {len(images)} zarr files")

zarr_root = zarr.open(images[0], mode = 'r')
arr = zarr_root['s0']
np_arr = np.array(arr)
print(np_arr.shape)

print("Removing unused channels...")
np_arr = np_arr[:, 0]
print(np_arr.shape)

# Normalize img using min and max values from the img
# was i suppose to do this for each frame? I dont think so
print("Normalizing frames...")
min_int_val = np_arr.min()
max_int_val = np_arr.max()
arr_norm = ((np_arr - min_int_val) / (max_int_val - min_int_val) * 255).astype(np.uint8)

# Get first frame and enhance its contrast
#         contrast: *enhance_contrast_AHE* - write this one in source utils? 
print("Computing flow...")
prev_frame = arr_norm[0] #print me to a png as 'raw'

viewer = napari.Viewer()
viewer.add_image(prev_frame, name='Raw')

# Enhance contrast
# Jen had a rationale for this kernel size (and clip_limit) before but does not 
# really remember so double check that the output image looks okay
# Jen said this looked good for my data - the orginal was 8x8 and we can change back to that if needed
kernel_size = [16,16] 
contrast_test = equalize_adapthist(prev_frame, kernel_size=kernel_size, clip_limit=0.01)
contrast_test = (contrast_test * 255).astype(np.uint8) #print me to a png as 'enhanced'
print("The first frame has enhanced contrast")
viewer.add_image(contrast_test, name='Enhanced 16x16 0.01')




napari.run()
# Open loop
#     get next frame, enhance its contrast, and start flow against previous frame 
#         flow: *calcOpticalFlowFarneback* OR *calcOpticalFlowPyrLK*


# write this to a png just to verity that it doesn't look super weird


# add empty frame to end of flow time series
# output zarr
# write zarr

