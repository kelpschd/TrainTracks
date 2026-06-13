import nd2
import zarr
from pathlib import Path
import ome_zarr
import numpy as np
from ome_zarr.io import parse_url
from ome_zarr.writer import write_image

# set this up in a loop and to take in inputs from bash
images_dir = Path("/mnt/efs/dl_jrc/student_data/S-DK/Sphere/220725_i11w-hT-M33-I76_sg1035_d10sphere")
images = list(sorted(images_dir.glob('*.nd2')))
print(f"Found {len(images)} ND2 files")

nd2_path = images[0]
zarr_path = nd2_path.with_suffix('.zarr').name

store = zarr.storage.LocalStore(zarr_path) 
root = zarr.group(store, overwrite=True)

with nd2.ND2File(nd2_path) as f:
    arr = f.asarray()
    print(zarr_path)
    print(f"Array shape: {arr.shape}")
    print(f"Array dtype: {arr.dtype}")
    print(f"Array min: {arr[0,0].min()} and max: {arr[0,0].max()}")

    # report micron per pixel on each axis
    px = f.metadata.channels[0].volume.axesCalibration
    print(f"The image is {px[0]} micrometers per pixel on Y")
    print(f"The image is {px[1]} micrometers per pixel on X")
    
    write_image(
        image=arr,
        group=root,
        axes="tcyx",
        scale_factors={}, 
        coordinate_transformations=[
            [{"type": "scale", "scale": [1.0, 1.0, px[0], px[1]]}]
        ],
    )
    print("Zarr'd")

z_array = zarr.open('/home/S-DK/TrainTracks/sg100_Well1_1001.zarr', mode='r')
np_array = np.asarray(z_array['s0'])

print(f"Zarr shape: {np_array.shape}")
print(f"Zarr dtype: {np_array.shape}")
print(f"Array min: {np_array[0,0].min()} and max: {np_array[0,0].max()}")
