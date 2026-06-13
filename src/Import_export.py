import nd2
import zarr
from pathlib import Path
import ome_zarr
from ome_zarr.io import parse_url
from ome_zarr.writer import write_image

# set this up in a loop
images_dir = Path("/mnt/efs/dl_jrc/student_data/S-DK/Sphere/220725_i11w-hT-M33-I76_sg1035_d10sphere")
images = list(sorted(images_dir.glob('*.nd2')))
print(f"Found {len(images)} ND2 files")

nd2_path = images[0]
zarr_path = nd2_path.with_suffix('.zarr').name
print(zarr_path)

store = zarr.storage.LocalStore(zarr_path) 
root = zarr.group(store, overwrite=True)

with nd2.ND2File(nd2_path) as f:
    arr = f.asarray()
    print(f"Shape: {arr.shape}")
    px = f.metadata.channels[0].volume.axesCalibration
    print(px)
    
    write_image(
        image=arr,
        group=root,
        axes="tcyx",
        n_lvls=3,  # stop at 512x512
        coordinate_transformations=[
            [{"type": "scale", "scale": [1.0, 1.0, px[0] * (2**i), px[1] * (2**i)]}]
            for i in range(5)
        ],
    )
    print("Zarr'd")


# set up a definition to take a nd2 file (using nd2 package)
# nd2 directly to zarr - inbetween step with np?
