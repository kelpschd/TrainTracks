# Utilizing some functions directly from Jen!
# mhat.opticalflow.visualization import generate_flow_frames

import cv2
import zarr
import numpy as np
from tqdm import tqdm
from pathlib import Path

def create_flow_color_wheel(width, height):
    # make a square legend with padding
    size = min(width, height)
    legend_size = int(size * 0.15) # Legend size as 15% of frame dimension
    min_size = 10
    legend_size = max(legend_size, min_size)
    legend = np.zeros((legend_size, legend_size, 3), dtype=np.uint8) + 20  # dark grey background
    
    # calculate center and radius
    center_x, center_y = legend_size // 2, legend_size // 2
    max_radius = (legend_size // 2) - 2 
    
    # create the color wheel
    for y in range(legend_size):
        for x in range(legend_size):
            # calculate distance from center
            dx, dy = x - center_x, y - center_y
            distance = np.sqrt(dx**2 + dy**2)
            
            # skip pixels outside the circle
            if distance > max_radius:
                continue
            
            # calculate angle using same method as main flow visualization
            # This matches: ang = cv2.cartToPolar(...) and hsv[..., 0] = ang * 180 / np.pi / 2
            angle_rad = np.arctan2(dy, dx)  # angle in radians (note: dy, dx for correct orientation)
            hue = angle_rad * 180 / np.pi / 2  # convert to HSV hue (0-180)
            
            # ensure hue is in valid range
            hue = hue % 180
            
            # normalize distance to 0-1 range for brightness
            normalized_distance = distance / max_radius
            
            # set HSV values based on angle and distance
            saturation = 255
            
            # makes the center dimmer, edges brighter
            value = int(normalized_distance * 255)
            
            # convert HSV to BGR for this pixel
            color = cv2.cvtColor(np.uint8([[[hue, saturation, value]]]), cv2.COLOR_HSV2BGR)[0][0]
            legend[y, x] = color
    
    # add a thin border around the wheel
    cv2.circle(legend, (center_x, center_y), max_radius, (200, 200, 200), 1)

    # add magnitude labels of 0 and 15 on the circle 
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    font_thickness = 1

    text_size = cv2.getTextSize("0", font, font_scale, font_thickness)[0]
    text_x = center_x - text_size[0] // 2
    text_y = center_y + text_size[1] // 2
    cv2.putText(legend, "0", (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)

    text_size = cv2.getTextSize("10", font, font_scale, font_thickness)[0]
    edge_x = center_x + max_radius - text_size[0] - 2  
    edge_y = center_y
    cv2.putText(legend, "10", (edge_x, edge_y), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)

    return legend


def generate_flow_frame(flow, scale_factor=1):
    depth, height, width, _ = flow.shape
    hsv = np.zeros((depth, height, width, 3), dtype=np.uint8)  # initialize hsv image
    hsv[..., 1] = 255  # set saturation to maximum

    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])  # calculate magnitude and angle
    hsv[..., 0] = ang * 180 / np.pi / 2  # set hue based on angle
    hsv[..., 2] = np.clip(mag * 255 * scale_factor, 0, 255).astype(np.uint8) # dim = 2550, medium = 25500, bright = 255000

    rgb = np.zeros_like(hsv, dtype=np.uint8)  # initialize rgb image
    for zslice in range(depth):
        rgb[zslice, ...] = cv2.cvtColor(hsv[zslice], cv2.COLOR_HSV2BGR)  # convert hsv to bgr

    # Create a copy of the flow visualization
    final_frame = rgb.copy()

    return final_frame


def generate_flow_frames(flow_zarr, scale_factor=0.1, color_wheel=False):
    flow_raw = flow_zarr['flow_raw']

    T, Y, X, D = flow_raw.shape

    # Calculate scale factor
    # percentile_75 = np.percentile(np.linalg.norm(flow_raw, axis=-1), 75)
    # print(f"75th percentile flow magnitude: {percentile_75}")
    # scale_factor = 255.0 / percentile_75
    print(f"Using scale factor for flow visualization: {scale_factor}")

    flow_zarr.create_dataset('flow_frames_XY', shape=(T, Y, X, 2), chunks=(1, Y, X, 2), dtype=np.uint8)
    flow_frames = np.zeros((T, Y, X, 2), dtype=np.uint8)
    for i in tqdm(range(T), desc="Generating flow visualization frames"):
        flow_frame = generate_flow_frame(flow_raw[i, ...], scale_factor=scale_factor)
        flow_frames[i, ...] = flow_frame

        # Add color wheel legend
    if color_wheel:
        legend = create_flow_color_wheel(X, Y)
        legend_h, legend_w = legend.shape[:2]

        # position in bottom right
        pos_x = X - legend_w
        pos_y = Y - legend_h

        flow_frames[..., pos_y:pos_y+legend_h, pos_x:pos_x+legend_w, :] = legend

    flow_zarr['flow_frames_XY'][:] = flow_frames


# import raw img for reference
images_dir = Path("/mnt/efs/dl_jrc/student_data/S-DK/Sphere/220725_i11w-hT-M33-I76_sg1035_d10sphere")
images = list(sorted(images_dir.glob('*.zarr')))
print(f"Found {len(images)} zarr files")

zarr_root = zarr.open(images[0], mode = 'r')
arr = zarr_root['s0']
np_arr = np.array(arr)
print(np_arr.shape)

import 