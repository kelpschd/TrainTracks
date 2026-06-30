from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import median_filter
from skimage.feature import blob_log
from dask import delayed, compute
import os
import numpy as np
import nd2
import imageio.v3 as iio
from PIL import Image, ImageDraw

import napari
import zarr
import pandas as pd
import numpy as np
import networkx as nx
from funtracks.import_export import import_from_geff
# from motile_tracker.motile.backend import MotileRun
from motile_tracker.data_views.views.tree_view.tree_widget import TreeWidget
from motile_tracker.data_views.views_coordinator.tracks_viewer import TracksViewer

# currently have a solution graph in GEFF and the raw image
# want to visualize solution graph tracks

if __name__ == "__main__":
    images_dir = Path("/Users/dankelpsch/Datatecnica/TTU/Data/Sphere/220725_i11w-hT-M33-I76_sg1035_d10sphere")
    images = list(sorted(images_dir.glob('*.zarr')))
    img_num = 9
    zarr_root = zarr.open(images[img_num], mode = 'r')
    arr = zarr_root['s0']
    np_arr = np.array(arr)
    print("Raw image loaded!")

    blobs_fp = "/Users/dankelpsch/Datatecnica/TTU/Data/Sphere/220725_i11w-hT-M33-I76_sg1035_d10sphere/sg100_Well5_1018_blobs.csv"
    blobs_df = pd.read_csv(blobs_fp)
    print(blobs_df)
    blobs_np = blobs_df.to_numpy().astype(np.uint16)
    print("Annotated blobs loaded!")

    # load in flow.zarr
    flow_path = Path("/Users/dankelpsch/Datatecnica/TrainTracks/flow.zarr")
    flow_root = zarr.open_group(flow_path, mode='r')
    flow_frames = flow_root['flow_frames_XY'] 
    print("flow frames loaded!")

    # load in tracks.geff
    solution_graph = import_from_geff(
        directory="/Users/dankelpsch/Datatecnica/TrainTracks/test_run_7.geff",
        node_name_map={"time": "t", "pos": ["y", "x"]})
    
    solution_graph_nx = solution_graph.graph
    print(len(solution_graph.nodes()))

    # check everything in napari
    viewer = napari.Viewer()

    widget = TreeWidget(viewer)
    viewer.window.add_dock_widget(widget, name="Lineage View", area="right")
    tracks_viewer = TracksViewer.get_instance(viewer)

    viewer.add_image(np_arr[:,0], name = "Raw image")
    viewer.add_points(blobs_df, size = 30, face_color = "transparent", border_color="red", border_width=0.1)
    viewer.add_image(np.array(flow_frames), name = "Flow")
    tracks_viewer.tracks_list.add_tracks(solution_graph, "test_data_7")
    napari.run()

"""
Self-contained: converts a funtracks SolutionTracks object (e.g. from
funtracks.import_export.import_from_geff) into the track_df shape your
save_track_gifs() expects, then calls it. save_track_gifs and its helpers
are your original code, unchanged -- only geff_tracks_to_track_df is new.

Verified against funtracks 2.0.2 / tracksdata 0.1.0rc4, by actually running
import_from_geff on a synthetic dividing track and inspecting the result.
solution_graph.graph is a tracksdata.graph.GraphView, not an nx.DiGraph --
that's why nx_graph.nodes(data=True) failed earlier with "'NodesAccessor'
object is not callable", which is a networkx-only calling convention.
funtracks auto-computes a "tracklet_id" node attribute on import: a stable id
that's constant along a lineage and increments at every division/merge --
exactly the "one continuous (frame, y, x) path per id" structure
save_track_gifs's `particle` column needs.
"""

import os
import numpy as np
import pandas as pd
import imageio.v3 as iio
from PIL import Image, ImageDraw


# ---------------------------
# geff -> track_df conversion
# ---------------------------

def geff_tracks_to_track_df(
    tracks,
    time_attr: str = "t",
    pos_attrs: tuple[str, ...] = ("pos_0", "pos_1"),
    track_attr: str = "tracklet_id",
) -> pd.DataFrame:
    """
    Convert a funtracks Tracks/SolutionTracks object into a DataFrame with
    columns [particle, frame, y, x], matching what save_track_gifs() expects.

    time_attr: with node_name_map={"time": "t", ...} at import, this ends up
        named "t" (not "time") -- the import code keeps the original GEFF
        property name for scalar attrs, confirmed empirically.
    pos_attrs: after node_attrs(unpack=True), a combined "pos" array attribute
        becomes "pos_0", "pos_1" (..."pos_2" for 3D), in the order you passed
        as node_name_map["pos"] at import time (e.g. ["y","x"] -> pos_0=y,
        pos_1=x).
    track_attr: "tracklet_id" is funtracks' auto-computed core feature for
        linear-segment track identity, also what colors/segments the napari
        Tracks layer -- using it here keeps GIFs consistent with napari.
    """
    if len(pos_attrs) not in (2, 3):
        raise ValueError(f"Expected 2 (y,x) or 3 (z,y,x) pos_attrs, got {pos_attrs}")

    attrs_df = tracks.graph.node_attrs(unpack=True).to_pandas()

    missing = [c for c in (time_attr, track_attr, *pos_attrs) if c not in attrs_df.columns]
    if missing:
        raise KeyError(
            f"Expected columns {missing} not found in node_attrs(); "
            f"available columns: {list(attrs_df.columns)}. "
            "Run tracks.graph.node_attrs(unpack=True) yourself to check names "
            "if your node_name_map/axis order differs from the default."
        )

    out = pd.DataFrame(
        {
            "particle": attrs_df[track_attr],
            "frame": attrs_df[time_attr].astype(int),
            "y": attrs_df[pos_attrs[-2]].astype(float),
            "x": attrs_df[pos_attrs[-1]].astype(float),
        }
    )
    # NOTE: if pos_attrs has 3 entries (z,y,x), z is currently dropped since
    # save_track_gifs works on (T,C,Y,X) images. Say if you need z preserved.

    return out.sort_values(["particle", "frame"]).reset_index(drop=True)


# ---------------------------
# Normalization + rendering helpers (yours, unchanged)
# ---------------------------

def normalize_crop_percentile(crop, lower=2, upper=98):
    """
    Robust per-channel percentile-based normalization.
    crop: (T, C, h, w) float/uint -> uint8 (T, C, h, w)
    Percentiles computed over ALL frames in the crop per channel.
    """
    T, C, h, w = crop.shape
    crop_norm = np.zeros((T, C, h, w), dtype=np.uint8)

    for c in range(C):
        channel_data = crop[:, c].reshape(-1).astype(np.float32)
        p_low, p_high = np.percentile(channel_data, [lower, upper])

        if (p_high - p_low) < 1e-3:
            crop_norm[:, c] = 0
        else:
            norm = (crop[:, c].astype(np.float32) - p_low) / (p_high - p_low)
            norm = np.clip(norm * 255.0, 0, 255)
            crop_norm[:, c] = norm.astype(np.uint8)

    return crop_norm

def _to_rgb(gray_u8):
    """(H,W) uint8 -> (H,W,3) uint8 grayscale RGB."""
    return np.stack([gray_u8, gray_u8, gray_u8], axis=-1)

def _apply_pseudocolor(gray_u8, rgb_color):
    """
    gray_u8: (H,W) uint8
    rgb_color: (R,G,B) in 0-255
    returns: (H,W,3) uint8
    """
    g = gray_u8.astype(np.float32) / 255.0
    color = np.array(rgb_color, dtype=np.float32) / 255.0
    out = g[..., None] * color[None, None, :]
    return (np.clip(out, 0, 1) * 255).astype(np.uint8)

def _merge_rgb(rgb_list):
    """Additive merge with clamp."""
    if len(rgb_list) == 0:
        raise ValueError("rgb_list is empty; nothing to merge.")
    acc = np.zeros_like(rgb_list[0], dtype=np.float32)
    for im in rgb_list:
        acc += im.astype(np.float32)
    return np.clip(acc, 0, 255).astype(np.uint8)

def load_marker(path, bg_threshold=10, size=None, transparent_frac_threshold=0.01):
    """
    Load a marker icon for use with _draw_overlays(marker_img=...).
    If the file already has real alpha transparency (e.g. a properly
    exported webp/png), that's used directly -- no chroma-keying, so no
    halo/fringing around the edges. Only falls back to chroma-keying a
    solid near-black background (see load_marker_with_chroma_key) if the
    image turns out to have no meaningful transparency of its own, which
    happens when a PNG was flattened against black at some point and lost
    its alpha channel.
    """
    im = Image.open(path).convert("RGBA")
    arr = np.array(im)
    alpha = arr[..., 3]
    has_real_alpha = (alpha < 255).mean() > transparent_frac_threshold

    if not has_real_alpha:
        rgb = arr[..., :3]
        new_alpha = np.where(np.all(rgb < bg_threshold, axis=-1), 0, 255).astype(np.uint8)
        arr = np.dstack([rgb, new_alpha])
        im = Image.fromarray(arr, mode="RGBA")

    if size is not None:
        im = im.resize(size, Image.LANCZOS)
    return im

def load_marker_with_chroma_key(path, bg_threshold=10, size=None):
    """
    Load a PNG marker icon and convert its solid background to transparency.
    Use this when your marker image is plain RGB with a flat black (or other
    solid color) background instead of real alpha transparency -- e.g. a PNG
    that was flattened at some point and lost its alpha channel.

    path: image path
    bg_threshold: pixels with all channels below this value become fully
        transparent. Only safe if the actual background is near-pure black
        AND the icon's own dark details are noticeably lighter than that
        (check with a quick pixel sample if unsure).
    size: optional (w, h) to resize to, e.g. (circle_r*2*upscale,)*2
    """
    im = Image.open(path).convert("RGB")
    arr = np.array(im)
    alpha = np.where(np.all(arr < bg_threshold, axis=-1), 0, 255).astype(np.uint8)
    rgba = np.dstack([arr, alpha])
    marker = Image.fromarray(rgba, mode="RGBA")
    if size is not None:
        marker = marker.resize(size, Image.LANCZOS)
    return marker

def _draw_overlays(rgb_u8, center_xy, track_xy,
                   circle_r=6, line_w=2,
                   circle_color=(255, 0, 0),      # red
                   line_color=(255, 255, 0),       # yellow
                   draw_line=True,
                   marker_img=None):
    """
    Draw a circle (or a marker image, if provided) at center, and optionally
    a polyline track.
    center_xy: (x,y) pixel coords, or None to skip the circle/marker
    track_xy: list[(x,y)] points (same coords)
    draw_line: if False, only the circle/marker is drawn (no track trail)
    marker_img: optional PIL RGBA Image (e.g. from load_marker_with_chroma_key)
        pasted centered at center_xy instead of drawing the default ellipse.
        Its own alpha channel is used as the paste mask.
    """
    if marker_img is not None:
        im = Image.fromarray(rgb_u8).convert("RGBA")

        if draw_line and len(track_xy) >= 2:
            ImageDraw.Draw(im).line(track_xy, fill=line_color, width=line_w, joint="curve")

        if center_xy is not None:
            cx, cy = center_xy
            mw, mh = marker_img.size
            paste_xy = (int(cx - mw / 2), int(cy - mh / 2))
            im.paste(marker_img, paste_xy, marker_img)

        return np.array(im.convert("RGB"), dtype=np.uint8)

    im = Image.fromarray(rgb_u8)
    dr = ImageDraw.Draw(im)

    if center_xy is not None:
        cx, cy = center_xy
        dr.ellipse((cx - circle_r, cy - circle_r, cx + circle_r, cy + circle_r),
                   outline=circle_color, width=line_w)

    if draw_line and len(track_xy) >= 2:
        dr.line(track_xy, fill=line_color, width=line_w, joint="curve")

    return np.array(im, dtype=np.uint8)

def _resize_nn(rgb_u8, scale):
    """Integer upscale via nearest-neighbor (keeps pixels crisp)."""
    if scale == 1:
        return rgb_u8
    im = Image.fromarray(rgb_u8)
    w, h = im.size
    im = im.resize((w * scale, h * scale), resample=Image.Resampling.NEAREST)
    return np.array(im, dtype=np.uint8)

def apply_gamma_u8(gray_u8, gamma=0.8):
    """
    Apply display gamma to uint8 grayscale image.
    gamma < 1  → brighten dim structures
    gamma > 1  → darken midtones
    """
    g = gray_u8.astype(np.float32) / 255.0
    g = np.power(g, gamma)
    return (np.clip(g, 0, 1) * 255).astype(np.uint8)



# ---------------------------
# Main writer (yours, unchanged)
# ---------------------------

def save_track_gifs(
    img_array,               # (T, C, Y, X)
    track_df,                # must contain: particle, frame, y, x
    image_name,
    output_dir,
    padding=40,              # <-- bbox padding around entire track
    fps=10,
    upscale=2,               # 2 is usually a nice default
    norm_lower=2,
    norm_upper=98,
    channel_colors=None,     # dict: {0:(...), 1:(...), 2:(...)}
    composite_channels=(0, 1, 2),
    circle_r=6,
    line_w=2,
    gamma=0.8,
    only_track_frames=True,  # write only frames where track exists
    preprocessed_array=None, # optional (T, Y, X) background-subtracted channel
    min_frames=4,            # skip tracks with <= 3 detections (set to 1 to disable)
):
    output_dir = str(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    if img_array.ndim != 4:
        raise ValueError(f"Expected img_array shape (T,C,Y,X), got {img_array.shape}")

    T, C, Y, X = img_array.shape

    if preprocessed_array is not None:
        if preprocessed_array.ndim != 3:
            raise ValueError(f"Expected preprocessed_array shape (T,Y,X), got {preprocessed_array.shape}")
        if preprocessed_array.shape != (T, Y, X):
            raise ValueError(f"preprocessed_array shape {preprocessed_array.shape} must match (T,Y,X)=({T},{Y},{X})")

    # Defaults requested
    if channel_colors is None:
        channel_colors = {
            0: (255, 0, 255),  # magenta
            1: (255, 0, 0),    # red
            2: (0, 255, 0),    # green
        }

    base_name = os.path.splitext(image_name)[0].replace(" ", "_")

    df = track_df.copy()
    df["frame"] = df["frame"].astype(int)
    df = df.sort_values(["particle", "frame"])

    duration = 1.0 / fps

    for pid, group in df.groupby("particle"):
        group = group.sort_values("frame")

        if len(group) < min_frames:
            continue

        # Track-level bbox in full-image coords
        y_min, y_max = int(np.floor(group["y"].min())), int(np.ceil(group["y"].max()))
        x_min, x_max = int(np.floor(group["x"].min())), int(np.ceil(group["x"].max()))

        # Apply padding (your 40 px requirement)
        y1 = max(y_min - padding, 0)
        y2 = min(y_max + padding, Y)
        x1 = max(x_min - padding, 0)
        x2 = min(x_max + padding, X)

        if y2 <= y1 or x2 <= x1:
            continue

        # Crop all frames then normalize per-channel over all timepoints
        crop = img_array[:, :, y1:y2, x1:x2]  # (T, C, h, w)
        crop_norm = normalize_crop_percentile(crop, lower=norm_lower, upper=norm_upper)
        h, w = crop_norm.shape[2], crop_norm.shape[3]

        # Preprocessed array crop + normalize (if provided)
        if preprocessed_array is not None:
            pre_crop = preprocessed_array[:, y1:y2, x1:x2]          # (T, h, w)
            pre_crop_norm = normalize_crop_percentile(
                pre_crop[:, np.newaxis, :, :], lower=norm_lower, upper=norm_upper
            )  # (T, 1, h, w)

        # Map frame -> center (in crop coords)
        centers = {
            int(r["frame"]): (float(r["x"]) - x1, float(r["y"]) - y1)
            for _, r in group.iterrows()
            if 0 <= int(r["frame"]) < T
        }

        # Progressive track line (grows over time)
        track_by_frame = {}
        running = []
        for t in sorted(centers.keys()):
            running.append(centers[t])
            track_by_frame[t] = running.copy()

        # Last frame with a known position — trail stays frozen after this
        last_track_frame = max(centers.keys()) if centers else None

        # Fill in mid-track gaps so the trail never disappears between detections
        if centers:
            last_known_pts = []
            for t in range(min(centers.keys()), last_track_frame + 1):
                if t in track_by_frame:
                    last_known_pts = track_by_frame[t]
                else:
                    track_by_frame[t] = last_known_pts

        # Which frames to write
        if only_track_frames:
            t_list = np.array(sorted(centers.keys()), dtype=int)
        else:
            t_list = np.arange(T, dtype=int)

        if len(t_list) == 0:
            continue

        # Frame buffers
        per_ch_raw = [[] for _ in range(C)]
        per_ch_ann = [[] for _ in range(C)]
        per_ch_circ = [[] for _ in range(C)]   # circle-only
        comp_raw = []
        comp_ann = []
        comp_circ = []                          # circle-only composite
        pre_raw  = []                           # preprocessed grayscale
        pre_ann  = []                           # preprocessed + circle + line
        pre_circ = []                           # preprocessed + circle only

        def scale_pt(p):
            return (p[0] * upscale, p[1] * upscale)

        for t in t_list:
            center = centers.get(int(t), None)
            if int(t) in track_by_frame:
                pts_now = track_by_frame[int(t)]
            elif last_track_frame is not None and int(t) > last_track_frame:
                pts_now = track_by_frame[last_track_frame]
            else:
                pts_now = []

            pseudo_list = []

            for c in range(C):
                gray = crop_norm[t, c]
                # gray = apply_gamma_u8(gray, gamma=gamma)

                # Per-channel grayscale
                rgb_raw = _to_rgb(gray)
                rgb_raw = _resize_nn(rgb_raw, upscale)

                rgb_ann = rgb_raw
                rgb_circ = rgb_raw
                if len(pts_now) >= 2 or center is not None:
                    scaled_center = scale_pt(center) if center is not None else None
                    scaled_pts = [scale_pt(p) for p in pts_now]
                    r_scaled = circle_r * upscale
                    w_scaled = max(1, line_w * upscale)

                    # Full annotation: circle + track line
                    rgb_ann = _draw_overlays(
                        rgb_raw.copy(),
                        center_xy=scaled_center,
                        track_xy=scaled_pts,
                        circle_r=r_scaled,
                        line_w=w_scaled,
                        circle_color=(255, 0, 0),
                        line_color=(255, 255, 0),
                        draw_line=True,
                    )
                    # Circle only: no track line
                    rgb_circ = _draw_overlays(
                        rgb_raw.copy(),
                        center_xy=scaled_center,
                        track_xy=scaled_pts,
                        circle_r=r_scaled,
                        line_w=w_scaled,
                        circle_color=(255, 0, 0),
                        draw_line=False,
                    )

                per_ch_raw[c].append(rgb_raw)
                per_ch_ann[c].append(rgb_ann)
                per_ch_circ[c].append(rgb_circ)

                # Composite uses only selected channels
                if c in composite_channels:
                    color = channel_colors.get(c, (255, 255, 255))
                    rgb_pc = _apply_pseudocolor(gray, color)
                    rgb_pc = _resize_nn(rgb_pc, upscale)
                    pseudo_list.append(rgb_pc)

            merged = _merge_rgb(pseudo_list) if len(pseudo_list) else np.zeros((h*upscale, w*upscale, 3), dtype=np.uint8)
            comp_raw.append(merged)

            merged_ann = merged
            merged_circ = merged
            if len(pts_now) >= 2 or center is not None:
                scaled_center = scale_pt(center) if center is not None else None
                scaled_pts = [scale_pt(p) for p in pts_now]
                r_scaled = circle_r * upscale
                w_scaled = max(1, line_w * upscale)

                merged_ann = _draw_overlays(
                    merged.copy(),
                    center_xy=scaled_center,
                    track_xy=scaled_pts,
                    circle_r=r_scaled,
                    line_w=w_scaled,
                    circle_color=(255, 0, 0),
                    line_color=(255, 255, 0),
                    draw_line=True,
                )
                merged_circ = _draw_overlays(
                    merged.copy(),
                    center_xy=scaled_center,
                    track_xy=scaled_pts,
                    circle_r=r_scaled,
                    line_w=w_scaled,
                    circle_color=(255, 0, 0),
                    draw_line=False,
                )
            comp_ann.append(merged_ann)
            comp_circ.append(merged_circ)

            # Preprocessed channel (background-subtracted, grayscale)
            if preprocessed_array is not None:
                gray_pre = pre_crop_norm[t, 0]
                rgb_pre_raw = _to_rgb(gray_pre)
                rgb_pre_raw = _resize_nn(rgb_pre_raw, upscale)

                rgb_pre_ann  = rgb_pre_raw
                rgb_pre_circ = rgb_pre_raw
                if len(pts_now) >= 2 or center is not None:
                    scaled_center = scale_pt(center) if center is not None else None
                    scaled_pts = [scale_pt(p) for p in pts_now]
                    r_scaled = circle_r * upscale
                    w_scaled = max(1, line_w * upscale)
                    rgb_pre_ann = _draw_overlays(
                        rgb_pre_raw.copy(),
                        center_xy=scaled_center,
                        track_xy=scaled_pts,
                        circle_r=r_scaled,
                        line_w=w_scaled,
                        circle_color=(255, 0, 0),
                        line_color=(255, 255, 0),
                        draw_line=True,
                    )
                    rgb_pre_circ = _draw_overlays(
                        rgb_pre_raw.copy(),
                        center_xy=scaled_center,
                        track_xy=scaled_pts,
                        circle_r=r_scaled,
                        line_w=w_scaled,
                        circle_color=(255, 0, 0),
                        draw_line=False,
                    )
                pre_raw.append(rgb_pre_raw)
                pre_ann.append(rgb_pre_ann)
                pre_circ.append(rgb_pre_circ)

        # Write GIFs
        for c in range(C):
            fn_raw  = os.path.join(output_dir, f"{base_name}_track{pid}_ch{c}_raw.gif")
            fn_ann  = os.path.join(output_dir, f"{base_name}_track{pid}_ch{c}_annot.gif")
            fn_circ = os.path.join(output_dir, f"{base_name}_track{pid}_ch{c}_circle.gif")
            iio.imwrite(fn_raw,  per_ch_raw[c],  duration=duration, loop=0)
            iio.imwrite(fn_ann,  per_ch_ann[c],  duration=duration, loop=0)
            iio.imwrite(fn_circ, per_ch_circ[c], duration=duration, loop=0)

        fn_comp_raw  = os.path.join(output_dir, f"{base_name}_track{pid}_composite_raw.gif")
        fn_comp_ann  = os.path.join(output_dir, f"{base_name}_track{pid}_composite_annot.gif")
        fn_comp_circ = os.path.join(output_dir, f"{base_name}_track{pid}_composite_circle.gif")
        iio.imwrite(fn_comp_raw,  comp_raw,  duration=duration, loop=0)
        iio.imwrite(fn_comp_ann,  comp_ann,  duration=duration, loop=0)
        iio.imwrite(fn_comp_circ, comp_circ, duration=duration, loop=0)

        if preprocessed_array is not None:
            fn_pre_raw  = os.path.join(output_dir, f"{base_name}_track{pid}_preprocessed_raw.gif")
            fn_pre_ann  = os.path.join(output_dir, f"{base_name}_track{pid}_preprocessed_annot.gif")
            fn_pre_circ = os.path.join(output_dir, f"{base_name}_track{pid}_preprocessed_circle.gif")
            iio.imwrite(fn_pre_raw,  pre_raw,  duration=duration, loop=0)
            iio.imwrite(fn_pre_ann,  pre_ann,  duration=duration, loop=0)
            iio.imwrite(fn_pre_circ, pre_circ, duration=duration, loop=0)

        print(f"[track {pid}] bbox: x=[{x1},{x2}] y=[{y1},{y2}] | wrote gifs to: {output_dir}")


def _build_track_frame_maps(group, T):
    """
    group: DataFrame sorted by frame for a single particle, columns frame, x, y.
    Same logic as the per-frame center/trail bookkeeping inside save_track_gifs,
    just without crop offsets (full-image pixel coords).
    Returns (centers, track_by_frame, last_track_frame).
    """
    centers = {
        int(r["frame"]): (float(r["x"]), float(r["y"]))
        for _, r in group.iterrows()
        if 0 <= int(r["frame"]) < T
    }

    track_by_frame = {}
    running = []
    for t in sorted(centers.keys()):
        running.append(centers[t])
        track_by_frame[t] = running.copy()

    last_track_frame = max(centers.keys()) if centers else None

    if centers:
        last_known_pts = []
        for t in range(min(centers.keys()), last_track_frame + 1):
            if t in track_by_frame:
                last_known_pts = track_by_frame[t]
            else:
                track_by_frame[t] = last_known_pts

    return centers, track_by_frame, last_track_frame


def save_full_frame_track_gif(
    img_array,                # (T, C, Y, X)
    track_df,                 # must contain: particle, frame, y, x
    image_name,
    output_dir,
    fps=10,
    upscale=1,                # whole-FOV frames are usually already large
    norm_lower=2,
    norm_upper=98,
    channel_colors=None,
    composite_channels=(0, 1, 2),
    circle_r=4,
    line_w=1,
    circle_color=(255, 0, 0),
    line_color=(255, 255, 0),
    draw_line=True,
    min_frames=4,              # tracks shorter than this aren't drawn at all
    preprocessed_array=None,   # optional (T, Y, X) background-subtracted channel
    marker_path=None,          # optional PNG to paste at each track's position instead of a circle
    marker_size=None,          # (w,h) in pixels for the marker; defaults to (circle_r*2*upscale,)*2
    marker_bg_threshold=10,    # passed to load_marker_with_chroma_key
    channels=None,             # e.g. [0] to only render that channel (skips composite); None = all + composite
):
    """
    Same overlay style as save_track_gifs, but one GIF per channel (+composite)
    covering the *entire* field of view across all T frames, with every
    qualifying track drawn simultaneously instead of one crop per track.
    """
    output_dir = str(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    if img_array.ndim != 4:
        raise ValueError(f"Expected img_array shape (T,C,Y,X), got {img_array.shape}")

    T, C, Y, X = img_array.shape

    if preprocessed_array is not None:
        if preprocessed_array.ndim != 3:
            raise ValueError(f"Expected preprocessed_array shape (T,Y,X), got {preprocessed_array.shape}")
        if preprocessed_array.shape != (T, Y, X):
            raise ValueError(f"preprocessed_array shape {preprocessed_array.shape} must match (T,Y,X)=({T},{Y},{X})")

    if channel_colors is None:
        channel_colors = {
            0: (255, 0, 255),  # magenta
            1: (255, 0, 0),    # red
            2: (0, 255, 0),    # green
        }

    base_name = os.path.splitext(image_name)[0].replace(" ", "_")

    df = track_df.copy()
    df["frame"] = df["frame"].astype(int)
    df = df.sort_values(["particle", "frame"])

    duration = 1.0 / fps

    # Precompute per-track frame->center/trail maps once, skipping short tracks
    track_maps = {}
    for pid, group in df.groupby("particle"):
        group = group.sort_values("frame")
        if len(group) < min_frames:
            continue
        track_maps[pid] = _build_track_frame_maps(group, T)

    # Normalize over the whole movie, no cropping
    full_norm = normalize_crop_percentile(img_array, lower=norm_lower, upper=norm_upper)

    if preprocessed_array is not None:
        pre_norm = normalize_crop_percentile(
            preprocessed_array[:, np.newaxis, :, :], lower=norm_lower, upper=norm_upper
        )  # (T, 1, Y, X)

    def scale_pt(p):
        return (p[0] * upscale, p[1] * upscale)

    render_channels = list(range(C)) if channels is None else list(channels)

    marker_img = None
    if marker_path is not None:
        size = marker_size if marker_size is not None else (circle_r * 2 * upscale,) * 2
        marker_img = load_marker(marker_path, bg_threshold=marker_bg_threshold, size=size)

    per_ch_raw = [[] for _ in range(C)]
    per_ch_ann = [[] for _ in range(C)]
    per_ch_circ = [[] for _ in range(C)]
    comp_raw, comp_ann, comp_circ = [], [], []
    pre_raw, pre_ann, pre_circ = [], [], []

    for t in range(T):
        # Gather active (center, trail) pairs for every qualifying track at frame t
        active = []
        for pid, (centers, track_by_frame, last_track_frame) in track_maps.items():
            if last_track_frame is None:
                continue
            if t in track_by_frame:
                pts_now = track_by_frame[t]
            elif t > last_track_frame:
                pts_now = track_by_frame[last_track_frame]
            else:
                continue  # before this track starts -- not drawn yet
            center = centers.get(t, None)
            if center is not None or len(pts_now) >= 2:
                active.append((center, pts_now))

        def draw_all(canvas, draw_line_flag):
            for center, pts_now in active:
                scaled_center = scale_pt(center) if center is not None else None
                scaled_pts = [scale_pt(p) for p in pts_now]
                canvas = _draw_overlays(
                    canvas,
                    center_xy=scaled_center,
                    track_xy=scaled_pts,
                    circle_r=circle_r * upscale,
                    line_w=max(1, line_w * upscale),
                    circle_color=circle_color,
                    line_color=line_color,
                    draw_line=draw_line_flag and draw_line,
                    marker_img=marker_img,
                )
            return canvas

        pseudo_list = []
        for c in render_channels:
            gray = full_norm[t, c]
            rgb_raw = _resize_nn(_to_rgb(gray), upscale)

            per_ch_raw[c].append(rgb_raw)
            per_ch_ann[c].append(draw_all(rgb_raw.copy(), True) if active else rgb_raw)
            per_ch_circ[c].append(draw_all(rgb_raw.copy(), False) if active else rgb_raw)

            if channels is None and c in composite_channels:
                color = channel_colors.get(c, (255, 255, 255))
                rgb_pc = _resize_nn(_apply_pseudocolor(gray, color), upscale)
                pseudo_list.append(rgb_pc)

        if channels is None:
            merged = _merge_rgb(pseudo_list) if pseudo_list else np.zeros((Y * upscale, X * upscale, 3), dtype=np.uint8)
            comp_raw.append(merged)
            comp_ann.append(draw_all(merged.copy(), True) if active else merged)
            comp_circ.append(draw_all(merged.copy(), False) if active else merged)

        if preprocessed_array is not None:
            gray_pre = pre_norm[t, 0]
            rgb_pre_raw = _resize_nn(_to_rgb(gray_pre), upscale)
            pre_raw.append(rgb_pre_raw)
            pre_ann.append(draw_all(rgb_pre_raw.copy(), True) if active else rgb_pre_raw)
            pre_circ.append(draw_all(rgb_pre_raw.copy(), False) if active else rgb_pre_raw)

    for c in render_channels:
        iio.imwrite(os.path.join(output_dir, f"{base_name}_fullframe_ch{c}_raw.gif"), per_ch_raw[c], duration=duration, loop=0)
        iio.imwrite(os.path.join(output_dir, f"{base_name}_fullframe_ch{c}_annot.gif"), per_ch_ann[c], duration=duration, loop=0)
        iio.imwrite(os.path.join(output_dir, f"{base_name}_fullframe_ch{c}_circle.gif"), per_ch_circ[c], duration=duration, loop=0)

    if channels is None:
        iio.imwrite(os.path.join(output_dir, f"{base_name}_fullframe_composite_raw.gif"), comp_raw, duration=duration, loop=0)
        iio.imwrite(os.path.join(output_dir, f"{base_name}_fullframe_composite_annot.gif"), comp_ann, duration=duration, loop=0)
        iio.imwrite(os.path.join(output_dir, f"{base_name}_fullframe_composite_circle.gif"), comp_circ, duration=duration, loop=0)

    if preprocessed_array is not None:
        iio.imwrite(os.path.join(output_dir, f"{base_name}_fullframe_preprocessed_raw.gif"), pre_raw, duration=duration, loop=0)
        iio.imwrite(os.path.join(output_dir, f"{base_name}_fullframe_preprocessed_annot.gif"), pre_ann, duration=duration, loop=0)
        iio.imwrite(os.path.join(output_dir, f"{base_name}_fullframe_preprocessed_circle.gif"), pre_circ, duration=duration, loop=0)

    print(f"[full frame] {len(track_maps)} tracks drawn (>= {min_frames} frames) | wrote gifs to: {output_dir}")


if __name__ == "__main__":
    # --- Example wiring, mirroring your napari script ---
    from pathlib import Path
    import zarr
    from funtracks.import_export import import_from_geff

    images_dir = Path("/Users/dankelpsch/Datatecnica/TTU/Data/Sphere/220725_i11w-hT-M33-I76_sg1035_d10sphere")
    images = list(sorted(images_dir.glob("*.zarr")))
    img_num = 9
    zarr_root = zarr.open(images[img_num], mode="r")
    np_arr = np.array(zarr_root["s0"])  # (T, C, Y, X) -- matches save_track_gifs' expected shape directly

    solution_graph = import_from_geff(
        directory="/Users/dankelpsch/Datatecnica/TrainTracks/test_run_7.geff",
        node_name_map={"time": "t", "pos": ["y", "x"]},
    )

    # sanity check before trusting the column names below -- run this once and
    # compare against the time_attr/pos_attrs/track_attr defaults:
    print(solution_graph.graph.node_attrs(unpack=True).columns)

    track_df = geff_tracks_to_track_df(solution_graph)

 # save_track_gifs(
    #     img_array=np_arr,
    #     track_df=track_df,
    #     padding=40,              
    #     fps=5,
    #     upscale=2,              
    #     norm_lower=0.1,
    #     norm_upper=99.99,
    #     only_track_frames=False,  
    #     image_name="test_run_7",
    #     output_dir="/Users/dankelpsch/Datatecnica/TrainTracks/track_gifs",
    # )

    # separate call: whole field of view, all qualifying tracks overlaid at once
    # save_full_frame_track_gif(
    #     img_array=np_arr,
    #     track_df=track_df,
    #     fps=5,
    #     circle_r=8,
    #     line_w=3,
    #     norm_lower=0.1,
    #     norm_upper=99.99,
    #     image_name="test_run_7",
    #     output_dir="/Users/dankelpsch/Datatecnica/TrainTracks/track_gifs",
    # )

    # save_full_frame_track_gif(
    #     img_array=np_arr,
    #     track_df=track_df,
    #     image_name="test_run_7",
    #     output_dir="/Users/dankelpsch/Datatecnica/TrainTracks/track_gifs/Thomas_web",
    #     norm_lower=0.1,
    #     norm_upper=99.99,
    #     line_w=3,
    #     marker_path="/Users/dankelpsch/Datatecnica/TrainTracks/ThomasTank.webp",
    #     marker_size=(100, 100),     # tune to taste relative to your cell/FOV scale
    #     composite_channels=False,
    #     channels=[0],
    # )