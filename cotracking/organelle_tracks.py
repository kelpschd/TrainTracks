"""
Organelle tracking, lifted from organelle_tracker_testing_20251219.ipynb.

The function and class bodies below are extracted from the notebook cells
programmatically and are UNCHANGED: generate_lysosome_mask,
generate_mitochondria_mask, process_frame, mask_to_centroids,
filter_dense_blobs, KalmanTrack, kalman_track_blobs,
filter_tracks_by_net_displacement.

Only `track_organelles` at the bottom is new; it strings the notebook's cells
together in the same order with the same parameters:

    masks (per frame, dask) -> centroids -> filter_dense_blobs(blob_filter=5)
    -> kalman_track_blobs (lyso: 25 px / 4 missed, mito: 25 px / 3 missed)
    -> filter_tracks_by_net_displacement (defaults: 50 px, 10 frames)
"""
import numpy as np
import pandas as pd
import dask
from scipy.ndimage import gaussian_laplace
from scipy import ndimage as ndi
from scipy.stats import norm
from scipy.spatial import cKDTree
from skimage import filters, morphology
from skimage.measure import label, regionprops_table
from filterpy.kalman import KalmanFilter
import trackpy as tp


def generate_lysosome_mask(
    frame,
    intensity_scaling_param=[3, 19],
    min_lysosome_area=15,
    blur_sigma=1,
    log_params=[(5, 0.09), (2.5, 0.07), (1, 0.01)],
    vesselness_sigma=[1],
    vesselness_cutoff=0.15):

    # Normalize to normal distribution
    m, s = norm.fit(frame.flatten())
    stretch_min = max(m - intensity_scaling_param[0] * s, frame.min())
    stretch_max = min(m + intensity_scaling_param[1] * s, frame.max())
    frame = np.clip(frame, stretch_min, stretch_max)
    image_norm = (frame - stretch_min) / (stretch_max - stretch_min)

    # Blur frame
    blurred = filters.gaussian(image_norm, sigma=blur_sigma)

    # LoG blob detection
    log_masks = []
    for sigma, cutoff in log_params:
        log_img = -1 * (sigma**2) * gaussian_laplace(blurred, sigma=sigma)
        log_masks.append(log_img > cutoff)
    log_mask = np.logical_or.reduce(log_masks)

    # Frangi vesselness
    vesselness = filters.frangi(blurred, sigmas=vesselness_sigma, black_ridges=False) > vesselness_cutoff

    # Combine and postprocess
    combined = np.logical_or(log_mask, vesselness)
    filled = ndi.binary_fill_holes(combined)
    cleaned = morphology.remove_small_objects(filled, min_size=min_lysosome_area)

    return cleaned.astype(np.uint8)

def generate_mitochondria_mask(
    frame,
    intensity_scaling_param=(3.5, 15),
    min_mitochondria_area=10,
    blur_sigma=1,
    log_params=((5, 0.09), (2.5, 0.07), (1, 0.01)),
    vesselness_sigma=(1.5,),
    vesselness_cutoff=0.16
):
    # Normalize using Gaussian fit
    m, s = norm.fit(frame.flatten())
    stretch_min = max(m - intensity_scaling_param[0] * s, frame.min())
    stretch_max = min(m + intensity_scaling_param[1] * s, frame.max())
    clipped = np.clip(frame, stretch_min, stretch_max)
    image_norm = (clipped - stretch_min) / (stretch_max - stretch_min + 1e-8)  # prevent div-by-zero

    # Blur frame
    blurred = filters.gaussian(image_norm, sigma=blur_sigma)

    # Multi-scale LoG filtering
    log_masks = []
    for sigma, cutoff in log_params:
        log_response = -1 * (sigma ** 2) * gaussian_laplace(blurred, sigma=sigma)
        log_masks.append(log_response > cutoff)

    log_mask = np.logical_or.reduce(log_masks)

    # Vesselness (Frangi)
    vesselness = filters.frangi(blurred, sigmas=vesselness_sigma, black_ridges=False)
    vessel_mask = vesselness > vesselness_cutoff

    # Combine masks
    combined = np.logical_or(log_mask, vessel_mask)

    # Remove small objects
    mitochondria_mask = morphology.remove_small_objects(combined, min_size=min_mitochondria_area)

    return mitochondria_mask

def process_frame(frame_4d, lysosome_channel, mitochondria_channel):
    lyso_frame = frame_4d[lysosome_channel].astype(np.float32)
    mito_frame = frame_4d[mitochondria_channel].astype(np.float32)

    lyso_mask = generate_lysosome_mask(lyso_frame)
    mito_mask = generate_mitochondria_mask(mito_frame)

    return lyso_mask, mito_mask

def mask_to_centroids(mask, frame_idx, organelle_type):
    """
    mask: 2D binary array
    frame_idx: int frame number
    organelle_type: "lysosome" or "mitochondria"
    """
    mask_bool = mask.astype(bool)
    labels = label(mask_bool)

    if labels.max() == 0:
        # No objects in this frame
        return pd.DataFrame(columns=["frame", "x", "y", "area", "label", "organelle"])

    props = regionprops_table(
        labels,
        properties=("label", "area", "centroid")
    )
    df = pd.DataFrame(props)

    # regionprops gives centroid-0 = row (y), centroid-1 = col (x)
    df.rename(columns={"centroid-0": "y", "centroid-1": "x"}, inplace=True)
    df["frame"] = frame_idx
    df["organelle"] = organelle_type

    return df[["frame", "x", "y", "area", "label", "organelle"]]

def filter_dense_blobs(blob_df, bin_size=5, blob_filter=10):
    """
    Filters out blobs that appear too frequently within a very small local neighborhood.

    Parameters:
        blob_df (pd.DataFrame): DataFrame with 'x' and 'y' columns of blob coordinates.
        bin_size (int): Pixel binning size for rounding positions.
        blob_filter (int): Maximum allowed count of blobs in a bin before filtering them out.

    Returns:
        pd.DataFrame: Filtered DataFrame with blobs below the density threshold.
    """
    rounded = blob_df.copy()
    rounded['x_rounded'] = (rounded['x'] // bin_size * bin_size).astype(int)
    rounded['y_rounded'] = (rounded['y'] // bin_size * bin_size).astype(int)

    counts = rounded.groupby(['x_rounded', 'y_rounded']).size().reset_index(name='count')
    merged = rounded.merge(counts, on=['x_rounded', 'y_rounded'])

    filtered_blobs = merged[merged['count'] < blob_filter].reset_index(drop=True)

    print(f"Puncta filtering complete: {len(filtered_blobs)} puncta remain")
    return filtered_blobs

class KalmanTrack:
    def __init__(self, initial_detection, particle_id, frame, max_missed=5):
        x, y = initial_detection
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        dt = 1
        self.kf.F = np.array([[1, 0, dt, 0],
                              [0, 1, 0, dt],
                              [0, 0, 1, 0],
                              [0, 0, 0, 1]])
        self.kf.H = np.array([[1, 0, 0, 0],
                              [0, 1, 0, 0]])
        self.kf.R *= 5
        self.kf.P *= 1000
        self.kf.Q = np.eye(4) * 0.1
        self.kf.x = np.array([x, y, 0, 0])

        self.id = particle_id
        self.history = [{'frame': frame, 'x': x, 'y': y, 'particle': self.id}]
        self.missed = 0
        self.max_missed = max_missed

    def predict(self):
        self.kf.predict()
        return self.kf.x[:2]

    def update(self, detection, frame):
        self.kf.update(np.array(detection))
        x, y = self.kf.x[:2]
        self.history.append({'frame': frame, 'x': x, 'y': y, 'particle': self.id})
        self.missed = 0

    def mark_missed(self):
        self.missed += 1

    def is_dead(self):
        return self.missed > self.max_missed

def kalman_track_blobs(all_blobs, max_distance=15, max_missed=5):
    active_tracks = []
    finished_tracks = []
    next_id = 0

    for frame, detections in all_blobs.groupby('frame'):
        detected_positions = detections[['x', 'y']].to_numpy()

        # Predict positions for all current tracks
        predictions = [track.predict() for track in active_tracks]
        assigned = set()

        if len(predictions) > 0 and len(detected_positions) > 0:
            tree = cKDTree(detected_positions)
            for i, pred in enumerate(predictions):
                dist, idx = tree.query(pred, distance_upper_bound=max_distance)
                if idx != len(detected_positions) and idx not in assigned:
                    active_tracks[i].update(detected_positions[idx], frame)
                    assigned.add(idx)
                else:
                    active_tracks[i].mark_missed()
        else:
            for track in active_tracks:
                track.mark_missed()

        # Start new tracks for unassigned detections
        for i, pos in enumerate(detected_positions):
            if i not in assigned:
                new_track = KalmanTrack(pos, next_id, frame, max_missed)
                active_tracks.append(new_track)
                next_id += 1

        # Remove dead tracks
        still_alive = []
        for track in active_tracks:
            if track.is_dead():
                finished_tracks.append(track)
            else:
                still_alive.append(track)
        active_tracks = still_alive

    # Add remaining active tracks to finished
    finished_tracks.extend(active_tracks)

    # Combine history
    all_tracks = [row for track in finished_tracks for row in track.history]        
    all_tracks = pd.DataFrame(all_tracks)
    print(f"Particle tracking complete: There are {len(all_tracks.groupby('particle'))} unique particle tracks")
    return all_tracks

def filter_tracks_by_net_displacement(tracks_df, displacement_threshold=50, min_frames=10):
    """
    Filter tracks by net displacement after removing short tracks.

    Parameters:
        tracks_df (pd.DataFrame): DataFrame with ['particle', 'x', 'y'].
        displacement_threshold (float): Minimum net displacement required to keep a track.
        min_frames (int): Minimum number of frames a track must appear in.

    Returns:
        pd.DataFrame: Filtered tracking DataFrame.
    """
    # Remove short tracks
    filtered_tracks = tp.filter_stubs(tracks_df, threshold=min_frames).reset_index(drop=True)
    print(f"Track stub removal complete: {len(filtered_tracks.groupby('particle'))} unique particle tracks")

    # Compute net displacement
    displacements = []
    for pid, group in filtered_tracks.groupby('particle'):
        start = group.iloc[0][['x', 'y']].values
        end = group.iloc[-1][['x', 'y']].values
        net_disp = np.linalg.norm(end - start)
        displacements.append({'particle': pid, 'net_displacement': net_disp})

    displacement_df = pd.DataFrame(displacements)

    # Filter by displacement
    valid_particles = displacement_df.loc[displacement_df['net_displacement'] > displacement_threshold, 'particle']
    print(f"Track filtering complete: {len(valid_particles)} unique particle tracks")

    return filtered_tracks[filtered_tracks['particle'].isin(valid_particles)].reset_index(drop=True)

# ---------------------------------------------------------------------------
# NEW: glue only. Same order and parameters as the notebook.
# ---------------------------------------------------------------------------
LYSO_LINK = dict(max_distance=25, max_missed=4)
MITO_LINK = dict(max_distance=25, max_missed=3)
BLOB_FILTER = 5


def _empty_tracks():
    return pd.DataFrame(columns=["frame", "x", "y", "particle"])


def track_organelles(img_array, lysosome_channel=1, mitochondria_channel=2):
    """img_array: (T, C, Y, X). Returns (tracks_lys, tracks_mito, lys_blobs, mito_blobs).

    Track DataFrames have columns frame, x, y, particle (as in the notebook).
    """
    T = img_array.shape[0]
    delayed_results = [
        dask.delayed(process_frame)(img_array[t], lysosome_channel, mitochondria_channel)
        for t in range(T)
    ]
    results = dask.compute(*delayed_results)
    lyso_masks, mito_masks = zip(*results)
    lysosome_mask_stack = np.stack(lyso_masks, axis=0)
    mitochondria_mask_stack = np.stack(mito_masks, axis=0)

    lys_dfs, mito_dfs = [], []
    for t in range(T):
        lys_dfs.append(mask_to_centroids(lysosome_mask_stack[t], frame_idx=t, organelle_type="lysosome"))
        mito_dfs.append(mask_to_centroids(mitochondria_mask_stack[t], frame_idx=t, organelle_type="mitochondria"))
    lys_blobs = filter_dense_blobs(pd.concat(lys_dfs, ignore_index=True), blob_filter=BLOB_FILTER)
    mito_blobs = filter_dense_blobs(pd.concat(mito_dfs, ignore_index=True), blob_filter=BLOB_FILTER)

    out = []
    for blobs, link in ((lys_blobs, LYSO_LINK), (mito_blobs, MITO_LINK)):
        if blobs.empty:
            out.append(_empty_tracks())
            continue
        tr = kalman_track_blobs(blobs, **link)
        try:
            tr = filter_tracks_by_net_displacement(tr)
        except Exception:          # e.g. nothing survives the stub filter
            tr = _empty_tracks()
        out.append(tr)
    return out[0], out[1], lys_blobs, mito_blobs
