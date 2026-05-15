"""GPU-oriented segmented Viterbi kymograph tracking (collaborator implementation).

This is an alternative to the wavelet-residual + clustering pipeline in
``nsm.track``. For side-by-side comparisons on the **same** data, use a shared
cropped HDF5 (for example from ``nsm-crop``) and either :func:`load_h5_file` with
``trim_trailing_rows=0`` or :func:`nsm.kymograph_io.load_kymograph_binned` with
identical ``binning_x``, ``binned_time_ms``, and time window.

Requires CuPy (CUDA) and Numba; install Numba via ``pip install nsm[collaborator]``
and install a CuPy wheel that matches your CUDA runtime.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numba
import numpy as np
try:
    import cupy as cp
except ImportError as _e:
    raise ImportError(
        "The nsm.collaborator module requires CuPy (CUDA). "
        "Install a CuPy wheel that matches your CUDA version: https://docs.cupy.dev/en/stable/install.html"
    ) from _e
from cupyx.scipy.ndimage import uniform_filter1d as cupy_uniform_filter1d
from cupyx.scipy.ndimage import gaussian_filter
from cupyx.scipy.special import erf as cp_erf
from cupyx.scipy.special import erfinv as cp_erfinv

from nsm.kymograph_io import DEFAULT_KYMOGRAPH_DATASET, load_kymograph_binned

# ------------------------------------------------------------------------------
# remove background by subtracting moving average in two directions

def remove_background(kymo, average_samples_x, average_samples_t):
    kymo = kymo / cupy_uniform_filter1d(kymo, size=average_samples_x, axis=1)
    kymo = kymo / cupy_uniform_filter1d(kymo, size=average_samples_t, axis=0)
    return - kymo + 1



# -----------------------------------------------------------------------------
# Functions for Viterbi tracking in segments


def calculate_probability_map(kymo_background_removed, gauss_blur_sigma, prior_particle_probability=0.5):
    """Calculate a probability map from the background-removed kymograph.
    Parameters:
    kymo_background_removed: 2D array of background-removed kymograph
    gauss_blur_sigma: sigma for Gaussian blurring to smooth the probability map
    prior_particle_probability: prior probability of a pixel being part of a track (between 0 and 1)
    Returns:
    pixel_track_prob_map: 2D array of probabilities that each pixel is part of a track
    """
    # z score the kymograph. This gives pixels at a high noise position a lower probability of being a track
    noise_det_length = np.max([kymo_background_removed.shape[0], 1000])
    sigma_n = cp.std(kymo_background_removed[:noise_det_length-1, :], axis=0)
    pixel_track_prob_map = kymo_background_removed / sigma_n
    # map to [0, 1] using gaussian cdf with prior probability adjustment
    # The prior_particle_probability sets the baseline probability (e.g., 0.1 for 10% prior)
    # We calculate the z-score shift needed to achieve this prior at z=0
    z_shift = -cp_erfinv(2 * prior_particle_probability - 1)
    pixel_track_prob_map = 0.5 * (1 + cp_erf(pixel_track_prob_map / cp.sqrt(2) - z_shift))
    pixel_track_prob_map = gaussian_filter(pixel_track_prob_map, sigma=(gauss_blur_sigma, gauss_blur_sigma))
    return pixel_track_prob_map


# Numba-accelerated Viterbi implementation
@numba.njit
def _viterbi_algorithm_core(emission_prob, transition_prob, dia_width):
    """Core Viterbi algorithm with Numba acceleration."""
    num_frames, num_positions = emission_prob.shape
    viterbi = np.zeros((num_frames, num_positions))
    backpointer = np.zeros((num_frames, num_positions), dtype=np.int32)
    
    # Initialization
    viterbi[0, :] = emission_prob[0, :]
    
    # Forward pass with diagonal width optimization
    for t in range(1, num_frames):
        for j in range(num_positions):
            max_prob = -1
            max_state = -1
            for i in range(max(0, j - dia_width // 2), min(num_positions, j + dia_width // 2 + 1)):
                prob = viterbi[t-1, i] * transition_prob[i, j] * emission_prob[t, j]
                if prob > max_prob:
                    max_prob = prob
                    max_state = i
            viterbi[t, j] = max_prob
            backpointer[t, j] = max_state
    
    # Backward pass to find the best path
    best_path = np.zeros(num_frames, dtype=np.int32)
    best_path[-1] = np.argmax(viterbi[-1, :])
    for t in range(num_frames - 2, -1, -1):
        best_path[t] = backpointer[t + 1, best_path[t + 1]]
    
    return best_path


@numba.njit
def _process_multiple_segments(emission_prob_segments, transition_prob, dia_width):
    """
    Process multiple kymograph segments using Numba.
    
    Parameters:
    - emission_prob_segments: 3D array of shape (n_segments, num_frames, num_positions)
    - transition_prob: 2D transition probability matrix
    - dia_width: diagonal width for sparse matrix optimization
    
    Returns:
    - paths: 2D array of shape (n_segments, num_frames) with best paths for each segment
    """
    n_segments = emission_prob_segments.shape[0]
    num_frames = emission_prob_segments.shape[1]
    paths = np.zeros((n_segments, num_frames), dtype=np.int32)
    
    for seg in range(n_segments):
        paths[seg] = _viterbi_algorithm_core(emission_prob_segments[seg], transition_prob, dia_width)
    
    return paths


def process_segments_batch(emission_prob_segments, sigma_transition=0.5, velocity=None):
    """
    Process multiple segments with Viterbi algorithm.
    
    Parameters:
    - emission_prob_segments: list or 3D array of emission probability maps
    - sigma_transition: transition model parameter (diffusion)
    - velocity: optional flow velocity shift
    
    Returns:
    - paths: list of best paths for each segment
    - probs: list of geometric mean probabilities
    """
    # Convert to numpy array
    if isinstance(emission_prob_segments, list):
        emission_prob_segments = np.array(emission_prob_segments, dtype=np.float64)
    if hasattr(emission_prob_segments, 'get'):  # CuPy array
        emission_prob_segments = emission_prob_segments.get()
    emission_prob_segments = np.asarray(emission_prob_segments, dtype=np.float64)
    
    n_segments, num_frames, num_positions = emission_prob_segments.shape
    
    # Calculate transition probability matrix once for all segments
    threshold = 0.001
    dia_width = int(np.sum(np.exp(-np.arange(0, 100)**2 / (2 * sigma_transition**2)) > threshold) * 2)
    
    positions = np.arange(num_positions)
    distance_matrix = np.abs(positions[:, None] - positions[None, :])
    if velocity is not None:
        distance_matrix = np.abs(distance_matrix - velocity)
    transition_prob = np.exp(-(distance_matrix**2) / (2 * sigma_transition**2))
    
    # Process all segments
    paths = _process_multiple_segments(emission_prob_segments, transition_prob, dia_width)
    
    # Calculate geometric mean probabilities
    probs = []
    for i in range(n_segments):
        path_emission_probs = emission_prob_segments[i, np.arange(num_frames), paths[i]]
        probs.append(np.prod(path_emission_probs) ** (1 / num_frames))
    
    return list(paths), probs


def segmented_multi_track_viterbi(kymo, segment_length=100, overlap_length=0, sigma_transition=0.5, velocity=None, num_passes=3, max_frames=None):
    """Run Viterbi on segments of the kymograph and stitch results together.
    
    Parameters:
    -----------
    kymo : array
        Kymograph data (time x position)
    segment_length : int
        Length of each segment in frames
    overlap_length : int
        Number of frames to overlap between consecutive segments
    sigma_transition : float
        Transition model parameter for Viterbi
    velocity : float, optional
        Estimated flow velocity
    num_passes : int, optional
        Number of passes for recursive Viterbi to find multiple tracks in each segment
    max_frames : int, optional
        Maximum number of frames to process. If None or 0, process all frames.
    Returns:
    --------
    all_tracks : list of lists
        Tracks for each segment
    all_probabilities : list of lists
        Probabilities for each track
    """

    num_frames, num_positions = kymo.shape
    
    # Limit frames if max_frames is specified
    if max_frames is not None and max_frames > 0:
        num_frames = min(num_frames, max_frames)
        print(f"Limiting track finding to first {num_frames} frames (of {kymo.shape[0]} total)")
    
    all_tracks = []
    all_probabilities = []
    
    # Calculate step size based on overlap
    step_size = segment_length - overlap_length
    if step_size <= 0:
        raise ValueError(f"overlap_length ({overlap_length}) must be less than segment_length ({segment_length})")
    
    # Collect all segments first for parallel processing
    # Only include full-length segments (dismiss incomplete last segment)
    kymo_segments = []
    for start in range(0, num_frames, step_size):
        end = start + segment_length
        
        # Skip if segment extends beyond available frames
        if end > num_frames:
            break
            
        kymo_segment = kymo[start:end, :]
        kymo_segments.append(kymo_segment)
    
    # Process each pass across all segments
    # Convert to NumPy for processing
    if hasattr(kymo, 'get'):  # CuPy array
        kymo_segments = [seg.get() if hasattr(seg, 'get') else seg for seg in kymo_segments]
    
    # Initialize tracking for multi-pass processing
    all_tracks = [[] for _ in range(len(kymo_segments))]
    all_probabilities = [[] for _ in range(len(kymo_segments))]
    
    # Process each pass
    for pass_num in range(num_passes):
        # All segments now have same shape, convert to 3D array
        segments_array = np.array(kymo_segments, dtype=np.float64)
        
        # Process all segments for this pass
        tracks, probs = process_segments_batch(segments_array, sigma_transition, velocity)
        
        # Store results
        for i, (track, prob) in enumerate(zip(tracks, probs)):
            all_tracks[i].append(track)
            all_probabilities[i].append(prob)
        
        # Mask detected tracks for next pass
        if pass_num < num_passes - 1:  # Don't mask on last pass
            mask_width = 35
            for i, track in enumerate(tracks):
                mask = np.zeros_like(kymo_segments[i], dtype=bool)
                for t, pos in enumerate(track):
                    mask[t, max(0, pos-mask_width//2):min(kymo_segments[i].shape[1], pos+mask_width//2+1)] = True
                kymo_segments[i][mask] = 0  # Suppress detected track

    
    # use elbow method global across all segments to determine how many tracks to keep in each segment
    all_log_probabilities_flat = [prob for segment_probs in all_probabilities for prob in segment_probs]
    all_log_probabilities_flat = np.log(np.array(all_log_probabilities_flat))
    # sort probabilities from best to worst
    sorted_indices = np.argsort(all_log_probabilities_flat)[::-1]  # Sort in descending order
    sorted_log_probabilities = all_log_probabilities_flat[sorted_indices]
    # Find the threshold
    method = 'median x 1.5'
    if method == 'elbow':
        diffs = np.diff(sorted_log_probabilities)
        elbow_index = np.argmin(diffs) + 1  # argmin finds most negative
        threshold_log_prob = sorted_log_probabilities[elbow_index]
    elif method == 'median x 1.5':
        threshold_log_prob = 1.5 * sorted_log_probabilities[len(sorted_log_probabilities)//2]
    print(f"threshold_log_prob: {threshold_log_prob:.4f}, corresponding to probability {np.exp(threshold_log_prob):.4e}")
    # Filter tracks in each segment based on the threshold
    filtered_tracks = []
    filtered_probabilities = []
    for segment_tracks, segment_probs in zip(all_tracks, all_probabilities):
        segment_log_probs = np.log(np.array(segment_probs))
        valid_indices = [i for i, log_prob in enumerate(segment_log_probs) if log_prob >= threshold_log_prob]
        filtered_tracks.append([segment_tracks[i] for i in valid_indices])
        filtered_probabilities.append([segment_probs[i] for i in valid_indices])

    return filtered_tracks, filtered_probabilities, all_log_probabilities_flat


# -----------------------------------------------------------------------------
# Functions for linking segments

@numba.njit
def _score_overlap(track1, track2, overlap_length):
    """Calculate 25th percentile of positional gaps in overlapping region."""
    gap = np.abs(track1[-overlap_length:] - track2[:overlap_length])
    sorted_gap = np.sort(gap)
    idx = int(0.25 * len(sorted_gap))
    return sorted_gap[idx]


@numba.njit
def _build_edge_matrix(tracks_current, tracks_next, overlap_length, max_gap):
    """
    Build edge weight matrix between consecutive segments.
    
    Returns matrix where edge_matrix[i, j] is the overlap score between
    track i in current segment and track j in next segment.
    Uses -1 for edges that exceed max_gap threshold.
    """
    n_current = len(tracks_current)
    n_next = len(tracks_next)
    edge_matrix = np.full((n_current, n_next), -1.0, dtype=np.float64)
    
    for i in range(n_current):
        for j in range(n_next):
            score = _score_overlap(tracks_current[i], tracks_next[j], overlap_length)
            if score <= max_gap:
                edge_matrix[i, j] = score
    
    return edge_matrix


@numba.njit
def _find_best_path(edge_matrices, nodes_per_segment, min_length):
    """
    Find best path across segments using dynamic programming (Viterbi-like).
    
    Parameters:
    - edge_matrices: list of 2D arrays where edge_matrices[s][i,j] is the cost
                     from node i in segment s to node j in segment s+1 (-1 if no edge)
    - nodes_per_segment: array with number of nodes in each segment
    - min_length: minimum number of segments required for a valid path
    
    Returns:
    - path_segments: segment indices along the path
    - path_nodes: node indices along the path
    - (length, total_cost): path statistics
    """
    n_segments = len(nodes_per_segment)
    max_nodes = np.max(nodes_per_segment)
    
    # Dynamic programming tables
    best_length = np.zeros((n_segments, max_nodes), dtype=np.int32)
    best_weight = np.zeros((n_segments, max_nodes), dtype=np.float64)
    backpointer = np.full((n_segments, max_nodes), -1, dtype=np.int32)
    
    # Initialize: each node can be a starting point
    for s in range(n_segments):
        for i in range(nodes_per_segment[s]):
            best_length[s, i] = 1
            best_weight[s, i] = 0.0
    
    # Forward pass: extend paths segment by segment
    for s in range(n_segments - 1):
        edge_matrix = edge_matrices[s]
        for i in range(nodes_per_segment[s]):
            cur_len = best_length[s, i]
            cur_w = best_weight[s, i]
            
            for j in range(nodes_per_segment[s + 1]):
                if edge_matrix[i, j] >= 0:  # Valid edge exists
                    new_len = cur_len + 1
                    new_w = cur_w + edge_matrix[i, j]
                    
                    # Update if this path is better (longer, or same length but lower cost)
                    if (new_len > best_length[s + 1, j] or 
                        (new_len == best_length[s + 1, j] and new_w < best_weight[s + 1, j])):
                        best_length[s + 1, j] = new_len
                        best_weight[s + 1, j] = new_w
                        backpointer[s + 1, j] = i
    
    # Find best endpoint across all segments
    best_seg = -1
    best_node = -1
    best_len = 0
    best_w = np.inf
    
    for s in range(n_segments):
        for i in range(nodes_per_segment[s]):
            length = best_length[s, i]
            weight = best_weight[s, i]
            if length >= min_length and (length > best_len or (length == best_len and weight < best_w)):
                best_len = length
                best_w = weight
                best_seg = s
                best_node = i
    
    if best_seg == -1:
        # No valid path found - return length 0 arrays
        path_segments = np.zeros(1, dtype=np.int32)
        path_nodes = np.zeros(1, dtype=np.int32)
        return path_segments, path_nodes, (0, 0.0)
    
    # Backtrack to construct the path
    path_segments = np.zeros(best_len, dtype=np.int32)
    path_nodes = np.zeros(best_len, dtype=np.int32)
    
    current_seg = best_seg
    current_node = best_node
    
    for i in range(best_len - 1, -1, -1):
        path_segments[i] = current_seg
        path_nodes[i] = current_node
        
        if i > 0:
            current_node = backpointer[current_seg, current_node]
            current_seg = current_seg - 1
    
    return path_segments, path_nodes, (best_len, best_w)


def score_overlap(track1, prob1, track2, prob2, overlap_length, sigma_gap):
    # Calculate positional gap in the overlapping region
    gap = np.abs(track1[-overlap_length:] - track2[:overlap_length])
    avg_gap = np.mean(gap)
    
    return avg_gap

def score_overlap_partial(track1, prob1, track2, prob2, overlap_length, sigma_gap):
    # Calculate positional gap in the overlapping region
    gap = np.abs(track1[-overlap_length:] - track2[:overlap_length])
    
    return np.percentile(gap, 25)  # 25th percentile also gives good overlap if only parts of the tracks overlap well


def build_segment_graph(segmented_tracks, segmented_probabilities, overlap_length, sigma_gap, max_average_gap):
    """
    Build graph connecting tracks across segments based on overlap quality.
    
    Parameters:
    - segmented_tracks: list of lists of tracks for each segment
    - segmented_probabilities: list of lists of probabilities for each track
    - overlap_length: number of overlapping frames between segments
    - sigma_gap: unused (kept for compatibility)
    - max_average_gap: maximum gap allowed for linking tracks
    
    Returns:
    - edge_matrices: list of edge weight matrices between consecutive segments
    - segments: list of lists of TrackNode objects
    """

    class TrackNode:
        def __init__(self, segment_idx, track_idx, track, prob):
            self.segment_idx = segment_idx
            self.track_idx = track_idx
            self.track = track
            self.prob = prob
            self.id = f"S{segment_idx}_N{track_idx}"
            
        def __repr__(self):
            return self.id
    
    # Create TrackNode objects for each track
    segments = [
        [TrackNode(s, n, segmented_tracks[s][n], segmented_probabilities[s][n]) 
         for n in range(len(segmented_tracks[s]))] 
        for s in range(len(segmented_tracks))
    ]

    # Build edge matrices between consecutive segments
    edge_matrices = []
    for s in range(len(segmented_tracks) - 1):
        tracks_current = [np.asarray(segmented_tracks[s][i], dtype=np.float64) 
                         for i in range(len(segmented_tracks[s]))]
        tracks_next = [np.asarray(segmented_tracks[s + 1][i], dtype=np.float64) 
                      for i in range(len(segmented_tracks[s + 1]))]
        
        # Ensure all tracks are the same length and convert to proper 2D array
        if len(tracks_current) > 0 and len(tracks_next) > 0:
            # Stack into 2D arrays (n_tracks x track_length)
            tracks_current_array = np.stack(tracks_current, axis=0).astype(np.float64)
            tracks_next_array = np.stack(tracks_next, axis=0).astype(np.float64)
            
            edge_matrix = _build_edge_matrix(tracks_current_array, tracks_next_array, 
                                             overlap_length, max_average_gap)
        else:
            # No tracks in one of the segments - create empty matrix with proper shape
            n_current = len(tracks_current)
            n_next = len(tracks_next)
            edge_matrix = np.full((n_current, n_next), -1.0, dtype=np.float64)
        
        edge_matrices.append(edge_matrix)

    return edge_matrices, segments


def find_best_path_across_segments(edge_matrices, segments, used_set, min_length):
    """
    Find the longest unused path connecting tracks across segments.
    
    Parameters:
    - edge_matrices: list of edge weight matrices from build_segment_graph
    - segments: list of lists of TrackNode objects
    - used_set: set of already used TrackNode objects
    - min_length: minimum number of segments required in a valid path
    
    Returns:
    - path: list of TrackNode objects forming the path, or None if no valid path exists
    """
    if len(segments) == 0:
        return None
    
    # Mask out already used nodes in edge matrices
    modified_edges = []
    for s in range(len(edge_matrices)):
        edge_matrix = edge_matrices[s].copy()
        
        # Block edges from used nodes
        for i, node in enumerate(segments[s]):
            if node in used_set:
                edge_matrix[i, :] = -1.0
        
        # Block edges to used nodes
        for j, node in enumerate(segments[s + 1]):
            if node in used_set:
                edge_matrix[:, j] = -1.0
        
        modified_edges.append(edge_matrix)
    
    # Find best path using dynamic programming
    nodes_per_segment = np.array([len(seg) for seg in segments], dtype=np.int32)
    path_segments, path_nodes, (length, weight) = _find_best_path(
        modified_edges, nodes_per_segment, min_length
    )
    
    if length == 0:
        return None
    
    # Convert indices back to TrackNode objects
    path = [segments[seg_idx][node_idx] 
            for seg_idx, node_idx in zip(path_segments, path_nodes)]
    
    return path


def link_segmented_tracks_with_viterbi(segmented_tracks, segmented_probabilities, segment_length, overlap_length, sigma_gap, max_average_gap, min_length=3, prob_map=None):
    """
    Repeatedly run Viterbi to find non-overlapping particle tracks across segments.
    
    Parameters:
    -----------
    segmented_tracks : list of lists
        Tracks for each segment
    segmented_probabilities : list of lists
        Probabilities for each track in each segment
    segment_length : int
        Length of each segment in frames
    overlap_length : int
        Number of overlapping frames between consecutive segments
    sigma_gap : float
        Parameter for scoring gaps between tracks
    max_average_gap : float
        Maximum average gap allowed for linking tracks
    min_length : int, optional
        Minimum number of segments in a valid path (default: 3)
    prob_map : np.ndarray, optional
        Probability map (time x position) for extracting probabilities along tracks
    
    Returns:
    --------
    tracks : list of ParticleTrack
        List of ParticleTrack objects, longest first
    """
    edge_matrices, segments = build_segment_graph(segmented_tracks, segmented_probabilities, overlap_length, sigma_gap, max_average_gap)
    used = set()
    paths = []
    iteration = 0
    while True:
        iteration += 1
        path = find_best_path_across_segments(edge_matrices, segments, used, min_length)
        if path is None:
            break
        paths.append(path)
        used.update(path)
        if iteration % 50 == 0:
            print(f" Iteration {iteration}: {len(paths)} paths found so far, "
                  f"{len(used):,} nodes used")
    
    # Convert paths to ParticleTrack objects
    tracks = []
    for path in paths:
        tracks.append(ParticleTrack(path, segment_length, overlap_length, prob_map))
    
    return tracks

def stitch_path_with_overlap(path, segment_length, overlap_length, trim_low_prob_segments=True):
    """
    Stitch together track segments from a path, handling overlapping regions.
    In overlapping regions, automatically selects the track whose end matches better
    with the beginning of the next track, creating smoother transitions.
    
    Parameters:
    - path: list of TrackNode objects
    - segment_length: length of each segment
    - overlap_length: number of overlapping frames between consecutive segments
    
    Returns:
    - stitched_track: numpy array of stitched positions
    - start_time: starting time index of the stitched track
    - avg_prob: average probability across all segments in the path
    """
    if len(path) == 0:
        return None, None, None
    
    if len(path) == 1:
        # Only one segment, no stitching needed
        return path[0].track, path[0].segment_idx * (segment_length - overlap_length), path[0].prob
    
    # Start with the first track
    stitched_track = path[0].track.copy()
    
    # Iteratively add subsequent segments, choosing best match in overlapping regions
    for i in range(1, len(path)):
        current_node = path[i]
        prev_node = path[i-1]
        
        # in the overlapping region, there are two options for stitching:
        # 1) use the first track's overlap (keep stitched_track as is)
        # 2) use the second track's overlap (discard first track's overlap)
        
        fist_track_overlap_last_node = stitched_track[-1]
        second_track_overlap_middle_node = current_node.track[len(current_node.track)//2]  # middle of the overlap region in the second track
        following_track_first_node = path[i+1].track[0] if i + 1 < len(path) else None

        gap_use_first = np.abs(fist_track_overlap_last_node - following_track_first_node) if following_track_first_node is not None else 0
        gap_use_second = np.abs(second_track_overlap_middle_node - following_track_first_node) if following_track_first_node is not None else 0
        
        if gap_use_first <= gap_use_second:
            # Use first track's overlap (keep stitched_track as is)
            stitched_track = np.concatenate([
                stitched_track,
                current_node.track[overlap_length:]
            ])
        else:
            # Use second track's overlap (discard first track's overlap)
            stitched_track = np.concatenate([
                stitched_track[:-overlap_length],
                current_node.track
            ])

    
    # Calculate start time and average probability
    start_time = path[0].segment_idx * (segment_length - overlap_length)
    avg_prob = np.mean([node.prob for node in path])
    
    return stitched_track, start_time, avg_prob


def load_envue_mat_file(filename, binning_x=1, binned_time_ms=2.857):
    import scipy.io as sio

    from nsm.kymograph_io import bin_kymograph_spatiotemporal_sum

    file = sio.loadmat(filename, simplify_cells=True)
    kymo = np.asarray(file["Im"], dtype=np.float64)
    raw_frame_rate = float(np.asarray(file["FrameRate"]) / np.asarray(file["AccumulateCount"]))
    binning_t = max(1, int(round(binned_time_ms * raw_frame_rate / 1000.0)))
    print(
        f"Binning time: {binned_time_ms:.3f} ms → binning_t = {binning_t} frames at {raw_frame_rate} fps"
    )
    kymo, binned_frame_rate = bin_kymograph_spatiotemporal_sum(
        kymo,
        binning_x=binning_x,
        binned_time_ms=binned_time_ms,
        raw_frame_rate_hz=raw_frame_rate,
    )
    print(f"Resulting frame rate after binning: {binned_frame_rate:.2f} fps")
    return kymo, binned_frame_rate


def load_h5_file(
    filename: str | Path,
    binning_x: int = 2,
    binned_time_ms: float = 2.857,
    *,
    trim_trailing_rows: int = 5000,
    dataset_name: str = DEFAULT_KYMOGRAPH_DATASET,
    max_time: int | None = None,
    default_fps: float = 2800.0,
) -> tuple[np.ndarray, float]:
    """Load kymograph from HDF5 with the same sum-binning as the collaborator notebook.

    Binning matches :func:`nsm.kymograph_io.load_kymograph_binned`. For comparisons
    with the main ``nsm`` CLI on a cropped file, set ``trim_trailing_rows=0`` (and
    use ``max_time`` or an ``nsm-crop`` output so both pipelines see the same window).

    Parameters
    ----------
    filename
        Path to the HDF5 file.
    trim_trailing_rows
        Rows dropped from the **end** of the raw dataset before binning. The
        historical notebook default was 5000; use ``0`` when the file is already
        cropped externally.
    dataset_name
        HDF5 dataset key (default matches :data:`nsm.kymograph_io.DEFAULT_KYMOGRAPH_DATASET`).
    max_time
        If set, read at most this many leading time rows (after ``trim_trailing_rows``).
    default_fps
        Used when the file has no ``fps`` attribute.
    """
    path = Path(filename).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"not a file: {path}")
    with h5py.File(path, "r", swmr=True, libver="latest", locking=False) as f:
        raw_frame_rate = float(f.attrs.get("fps", default_fps))
    binning_t = max(1, int(round(binned_time_ms * raw_frame_rate / 1000.0)))
    print(
        f"Binning time: {binned_time_ms:.3f} ms → binning_t = {binning_t} frames "
        f"at {raw_frame_rate} fps"
    )
    kymo, binned_frame_rate = load_kymograph_binned(
        path,
        dataset_name=dataset_name,
        binning_x=binning_x,
        binned_time_ms=binned_time_ms,
        default_fps=default_fps,
        trim_trailing_rows=trim_trailing_rows,
        max_time=max_time,
    )
    print(f"Resulting frame rate after binning: {binned_frame_rate:.2f} fps")
    return kymo, binned_frame_rate


# -----------------------------------------------------------------------------
# Particle Track Class


def estimate_diffusion_cve(time_idx: np.ndarray, 
                          position: np.ndarray,
                          dt: float,
                          remove_drift: bool = True):
    """
    Covariance-based estimator for diffusion coefficient with drift correction
    
    This is an unbiased estimator that works directly on position data
    without computing MSDs. Optimal for SNR > 1.
    
    Drift correction: Linear detrending removes constant velocity drift before
    calculating diffusion coefficient, preventing drift from inflating the estimate.
    
    Parameters
    ----------
    time_idx : np.ndarray
        Frame indices
    position : np.ndarray
        Spatial positions (pixels)
    dt : float
        Time step between frames (in seconds)
    remove_drift : bool
        If True, remove linear drift before calculating D
        
    Returns
    -------
    D : float
        Diffusion coefficient (pixels²/s)
    D_std : float
        Standard error of diffusion estimate
    localization_var : float
        Estimated localization error variance
    drift_velocity : float
        Estimated drift velocity (pixels/frame)
        
    Reference
    ---------
    Vestergaard, C. L., Blainey, P. C., & Flyvbjerg, H. (2014). 
    Optimal estimation of diffusion coefficients from single-particle trajectories. 
    Physical Review E, 89(2), 022726.
    """
    if len(position) < 3:
        return np.nan, np.nan, np.nan, np.nan
    
    # Estimate and remove drift (constant velocity)
    if remove_drift:
        # Linear fit: position = drift_velocity * time_idx + offset
        coeffs = np.polyfit(time_idx, position, 1)
        drift_velocity = coeffs[0]  # pixels per frame
        drift_trend = np.polyval(coeffs, time_idx)
        position_detrended = position - drift_trend
    else:
        drift_velocity = 0.0
        position_detrended = position
    
    # Calculate average frame step (handles missing frames)
    frame_steps = np.diff(time_idx)
    avg_frame_step = np.mean(frame_steps)
    avg_dt = avg_frame_step * dt
    
    # Calculate position differences (on detrended data)
    dx = np.diff(position_detrended)
    mean_dx_squared = np.mean(dx**2)
    
    # For unknown localization variance (estimate from data)
    # Equation 14 from Vestergaard et al. 2014
    mean_dx_consecutive = np.mean(dx[1:] * dx[:-1])
    
    # Diffusion coefficient estimate (Eq 14)
    D = mean_dx_squared / (2 * avg_dt) + mean_dx_consecutive / avg_dt
    
    # Localization variance estimate (Eq 15)
    localization_var = mean_dx_squared + 2 * mean_dx_consecutive
    
    # Variance of diffusion estimate (Eq 17)
    n = len(position)
    epsilon = localization_var / dt
    term1 = (6 * D**2 * avg_frame_step**2 + 4 * epsilon * D * avg_frame_step + 2 * epsilon**2) / (n * avg_frame_step**2)
    term2 = 4 * (D * avg_frame_step + epsilon)**2 / (n**2 * avg_frame_step**2)
    D_var = term1 + term2
    D_std = np.sqrt(np.abs(D_var))
    
    return D, D_std, localization_var, drift_velocity




class ParticleTrack:
    """
    A class to represent a particle track with segments, probabilities, and analysis methods.
    
    Attributes:
    -----------
    path : list of TrackNode
        List of TrackNode objects representing the segmented track
    segment_length : int
        Length of each segment in frames
    overlap_length : int
        Number of overlapping frames between consecutive segments
    stitched_track : np.ndarray or None
        Stitched positions along the track (time -> position)
    start_time : int or None
        Starting time index of the stitched track
    segment_probabilities : np.ndarray
        Probabilities of each segment in the path
    track_probabilities : np.ndarray or None
        Probabilities sampled from the probability map along the stitched track
    prob_map : np.ndarray or None
        The probability map used for extracting track probabilities
    """
    
    def __init__(self, path, segment_length, overlap_length, prob_map):
        """
        Initialize a ParticleTrack from a path of TrackNode objects.
        
        Parameters:
        -----------
        path : list of TrackNode
            List of TrackNode objects representing the segmented track
        segment_length : int
            Length of each segment in frames
        overlap_length : int
            Number of overlapping frames between consecutive segments
        prob_map : np.ndarray
            Probability map (time x position) for extracting probabilities along track
        """
        self.path = path
        self.segment_length = segment_length
        self.overlap_length = overlap_length
        self.prob_map = prob_map
        
        # Extract segment probabilities
        self.segment_probabilities = np.array([node.prob for node in path])
        
        # Stitch the track
        self.stitched_track, self.start_time, self.avg_segment_prob = stitch_path_with_overlap(
            path, segment_length, overlap_length
        )

        # Extract probabilities along the stitched track if prob_map is provided
        self.track_probabilities = None
        if self.stitched_track is not None:
            self.track_probabilities = self._extract_track_probabilities()

    
    def _extract_track_probabilities(self):
        """Extract probabilities from the probability map along the stitched track.
        
        Vectorized version - assumes prob_map is already on CPU (passed from link_segmented_tracks_with_viterbi).
        """
        if self.prob_map is None or self.stitched_track is None:
            return None
        
        # Vectorized extraction - much faster than Python loop!
        time_indices = self.start_time + cp.arange(len(self.stitched_track))
        pos_indices = cp.asarray(self.stitched_track.astype(int))
        
        # Create output array
        track_probs = cp.full(len(self.stitched_track), np.nan, dtype=np.float32)
        
        # Find valid indices (within bounds)
        valid_mask = (
            (time_indices >= 0) & (time_indices < self.prob_map.shape[0]) &
            (pos_indices >= 0) & (pos_indices < self.prob_map.shape[1])
        )
        
        # Extract probabilities for valid indices only
        track_probs[valid_mask] = self.prob_map[time_indices[valid_mask], pos_indices[valid_mask]]
        
        return track_probs
    
    @property
    def num_segments(self):
        """Number of segments in the track."""
        return len(self.path)
    
    @property
    def length(self):
        """Length of the stitched track in frames."""
        return len(self.stitched_track) if self.stitched_track is not None else 0
    
    @property
    def time_indices(self):
        """Time indices corresponding to the stitched track."""
        if self.stitched_track is None:
            return None
        return self.start_time + np.arange(len(self.stitched_track))
    
    @property
    def end_time(self):
        """End time index of the track."""
        if self.start_time is None or self.stitched_track is None:
            return None
        return self.start_time + len(self.stitched_track) - 1
    
    
    def calculate_diffusion(self, framerate=1.0, pixel_size=1.0, remove_drift=True):
        """
        Calculate the diffusion coefficient using covariance-based estimation (CVE).
        The method is optimal for SNR > 1 and works on tracks with as few as 3 points.
        
        Parameters:
        -----------
        framerate : float, optional
            Frame rate in Hz (default: 1.0)
        pixel_size : float, optional
            Physical size of one pixel in μm (default: 1.0)
        remove_drift : bool, optional
            If True, remove linear drift before calculating diffusion (default: True)
        
        Returns:
        --------
        result : dict or None
            Dictionary containing:
            - 'D': Diffusion coefficient in μm²/s
            - 'D_std': Standard error of diffusion estimate in μm²/s
            - 'localization_var': Localization error variance in μm²
            - 'drift_velocity': Drift velocity in pixels/frame
            Returns None if track is too short
        """
        if self.stitched_track is None or len(self.stitched_track) < 3:
            return None
        
        time_indices = self.time_indices
        dt = 1.0 / framerate
        
        # Use covariance-based estimator
        D_pixels, D_std_pixels, localization_var_pixels, drift_velocity = estimate_diffusion_cve(
            time_indices, self.stitched_track, dt, remove_drift=remove_drift
        )
        
        # Convert to physical units
        D = D_pixels * pixel_size**2 / dt
        D_std = D_std_pixels * pixel_size**2 / dt
        localization_var = localization_var_pixels * pixel_size**2
        
        return {
            'D': D,
            'D_std': D_std,
            'localization_var': localization_var,
            'drift_velocity': drift_velocity
        }
    
    def calculate_image_contrast_along_track(self, kymo, gaussian_blur_sigma=1):
        """
        Calculate the average image contrast along the track.
        
        Parameters:
        -----------
        kymo : np.ndarray or cp.ndarray
            Kymograph data (time x position)
        
        Returns:
        --------
        avg_contrast : float or None
            Average contrast along the track, or None if track is empty
        """
        
        # Ensure kymo is a CuPy array for gaussian_filter
        if not isinstance(kymo, cp.ndarray):
            kymo = cp.asarray(kymo)

        kymo = gaussian_filter(kymo, sigma=gaussian_blur_sigma)

        if self.stitched_track is None or len(self.stitched_track) == 0:
            return None
        
        # Extract pixel values along the track
        time_indices = self.time_indices
        pos_indices = self.stitched_track.astype(int)
        
        # Ensure indices are within bounds
        valid_mask = (
            (time_indices >= 0) & (time_indices < kymo.shape[0]) &
            (pos_indices >= 0) & (pos_indices < kymo.shape[1])
        )
        
        if not np.any(valid_mask):
            return None
        
        pixel_values = kymo[time_indices[valid_mask], pos_indices[valid_mask]]
        
        avg_contrast = np.mean(pixel_values)
        
        return avg_contrast, pixel_values

       
    def trim_track_by_probability(self, threshold_factor=0.66, window_width=10):
        """
        Trim low-probability regions from the start and end of the track based on 
        track probabilities (from prob_map). More precise than segment-based trimming.
        
        This method requires that prob_map was provided during initialization.
        
        Parameters:
        -----------
        threshold_factor : float, optional
            Factor to determine trimming threshold. 
            threshold = (median - min) * threshold_factor + min
            Default: 0.66
        window_width : int, optional
            Width of smoothing window for probabilities (default: 10)
        
        Returns:
        --------
        trimmed_track : ParticleTrack
            New ParticleTrack object with trimmed track, or None if track is too low probability
        """
        if self.track_probabilities is None:
            trimmed_path = trim_low_probability_segments(self.path, threshold_factor)
            if not trimmed_path:
                return None
            return ParticleTrack(
                trimmed_path,
                self.segment_length,
                self.overlap_length,
                self.prob_map,
            )
        
        if self.stitched_track is None or len(self.stitched_track) == 0:
            return None
        
        # Smooth probabilities to avoid cutting on single low-probability pixels
        if len(self.track_probabilities) < window_width:
            smoothed_prob = self.track_probabilities
        else:
            smoothed_prob = cupy_uniform_filter1d(
                self.track_probabilities, 
                size=window_width
            )
            smoothed_prob = cp.asnumpy(smoothed_prob)
        
        # Calculate threshold
        valid_probs = self.track_probabilities[~np.isnan(self.track_probabilities)]
        if len(valid_probs) == 0:
            return None
        
        median_prob = np.median(valid_probs)
        min_prob = np.percentile(valid_probs, 2)  # Use 2nd percentile to avoid outliers
        threshold = (median_prob - min_prob) * threshold_factor + min_prob
        
        # Find first position above threshold
        start_idx = 0
        for i in range(len(smoothed_prob)):
            if not np.isnan(smoothed_prob[i]) and smoothed_prob[i] >= threshold:
                start_idx = i
                break
        
        # Find last position above threshold
        end_idx = len(smoothed_prob) - 1
        for i in range(len(smoothed_prob) - 1, -1, -1):
            if not np.isnan(smoothed_prob[i]) and smoothed_prob[i] >= threshold:
                end_idx = i
                break
        
        # Check if track is completely below threshold
        if start_idx > end_idx or end_idx - start_idx < 10:  # Require at least 10 frames
            return None
        
        # Calculate which segments to keep based on the trimmed time range
        new_start_time = self.start_time + start_idx
        new_end_time = self.start_time + end_idx
        
        # Find segments that overlap with the trimmed time range
        step_size = self.segment_length - self.overlap_length
        trimmed_path = []
        
        for node in self.path:
            segment_start = node.segment_idx * step_size
            segment_end = segment_start + self.segment_length
            
            # Keep segments that overlap with the trimmed range
            if segment_end >= new_start_time and segment_start <= new_end_time:
                trimmed_path.append(node)
        
        if len(trimmed_path) == 0:
            return None
        
        # Return new ParticleTrack with trimmed path
        return ParticleTrack(trimmed_path, self.segment_length, self.overlap_length, self.prob_map)
    
    
    def plot_on_kymograph(self, kymo, ax=None, color='red', alpha=0.8, linewidth=2, label=None):
        """
        Plot the track on a kymograph.
        
        Parameters:
        -----------
        kymo : np.ndarray or cp.ndarray
            Kymograph data (time x position)
        ax : matplotlib axis, optional
            Axis to plot on. If None, creates a new figure.
        color : str, optional
            Color of the track line
        alpha : float, optional
            Transparency of the track line
        linewidth : float, optional
            Width of the track line
        label : str, optional
            Label for the track in the legend
        
        Returns:
        --------
        ax : matplotlib axis
            The axis with the plot
        """
        import matplotlib.pyplot as plt
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(14, 6))
        
        # Convert kymo to CPU if needed
        kymo_cpu = kymo.get() if hasattr(kymo, 'get') else kymo
        
        # Plot kymograph
        ax.imshow(kymo_cpu.T, cmap='gray', aspect='auto')
        
        # Plot track
        if self.stitched_track is not None:
            time_indices = self.time_indices
            ax.plot(time_indices, self.stitched_track, '-', 
                   linewidth=linewidth, alpha=alpha, color=color, label=label)
        
        ax.set_xlabel('Time (frames)')
        ax.set_ylabel('Position (pixels)')
        
        if label is not None:
            ax.legend()
        
        return ax
    
    
    def __repr__(self):
        return (f"ParticleTrack(segments={self.num_segments}, "
                f"length={self.length} frames, "
                f"start_time={self.start_time}, "
                f"avg_prob={self.avg_segment_prob:.4f})")
    
    def __len__(self):
        """Length of the track in frames."""
        return self.length


# Helper function to create tracks from paths
# only used in notebook. should be removed as soon as possible
def trim_low_probability_segments(path, threshold_factor=0.66):
    """
    Trim low-probability segments from the start and end of a path.
    
    This is a convenience function that works with paths (list of TrackNode objects).
    
    Parameters:
    -----------
    path : list of TrackNode
        List of TrackNode objects representing the segmented track
    threshold_factor : float, optional
        Factor to determine trimming threshold. 
        threshold = (median - min) * threshold_factor + min
        Default: 0.66
    
    Returns:
    --------
    trimmed_path : list of TrackNode
        Trimmed path with low-probability segments removed
    """
    if len(path) == 0:
        return []
    
    # Calculate threshold based on segment probabilities
    segment_probs = np.array([node.prob for node in path])
    median_prob = np.median(segment_probs)
    min_prob = np.min(segment_probs)
    threshold = (median_prob - min_prob) * threshold_factor + min_prob
    
    # Find first segment above threshold
    start_idx = 0
    for i, prob in enumerate(segment_probs):
        if prob >= threshold:
            start_idx = i
            break
    
    # Find last segment above threshold
    end_idx = len(segment_probs) - 1
    for i in range(len(segment_probs) - 1, -1, -1):
        if segment_probs[i] >= threshold:
            end_idx = i
            break
    
    # Check if track is completely below threshold
    if start_idx > end_idx:
        return []
    
    # Return trimmed path
    return path[start_idx:end_idx + 1]
