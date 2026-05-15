"""Segmented-Viterbi command and helpers.

This file now contains what was previously spread across
``segmented_viterbi_io.py``, ``segmented_viterbi_model.py``,
``segmented_viterbi_core.py``, and ``segmented_viterbi_algo.py``.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter, uniform_filter1d
from scipy.special import erf as sp_erf, erfinv as sp_erfinv

import numba

from nsm.core import (
    DEFAULT_KYMOGRAPH_DATASET,
    DEFAULT_PLOTS_DIR,
    FIGSIZE_INCHES,
    IMAGE_CMAP,
    bin_kymograph_spatiotemporal_sum,
    load_kymograph,
    load_kymograph_binned,
    resolve_output_directory,
    write_kymograph_png,
    write_raw_preview,
    write_trajectory_overlay_png,
)

METHOD = "segmented_viterbi"
NAME = "segmented-viterbi"
HELP = "Run segmented-Viterbi directly on a kymograph and export tracks."


# ---------------------------------------------------------------------------
# I/O helpers (from segmented_viterbi_io.py)


def load_envue_mat_file(filename, binning_x=1, binned_time_ms=2.857):
    import scipy.io as sio

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
    """Load HDF5 kymograph with sum-binning used by segmented Viterbi."""
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


# ---------------------------------------------------------------------------
# Track model (from segmented_viterbi_model.py)

from dataclasses import dataclass


@dataclass(frozen=True, eq=False)
class TrackNode:
    """Node in graph for one segmented track hypothesis."""

    segment_idx: int
    track_idx: int
    track: np.ndarray
    prob: float

    def __repr__(self) -> str:
        return f"S{self.segment_idx}_N{self.track_idx}"


def estimate_diffusion_cve(
    time_idx: np.ndarray,
    position: np.ndarray,
    dt: float,
    remove_drift: bool = True,
):
    """Covariance-based estimator used for CVE-style post-fit diagnostics."""
    if len(position) < 3:
        return np.nan, np.nan, np.nan, np.nan

    if remove_drift:
        coeffs = np.polyfit(time_idx, position, 1)
        drift_velocity = coeffs[0]
        drift_trend = np.polyval(coeffs, time_idx)
        position_detrended = position - drift_trend
    else:
        drift_velocity = 0.0
        position_detrended = position

    frame_steps = np.diff(time_idx)
    avg_frame_step = np.mean(frame_steps)
    avg_dt = avg_frame_step * dt

    dx = np.diff(position_detrended)
    mean_dx_squared = np.mean(dx**2)
    mean_dx_consecutive = np.mean(dx[1:] * dx[:-1])

    D = mean_dx_squared / (2 * avg_dt) + mean_dx_consecutive / avg_dt
    localization_var = mean_dx_squared + 2 * mean_dx_consecutive

    n = len(position)
    epsilon = localization_var / dt
    term1 = (6 * D**2 * avg_frame_step**2 + 4 * epsilon * D * avg_frame_step + 2 * epsilon**2) / (n * avg_frame_step**2)
    term2 = 4 * (D * avg_frame_step + epsilon)**2 / (n**2 * avg_frame_step**2)
    D_var = term1 + term2
    D_std = np.sqrt(np.abs(D_var))

    return D, D_std, localization_var, drift_velocity


def stitch_path_with_overlap(path, segment_length, overlap_length, trim_low_prob_segments=True):
    if len(path) == 0:
        return None, None, None
    if len(path) == 1:
        return path[0].track, path[0].segment_idx * (segment_length - overlap_length), path[0].prob

    stitched_track = path[0].track.copy()
    for i in range(1, len(path)):
        current_node = path[i]
        fist_track_overlap_last_node = stitched_track[-1]
        second_track_overlap_middle_node = current_node.track[len(current_node.track) // 2]
        following_track_first_node = path[i + 1].track[0] if i + 1 < len(path) else None

        gap_use_first = np.abs(fist_track_overlap_last_node - following_track_first_node) if following_track_first_node is not None else 0
        gap_use_second = np.abs(second_track_overlap_middle_node - following_track_first_node) if following_track_first_node is not None else 0

        if gap_use_first <= gap_use_second:
            stitched_track = np.concatenate([
                stitched_track,
                current_node.track[overlap_length:]
            ])
        else:
            stitched_track = np.concatenate([
                stitched_track[:-overlap_length],
                current_node.track
            ])

    start_time = path[0].segment_idx * (segment_length - overlap_length)
    avg_prob = np.mean([node.prob for node in path])
    return stitched_track, start_time, avg_prob


class ParticleTrack:
    """A particle track with segments and convenience properties."""

    def __init__(self, path, segment_length, overlap_length, prob_map):
        self.path = path
        self.segment_length = segment_length
        self.overlap_length = overlap_length
        self.prob_map = prob_map

        self.segment_probabilities = np.array([node.prob for node in path])
        self.stitched_track, self.start_time, self.avg_segment_prob = stitch_path_with_overlap(
            path, segment_length, overlap_length
        )

        self.track_probabilities = None
        if self.stitched_track is not None:
            self.track_probabilities = self._extract_track_probabilities()

    def _extract_track_probabilities(self):
        if self.prob_map is None or self.stitched_track is None:
            return None

        time_indices = self.start_time + np.arange(len(self.stitched_track))
        pos_indices = np.asarray(self.stitched_track.astype(int))

        track_probs = np.full(len(self.stitched_track), np.nan, dtype=np.float32)
        valid_mask = (
            (time_indices >= 0) & (time_indices < self.prob_map.shape[0]) &
            (pos_indices >= 0) & (pos_indices < self.prob_map.shape[1])
        )
        track_probs[valid_mask] = self.prob_map[time_indices[valid_mask], pos_indices[valid_mask]]
        return track_probs

    @property
    def num_segments(self):
        return len(self.path)

    @property
    def length(self):
        return len(self.stitched_track) if self.stitched_track is not None else 0

    @property
    def time_indices(self):
        if self.stitched_track is None:
            return None
        return self.start_time + np.arange(len(self.stitched_track))

    @property
    def end_time(self):
        if self.start_time is None or self.stitched_track is None:
            return None
        return self.start_time + len(self.stitched_track) - 1

    def calculate_diffusion(
        self,
        framerate=1.0,
        pixel_size=1.0,
        remove_drift=True,
    ):
        if self.stitched_track is None or len(self.stitched_track) < 3:
            return None

        time_indices = self.time_indices
        dt = 1.0 / framerate

        D_pixels, D_std_pixels, localization_var_pixels, drift_velocity = estimate_diffusion_cve(
            time_indices, self.stitched_track, dt, remove_drift=remove_drift
        )

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
        kymo = np.asarray(kymo)
        kymo = gaussian_filter(kymo, sigma=gaussian_blur_sigma)

        if self.stitched_track is None or len(self.stitched_track) == 0:
            return None

        time_indices = self.time_indices
        pos_indices = self.stitched_track.astype(int)

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

        if len(self.track_probabilities) < window_width:
            smoothed_prob = self.track_probabilities
        else:
            smoothed_prob = uniform_filter1d(self.track_probabilities, size=window_width)

        valid_probs = self.track_probabilities[~np.isnan(self.track_probabilities)]
        if len(valid_probs) == 0:
            return None

        median_prob = np.median(valid_probs)
        min_prob = np.percentile(valid_probs, 2)
        threshold = (median_prob - min_prob) * threshold_factor + min_prob

        start_idx = 0
        for i in range(len(smoothed_prob)):
            if not np.isnan(smoothed_prob[i]) and smoothed_prob[i] >= threshold:
                start_idx = i
                break

        end_idx = len(smoothed_prob) - 1
        for i in range(len(smoothed_prob) - 1, -1, -1):
            if not np.isnan(smoothed_prob[i]) and smoothed_prob[i] >= threshold:
                end_idx = i
                break

        if start_idx > end_idx or end_idx - start_idx < 10:
            return None

        new_start_time = self.start_time + start_idx
        new_end_time = self.start_time + end_idx

        step_size = self.segment_length - self.overlap_length
        trimmed_path = []

        for node in self.path:
            segment_start = node.segment_idx * step_size
            segment_end = segment_start + self.segment_length
            if segment_end >= new_start_time and segment_start <= new_end_time:
                trimmed_path.append(node)

        if len(trimmed_path) == 0:
            return None

        return ParticleTrack(trimmed_path, self.segment_length, self.overlap_length, self.prob_map)

    def plot_on_kymograph(self, kymo, ax=None, color='red', alpha=0.8, linewidth=2, label=None):
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots(figsize=(14, 6))

        kymo_cpu = np.asarray(kymo)
        ax.imshow(kymo_cpu.T, cmap='gray', aspect='auto')

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
        return self.length


def trim_low_probability_segments(path, threshold_factor=0.66):
    if len(path) == 0:
        return []
    segment_probs = np.array([node.prob for node in path])
    median_prob = np.median(segment_probs)
    min_prob = np.min(segment_probs)
    threshold = (median_prob - min_prob) * threshold_factor + min_prob

    start_idx = 0
    for i, prob in enumerate(segment_probs):
        if prob >= threshold:
            start_idx = i
            break

    end_idx = len(segment_probs) - 1
    for i in range(len(segment_probs) - 1, -1, -1):
        if segment_probs[i] >= threshold:
            end_idx = i
            break

    if start_idx > end_idx:
        return []
    return path[start_idx:end_idx + 1]


# ---------------------------------------------------------------------------
# Core DP/Viterbi helpers (from segmented_viterbi_core.py)


def remove_background(kymo, average_samples_x, average_samples_t):
    kymo = kymo / uniform_filter1d(kymo, size=average_samples_x, axis=1)
    kymo = kymo / uniform_filter1d(kymo, size=average_samples_t, axis=0)
    return -kymo + 1


def calculate_probability_map(kymo_background_removed, gauss_blur_sigma, prior_particle_probability=0.5):
    """Map background-removed kymograph into probability-like weights."""
    noise_det_length = np.max([kymo_background_removed.shape[0], 1000])
    sigma_n = np.std(kymo_background_removed[:noise_det_length-1, :], axis=0)
    pixel_track_prob_map = kymo_background_removed / sigma_n
    z_shift = -sp_erfinv(2 * prior_particle_probability - 1)
    pixel_track_prob_map = 0.5 * (1 + sp_erf(pixel_track_prob_map / np.sqrt(2) - z_shift))
    pixel_track_prob_map = gaussian_filter(pixel_track_prob_map, sigma=(gauss_blur_sigma, gauss_blur_sigma))
    return pixel_track_prob_map


@numba.njit
def _viterbi_algorithm_core(emission_prob, transition_prob, dia_width):
    num_frames, num_positions = emission_prob.shape
    viterbi = np.zeros((num_frames, num_positions))
    backpointer = np.zeros((num_frames, num_positions), dtype=np.int32)

    viterbi[0, :] = emission_prob[0, :]

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

    best_path = np.zeros(num_frames, dtype=np.int32)
    best_path[-1] = np.argmax(viterbi[-1, :])
    for t in range(num_frames - 2, -1, -1):
        best_path[t] = backpointer[t + 1, best_path[t + 1]]

    return best_path


@numba.njit
def _process_multiple_segments(emission_prob_segments, transition_prob, dia_width):
    n_segments = emission_prob_segments.shape[0]
    num_frames = emission_prob_segments.shape[1]
    paths = np.zeros((n_segments, num_frames), dtype=np.int32)

    for seg in range(n_segments):
        paths[seg] = _viterbi_algorithm_core(emission_prob_segments[seg], transition_prob, dia_width)

    return paths


def process_segments_batch(
    emission_prob_segments,
    sigma_transition=0.5,
    velocity=None,
):
    if isinstance(emission_prob_segments, list):
        emission_prob_segments = np.array(emission_prob_segments, dtype=np.float64)
    emission_prob_segments = np.asarray(emission_prob_segments, dtype=np.float64)

    n_segments, num_frames, num_positions = emission_prob_segments.shape
    threshold = 0.001
    dia_width = int(np.sum(np.exp(-np.arange(0, 100)**2 / (2 * sigma_transition**2)) > threshold) * 2)

    positions = np.arange(num_positions)
    distance_matrix = np.abs(positions[:, None] - positions[None, :])
    if velocity is not None:
        distance_matrix = np.abs(distance_matrix - velocity)
    transition_prob = np.exp(-(distance_matrix**2) / (2 * sigma_transition**2))

    paths = _process_multiple_segments(emission_prob_segments, transition_prob, dia_width)

    probs = []
    for i in range(n_segments):
        path_emission_probs = emission_prob_segments[i, np.arange(num_frames), paths[i]]
        probs.append(np.prod(path_emission_probs) ** (1 / num_frames))

    return list(paths), probs


def segmented_multi_track_viterbi(
    kymo,
    segment_length=100,
    overlap_length=0,
    sigma_transition=0.5,
    velocity=None,
    num_passes=3,
    max_frames=None,
):
    num_frames, num_positions = kymo.shape

    if max_frames is not None and max_frames > 0:
        num_frames = min(num_frames, max_frames)
        print(f"Limiting track finding to first {num_frames} frames (of {kymo.shape[0]} total)")

    all_tracks = []
    all_probabilities = []

    step_size = segment_length - overlap_length
    if step_size <= 0:
        raise ValueError(f"overlap_length ({overlap_length}) must be less than segment_length ({segment_length})")

    kymo_segments = []
    for start in range(0, num_frames, step_size):
        end = start + segment_length
        if end > num_frames:
            break
        kymo_segments.append(np.asarray(kymo[start:end, :], dtype=np.float64))

    all_tracks = [[] for _ in range(len(kymo_segments))]
    all_probabilities = [[] for _ in range(len(kymo_segments))]

    for pass_num in range(num_passes):
        segments_array = np.array(kymo_segments, dtype=np.float64)
        tracks, probs = process_segments_batch(segments_array, sigma_transition, velocity)
        for i, (track, prob) in enumerate(zip(tracks, probs)):
            all_tracks[i].append(track)
            all_probabilities[i].append(prob)

        if pass_num < num_passes - 1:
            mask_width = 35
            for i, track in enumerate(tracks):
                mask = np.zeros_like(kymo_segments[i], dtype=bool)
                for t, pos in enumerate(track):
                    mask[t, max(0, pos-mask_width//2):min(kymo_segments[i].shape[1], pos+mask_width//2+1)] = True
                kymo_segments[i][mask] = 0

    all_log_probabilities_flat = [prob for segment_probs in all_probabilities for prob in segment_probs]
    all_log_probabilities_flat = np.log(np.array(all_log_probabilities_flat))
    sorted_indices = np.argsort(all_log_probabilities_flat)[::-1]
    sorted_log_probabilities = all_log_probabilities_flat[sorted_indices]

    method = 'median x 1.5'
    if method == 'elbow':
        diffs = np.diff(sorted_log_probabilities)
        elbow_index = np.argmin(diffs) + 1
        threshold_log_prob = sorted_log_probabilities[elbow_index]
    elif method == 'median x 1.5':
        threshold_log_prob = 1.5 * sorted_log_probabilities[len(sorted_log_probabilities)//2]
    print(f"threshold_log_prob: {threshold_log_prob:.4f}, corresponding to probability {np.exp(threshold_log_prob):.4e}")

    filtered_tracks = []
    filtered_probabilities = []
    for segment_tracks, segment_probs in zip(all_tracks, all_probabilities):
        segment_log_probs = np.log(np.array(segment_probs))
        valid_indices = [i for i, log_prob in enumerate(segment_log_probs) if log_prob >= threshold_log_prob]
        filtered_tracks.append([segment_tracks[i] for i in valid_indices])
        filtered_probabilities.append([segment_probs[i] for i in valid_indices])

    return filtered_tracks, filtered_probabilities, all_log_probabilities_flat


@numba.njit
def _score_overlap(track1, track2, overlap_length):
    gap = np.abs(track1[-overlap_length:] - track2[:overlap_length])
    sorted_gap = np.sort(gap)
    idx = int(0.25 * len(sorted_gap))
    return sorted_gap[idx]


@numba.njit
def _build_edge_matrix(tracks_current, tracks_next, overlap_length, max_gap):
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
    n_segments = len(nodes_per_segment)
    max_nodes = np.max(nodes_per_segment)

    best_length = np.zeros((n_segments, max_nodes), dtype=np.int32)
    best_weight = np.zeros((n_segments, max_nodes), dtype=np.float64)
    backpointer = np.full((n_segments, max_nodes), -1, dtype=np.int32)

    for s in range(n_segments):
        for i in range(nodes_per_segment[s]):
            best_length[s, i] = 1
            best_weight[s, i] = 0.0

    for s in range(n_segments - 1):
        edge_matrix = edge_matrices[s]
        for i in range(nodes_per_segment[s]):
            cur_len = best_length[s, i]
            cur_w = best_weight[s, i]

            for j in range(nodes_per_segment[s + 1]):
                if edge_matrix[i, j] >= 0:
                    new_len = cur_len + 1
                    new_w = cur_w + edge_matrix[i, j]
                    if (new_len > best_length[s + 1, j] or
                        (new_len == best_length[s + 1, j] and new_w < best_weight[s + 1, j])):
                        best_length[s + 1, j] = new_len
                        best_weight[s + 1, j] = new_w
                        backpointer[s + 1, j] = i

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
        path_segments = np.zeros(1, dtype=np.int32)
        path_nodes = np.zeros(1, dtype=np.int32)
        return path_segments, path_nodes, (0, 0.0)

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
    gap = np.abs(track1[-overlap_length:] - track2[:overlap_length])
    avg_gap = np.mean(gap)
    return avg_gap


def score_overlap_partial(track1, prob1, track2, prob2, overlap_length, sigma_gap):
    gap = np.abs(track1[-overlap_length:] - track2[:overlap_length])
    return np.percentile(gap, 25)


def build_segment_graph(
    segmented_tracks,
    segmented_probabilities,
    overlap_length,
    sigma_gap,
    max_average_gap,
):
    segments = [
        [TrackNode(s, n, segmented_tracks[s][n], segmented_probabilities[s][n])
         for n in range(len(segmented_tracks[s]))]
        for s in range(len(segmented_tracks))
    ]

    edge_matrices = []
    for s in range(len(segmented_tracks) - 1):
        tracks_current = [np.asarray(segmented_tracks[s][i], dtype=np.float64)
                         for i in range(len(segmented_tracks[s]))]
        tracks_next = [np.asarray(segmented_tracks[s + 1][i], dtype=np.float64)
                      for i in range(len(segmented_tracks[s + 1]))]

        if len(tracks_current) > 0 and len(tracks_next) > 0:
            tracks_current_array = np.stack(tracks_current, axis=0).astype(np.float64)
            tracks_next_array = np.stack(tracks_next, axis=0).astype(np.float64)
            edge_matrix = _build_edge_matrix(tracks_current_array, tracks_next_array,
                                            overlap_length, max_average_gap)
        else:
            n_current = len(tracks_current)
            n_next = len(tracks_next)
            edge_matrix = np.full((n_current, n_next), -1.0, dtype=np.float64)

        edge_matrices.append(edge_matrix)

    return edge_matrices, segments


def find_best_path_across_segments(edge_matrices, segments, used_set, min_length):
    if len(segments) == 0:
        return None
    if all(len(seg) == 0 for seg in segments):
        return None

    modified_edges = []
    for s in range(len(edge_matrices)):
        edge_matrix = edge_matrices[s].copy()
        for i, node in enumerate(segments[s]):
            if node in used_set:
                edge_matrix[i, :] = -1.0
        for j, node in enumerate(segments[s + 1]):
            if node in used_set:
                edge_matrix[:, j] = -1.0
        modified_edges.append(edge_matrix)

    nodes_per_segment = np.array([len(seg) for seg in segments], dtype=np.int32)
    if np.any(nodes_per_segment <= 0):
        return None
    if len(modified_edges) == 0:
        return None

    path_segments, path_nodes, (length, weight) = _find_best_path(
        modified_edges, nodes_per_segment, min_length
    )

    if length == 0:
        return None

    path = [segments[seg_idx][node_idx]
            for seg_idx, node_idx in zip(path_segments, path_nodes)]
    return path


def link_segmented_tracks_with_viterbi(
    segmented_tracks,
    segmented_probabilities,
    segment_length,
    overlap_length,
    sigma_gap,
    max_average_gap,
    min_length=3,
    prob_map=None,
):
    if len(segmented_tracks) == 0 or all(len(seg) == 0 for seg in segmented_tracks):
        return []

    edge_matrices, segments = build_segment_graph(
        segmented_tracks,
        segmented_probabilities,
        overlap_length,
        sigma_gap,
        max_average_gap,
    )
    if len(segments) == 0 or all(len(seg) == 0 for seg in segments):
        return []

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
        if iteration % 500 == 0:
            print(
                f"Iteration {iteration}: {len(paths)} paths found so far, "
                f"{len(used):,} nodes used"
            )

    tracks = []
    for path in paths:
        tracks.append(ParticleTrack(path, segment_length, overlap_length, prob_map))

    return tracks


# ---------------------------------------------------------------------------
# Lightweight wrappers and CLI helpers


def write_tracks_csv(path: Path, tracks: list, track_rows: list[tuple[int, float, float, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not track_rows:
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["id", "x", "t", "intensity"])
        return

    arr = np.array(track_rows, dtype=np.float64)
    order = np.lexsort((arr[:, 1], arr[:, 2], arr[:, 0]))
    arr = arr[order]

    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "x", "t", "intensity"])
        for row in arr:
            pid, x, t, intensity = row
            w.writerow([
                int(round(pid)),
                float(x),
                int(round(t)),
                float(intensity),
            ])


def tracks_to_rows(tracks: list) -> list[tuple[float, float, float, float]]:
    rows: list[tuple[float, float, float, float]] = []
    for tid, track in enumerate(tracks):
        if track.stitched_track is None or track.time_indices is None:
            continue
        if len(track.stitched_track) == 0:
            continue

        tids = np.full(len(track.stitched_track), float(tid), dtype=np.float64)
        xs = np.asarray(track.stitched_track, dtype=np.float64)
        ts = np.asarray(track.time_indices, dtype=np.float64)
        if track.track_probabilities is not None:
            intensity = np.asarray(track.track_probabilities, dtype=np.float64)
            if len(intensity) != len(ts):
                intensity = np.full_like(ts, np.nan, dtype=np.float64)
        else:
            intensity = np.full_like(ts, np.nan, dtype=np.float64)

        rows.extend((float(tid), float(x), float(t), float(i)) for tid, x, t, i in zip(tids, xs, ts, intensity))
    return rows


def _filter_edge_tracks(
    segmented_tracks,
    segmented_probs,
    edge_region_percent: float,
    n_pixels: int,
):
    if edge_region_percent <= 0:
        return segmented_tracks, segmented_probs

    edge_width = int(max(0, n_pixels) * (edge_region_percent / 100.0))
    if edge_width <= 0:
        return segmented_tracks, segmented_probs

    lower = edge_width
    upper = max(lower, n_pixels - edge_width)

    filtered_tracks = []
    filtered_probs = []
    for seg_tracks, seg_probs in zip(segmented_tracks, segmented_probs):
        tks = []
        prs = []
        for track, prob in zip(seg_tracks, seg_probs):
            if not (np.max(track) <= lower or np.min(track) >= upper):
                tks.append(track)
                prs.append(prob)
        filtered_tracks.append(tks)
        filtered_probs.append(prs)
    return filtered_tracks, filtered_probs


def _write_preprocessed_h5(path: Path, prob_map: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as fw:
        fw.create_dataset(
            "probability_map",
            data=prob_map.astype(np.float32, copy=False),
        )


def run_command(
    raw_h5: Path,
    output: Path = DEFAULT_PLOTS_DIR,
    max_time: int = 1024,
    overlay: bool = True,
    keep_preprocessed: bool = True,
) -> None:
    """Run segmented-Viterbi with CLI-friendly defaults."""
    args = [
        str(raw_h5),
        "-o",
        str(output),
        "--max-time",
        str(max_time),
        "--binning-x",
        "1",
        "--binned-time-ms",
        "2.0",
        "--segment-length",
        "100",
        "--overlap-length",
        "50",
        "--average-samples-x",
        "20",
        "--average-samples-t",
        "50",
        "--prior-particle-probability",
        "0.3",
        "--probability-sigma",
        "2.0",
        "--sigma-transition",
        "1.0",
        "--num-passes",
        "3",
        "--trim-window",
        "100",
        "--min-path-length",
        "3",
        "--sigma-gap",
        "2.0",
        "--max-average-gap",
        "5.0",
        "--edge-region-percent",
        "0.0",
        "--no-summary",
    ]
    if not keep_preprocessed:
        args.append("--no-preprocessed")
    if not overlay:
        args.append("--no-overlay")
    run(args)


def run(argv: list[str] | None = None) -> None:
    """Full argument-parsing CLI retained from historical implementation."""
    parser = argparse.ArgumentParser(
        description=(
            "Run segmented Viterbi tracking directly on a kymograph and export tracks. "
            "Structured, CPU-only alternative to the clustering pipeline."
        )
    )
    parser.add_argument("h5_path", type=Path)
    parser.add_argument("-o", "--output", type=Path, default=".")
    parser.add_argument("--binning-x", type=int, default=2)
    parser.add_argument("--binned-time-ms", type=float, default=2.0)
    parser.add_argument("--trim-trailing-rows", type=int, default=0)
    parser.add_argument("--max-time", type=int, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--average-samples-x", type=int, default=20)
    parser.add_argument("--average-samples-t", type=int, default=50)
    parser.add_argument("--prior-particle-probability", type=float, default=0.3)
    parser.add_argument("--probability-sigma", type=float, default=2.0)
    parser.add_argument("--segment-length", type=int, default=100)
    parser.add_argument("--overlap-length", type=int, default=50)
    parser.add_argument("--sigma-transition", type=float, default=1.0)
    parser.add_argument("--velocity", type=float, default=0.0)
    parser.add_argument("--num-passes", type=int, default=3)
    parser.add_argument("--sigma-gap", type=float, default=2.0)
    parser.add_argument("--max-average-gap", type=float, default=5.0)
    parser.add_argument("--min-path-length", type=int, default=3)
    parser.add_argument("--threshold-factor", type=float, default=1.0)
    parser.add_argument("--trim-window", type=int, default=100)
    parser.add_argument("--edge-region-percent", type=float, default=0.0)
    parser.add_argument("--preprocessed", dest="keep_preprocessed", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--summary", dest="write_summary", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--no-overlay", action="store_true")
    args = parser.parse_args(argv)

    if args.binning_x < 1:
        parser.error("--binning-x must be >= 1")
    if args.overlap_length < 0:
        parser.error("--overlap-length must be >= 0")
    if args.segment_length <= args.overlap_length:
        parser.error("--segment-length must be greater than --overlap-length")
    if args.max_time is not None and args.max_time <= 0:
        parser.error("--max-time must be > 0")
    if args.max_frames is not None and args.max_frames <= 0:
        parser.error("--max-frames must be > 0")
    if args.edge_region_percent < 0 or args.edge_region_percent >= 100:
        parser.error("--edge-region-percent must be in [0, 100)")

    src = args.h5_path.expanduser().resolve()
    if not src.is_file():
        raise FileNotFoundError(f"not a file: {src}")

    out_dir = resolve_output_directory(args.output)
    stem = src.stem

    raw_preview_png = out_dir / f"{stem}_raw.png"
    if not raw_preview_png.exists():
        raw, _ = load_kymograph(
            src,
            dataset_name=DEFAULT_KYMOGRAPH_DATASET,
            max_time=args.max_time,
        )
        write_raw_preview(
            raw,
            raw_preview_png,
            title=(
                f"{src.name}\n"
                f"cropped raw ({tuple(raw.shape)}) dataset={DEFAULT_KYMOGRAPH_DATASET}"
            ),
        )
        print(f"Wrote {raw_preview_png.resolve()}")

    kymo, fps = load_h5_file(
        src,
        binning_x=args.binning_x,
        binned_time_ms=args.binned_time_ms,
        trim_trailing_rows=args.trim_trailing_rows,
        max_time=args.max_time,
    )

    kymo = np.asarray(kymo, dtype=np.float64)
    bg = remove_background(
        kymo,
        average_samples_x=args.average_samples_x,
        average_samples_t=args.average_samples_t,
    )
    prob_map = calculate_probability_map(
        bg,
        gauss_blur_sigma=args.probability_sigma,
        prior_particle_probability=args.prior_particle_probability,
    )

    if args.keep_preprocessed:
        pre_h5 = out_dir / f"{stem}_{METHOD}_preprocessed.h5"
        pre_png = out_dir / f"{stem}_{METHOD}_preprocessed.png"
        _write_preprocessed_h5(pre_h5, prob_map)
        write_kymograph_png(
            prob_map,
            pre_png,
            title=(
                f"{src.name}\n"
                f"{METHOD} preprocessing: background_removed → probability_map\n"
                f"fps={fps:.2f}, binning_x={args.binning_x}, binned_time_ms={args.binned_time_ms}"
            ),
        )

    segmented_tracks, segmented_probs, all_log_probs = segmented_multi_track_viterbi(
        prob_map,
        segment_length=args.segment_length,
        overlap_length=args.overlap_length,
        sigma_transition=args.sigma_transition,
        velocity=(args.velocity if args.velocity != 0.0 else None),
        num_passes=args.num_passes,
        max_frames=args.max_frames,
    )
    del all_log_probs

    segmented_tracks, segmented_probs = _filter_edge_tracks(
        segmented_tracks,
        segmented_probs,
        edge_region_percent=args.edge_region_percent,
        n_pixels=int(prob_map.shape[1]),
    )

    linked_tracks = link_segmented_tracks_with_viterbi(
        segmented_tracks,
        segmented_probs,
        segment_length=args.segment_length,
        overlap_length=args.overlap_length,
        sigma_gap=args.sigma_gap,
        max_average_gap=args.max_average_gap,
        min_length=args.min_path_length,
        prob_map=prob_map,
    )

    tracks = [
        tr.trim_track_by_probability(args.threshold_factor, args.trim_window)
        for tr in linked_tracks
        if tr is not None
    ]
    tracks = [tr for tr in tracks if tr is not None]

    track_rows = tracks_to_rows(tracks)

    out_csv = out_dir / f"{stem}_{METHOD}_tracks.csv"
    write_tracks_csv(out_csv, tracks, track_rows)
    print(f"Wrote {out_csv.resolve()}")

    if args.write_summary:
        out_summary = out_dir / f"{stem}_{METHOD}_track_summary.csv"
        with out_summary.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["id", "n_frames", "start_frame", "end_frame"])
            for i, track in enumerate(tracks):
                start = track.start_time if track.start_time is not None else 0
                end = track.end_time if track.end_time is not None else start
                w.writerow([i, int(len(track)), int(start), int(end)])
        print(f"Wrote {out_summary.resolve()}")

    if not args.no_overlay and tracks:
        out_plot = out_dir / f"{stem}_{METHOD}_tracks_overlay.png"
        plot_track_overlay(prob_map, tracks, out_plot)
        print(f"Wrote {out_plot.resolve()}")

    print(
        f"Segmented-Viterbi: linked {len(linked_tracks)} paths, kept {len(tracks)} trimmed tracks "
        f"from {src.name} (binned fps={fps:.3f}, seed-shape={prob_map.shape})"
    )


def plot_track_overlay(kymo: np.ndarray, tracks: list, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=FIGSIZE_INCHES)
    im = ax.imshow(
        kymo.T,
        aspect="auto",
        origin="upper",
        cmap=IMAGE_CMAP,
    )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for ti, track in enumerate(tracks):
        if track.stitched_track is None or len(track) == 0:
            continue
        times = track.time_indices
        ax.plot(
            times,
            track.stitched_track,
            linewidth=1.2,
            alpha=0.8,
            label=str(ti),
        )

    ax.set_xlabel("Time (frame)")
    ax.set_ylabel("Position (pixels)")
    ax.set_title(f"{METHOD} tracks overlay")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
