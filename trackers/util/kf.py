"""
    SORT: A Simple, Online and Realtime Tracker
    Copyright (C) 2016-2020 Alex Bewley alex@bewley.ai

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
    
    This is a modified version by Esslingen University of Applied Sciences.
"""
from __future__ import print_function
from collections import defaultdict

import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

USE_LAP = True
try:
    import lap
except ImportError:
    from scipy.optimize import linear_sum_assignment
    USE_LAP = False

def linear_assignment(cost_matrix):
    if USE_LAP:
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[y[i],i] for i in x if i >= 0])
    else:
        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))


def convert_rv_to_z(rv):
    return np.array([rv[0], rv[1]]).reshape((2, 1))


def convert_x_to_rv(x, score=None):
    if score is None:
        return np.array([x[0], x[1]]).reshape((1, 2))
    else:
        return np.array([x[0], x[1], score]).reshape((1, 3))

def _get_hard_identity_matches(
        detection_ids: np.ndarray,
        track_ids: np.ndarray,
        dist_matrix: np.ndarray,
        dist_threshold: float
    ) -> np.ndarray:

    ids_to_det = defaultdict(list)
    for det_idx, det_id in enumerate(detection_ids):
        if det_id >= 0:
            ids_to_det[int(det_id)].append(det_idx)

    ids_to_track = defaultdict(list)
    for track_idx, track_id in enumerate(track_ids):
        if track_id >= 0:
            ids_to_track[int(track_id)].append(track_idx)

    forced_matches = {}
    for det_id, det_indices in ids_to_det.items():
        if len(det_indices) != 1:
            continue
        track_indices = ids_to_track.get(det_id, [])
        if len(track_indices) != 1:
            continue

        det_idx = det_indices[0]
        track_idx = track_indices[0]

        # Check distance between new detection and currently tracked object
        if dist_matrix[det_idx, track_idx] <= dist_threshold:
            # Forcefully matching new detection to currently tracked object
            forced_matches[det_idx] = track_idx

    used_dets = set(forced_matches.keys())
    remaining_dets = np.array([i for i in range(len(detection_ids)) if i not in used_dets])

    used_trackers = set(forced_matches.values())
    remaining_trackers = np.array([i for i in range(len(track_ids)) if i not in used_trackers])

    # Ordinary association between remaining dets and trackers
    ordinary_matches = []
    if len(remaining_dets) > 0 and len(remaining_trackers) > 0:
        remaining_dist_matrix = dist_matrix[remaining_dets][:, remaining_trackers]
        remaining_indices = linear_assignment(remaining_dist_matrix)
        # Translating indices
        for det_idx, track_idx in remaining_indices:
            ordinary_matches.append([remaining_dets[det_idx], remaining_trackers[track_idx]])

    total_matched_indices = np.array(list(forced_matches.items()) + ordinary_matches).reshape(-1, 2)

    return total_matched_indices

def _filter_matches_by_distance(
        total_matched_indices: np.ndarray,
        dist_matrix: np.ndarray,
        dist_threshold: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    # Identify trackers and detections that were not matched at all
    unmatched_dets = []
    for d in range(dist_matrix.shape[0]):
        if d not in total_matched_indices[:, 0]:
            unmatched_dets.append(d)
    unmatched_trackers = []
    for t in range(dist_matrix.shape[1]):
        if t not in total_matched_indices[:, 1]:
            unmatched_trackers.append(t)

    # Filter out detections and trackers whose distance exceeds the accepted threshold
    matches = []
    for match in total_matched_indices:
        if dist_matrix[match[0], match[1]] > dist_threshold:
            unmatched_dets.append(match[0])
            unmatched_trackers.append(match[1])
        else:
            matches.append(match.reshape(1, 2))

    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_dets), np.array(unmatched_trackers)


def associate_detections_to_trackers(
        detections: np.ndarray,
        trackers: np.ndarray,
        dist_threshold: float,
        detection_ids: np.ndarray | None = None,
        track_ids: np.ndarray | None = None,
        association_mode: str = 'none',
        weight: float = 0.5 # the greater it is, the more cost decreases
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    # Explicit input validation checks
    if association_mode not in ('none', 'soft', 'hard'):
        raise ValueError(f'Unknown association mode: {association_mode}')
    if association_mode == 'soft' and not 0.0 <= weight <= 1.0:
        raise ValueError('weight must be between 0 and 1.')
    if detection_ids is not None:
        detection_ids = np.asarray(detection_ids)
        if len(detection_ids) != len(detections):
            raise ValueError('detection_ids must have the same length as detections.')
    if track_ids is not None:
        track_ids = np.asarray(track_ids)
        if len(track_ids) != len(trackers):
            raise ValueError('track_ids must have the same length as trackers.')
    # Early exit
    if len(detections) == 0 or len(trackers) == 0:
        return (np.empty((0, 2), dtype=int), np.arange(len(detections)), np.arange(len(trackers)))

    ids_present = detection_ids is not None and track_ids is not None

    # Compare only range and velocity
    dist_matrix = cdist(detections[:,:2], trackers[:,:2], 'euclidean')
    if association_mode == 'hard' and ids_present:
        total_matched_indices = _get_hard_identity_matches(
            detection_ids,
            track_ids,
            dist_matrix,
            dist_threshold
        )
    else:
        cost_matrix = dist_matrix.copy()
        if association_mode == 'soft' and ids_present:
            matching_ids = ((detection_ids[:, None] == track_ids[None, :]) & (detection_ids[:, None] >= 0))
            cost_matrix[matching_ids] *= (1.0 - weight)

        total_matched_indices = linear_assignment(cost_matrix)
    return _filter_matches_by_distance(total_matched_indices, dist_matrix, dist_threshold)


class KalmanBoxTracker:
    """
    This class represents the internal state of individual tracked objects observed as bbox.
    """
    def __init__(self, rv: np.ndarray, id: int, dt: float):
        """
        Initializes a tracker using initial bounding box.
        """
        self.kf = KalmanFilter(dim_x=4, dim_z=2)

        # State: [r, v, r', v']
        # where v is doppler velocity, r is range
        # v' is change in velocity (acceleration), r' change of real velocity (not radial/doppler velocity)

        self.kf.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])
        self.kf.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ])

        self.kf.R[2:,2:] *= 10.
        self.kf.P[2:,2:] *= 1000. #give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.

        self.kf.x[:2] = convert_rv_to_z(rv)
        self.time_since_update: int = 0
        self.id = id
        self.history = []
        self.hits: int = 0
        self.hit_streak: int = 0
        self.age: int = 0

        self.original_idx = rv[2]

    def update(self, rv):
        """
        Updates the state vector with observed bbox.
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.original_idx = rv[2]
        self.kf.update(convert_rv_to_z(rv))

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        self.kf.predict()
        self.age += 1

        if self.time_since_update > 0:
            self.hit_streak = 0

        self.time_since_update += 1
        self.history.append(convert_x_to_rv(self.kf.x))

        return self.history[-1]

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return convert_x_to_rv(self.kf.x)


class RDSort:
    def __init__(self, max_age: int = 1, min_hits: int = 3, dist_threshold: float = 5, fps: int = 1):
        """
        Sets key parameters for SORT
        """
        self.max_age: int = max_age
        self.min_hits: int = min_hits
        self.dist_threshold: float = dist_threshold
        self.fps: int = fps

        self.trackers: list[KalmanBoxTracker] = []
        self.frame_count: int = 0
        self.tracker_count: int = 0

    def update(self, 
        dets: np.ndarray = np.empty((0, 3)),
        detection_ids: np.ndarray | None = None,
        association_mode: str = 'none',
        weight: float = 0.5
    ) -> np.ndarray:
        """
        Params:
        dets - a numpy array of detections in the format [[r,v,idx],[r,v,idx],...]
        detection_ids - optional KF detection id hints, one per detection
        association_mode - association strategy: 'none', 'soft', or 'hard'
        weight - weight used for soft association
        
        Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 3)) for frames without detections).
        Returns tracking results in the format [[r,v,id,idx],...], where id is the KF track ID and idx is the original proposal index.

        NOTE: The number of objects returned may differ from the number of detections provided.
        """
        if detection_ids is not None:
            detection_ids = np.asarray(detection_ids)
            if len(detection_ids) != len(dets):
                raise ValueError('detection_ids must have the same length as dets.')
        
        self.frame_count += 1
        # get predicted locations from existing trackers.
        tracks = np.zeros((len(self.trackers), 3))
        to_del = []
        ret = []

        for t, track in enumerate(tracks):
            pos = self.trackers[t].predict()[0]
            track[:] = [pos[0], pos[1], 0]
            if np.any(~np.isfinite(pos)):
                to_del.append(t)

        tracks = np.ma.compress_rows(np.ma.masked_invalid(tracks))

        for t in reversed(to_del):
            self.trackers.pop(t)

        track_ids = np.array([track.id + 1 for track in self.trackers], dtype=int)
        matched, unmatched_dets, unmatched_tracks = associate_detections_to_trackers(
            dets,
            tracks,
            self.dist_threshold,
            detection_ids=detection_ids,
            track_ids=track_ids,
            association_mode=association_mode,
            weight=weight
        )

        # update matched trackers with assigned detections
        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :])

        # create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            track = KalmanBoxTracker(dets[i, :], self.tracker_count, dt=(1 / self.fps))
            self.tracker_count += 1
            self.trackers.append(track)

        i = len(self.trackers)
        for track in reversed(self.trackers):
            d = track.get_state()[0]

            if (track.time_since_update < 1) and ((track.hit_streak >= self.min_hits) or (self.frame_count <= self.min_hits)):
                ret.append(np.concatenate((d, [track.id + 1], [track.original_idx])).reshape(1, -1)) # +1 as MOT benchmark requires positive

            i -= 1
            # remove dead tracklet
            if track.time_since_update > self.max_age:
                self.trackers.pop(i)

        if len(ret) > 0:
            return np.concatenate(ret)

        return np.empty((0,4))

    def reset(self):
        self.trackers = []
        self.frame_count = 0
        self.tracker_count = 0
