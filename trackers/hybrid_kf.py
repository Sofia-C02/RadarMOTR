from argparse import Namespace

import torch
import numpy as np

from trackers.kf import KFTracker
from trackers.radarmotr import RadarMOTRTracker
from trackers.tracker_base import Tracker


class HybridKFTracker(Tracker):
    def __init__(self, args: Namespace):
        super().__init__()

        self.kf_tracker = KFTracker(args)
        self.radarmotr_tracker = RadarMOTRTracker(args)

        self.association_mode = args.association_mode
        self.weight = args.weight

        # Map track IDs (RadarMOTR <-> KF)
        self.motr_to_kf: dict[int, int] = {}

    def map_motr_ids_to_kf_ids(self, motr_ids: np.ndarray) -> np.ndarray | None:
        if not self.motr_to_kf:
            return None

        kf_ids = np.full(len(motr_ids), -1)

        for i, motr_id in enumerate(motr_ids):
            if motr_id >= 0:
                kf_ids[i] = self.motr_to_kf.get(int(motr_id), -1)

        return kf_ids

    def update_id_mapping(self, motr_ids, kf_ids):
        if len(motr_ids) != len(kf_ids):
            raise ValueError('RadarMOTR and KF proposal ID arrays must have equal length.')

        for motr_id, kf_id in zip(motr_ids, kf_ids):
            if motr_id < 0 or kf_id < 0:
                continue

            motr_id = int(motr_id)
            kf_id = int(kf_id)

            # Only change mapping if the ids are already not in the dict
            # This is a choice, perhaps some other policy could be used
            if motr_id not in self.motr_to_kf:
                self.motr_to_kf[motr_id] = kf_id
                
    def track_frame(
        self,
        frame: int,
        data: tuple[torch.Tensor, dict[str, torch.Tensor]]
    ):
        image, targets = data

        kf_targets = targets.copy()
        kf_targets['proposals'] = targets['proposals'].clone()

        # KF preprocessing
        img_h, img_w = kf_targets['orig_size'].tolist()
        scale = kf_targets['proposals'].new_tensor([img_w, img_h, img_w, img_h])
        kf_targets['proposals'][:, :4] *= scale
        kf_data = (image, kf_targets)

        _, _, motr_ids = self.radarmotr_tracker.track_frame_with_proposal_ids(frame, data)
        # RadarMOTR IDs to KF IDs
        detection_ids = self.map_motr_ids_to_kf_ids(np.asarray(motr_ids, dtype=int))

        bbox_xywh, identities, kf_proposal_ids = self.kf_tracker.track_frame_with_proposal_ids(
            frame,
            kf_data,
            detection_ids,
            self.association_mode,
            self.weight
        )
        self.update_id_mapping(np.asarray(motr_ids, dtype=int), np.asarray(kf_proposal_ids, dtype=int))

        return bbox_xywh, identities

    def reset(self):
        self.kf_tracker.reset()
        self.radarmotr_tracker.reset()
        self.motr_to_kf.clear()