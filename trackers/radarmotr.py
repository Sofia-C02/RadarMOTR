from argparse import Namespace
from copy import deepcopy

import torch
import numpy as np
from scipy.optimize import linear_sum_assignment

from trackers.tracker_base import Tracker
from models.radarmotr import RadarMOTR
from models.structures import Instances
from datasets.data_prefetcher import data_dict_to_device
from models import build_model
from util.tool import load_model


class RadarMOTRTracker(Tracker):
    def __init__(self, args: Namespace):
        super().__init__()

        self.device = torch.device(args.device)
        self.score_threshold = args.score_threshold
        self.area_threshold = args.area_threshold

        # New functionality
        self.proposal_assisted = args.proposal_assisted
        self.proposal_iou_threshold = args.proposal_iou_threshold
        self.proposals_normalized = args.proposals_normalized

        self.model = self.build_model(self.device, args)

        self.tracks: Instances = None

    def build_model(self, device, args: Namespace) -> RadarMOTR:
        model, _ = build_model(args)
        model = load_model(model, args.model_path)
        model.eval()

        model.to(device, non_blocking=True)

        return model

    @staticmethod
    def filter_by_score(dt_instances: Instances, score_threshold: float) -> Instances:
        keep = dt_instances.scores > score_threshold
        keep &= dt_instances.obj_idxes >= 0
        return dt_instances[keep]

    @staticmethod
    def filter_by_area(dt_instances: Instances, area_threshold: float) -> Instances:
        areas = dt_instances.boxes[:, 2] * dt_instances.boxes[:, 3]
        keep = areas > area_threshold
        return dt_instances[keep]

    def track_frame_with_proposal_ids(self, frame: int, data: tuple[torch.Tensor, dict[str, torch.Tensor]]):
        image, targets = data
        data = data_dict_to_device({'image': image, 'targets': targets.copy()}, self.device, non_blocking=True)
        image = data['image']
        targets = data['targets']

        proposals = targets['proposals']
        image_size = tuple(targets['orig_size'].tolist())

        if self.tracks is not None:
            self.tracks.remove('boxes')
            self.tracks.remove('labels')

        res = self.model.predict(image, image_size, self.tracks, proposals)
        self.tracks = res['track_instances']

        dt_instances = deepcopy(self.tracks)

        dt_instances = self.filter_by_score(dt_instances, self.score_threshold)
        dt_instances = self.filter_by_area(dt_instances, self.area_threshold)

        proposal_track_ids = self.get_proposal_track_ids(
            dt_instances,
            proposals,
            image_size
        )

        # Replace predicted boxes with original detection proposals
        if self.proposal_assisted:
            self.replace_predictions_with_proposals(
                dt_instances,
                proposals,
                image_size
        )

        bbox_xywh = dt_instances.boxes
        identities = dt_instances.obj_idxes

        return bbox_xywh, identities, proposal_track_ids

    def track_frame(self, frame: int, data: tuple[torch.Tensor, dict[str, torch.Tensor]]):
        bbox_xywh, identities, _ = self.track_frame_with_proposal_ids(frame, data)
        return bbox_xywh, identities

    @staticmethod
    def proposals_to_xywh(
        proposals: torch.Tensor,
        image_size: tuple[int, int],
        normalized: bool
    ) -> torch.Tensor:

        if len(proposals) == 0:
            return proposals.new_empty((0, 4))

        # Skip confidence dimension
        boxes = proposals[:, :4].clone()

        if normalized:
            img_h, img_w = image_size
            scale = boxes.new_tensor([
                img_w, img_h, img_w, img_h
            ])
            boxes *= scale

        cx = boxes[:, 0]
        cy = boxes[:, 1]
        w = boxes[:, 2]
        h = boxes[:, 3]

        x = cx - w / 2
        y = cy - h / 2
        
        return torch.column_stack([x, y, w, h])

    @staticmethod
    def calculate_iou(
        pred_boxes: torch.Tensor,
        orig_boxes: torch.Tensor
    ) -> torch.Tensor:

        if len(pred_boxes) == 0 or len(orig_boxes) == 0:
            return pred_boxes.new_zeros((len(pred_boxes), len(orig_boxes)))

        pred = pred_boxes[:, None, :]
        orig = orig_boxes[None, :, :]

        # Top-left corner
        left = torch.maximum(pred[..., 0], orig[..., 0])
        top = torch.maximum(pred[..., 1], orig[..., 1])

        # Bottom-right corner
        right = torch.minimum(
            pred[..., 0] + pred[..., 2],
            orig[..., 0] + orig[..., 2]
        )
        bottom = torch.minimum(
            pred[..., 1] + pred[..., 3],
            orig[..., 1] + orig[..., 3]
        )

        intersection_area = (
            torch.clamp(right - left, min=0) *
            torch.clamp(bottom - top, min=0)
        )
        pred_area = pred[..., 2] * pred[..., 3]
        orig_area = orig[..., 2] * orig[..., 3]
        union_area = pred_area + orig_area - intersection_area

        iou = torch.zeros_like(union_area)
        valid = union_area > 0
        iou[valid] = intersection_area[valid] / union_area[valid]

        return iou

    @staticmethod
    def match_predictions_to_proposals(
        pred_boxes: torch.Tensor,
        orig_boxes: torch.Tensor,
        iou_threshold: float
    ) -> list[tuple[int, int]]:

        if len(pred_boxes) == 0 or len(orig_boxes) == 0:
            return []

        iou = RadarMOTRTracker.calculate_iou(
            pred_boxes,
            orig_boxes
        )
        iou_np = iou.detach().cpu().numpy()

        cost_matrix = 1.0 - iou_np
        cost_matrix[iou_np < iou_threshold] = 1e6

        pred_indices, orig_indices = linear_sum_assignment(cost_matrix)

        valid_matches = [(int(pred_idx), int(orig_idx)) for pred_idx, orig_idx in zip(pred_indices, orig_indices)
            if iou_np[pred_idx, orig_idx] >= iou_threshold]

        return valid_matches

    def replace_predictions_with_proposals(
        self,
        dt_instances: Instances,
        proposals: torch.Tensor,
        image_size: tuple[int, int]
    ) -> None:

        if len(dt_instances) == 0 or len(proposals) == 0:
            return

        orig_boxes = self.proposals_to_xywh(
            proposals,
            image_size,
            self.proposals_normalized
        )
        pred_boxes = dt_instances.boxes

        valid_matches = self.match_predictions_to_proposals(
            pred_boxes,
            orig_boxes,
            self.proposal_iou_threshold
        )

        if not valid_matches:
            return

        pred_indices = torch.as_tensor([pred_idx for (pred_idx, _) in valid_matches], dtype=torch.long, device=pred_boxes.device)
        orig_indices = torch.as_tensor([orig_idx for (_, orig_idx) in valid_matches], dtype=torch.long, device=orig_boxes.device)

        updated_boxes = pred_boxes.clone()
        updated_boxes[pred_indices] = orig_boxes[orig_indices]

        dt_instances.boxes = updated_boxes

    def get_proposal_track_ids(
        self,
        dt_instances: Instances,
        proposals: torch.Tensor,
        image_size: tuple[int, int]
    ) -> np.ndarray:

        proposal_track_ids = np.full(len(proposals), -1, dtype=np.int64)

        if len(dt_instances) == 0 or len(proposals) == 0:
            return proposal_track_ids

        orig_boxes = self.proposals_to_xywh(
            proposals,
            image_size,
            self.proposals_normalized
        )

        valid_matches = self.match_predictions_to_proposals(
            dt_instances.boxes,
            orig_boxes,
            self.proposal_iou_threshold
        )

        if not valid_matches:
            return proposal_track_ids

        pred_indices = torch.as_tensor([pred_idx for pred_idx, _ in valid_matches], dtype=torch.long, device=dt_instances.obj_idxes.device)
        proposal_indices = np.array([orig_idx for _, orig_idx in valid_matches], dtype=np.int64)

        # neural_ids -> preds -> proposals
        neural_ids = dt_instances.obj_idxes[pred_indices].detach().cpu().numpy()
        proposal_track_ids[proposal_indices] = neural_ids

        return proposal_track_ids
    
    def reset(self):
        self.model.reset()
        self.tracks = None

