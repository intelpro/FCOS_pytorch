import torch
from torch import nn

from detectron2.layers import ShapeSpec
from detectron2.modeling.proposal_generator.build import PROPOSAL_GENERATOR_REGISTRY
from detectron2.structures import Boxes, Instances
from fcos.layers import IOULoss, ml_nms
from typing import Dict

from .fcos_head import FCOSHead
from .fcos_losses import FCOSLosses
from .fcos_targets import FCOSTargets, get_points

__all__ = ["FCOS"]

INF = 100000000

"""
Shape shorthand in this module:
    N: number of images in the minibatch.
    Hi, Wi: height and width of the i-th level feature map.
    4: size of the box parameterization.
Naming convention:
    labels: refers to the ground-truth class of an position.
    reg_targets: refers to the 4-d (left, top, right, bottom) distances that parameterize the
        ground-truth box.
    logits_pred: predicted classification scores in [-inf, +inf];
    reg_pred: the predicted (left, top, right, bottom), corresponding to reg_targets
    ctrness_pred: predicted centerness scores
"""


@PROPOSAL_GENERATOR_REGISTRY.register()
class FCOS(nn.Module):
    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super().__init__()
        # fmt: off
        self.mask_on               = cfg.MODEL.MASK_ON
        self.num_classes           = cfg.MODEL.FCOS.NUM_CLASSES
        self.in_features           = cfg.MODEL.FCOS.IN_FEATURES
        self.fpn_strides           = cfg.MODEL.FCOS.FPN_STRIDES
        self.normalize_reg_targets = cfg.MODEL.FCOS.NORMALIZE_REG_TARGETS
        # inference parameters
        self.score_threshold       = cfg.MODEL.FCOS.SCORE_THRESH_TEST
        self.nms_pre_topk          = cfg.MODEL.FCOS.NMS_PRE_TOPK
        self.nms_threshold         = cfg.MODEL.FCOS.NMS_THRESH_TEST
        self.nms_post_topk         = cfg.MODEL.FCOS.NMS_POST_TOPK
        # fmt: on
        self.cfg = cfg
        self.fcos_head = FCOSHead(cfg, [input_shape[f] for f in self.in_features])

        reg_loss_type = cfg.MODEL.FCOS.LOC_LOSS_TYPE
        self.reg_loss = IOULoss(reg_loss_type)

    def forward(self, images, features, gt_instances):
        """
        Args:
            images (ImageList): input images of length `N`
            features (dict[str: Tensor]): input data as a mapping from feature
                map name to tensor. Axis 0 represents the number of images `N` in
                the input data; axes 1-3 are channels, height, and width, which may
                vary between feature maps (e.g., if a feature pyramid is used).
            gt_instances (list[Instances], optional): a length `N` list of `Instances`s.
                Each `Instances` stores ground-truth instances for the corresponding image.

        Returns:
            proposals: list[Instances]: contains fields "pred_boxes", "scores", "class_id",
            "locations".
            loss: dict[Tensor] or None
        """
        features = [features[f] for f in self.in_features]
        # Step 1. FCOS head implementation
        fcos_preds = self.fcos_head(features)
        all_level_points = get_points(features, self.fpn_strides)

        if self.training:
            # Step 2. training target generation
            training_targets = FCOSTargets(all_level_points, gt_instances, self.cfg)

            # Step 3. loss computation
            loss_inputs = fcos_preds + training_targets
            losses = FCOSLosses(*loss_inputs, self.reg_loss, self.cfg)
            if self.mask_on:  # Proposal generation for Instance Segmentation (ExtraExtra)
                # compute proposals for ROI sampling
                proposals = self.predict_proposals(
                    *fcos_preds,
                    all_level_points,
                    images.image_sizes,
                )
                return proposals, losses
            else:
                return None, losses

        # Step 4. Inference phase
        proposals = self.predict_proposals(*fcos_preds, all_level_points, images.image_sizes)
        return proposals, None

    def predict_proposals(
        self,
        cls_scores,
        bbox_preds,
        centernesses,
        all_level_points,
        image_sizes
    ):
        """
        Arguments:
            cls_scores, bbox_preds, centernesses: Same as the output of :meth:`FCOSHead.forward`
            all_level_points (list[Tensor]): list of #feature levels. Each entry contains
                tensor of size (Hi*Wi, 2), a set of point coordinates (xi, yi) of all feature map
                locations on 'feature level i' in image coordinate.
            image_sizes (List[torch.Size]): the input image sizes

        Returns:
            results (List[Instances]): a list of #images elements.
        """
        num_imgs = len(image_sizes)
        num_levels = len(cls_scores)

        # recall that during training, we normalize regression targets with FPN's stride.
        # we denormalize them here.
        if self.normalize_reg_targets:
            bbox_preds = [bbox_preds[i] * self.fpn_strides[i] for i in range(num_levels)]

        result_list = []
        for img_id in range(num_imgs):
            # each entry of list corresponds to per-level feature tensor of single image.
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_list = [
                bbox_preds[i][img_id].detach() for i in range(num_levels)
            ]
            centerness_pred_list = [
                centernesses[i][img_id].detach() for i in range(num_levels)
            ]

            # per-image proposal comutation
            det_bboxes = self.predict_proposals_single_image(
                cls_score_list,
                bbox_pred_list,
                centerness_pred_list,
                all_level_points,
                image_sizes[img_id]
            )
            result_list.append(det_bboxes)
        return result_list

    def predict_proposals_single_image(
        self,
        cls_scores,
        bbox_preds,
        centernesses,
        all_level_points,
        image_size
    ):

        assert len(cls_scores) == len(bbox_preds) == len(all_level_points)
        bboxes_list = [] 
        # Iterate over every feature level
        for (cls_score, bbox_pred, centerness, points) in zip(cls_scores, bbox_preds, centernesses, all_level_points):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            # (C, Hi, Wi) -> (Hi*Wi, C)
            scores = cls_score.permute(1, 2, 0).reshape(-1, self.num_classes).sigmoid()
            # (4, Hi, Wi) -> (Hi*Wi, 4)
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            # (1, Hi, Wi) -> (Hi*Wi, )
            centerness = centerness.permute(1, 2, 0).reshape(-1).sigmoid()
            max_scores, idx = (scores * centerness[:, None]).max(dim=1)
            if  max_scores.shape[0] > self.nms_pre_topk:
                nms_pre_topk = max_scores.shape[0]
                top_k_scores, topk_inds = max_scores.topk(nms_pre_topk)
                points = points[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]
                class_id = idx[topk_inds]
                bboxes = distance2bbox(points, bbox_pred, max_shape=image_size)
            else:
                nms_pre_topk = max_scores.shape[0]
                top_k_scores, topk_inds = max_scores.topk(nms_pre_topk)
                points = points[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]
                class_id = idx[topk_inds] 
                bboxes = distance2bbox(points, bbox_pred, max_shape=image_size)
            boxlist = Instances(image_size)
            boxlist = Instances(image_size)
            boxlist.pred_classes = class_id
            boxlist.pred_boxes = Boxes(bboxes)
            boxlist.scores = torch.sqrt(top_k_scores)
            bboxes_list.append(boxlist)
        bboxes_list = Instances.cat(bboxes_list)
        # non-maximum suppression per-image.
        results = ml_nms(
            bboxes_list,
            self.nms_threshold,
            # Limit to max_per_image detections **over all classes**
            max_proposals=self.nms_post_topk
        )
        return results

def distance2bbox(points, distance, max_shape=None):
    """Decode distance prediction to bounding box.
    Args:
        points (Tensor): Shape (n, 2), [x, y].
        distance (Tensor): Distance from the given point to 4
            boundaries (left, top, right, bottom).
        max_shape (tuple): Shape of the image.
    Returns:
        Tensor: Decoded bboxes.
    """
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    if max_shape is not None:
        x1 = x1.clamp(min=0, max=max_shape[1] - 1)
        y1 = y1.clamp(min=0, max=max_shape[0] - 1)
        x2 = x2.clamp(min=0, max=max_shape[1] - 1)
        y2 = y2.clamp(min=0, max=max_shape[0] - 1)
    return torch.stack([x1, y1, x2, y2], -1)
