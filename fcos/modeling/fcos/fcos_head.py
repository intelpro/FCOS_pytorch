import math
import torch.nn as nn
from torch.nn import functional as F
import torch as th
from detectron2.layers import Conv2d, DeformConv, ShapeSpec
from fcos.layers import Scale, normal_init
from typing import List


class FCOSHead(nn.Module):
    """
    Fully Convolutional One-Stage Object Detection head from [1]_.

    The FCOS head does not use anchor boxes. Instead bounding boxes are
    predicted at each pixel and a centerness measure is used to supress
    low-quality predictions.

    References:
        .. [1] https://arxiv.org/abs/1904.01355

    In our Implementation, schemetic structure is as following:

                                    /-> logits
                    /-> cls convs ->
                   /                \-> centerness
    shared convs ->
                    \-> reg convs -> regressions
    """
    def __init__(self, cfg, input_shape: List[ShapeSpec]):
        super().__init__()
        # fmt: off
        self.in_channels       = input_shape[0].channels
        self.num_classes       = cfg.MODEL.FCOS.NUM_CLASSES
        self.fpn_strides       = cfg.MODEL.FCOS.FPN_STRIDES
        self.num_shared_convs  = cfg.MODEL.FCOS.NUM_SHARED_CONVS
        self.num_stacked_convs = cfg.MODEL.FCOS.NUM_STACKED_CONVS
        self.prior_prob        = cfg.MODEL.FCOS.PRIOR_PROB
        self.use_deformable    = cfg.MODEL.FCOS.USE_DEFORMABLE
        self.norm_layer        = cfg.MODEL.FCOS.NORM
        self.ctr_on_reg        = cfg.MODEL.FCOS.CTR_ON_REG
        # fmt: on

        self._init_layers()
        self._init_weights()

    def _init_layers(self):
        """
        Initializes six convolutional layers for FCOS head and a scaling layer for bbox predictions.
        """
        activation = nn.ReLU()

        """ your code starts here """
        conv_stack1, conv_stack2, conv_stack3 = [], [], []
        if self.use_deformable:
            conv_func = DeformConv
        else:
            conv_func = nn.Conv2d
        for i in range(self.num_stacked_convs):
            conv_stack1.append(conv_func(self.in_channels, self.in_channels, kernel_size=3, stride=1, padding=1, bias=True))
            conv_stack2.append(conv_func(self.in_channels, self.in_channels, kernel_size=3, stride=1, padding=1, bias=True))
            conv_stack3.append(conv_func(self.in_channels, self.in_channels, kernel_size=3, stride=1, padding=1, bias=True))
            if self.norm_layer=='GN': 
                conv_stack1.append(nn.GroupNorm(32, self.in_channels))
                conv_stack2.append(nn.GroupNorm(32, self.in_channels))
                conv_stack3.append(nn.GroupNorm(32, self.in_channels))
            conv_stack1.append(activation)
            conv_stack2.append(activation)
            conv_stack3.append(activation)
        self.shared_convs = nn.Sequential(*conv_stack1)
        self.cls_convs = nn.Sequential(*conv_stack2)
        self.reg_convs = nn.Sequential(*conv_stack3)
        self.cls_logits = nn.Conv2d(self.in_channels, self.num_classes, kernel_size=3, padding=1)
        self.bbox_pred = nn.Conv2d(self.in_channels, 4, kernel_size=3, padding=1)
        self.centerness = nn.Conv2d(self.in_channels, 1, kernel_size=3, padding=1)
        self.scales = nn.ModuleList([Scale(1.0) for _ in self.fpn_strides])
        """ your code ends here """


    def _init_weights(self):
        for modules in [
            self.shared_convs, self.cls_convs, self.reg_convs,
            self.cls_logits, self.bbox_pred, self.centerness
        ]:
            # weight initialization with mean=0, std=0.01
            pass

        # initialize the bias for classification logits
        bias_cls = -math.log((1-self.prior_prob)/self.prior_prob)# calculate proper value that makes cls_probability with `self.prior_prob`
        # In other words, make the initial 'sigmoid' activation of cls_logits as `self.prior_prob`
        # by controlling bias initialization
        nn.init.constant_(self.cls_logits.bias, bias_cls)

    def forward(self, features):
        """
        Arguments:
            features (list[Tensor]): FPN feature map tensors in high to low resolution.
                Each tensor in the list correspond to different feature levels.

        Returns:
            cls_scores (list[Tensor]): list of #feature levels, each has shape (N, C, Hi, Wi).
                The tensor predicts the classification probability
                at each spatial position for each of C object classes.
            bbox_preds (list[Tensor]): list of #feature levels, each has shape (N, 4, Hi, Wi).
                The tensor predicts 4-vector (l, t, r, b) box regression values for
                every position of featur map. These values are the distances from
                a specific point to each (left, top, right, bottom) edge
                of the corresponding ground truth box that the point belongs to.
            centernesses (list[Tensor]): list of #feature levels, each has shape (N, 1, Hi, Wi).
                The tensor predicts the centerness logits, where these values used to
                downweight the bounding box scores far from the center of an object.
        """
        cls_scores = []
        bbox_preds = []
        centernesses = []
        for feat_level, feature in enumerate(features):
            """ your code starts here """
            convs_feat = self.shared_convs(feature)
            cls_output = self.cls_convs(convs_feat)
            # find logit
            cls_scores.append(self.cls_logits(cls_output))
            # find centerness
            centernesses.append(self.centerness(cls_output))
            # find bbox pred
            reg_feat = self.reg_convs(convs_feat)
            bbox_pred = th.exp(self.scales[feat_level](self.bbox_pred(reg_feat)))
            bbox_preds.append(bbox_pred)
            import pdb; pdb.set_trace()
            """ your code ends here """
        return cls_scores, bbox_preds, centernesses
