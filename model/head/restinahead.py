import math
from collections import OrderedDict
import warnings

import torch
import torch.nn as nn
from torch import Tensor
from torch.jit.annotations import Dict, List, Tuple


from utils import boxes as det_utils
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

from utils.anchor import AnchorGenerator

from torchvision.ops import boxes as box_ops



from torchvision.ops import boxes as box_ops



from loss.focal import sigmoid_focal_loss


class RetinaNetClassificationHead(nn.Module):
    def __init__(self,in_channels, num_anchors, num_classes, prior_probability=0.01):
        super().__init__()
        conv = []
        for _ in range(4):
            conv.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1))
            conv.append(nn.ReLU())
        self.conv = nn.Sequential(*conv)

        for layer in self.conv.children():
            if isinstance(layer, nn.Conv2d):
                torch.nn.init.normal_(layer.weight, std=0.01)
                torch.nn.init.constant_(layer.bias, 0)

        self.cls_logits = nn.Conv2d(in_channels, num_anchors * num_classes, kernel_size=3, stride=1, padding=1)
        torch.nn.init.normal_(self.cls_logits.weight, std=0.01)
        torch.nn.init.constant_(self.cls_logits.bias, -math.log((1 - prior_probability) / prior_probability))

        self.num_classes = num_classes
        self.num_anchors = num_anchors

        # This is to fix using det_utils.Matcher.BETWEEN_THRESHOLDS in TorchScript.
        # TorchScript doesn't support class attributes.
        # https://github.com/pytorch/vision/pull/1697#issuecomment-630255584
        self.BETWEEN_THRESHOLDS = det_utils.Matcher.BETWEEN_THRESHOLDS

    
    def compute_loss(self, targets, head_outputs, matched_idxs):
        # type: (List[Dict[str, Tensor]], Dict[str, Tensor], List[Tensor]) -> Tensor
        loss = torch.tensor(0.0)

        cls_logits = head_outputs['cls_logits']

        for targets_per_image, cls_logits_per_image, matched_idxs_per_image in zip(targets, cls_logits, matched_idxs):
            # no matched_idxs means there were no annotations in this image
            if matched_idxs_per_image.numel() == 0:
                gt_classes_target = torch.zeros_like(cls_logits_per_image)
                valid_idxs_per_image = torch.arange(cls_logits_per_image.shape[0])
                num_foreground = torch.tensor(0.0)
            else:
                # determine only the foreground
                foreground_idxs_per_image = matched_idxs_per_image >= 0
                num_foreground = foreground_idxs_per_image.sum()

                # create the target classification
                gt_classes_target = torch.zeros_like(cls_logits_per_image)
                gt_classes_target[
                    foreground_idxs_per_image,
                    targets_per_image['labels'][matched_idxs_per_image[foreground_idxs_per_image]]
                ] = torch.tensor(1.0)

                # find indices for which anchors should be ignored
                valid_idxs_per_image = matched_idxs_per_image != self.BETWEEN_THRESHOLDS

            # compute the classification loss
            loss += sigmoid_focal_loss(
                cls_logits_per_image[valid_idxs_per_image],
                gt_classes_target[valid_idxs_per_image],
                reduction='sum',
            ) / max(1, num_foreground)

        return loss / len(targets)

    def forward(self, x):
        # type: (List[Tensor]) -> Tensor
        all_cls_logits = []

        for features in x:
            cls_logits = self.conv(features)
            cls_logits = self.cls_logits(cls_logits)

            # Permute classification output from (N, A * K, H, W) to (N, HWA, K).
            N, _, H, W = cls_logits.shape
            cls_logits = cls_logits.view(N, -1, self.num_classes, H, W)
            cls_logits = cls_logits.permute(0, 3, 4, 1, 2)
            cls_logits = cls_logits.reshape(N, -1, self.num_classes)  # Size=(N, HWA, 4)

            all_cls_logits.append(cls_logits)

        return torch.cat(all_cls_logits, dim=1)


class RetinaNetRegressionHead(nn.Module):
    """
    A regression head for use in RetinaNet.
    Arguments:
        in_channels (int): number of channels of the input feature
        num_anchors (int): number of anchors to be predicted
    """
    __annotations__ = {
        'box_coder': det_utils.BoxCoder,
    }

    def __init__(self, in_channels, num_anchors):
        super().__init__()

        conv = []
        for _ in range(4):
            conv.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1))
            conv.append(nn.ReLU())
        self.conv = nn.Sequential(*conv)

        self.bbox_reg = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=3, stride=1, padding=1)
        torch.nn.init.normal_(self.bbox_reg.weight, std=0.01)
        torch.nn.init.zeros_(self.bbox_reg.bias)

        for layer in self.conv.children():
            if isinstance(layer, nn.Conv2d):
                torch.nn.init.normal_(layer.weight, std=0.01)
                torch.nn.init.zeros_(layer.bias)

        self.box_coder = det_utils.BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))

    def compute_loss(self, targets, head_outputs, anchors, matched_idxs):
        # type: (List[Dict[str, Tensor]], Dict[str, Tensor], List[Tensor], List[Tensor]) -> Tensor
        loss = torch.tensor(0.0)

        bbox_regression = head_outputs['bbox_regression']

        for targets_per_image, bbox_regression_per_image, anchors_per_image, matched_idxs_per_image in \
                zip(targets, bbox_regression, anchors, matched_idxs):
            # no matched_idxs means there were no annotations in this image
            if matched_idxs_per_image.numel() == 0:
                continue

            # get the targets corresponding GT for each proposal
            # NB: need to clamp the indices because we can have a single
            # GT in the image, and matched_idxs can be -2, which goes
            # out of bounds
            matched_gt_boxes_per_image = targets_per_image['boxes'][matched_idxs_per_image.clamp(min=0)]

            # determine only the foreground indices, ignore the rest
            foreground_idxs_per_image = matched_idxs_per_image >= 0
            num_foreground = foreground_idxs_per_image.sum()

            # select only the foreground boxes
            matched_gt_boxes_per_image = matched_gt_boxes_per_image[foreground_idxs_per_image, :]
            bbox_regression_per_image = bbox_regression_per_image[foreground_idxs_per_image, :]
            anchors_per_image = anchors_per_image[foreground_idxs_per_image, :]

            # compute the regression targets
            target_regression = self.box_coder.encode_single(matched_gt_boxes_per_image, anchors_per_image)

            # compute the loss
            loss += det_utils.smooth_l1_loss(
                bbox_regression_per_image,
                target_regression,
                size_average=False
            ) / max(1, num_foreground)

        return loss / max(1, len(targets))

    def forward(self, x):
        # type: (List[Tensor]) -> Tensor
        all_bbox_regression = []

        for features in x:
            bbox_regression = self.conv(features)
            bbox_regression = self.bbox_reg(bbox_regression)

            # Permute bbox regression output from (N, 4 * A, H, W) to (N, HWA, 4).
            N, _, H, W = bbox_regression.shape
            bbox_regression = bbox_regression.view(N, -1, 4, H, W)
            bbox_regression = bbox_regression.permute(0, 3, 4, 1, 2)
            bbox_regression = bbox_regression.reshape(N, -1, 4)  # Size=(N, HWA, 4)

            all_bbox_regression.append(bbox_regression)

        return torch.cat(all_bbox_regression, dim=1)


class RetinaNetHead(nn.Module):
    """
    A regression and classification head for use in RetinaNet.
    Arguments:
        in_channels (int): number of channels of the input feature
        num_anchors (int): number of anchors to be predicted
        num_classes (int): number of classes to be predicted
    """

    def __init__(self, in_channels, num_anchors, num_classes):
        super().__init__()
        self.classification_head = RetinaNetClassificationHead(in_channels, num_anchors, num_classes)
        self.regression_head = RetinaNetRegressionHead(in_channels, num_anchors)

    def compute_loss(self, targets, head_outputs, anchors, matched_idxs):
        # type: (List[Dict[str, Tensor]], Dict[str, Tensor], List[Tensor], List[Tensor]) -> Dict[str, Tensor]
        return {
            'classification': self.classification_head.compute_loss(targets, head_outputs, matched_idxs),
            'bbox_regression': self.regression_head.compute_loss(targets, head_outputs, anchors, matched_idxs),
        }

    def forward(self, x):
        # type: (List[Tensor]) -> Dict[str, Tensor]
        return {
            'cls_logits': self.classification_head(x),
            'bbox_regression': self.regression_head(x)
        }


