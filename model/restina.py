
from .head.restinahead import RetinaNetHead, RetinaNetClassificationHead, RetinaNetRegressionHead

import math
from collections import OrderedDict
import warnings

import torch
import torch.nn as nn
from torch import Tensor
from torch.jit.annotations import Dict, List, Tuple

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

from utils import boxes as det_utils


from utils.anchor import AnchorGenerator
from transform.generalizedRCNNtransform import GeneralizedRCNNTransform

from torchvision.ops import boxes as box_ops



class RetinaNet(nn.Module):
    """
    Implements RetinaNet.
    The input to the model is expected to be a list of tensors, each of shape [C, H, W], one for each
    image, and should be in 0-1 range. Different images can have different sizes.
    The behavior of the model changes depending if it is in training or evaluation mode.
    During training, the model expects both the input tensors, as well as a targets (list of dictionary),
    containing:
        - boxes (FloatTensor[N, 4]): the ground-truth boxes in [x1, y1, x2, y2] format, with values
          between 0 and H and 0 and W
        - labels (Int64Tensor[N]): the class label for each ground-truth box
    The model returns a Dict[Tensor] during training, containing the classification and regression
    losses.
    During inference, the model requires only the input tensors, and returns the post-processed
    predictions as a List[Dict[Tensor]], one for each input image. The fields of the Dict are as
    follows:
        - boxes (FloatTensor[N, 4]): the predicted boxes in [x1, y1, x2, y2] format, with values between
          0 and H and 0 and W
        - labels (Int64Tensor[N]): the predicted labels for each image
        - scores (Tensor[N]): the scores for each prediction
    Arguments:
        backbone (nn.Module): the network used to compute the features for the model.
            It should contain an out_channels attribute, which indicates the number of output
            channels that each feature map has (and it should be the same for all feature maps).
            The backbone should return a single Tensor or an OrderedDict[Tensor].
        num_classes (int): number of output classes of the model (excluding the background).
        min_size (int): minimum size of the image to be rescaled before feeding it to the backbone
        max_size (int): maximum size of the image to be rescaled before feeding it to the backbone
        image_mean (Tuple[float, float, float]): mean values used for input normalization.
            They are generally the mean values of the dataset on which the backbone has been trained
            on
        image_std (Tuple[float, float, float]): std values used for input normalization.
            They are generally the std values of the dataset on which the backbone has been trained on
        anchor_generator (AnchorGenerator): module that generates the anchors for a set of feature
            maps.
        head (nn.Module): Module run on top of the feature pyramid.
            Defaults to a module containing a classification and regression module.
        score_thresh (float): Score threshold used for postprocessing the detections.
        nms_thresh (float): NMS threshold used for postprocessing the detections.
        detections_per_img (int): Number of best detections to keep after NMS.
        fg_iou_thresh (float): minimum IoU between the anchor and the GT box so that they can be
            considered as positive during training.
        bg_iou_thresh (float): maximum IoU between the anchor and the GT box so that they can be
            considered as negative during training.
    Example:
        >>> import torch
        >>> import torchvision
        >>> from torchvision.models.detection import RetinaNet
        >>> from torchvision.models.detection.anchor_utils import AnchorGenerator
        >>> # load a pre-trained model for classification and return
        >>> # only the features
        >>> backbone = torchvision.models.mobilenet_v2(pretrained=True).features
        >>> # RetinaNet needs to know the number of
        >>> # output channels in a backbone. For mobilenet_v2, it's 1280
        >>> # so we need to add it here
        >>> backbone.out_channels = 1280
        >>>
        >>> # let's make the network generate 5 x 3 anchors per spatial
        >>> # location, with 5 different sizes and 3 different aspect
        >>> # ratios. We have a Tuple[Tuple[int]] because each feature
        >>> # map could potentially have different sizes and
        >>> # aspect ratios
        >>> anchor_generator = AnchorGenerator(
        >>>     sizes=tuple((x, int(x * 2 ** (1.0 / 3)), int(x * 2 ** (2.0 / 3))) for x in [32, 64, 128, 256, 512]),
        >>>     aspect_ratios=((0.5, 1.0, 2.0),) * 5
        >>> )
        >>>
        >>> # put the pieces together inside a RetinaNet model
        >>> model = RetinaNet(backbone,
        >>>                   num_classes=2,
        >>>                   anchor_generator=anchor_generator)
        >>> model.eval()
        >>> x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
        >>> predictions = model(x)
    """
    __annotations__ = {
        'box_coder': det_utils.BoxCoder,
        'proposal_matcher': det_utils.Matcher,
    }

    def __init__(self, backbone, num_classes,
                 # transform parameters
                 min_size=800, max_size=1333,
                 image_mean=None, image_std=None,
                 # Anchor parameters
                 anchor_generator=None, head=None,
                 proposal_matcher=None,
                 score_thresh=0.05,
                 nms_thresh=0.5,
                 detections_per_img=300,
                 fg_iou_thresh=0.5, bg_iou_thresh=0.4):
        super().__init__()

        if not hasattr(backbone, "out_channels"):
            raise ValueError(
                "backbone should contain an attribute out_channels "
                "specifying the number of output channels (assumed to be the "
                "same for all the levels)")
        self.backbone = backbone

        assert isinstance(anchor_generator, (AnchorGenerator, type(None)))

        if anchor_generator is None:
            anchor_sizes = tuple((x, int(x * 2 ** (1.0 / 3)), int(x * 2 ** (2.0 / 3))) for x in [32, 64, 128, 256, 512])
            aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
            anchor_generator = AnchorGenerator(
                anchor_sizes, aspect_ratios
            )
        self.anchor_generator = anchor_generator

        if head is None:
            head = RetinaNetHead(backbone.out_channels, anchor_generator.num_anchors_per_location()[0], num_classes)
        self.head = head

        if proposal_matcher is None:
            proposal_matcher = det_utils.Matcher(
                fg_iou_thresh,
                bg_iou_thresh,
                allow_low_quality_matches=True,
            )
        self.proposal_matcher = proposal_matcher

        self.box_coder = det_utils.BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))

        if image_mean is None:
            image_mean = [0.485, 0.456, 0.406]
        if image_std is None:
            image_std = [0.229, 0.224, 0.225]

        self.transform = GeneralizedRCNNTransform(min_size, max_size, image_mean, image_std)

        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.detections_per_img = detections_per_img

        # used only on torchscript mode
        self._has_warned = False

    @torch.jit.unused
    def eager_outputs(self, losses, detections):
        # type: (Dict[str, Tensor], List[Dict[str, Tensor]]) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]
        if self.training:
            return losses

        return detections

    def compute_loss(self, targets, head_outputs, anchors):
        # type: (List[Dict[str, Tensor]], Dict[str, Tensor], List[Tensor]) -> Dict[str, Tensor]
        matched_idxs = []
        for anchors_per_image, targets_per_image in zip(anchors, targets):
            if targets_per_image['boxes'].numel() == 0:
                matched_idxs.append(torch.empty((0,), dtype=torch.int32))
                continue

            match_quality_matrix = box_ops.box_iou(targets_per_image['boxes'], anchors_per_image)
            matched_idxs.append(self.proposal_matcher(match_quality_matrix))

        return self.head.compute_loss(targets, head_outputs, anchors, matched_idxs)

    def postprocess_detections(self, head_outputs, anchors, image_shapes):
        # type: (Dict[str, Tensor], List[Tensor], List[Tuple[int, int]]) -> List[Dict[str, Tensor]]
        # TODO: Merge this with roi_heads.RoIHeads.postprocess_detections ?

        class_logits = head_outputs.pop('cls_logits')
        box_regression = head_outputs.pop('bbox_regression')
        other_outputs = head_outputs

        device = class_logits.device
        num_classes = class_logits.shape[-1]

        scores = torch.sigmoid(class_logits)

        # create labels for each score
        labels = torch.arange(num_classes, device=device)
        labels = labels.view(1, -1).expand_as(scores)

        detections = torch.jit.annotate(List[Dict[str, Tensor]], [])

        for index, (box_regression_per_image, scores_per_image, labels_per_image, anchors_per_image, image_shape) in \
                enumerate(zip(box_regression, scores, labels, anchors, image_shapes)):

            boxes_per_image = self.box_coder.decode_single(box_regression_per_image, anchors_per_image)
            boxes_per_image = box_ops.clip_boxes_to_image(boxes_per_image, image_shape)

            other_outputs_per_image = [(k, v[index]) for k, v in other_outputs.items()]

            image_boxes = []
            image_scores = []
            image_labels = []
            image_other_outputs = torch.jit.annotate(Dict[str, List[Tensor]], {})

            for class_index in range(num_classes):
                # remove low scoring boxes
                inds = torch.gt(scores_per_image[:, class_index], self.score_thresh)
                boxes_per_class, scores_per_class, labels_per_class = \
                    boxes_per_image[inds], scores_per_image[inds, class_index], labels_per_image[inds, class_index]
                other_outputs_per_class = [(k, v[inds]) for k, v in other_outputs_per_image]

                # remove empty boxes
                keep = box_ops.remove_small_boxes(boxes_per_class, min_size=1e-2)
                boxes_per_class, scores_per_class, labels_per_class = \
                    boxes_per_class[keep], scores_per_class[keep], labels_per_class[keep]
                other_outputs_per_class = [(k, v[keep]) for k, v in other_outputs_per_class]

                # non-maximum suppression, independently done per class
                keep = box_ops.nms(boxes_per_class, scores_per_class, self.nms_thresh)

                # keep only topk scoring predictions
                keep = keep[:self.detections_per_img]
                boxes_per_class, scores_per_class, labels_per_class = \
                    boxes_per_class[keep], scores_per_class[keep], labels_per_class[keep]
                other_outputs_per_class = [(k, v[keep]) for k, v in other_outputs_per_class]

                image_boxes.append(boxes_per_class)
                image_scores.append(scores_per_class)
                image_labels.append(labels_per_class)

                for k, v in other_outputs_per_class:
                    if k not in image_other_outputs:
                        image_other_outputs[k] = []
                    image_other_outputs[k].append(v)

            detections.append({
                'boxes': torch.cat(image_boxes, dim=0),
                'scores': torch.cat(image_scores, dim=0),
                'labels': torch.cat(image_labels, dim=0),
            })

            for k, v in image_other_outputs.items():
                detections[-1].update({k: torch.cat(v, dim=0)})

        return detections

    def forward(self, images, targets=None):
        # type: (List[Tensor], Optional[List[Dict[str, Tensor]]]) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]
        """
        Arguments:
            images (list[Tensor]): images to be processed
            targets (list[Dict[Tensor]]): ground-truth boxes present in the image (optional)
        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).
        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")

        # get the original image sizes
        original_image_sizes = torch.jit.annotate(List[Tuple[int, int]], [])
        for img in images:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))

        # transform the input
        images, targets = self.transform(images, targets)

        # get the features from the backbone
        features = self.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([('0', features)])

        # TODO: Do we want a list or a dict?
        features = list(features.values())

        # TODO: Is there a better way to check for [P3, P4, P5, P6, P7]?
        if len(features) == 6:
            # skip P2 because it generates too many anchors (according to their paper)
            features = features[1:]

        # compute the retinanet heads outputs using the features
        head_outputs = self.head(features)

        # create the set of anchors
        anchors = self.anchor_generator(images, features)

        losses = {}
        detections = torch.jit.annotate(List[Dict[str, Tensor]], [])
        if self.training:
            assert targets is not None

            # compute the losses
            losses = self.compute_loss(targets, head_outputs, anchors)
        else:
            # compute the detections
            detections = self.postprocess_detections(head_outputs, anchors, images.image_sizes)
            detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)

        if torch.jit.is_scripting():
            if not self._has_warned:
                warnings.warn("RetinaNet always returns a (Losses, Detections) tuple in scripting")
                self._has_warned = True
            return (losses, detections)
        return self.eager_outputs(losses, detections)