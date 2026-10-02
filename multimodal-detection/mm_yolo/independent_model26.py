"""YOLO26m-P2 detector with the V4.4 three-modality fusion path.

The original IndependentMMYOLO remains the YOLO11 checkpoint implementation.
This class replaces only its detector family after building the shared fusion
modules, so old checkpoints retain their module names and forward behavior.
"""
from __future__ import annotations

import copy
from pathlib import Path

import torch
from torch import nn

from ultralytics import YOLO
from ultralytics.nn.modules import Detect
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import YAML

from independent_model import IndependentMMYOLO, SCALES
from model import MMYOLO


class SpatialDetect26(Detect):
    """Native YOLO26 dual head with V4.4's localization-only geometry path.

    Validation uses the one-to-many branch and NMS so its scoring protocol is
    comparable to V4.4. Both native branches are optimized during training.
    """

    def _split_head(self, features, localization, box_head, cls_head):
        batch = features[0].shape[0]
        return {
            "boxes": torch.cat(
                [box_head[i](localization[i]).view(batch, 4 * self.reg_max, -1)
                 for i in range(self.nl)], -1),
            "scores": torch.cat(
                [cls_head[i](features[i]).view(batch, self.nc, -1)
                 for i in range(self.nl)], -1),
            "feats": features,
        }

    def forward(self, features, localization=None):
        localization = features if localization is None else localization
        one2many = self._split_head(features, localization, **self.one2many)
        if self.end2end:
            cls_features = [x.detach() for x in features] if self.training else features
            loc_features = [x.detach() for x in localization] if self.training else localization
            one2one = self._split_head(cls_features, loc_features, **self.one2one)
            preds = {"one2many": one2many, "one2one": one2one}
        else:
            preds = one2many
        if self.training:
            return preds
        # This project's validation/submission path applies class-aware NMS to
        # (B, 4+nc, anchors) xywh predictions. Decode with that same contract.
        end2end = self.end2end
        self.end2end = False
        try:
            decoded = self._inference(one2many)
        finally:
            self.end2end = end2end
        return decoded if self.export else (decoded, preds)


def _run_p2_neck(layers, raw, p2_hook=None):
    """Execute the YOLO26 P2 neck with fused or single-modality backbone maps."""
    saved = {2: raw["p2"], 4: raw["p3"], 6: raw["p4"], 10: raw["p5"]}
    value = raw["p5"]
    for index in range(11, 29):
        layer = layers[index - 11]
        source = layer.f
        if isinstance(source, (list, tuple)):
            inputs = [value if item == -1 else saved[item] for item in source]
        elif source == -1:
            inputs = value
        else:
            inputs = saved[source]
        value = layer(inputs)
        if index == 19 and p2_hook is not None:
            value = p2_hook(value)
        saved[index] = value
    return [saved[index] for index in (19, 22, 25, 28)]


class IndependentBranchDetector26(nn.Module):
    """Training-only native YOLO26m-P2 neck and detector for one sensor."""

    def __init__(self, backbone):
        super().__init__()
        self.neck = nn.ModuleList(copy.deepcopy(list(backbone.model[11:29])))
        self.detector = copy.deepcopy(backbone.model[-1])

    def forward(self, evidence):
        return self.detector(_run_p2_neck(self.neck, evidence))


def _pretrained_p2(cfg):
    """Transfer same-shaped YOLO26m weights, including shifted P3-P5 neck/head."""
    source = YOLO(cfg.resolve_weights()).model.float()
    yaml_path = (Path(__file__).resolve().parents[1] / "vendor" / "ultralytics" /
                 "cfg" / "models" / "26" / "yolo26-p2.yaml")
    description = YAML.load(str(yaml_path))
    description["scale"] = "m"
    target = DetectionModel(description, ch=3, nc=int(source.model[-1].nc), verbose=False).float()
    if len(source.model) != 24 or len(target.model) != 30:
        raise RuntimeError("unexpected YOLO26m/P2 layer layout; refusing partial transfer")
    state = target.state_dict()
    matched, transferred = 0, 0
    for key, value in source.state_dict().items():
        pieces = key.split(".")
        if len(pieces) < 3 or pieces[0] != "model":
            continue
        layer = int(pieces[1])
        if layer <= 16:
            target_key = key
        elif layer <= 22:
            target_key = f"model.{layer + 6}." + ".".join(pieces[2:])
        elif layer == 23 and len(pieces) >= 5 and pieces[2] in (
                "cv2", "cv3", "one2one_cv2", "one2one_cv3"):
            target_key = f"model.29.{pieces[2]}.{int(pieces[3]) + 1}." + ".".join(pieces[4:])
        else:
            continue
        if target_key in state and state[target_key].shape == value.shape:
            state[target_key] = value.detach().clone()
            matched += 1
            transferred += value.numel()
    target.load_state_dict(state, strict=True)
    neck_channels = [int(head[0].conv.in_channels) for head in target.model[-1].cv2]
    old_head = target.model[-1]
    head = SpatialDetect26(nc=old_head.nc, reg_max=old_head.reg_max,
                           end2end=True, ch=tuple(neck_channels))
    head.load_state_dict(old_head.state_dict(), strict=True)
    head.stride = old_head.stride.clone()
    target.model[-1] = head
    target.stride = head.stride
    target.names = source.names
    print(f"[yolo26m-p2] transferred {matched} tensors, {transferred:,} values "
          f"from YOLO26m; P2 layers are newly initialized")
    if transferred < 10_000_000:
        raise RuntimeError("YOLO26m pretrained transfer too small; refusing random model")
    return target, neck_channels


class IndependentMMYOLO26(IndependentMMYOLO):
    """YOLO26m-P2 main/auxiliary detectors using V4.4 multimodal evidence."""

    def __init__(self, cfg):
        if cfg.fusion.architecture != "independent_p2_memory_yolo26m":
            raise ValueError("YOLO26m requires independent_p2_memory_yolo26m architecture")
        # The shared V4.4 initializer creates evidence, alignment, memory and
        # residual modules. Its temporary YOLO11-style head is discarded below.
        super().__init__(cfg)
        pretrained, neck_channels = _pretrained_p2(cfg)
        if neck_channels != self.neck_channels:
            raise RuntimeError(f"P2 neck widths differ: {neck_channels} vs {self.neck_channels}")
        self.backbone = pretrained
        # _adapt_detect_cls reads self.backbone.names for COCO class mapping.
        # Keep the source names until the new classifier has been remapped.
        MMYOLO._adapt_detect_cls(self, pretrained.model[-1], self.nc, cfg.cls_remap)
        pretrained.names = dict(enumerate(self.class_names))
        pretrained.nc = self.nc
        if self.semantic_detect is not None:
            self.semantic_detect = copy.deepcopy(pretrained.model[-1])
        if len(self.independent_aux):
            self.independent_aux = nn.ModuleDict({
                name: IndependentBranchDetector26(pretrained) for name in ("ir", "dep")
            })
        # The V4.4 custom P2 neck is replaced by native YOLO26m-P2 layers.
        del self.p2_lateral, self.p2_neck, self.p2_down, self.p3_refine, self.neck_gain

    def _run_detection(self, fused, geometry, state, present):
        features = _run_p2_neck(
            self.backbone.model[11:29], fused,
            p2_hook=lambda x: self.neck_memory(x, state, present),
        )
        if len(self.occlusion_context):
            features[0] = features[0] + self.occlusion_context[0](features[0])
            features[1] = features[1] + self.occlusion_context[1](features[1])
        localization = [
            feature + .15 * self.localization_scale[index].tanh() *
            self.localization[index](geometry[scale])
            for index, (scale, feature) in enumerate(zip(SCALES, features))
        ]
        return self.backbone.model[-1](features, localization)
