"""Cutie 2D mask/bbox tracker; Cutie code lives under FoundationPose/third_party/Cutie."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
from torchvision.transforms.functional import to_tensor

_FOUNDATIONPOSE_ROOT = Path(__file__).resolve().parents[2]
_CUTIE_ROOT = _FOUNDATIONPOSE_ROOT / "third_party" / "Cutie"
if _CUTIE_ROOT.is_dir() and str(_CUTIE_ROOT) not in sys.path:
  sys.path.insert(0, str(_CUTIE_ROOT))


def _visualize_mask(image: np.ndarray, mask: np.ndarray, save_path: str) -> None:
  overlay = image.copy()
  color = np.zeros_like(overlay)
  color[..., 1] = (mask > 0).astype(np.uint8) * 200
  cv2.addWeighted(overlay, 0.7, color, 0.3, 0, overlay)
  cv2.imwrite(save_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))


def _visualize_bbox(image: np.ndarray, bbox: List[int], save_path: str) -> None:
  if bbox is None or bbox[0] < 0 or len(bbox) < 4:
    return
  x, y, w, h = bbox
  vis = image.copy()
  cv2.rectangle(vis, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
  cv2.imwrite(save_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))


class CutieTracker:
  """Video object segmentation bbox (xywh), compatible with FoundationPose++ Cutie usage."""

  def __init__(
    self,
    cutie_seg_threshold: float = 0.1,
    erosion_size: int = 5,
    max_internal_size: int = 480,
  ):
    if not _CUTIE_ROOT.is_dir():
      raise FileNotFoundError(
        f"Cutie not found at {_CUTIE_ROOT}. Clone or copy third_party/Cutie (see third_party/README.md)."
      )
    from cutie.inference.inference_core import InferenceCore
    from cutie.utils.get_default_model import get_default_model

    self.cutie_seg_threshold = cutie_seg_threshold
    self.erosion_size = erosion_size
    self.cutie = get_default_model()
    self.cutie_processor = InferenceCore(self.cutie, cfg=self.cutie.cfg)
    self.cutie_processor.max_internal_size = int(max_internal_size)

  def initialize(
    self,
    init_frame: np.ndarray,
    init_info: Dict[str, np.ndarray],
    mask_visualization_path: Optional[str] = None,
    bbox_visualization_path: Optional[str] = None,
  ) -> List[int]:
    with torch.no_grad():
      init_frame_tensor = to_tensor(init_frame).cuda().float()
      init_mask_tensor = torch.from_numpy(init_info["mask"]).cuda()
      objects = np.unique(init_info["mask"])
      objects = objects[objects != 0].tolist()
      output_prob = self.cutie_processor.step(init_frame_tensor, init_mask_tensor, objects=objects)
      mask = self.cutie_processor.output_prob_to_mask(output_prob, segment_threshold=self.cutie_seg_threshold)
      mask_np = mask.cpu().numpy()
    bbox_xywh = self._parse_output(mask_np)
    init_frame = init_frame.copy()
    if mask_visualization_path is not None:
      _visualize_mask(init_frame, mask_np * 255, mask_visualization_path)
    if bbox_visualization_path is not None:
      _visualize_bbox(init_frame, bbox_xywh, bbox_visualization_path)
    torch.cuda.empty_cache()
    return bbox_xywh

  def track(
    self,
    frame: np.ndarray,
    mask_visualization_path: Optional[str] = None,
    bbox_visualization_path: Optional[str] = None,
  ) -> List[int]:
    with torch.no_grad():
      frame_tensor = to_tensor(frame).cuda().float()
      output_prob = self.cutie_processor.step(frame_tensor)
      mask = self.cutie_processor.output_prob_to_mask(output_prob, segment_threshold=self.cutie_seg_threshold)
      mask_np = mask.cpu().numpy()
    bbox_xywh = self._parse_output(mask_np)
    frame = frame.copy()
    if mask_visualization_path is not None:
      _visualize_mask(frame, mask_np * 255, mask_visualization_path)
    if bbox_visualization_path is not None:
      _visualize_bbox(frame, bbox_xywh, bbox_visualization_path)
    torch.cuda.empty_cache()
    return bbox_xywh

  def _parse_output(self, mask_np: np.ndarray) -> List[int]:
    kernel = np.ones((self.erosion_size, self.erosion_size), np.uint8)
    mask_np = cv2.erode(mask_np.astype(np.uint8), kernel, iterations=1)
    rows = np.any(mask_np, axis=1)
    cols = np.any(mask_np, axis=0)
    if np.any(rows) and np.any(cols):
      y_min, y_max = np.where(rows)[0][[0, -1]]
      x_min, x_max = np.where(cols)[0][[0, -1]]
      bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
    else:
      bbox = [-1, -1, 0, 0]
    return bbox
