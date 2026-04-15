from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image
import torch


_FOUNDATIONPOSE_ROOT = Path(__file__).resolve().parents[2]
FASTSAM_ROOT = _FOUNDATIONPOSE_ROOT / "third_party" / "FastSAM"
if str(FASTSAM_ROOT) not in __import__("sys").path:
  __import__("sys").path.insert(0, str(FASTSAM_ROOT))

from fastsam import FastSAM, FastSAMPrompt  # noqa: E402


def to_mono8_mask(mask: np.ndarray) -> np.ndarray:
  mask_bool = mask.astype(bool)
  return (mask_bool.astype(np.uint8) * 255)


def _resize_mask_to_rgb(mask: np.ndarray, rgb_shape_hw: Tuple[int, int]) -> np.ndarray:
  target_h, target_w = int(rgb_shape_hw[0]), int(rgb_shape_hw[1])
  if mask.shape[:2] == (target_h, target_w):
    return mask
  return cv2.resize(mask.astype(np.uint8), (target_w, target_h), interpolation=cv2.INTER_NEAREST)


class FastSAMBridge:
  """Run FastSAM and always return mono8(0/255) mask."""

  def __init__(
    self,
    model_path: str,
    device: str = "cuda",
    imgsz: int = 1024,
    conf: float = 0.4,
    iou: float = 0.9,
    retina_masks: bool = True,
  ):
    if device == "auto":
      device = "cuda" if torch.cuda.is_available() else "cpu"
    self.device = device
    self.imgsz = int(imgsz)
    self.conf = float(conf)
    self.iou = float(iou)
    self.retina_masks = bool(retina_masks)
    self.model = FastSAM(model_path)

  def _run_everything(self, rgb: np.ndarray):
    image = Image.fromarray(rgb).convert("RGB")
    return self.model(
      image,
      device=self.device,
      retina_masks=self.retina_masks,
      imgsz=self.imgsz,
      conf=self.conf,
      iou=self.iou,
    )

  def infer_point_prompt(
    self,
    rgb: np.ndarray,
    points_xy: Sequence[Tuple[int, int]],
    point_labels: Sequence[int],
  ) -> np.ndarray:
    if len(points_xy) == 0:
      raise ValueError("Point prompt is empty.")
    if len(points_xy) != len(point_labels):
      raise ValueError("points_xy and point_labels must have the same length.")
    results = self._run_everything(rgb)
    prompt = FastSAMPrompt(rgb, results, device=self.device)
    ann = prompt.point_prompt(points=[list(p) for p in points_xy], pointlabel=list(point_labels))
    if ann is None or len(ann) == 0:
      return np.zeros(rgb.shape[:2], dtype=np.uint8)
    mask = np.asarray(ann[0]).astype(np.uint8)
    mask = _resize_mask_to_rgb(mask, rgb.shape[:2])
    return to_mono8_mask(mask)

  def infer_everything_largest(self, rgb: np.ndarray) -> np.ndarray:
    results = self._run_everything(rgb)
    prompt = FastSAMPrompt(rgb, results, device=self.device)
    ann = prompt.everything_prompt()
    if ann is None or len(ann) == 0:
      return np.zeros(rgb.shape[:2], dtype=np.uint8)
    if torch.is_tensor(ann):
      masks = ann.detach().cpu().numpy().astype(np.uint8)
    else:
      masks = np.asarray(ann).astype(np.uint8)
    if masks.ndim == 2:
      masks = _resize_mask_to_rgb(masks, rgb.shape[:2])
      return to_mono8_mask(masks)
    areas = masks.reshape(masks.shape[0], -1).sum(axis=1)
    idx = int(np.argmax(areas))
    mask = _resize_mask_to_rgb(masks[idx], rgb.shape[:2])
    return to_mono8_mask(mask)

  def ensure_mono8(self, mask: np.ndarray) -> np.ndarray:
    return to_mono8_mask(mask)


def build_fastsam_bridge(cfg: Dict, args) -> FastSAMBridge:
  """Load FastSAM from `cfg["fastsam"]` plus CLI `--fastsam_model_path` / `--device` overrides."""
  fs = dict(cfg.get("fastsam", {}))
  if getattr(args, "fastsam_model_path", None):
    fs["model_path"] = args.fastsam_model_path
  if getattr(args, "device", None) and args.device != "auto":
    fs["device"] = args.device
  if not fs.get("model_path"):
    raise ValueError("Set `fastsam.model_path` in the demo YAML or pass --fastsam_model_path.")
  return FastSAMBridge(
    model_path=fs["model_path"],
    device=fs.get("device", "auto"),
    imgsz=int(fs.get("imgsz", 1024)),
    conf=float(fs.get("conf", 0.4)),
    iou=float(fs.get("iou", 0.9)),
    retina_masks=bool(fs.get("retina_masks", True)),
  )
