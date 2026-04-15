"""6D pose helpers aligned with FoundationPose-plus-plus/src/obj_pose_track.py."""

from __future__ import annotations

import numpy as np
import torch
from scipy.spatial.transform import Rotation


def get_mat_from_6d_pose_arr(pose_arr: np.ndarray) -> np.ndarray:
  xyz = pose_arr[:3]
  euler_angles = pose_arr[3:6]
  rotation = Rotation.from_euler("xyz", euler_angles, degrees=False)
  rotation_matrix = rotation.as_matrix()
  transformation_matrix = np.eye(4, dtype=np.float64)
  transformation_matrix[:3, :3] = rotation_matrix
  transformation_matrix[:3, 3] = xyz
  return transformation_matrix


def get_6d_pose_arr_from_mat(pose) -> np.ndarray:
  if torch.is_tensor(pose):
    if pose.ndim == 3:
      pose_np = pose[0].detach().cpu().numpy()
    else:
      pose_np = pose.detach().cpu().numpy()
  else:
    pose_np = np.asarray(pose)
  xyz = pose_np[:3, 3]
  rotation_matrix = pose_np[:3, :3]
  euler_angles = Rotation.from_matrix(rotation_matrix).as_euler("xyz", degrees=False)
  return np.r_[xyz, euler_angles]


def get_pose_xy_from_image_point(
  ob_in_cam: torch.Tensor,
  K: torch.Tensor,
  x: float = -1.0,
  y: float = -1.0,
):
  if x == -1.0 or y == -1.0:
    return x, y
  row = ob_in_cam[0] if ob_in_cam.ndim == 3 else ob_in_cam
  if not torch.is_tensor(K):
    K = torch.as_tensor(K, dtype=row.dtype, device=row.device)
  elif K.device != row.device or K.dtype != row.dtype:
    K = K.to(device=row.device, dtype=row.dtype)
  t = row[:3, 3]
  fx = K[0, 0]
  fy = K[1, 1]
  cx = K[0, 2]
  cy = K[1, 2]
  tz = t[2]
  tx = (x - cx) * tz / fx
  ty = (y - cy) * tz / fy
  return tx, ty


def adjust_pose_to_image_point(
  ob_in_cam: torch.Tensor,
  K: torch.Tensor,
  x: float = -1.0,
  y: float = -1.0,
) -> torch.Tensor:
  device = ob_in_cam.device
  dtype = ob_in_cam.dtype
  if not torch.is_tensor(K):
    K = torch.as_tensor(K, dtype=dtype, device=device)
  elif K.device != device:
    K = K.to(device=device, dtype=dtype)

  is_batched = ob_in_cam.ndim == 3
  if not is_batched:
    ob_in_cam = ob_in_cam.unsqueeze(0)
  B = ob_in_cam.shape[0]
  ob_in_cam_new = torch.eye(4, device=device, dtype=dtype).repeat(B, 1, 1)
  for i in range(B):
    R = ob_in_cam[i, :3, :3]
    t = ob_in_cam[i, :3, 3]
    tx, ty = get_pose_xy_from_image_point(ob_in_cam[i], K, x, y)
    t_new = torch.tensor([float(tx), float(ty), float(t[2])], device=device, dtype=dtype)
    ob_in_cam_new[i, :3, :3] = R
    ob_in_cam_new[i, :3, 3] = t_new
  return ob_in_cam_new if is_batched else ob_in_cam_new[0]
