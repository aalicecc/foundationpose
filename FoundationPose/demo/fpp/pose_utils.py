"""6D pose helpers aligned with FoundationPose-plus-plus/src/obj_pose_track.py."""

from __future__ import annotations

from typing import Optional, Tuple

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


def compute_depth_median_and_valid_ratio(
  depth: np.ndarray,
  mask_u8: Optional[np.ndarray],
  min_depth_m: float = 0.001,
) -> Tuple[float, float]:
  """Median Z in meters over mask∩valid depth; ratio of valid depth pixels in mask."""
  if mask_u8 is None or depth is None:
    return 0.0, 1.0
  m = np.asarray(mask_u8) > 0
  if not np.any(m):
    return 0.0, 0.0
  d = np.asarray(depth, dtype=np.float64)[m]
  valid = d >= float(min_depth_m)
  n = float(d.size)
  valid_ratio = float(np.count_nonzero(valid)) / max(n, 1.0)
  if not np.any(valid):
    return 0.0, valid_ratio
  median_z = float(np.median(d[valid]))
  return median_z, valid_ratio


def slerp_rotation_matrices(R0: np.ndarray, R1: np.ndarray, t: float) -> np.ndarray:
  """Spherical linear interpolation between two rotation matrices, t in [0,1]."""
  t = float(np.clip(t, 0.0, 1.0))
  if t <= 1e-12:
    return R0.copy()
  if t >= 1.0 - 1e-12:
    return R1.copy()
  r0 = Rotation.from_matrix(R0)
  r1 = Rotation.from_matrix(R1)
  q0 = r0.as_quat()
  q1 = r1.as_quat()
  if np.dot(q0, q1) < 0.0:
    q1 = -q1
  q = np.zeros(4, dtype=np.float64)
  q[0] = (1.0 - t) * q0[0] + t * q1[0]
  q[1] = (1.0 - t) * q0[1] + t * q1[1]
  q[2] = (1.0 - t) * q0[2] + t * q1[2]
  q[3] = (1.0 - t) * q0[3] + t * q1[3]
  n = np.linalg.norm(q)
  if n < 1e-12:
    return R0.copy()
  q /= n
  return Rotation.from_quat(q).as_matrix()


def lock_roll_pitch_keep_yaw(R_new: np.ndarray, R_ref: np.ndarray) -> np.ndarray:
  """Keep roll and pitch from R_ref, yaw component from R_new (xyz Euler decomposition)."""
  e_new = Rotation.from_matrix(R_new).as_euler("xyz", degrees=False)
  e_ref = Rotation.from_matrix(R_ref).as_euler("xyz", degrees=False)
  e_blend = np.array([e_ref[0], e_ref[1], e_new[2]], dtype=np.float64)
  return Rotation.from_euler("xyz", e_blend, degrees=False).as_matrix()


def apply_z_blend_and_rotation_slerp(
  pose_4x4: np.ndarray,
  median_depth_z: float,
  R_smooth_prev: Optional[np.ndarray],
  z_blend: float,
  rot_slerp: float,
) -> Tuple[np.ndarray, np.ndarray]:
  """Soft correction: blend tz toward depth median; slerp rotation toward current estimate."""
  out = pose_4x4.copy().astype(np.float64)
  R_cur = out[:3, :3].copy()
  tz = float(out[2, 3])
  w = float(np.clip(z_blend, 0.0, 1.0))
  if median_depth_z > 1e-6:
    out[2, 3] = (1.0 - w) * tz + w * median_depth_z
  if R_smooth_prev is None:
    R_out = R_cur
  else:
    alpha = float(np.clip(rot_slerp, 0.0, 1.0))
    R_out = slerp_rotation_matrices(R_smooth_prev, R_cur, alpha)
  out[:3, :3] = R_out
  return out.astype(np.float32), R_out.astype(np.float64)


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
