# Realtime demo: frame rate notes

Official FoundationPose reports ~30 Hz pose tracking at 640×480 on capable GPUs. Your **end-to-end** loop often runs slower because it includes:

- **Depth filtering** in `estimater.FoundationPose.track_one` (`erode_depth` + `bilateral_filter_depth`). Enable `performance.fast_depth_filter: true` in `configs/demo.yaml` to use a lighter path (smaller erode, no bilateral).
- **Cutie** 2D segmentation (`cutie.max_internal_size`): lower values (e.g. 384–480) reduce GPU time with little impact on bbox quality.
- **FastSAM** runs only on **click** (initialization); it is not per-frame.
- **OpenCV** `imshow` / `waitKey` can block; use `--no_gui` to measure pure inference rate via logs.
- **RealSense `wait_for_frames`**: `camera.async_frame_timeout_ms` sets the per-call timeout for **both** synchronous capture and the async grab thread (default 10000 ms). If you still see `Frame didn't arrive within …`, raise it or increase `camera.wait_for_frames_retries` (retries only apply when the error message contains `didn't arrive`).
- **Main-loop camera stall handling**: `camera.main_loop_frame_timeout_ms` controls how long the demo loop waits for one frame (default 1000 ms). On repeated timeouts, the loop auto-restarts the RealSense pipeline after `camera.recover_after_consecutive_timeouts` (default 3), then sleeps `camera.camera_restart_cooldown_s` before retrying.
- **Async camera** (`performance.async_camera`): default off for stability. When on, a background thread fills a size-1 queue (latest frame); use `camera.async_first_frame_timeout_ms` (cold start) and `camera.async_frame_timeout_ms` (steady) if you see queue timeouts. Logged `pred_hz` reflects the processing loop, not the camera’s nominal FPS.

Use the built-in `pred_hz` / `box_hz` logs to compare configurations. Profile GPU with `nvidia-smi` while tuning `track_refine_iter`, Cutie size, and `fast_depth_filter`.

## New recovery and memory options

- `fastsam.release_fastsam_after_init` (default `true`): release FastSAM model after the first successful registration to free GPU memory. After release, click re-init via FastSAM is disabled until restart.
- `fpp.pose_recovery_mode`:
  - `reuse_mask` (default): existing behavior, re-initialize with cached init mask after `reinit_required` (unless mask is cleared).
  - `kf_continue`: do not re-register with cached mask after pose jump; keep pose output via Kalman continuation (bounded by `kf_continue_max_frames`).
- `fpp.kf_continue_max_frames` (default `30`): maximum consecutive frames allowed in `kf_continue` fallback.

## New Kalman logs

- `kf_hz_update`: rate of Kalman update calls that fuse current 2D Cutie bbox and 6D state.
- `kf_hz_predict`: rate of Kalman predict calls used for next-frame prior (and `kf_continue` fallback path).

## dev_1 profile and on-screen overlays

- Use `python3 demo/run_demo_realsense.py --mode online --config demo/configs/dev_1.yaml` to run with the saved `dev_1` profile.
- On-screen text now includes:
  - `kf_upd_ema / kf_upd`: Kalman update frequency (EMA and instant).
  - `kf_pred_ema / kf_pred`: Kalman predict frequency (EMA and instant).
  - `pose x=... y=... z=...`: current pose translation in meters, formatted to 2 decimals (`pose=NA` when unavailable).
