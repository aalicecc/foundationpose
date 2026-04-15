# FoundationPose Tracking Demo

## 功能
- 在线模式：D435i 采集 `RGB + 对齐Depth`，点击点提示触发 FastSAM，首帧注册后进入 FoundationPose 跟踪。
- 离线模式：回放录制数据，便于稳定调参与复现问题。
- `mask` 全流程统一为 `mono8`，且像素值仅 `0/255`。

## 目录约定
- Mesh 资产目录：`FoundationPose/assets/meshes/<object_name>/<mesh_file>`
- Demo 配置文件：`FoundationPose/demo/configs/demo.yaml`

## 运行前准备
1. 配置 `demo.yaml` 里的 `mesh.file`（你的目标 mesh 路径）。
2. 确认 `fastsam.model_path` 指向 `FastSAM.pt`。
3. 在线模式需安装 `pyrealsense2` 并连接 D435i。
4. `demo.yaml` 中所有路径会按“配置文件所在目录”解析；建议优先写相对路径，避免受 shell 当前工作目录影响。

示例（以 `demo/configs/demo.yaml` 为基准）：
- `mesh.file: ../../assets/meshes/pika/pika_origin.obj`
- `fastsam.model_path: ../../../FastSAM/weights/FastSAM-s.pt`
- `debug.dir: ../../debug/demo`

## 在线模式
```bash
cd /home/agilex/work/src/FoundationPose
python demo/run_demo_realsense.py --mode online
```

如需临时覆盖 mesh 路径：
```bash
python demo/run_demo_realsense.py --mode online --mesh_file /abs/path/to/your.obj
```

## 在线录制 + 离线回放
录制：
```bash
cd /home/agilex/work/src/FoundationPose
python demo/run_demo_realsense.py --mode online --record_dir /tmp/fp_demo_record
```

回放：
```bash
cd /home/agilex/work/src/FoundationPose
python demo/run_demo_realsense.py --mode offline --replay_dir /tmp/fp_demo_record
```

## 交互说明
- 左键：正点（目标点）
- 右键：负点（排除点）
- `c`：清除当前初始化，回到等待首帧
- `q`：退出

## 调试日志关注点
- `state`: `waiting_init` / `tracking` / `reinit_required`
- `sync_err`: `rgb/depth/mask` 同步误差（毫秒）
- `reason`: 当前状态原因（例如 `register_ok`、`pose_jump`）
