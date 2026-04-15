# Third-party bundles

Use **plain `git clone`** (no git submodule required). Place repositories directly under this directory.

## Cutie (video object segmentation)

Used for 2D bbox tracking in [`demo/run_demo_realsense.py`](../demo/run_demo_realsense.py).

```bash
cd third_party
git clone https://github.com/hkchengrex/Cutie.git
cd Cutie
pip install -e .
```

Download pretrained weights per Cutie’s documentation (e.g. `cutie/utils/download_models.py`).

## FastSAM

Optional: clone FastSAM here so the demo resolves imports from `third_party/FastSAM` first (see [`demo/seg/fastsam_bridge.py`](../demo/seg/fastsam_bridge.py)).

```bash
cd third_party
git clone https://github.com/CASIA-IVA-Lab/FastSAM.git
# Place weights under third_party/FastSAM/weights/ and point demo.yaml `model_path` accordingly.
```

If `third_party/FastSAM` is missing, the demo falls back to a sibling `FastSAM` folder next to the `FoundationPose` repo (previous layout).
