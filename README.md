# SpectraBreast Sample 3D Reconstruction 

### Repository Structure
- `rgb_images/`: contains RGB data from camera, indexes `0000` to `0020`
- `camera_poses/`: contains camera poses in robot space, as `.txt` files. Each file has one line with 6 floats, in the order: `[X Y Z Roll Pitch Yaw]`
- `checkerboard`: contains RGB images with a checkerboard pattern, used for calibration
- `camera_parameters`: contains `.npy` files with the intrinsic paramters (`[3x3]`), the camera-to-endeffector matrix  (`[4x4]`) and the  distortion parameters  (`[1x5]`)
- `reconstruction`: 3D reconstruction outputs
- `depth-anything-3`, `nerf_data`, `vggt`, `mast3r`: 3D reconstruction methods code cloned by each method repo
---
- `helpers.py`: Helper functions, angle conversion, geometry, etc...
- `intrinsic_calibration.py`: Camera calibration code. Loads the images from the `checkerboard` folder, searches for the checkerboard pattern and saves the calibration matrix and the distortion coefficients in `camera_parameters/` as `.npy` files

---
### MASt3R ⋅ CLI Usage

The script `mast3r_pipeline.py` runs the [MASt3R-SfM](https://github.com/naver/mast3r#mast3r-sfm) pipeline (sparse global alignment) and writes a fused point cloud plus a JSON of the parameters used.

**Basic usage**
```bash
python mast3r_pipeline.py
```

**Outputs** (in `--out_dir`, default `reconstruction/`):
- `mast3r_output.ply` — colored point cloud (ASCII PLY)
- `mast3r_params.json` — parameters used for that run (for reproducibility)
- `cache/` — MASt3R-SfM cache (pairs, matches, etc.)

**CLI arguments**

| Argument | Default | Description |
|----------|---------|-------------|
| `--rgb_dir` | `rgb_images/` | Directory of RGB images (`image_*.png`) |
| `--pose_dir` | `camera_poses/` | Directory of camera pose files (`pose_*.txt`) |
| `--out_dir` | `reconstruction` | Output directory for PLY and params JSON |
| `--image_size` | `512` | Image size for MASt3R (longer side) |
| `--scene_graph` | *(auto)* | `complete`, `swin-N`, `logwin-N`; empty = auto (complete if &lt;40 images else swin-5) |
| `--subsample` | `4` | Dense grid step: `1`, `2`, `4`, or `8`. Smaller = finer/smoother (less 8×8 blockiness), larger = faster |
| `--lr1` | `0.07` | Coarse alignment learning rate |
| `--niter1` | `300` | Coarse alignment iterations |
| `--lr2` | `0.01` | Fine refinement learning rate |
| `--niter2` | `300` | Fine refinement iterations |
| `--opt_depth` | on | Optimize depth in alignment (use `--no_opt_depth` to disable) |
| `--shared_intrinsics` | on | Shared intrinsics across views (use `--no_shared_intrinsics` to disable) |
| `--matching_conf_thr` | `5.0` | Matching confidence threshold before fallback |
| `--min_conf_thr` | `1.5` | Min confidence to keep a point in the output cloud |
| `--max_points` | `2000000` | Max points to keep (subsample if exceeded) |
| `--grpc_port` | `9876` | Rerun gRPC port |
| `--no_wait` | off | Do not wait for user input after logging to Rerun |

**Examples**
```bash
# Defaults
python mast3r_pipeline.py

# Custom output dir and stronger alignment
python mast3r_pipeline.py --out_dir reconstruction_run2 --niter1 300 --niter2 300 --min_conf_thr 2.0

# Force complete scene graph, skip interactive wait
python mast3r_pipeline.py --scene_graph complete --no_wait
```
