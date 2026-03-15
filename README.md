# SpectraBreast Sample 3D Reconstruction 

### Repository Structure
- `rgb_images/`: contains RGB data from camera, indexes `0000` to `0020`
- `camera_poses/`: contains camera poses in robot space, as `.txt` files. Each file has one line with 6 floats, in the order: `[X Y Z Roll Pitch Yaw]`
- `checkerboard`: contains RGB images with a checkerboard pattern, used for calibration
- `camera_parameters`: contains `.npy` files with the intrinsic paramters (`[3x3]`), the camera-to-endeffector matrix  (`[4x4]`) and the  distortion parameters  (`[1x5]`)
- `reconstruction`: 3D reconstruction outputs
- `depth-anything-3`, `vggt`, `mast3r`, `nerf_data`: 3D reconstruction methods code cloned by each method repo
---
- `helpers.py`: Helper functions, angle conversion, geometry, etc...
- `intrinsic_calibration.py`: Camera calibration code. Loads the images from the `checkerboard` folder, searches for the checkerboard pattern and saves the calibration matrix and the distortion coefficients in `camera_parameters/` as `.npy` files
- `vggt_pipeline.py`, `mast3r_pipeline.py`, `gsplat_pipeline.py`: 3D Reconstruction pipelines, one for each of the methods tested
---
<details>
<summary><strong>VGGT [BEST]</strong></summary>

<br/>

The script <code>vggt_pipeline.py</code> runs the <a href="https://github.com/facebookresearch/vggt">VGGT</a> (Visual Geometry Grounded Transformer) pipeline: feedforward depth/point-map prediction and optional alignment to GT camera poses. It logs the 3D scene (point cloud, GT and predicted cameras) and a frame-by-frame 2D view to <a href="https://rerun.io/">Rerun</a>.

**Basic usage**
```bash
python vggt_pipeline.py
```

**Outputs** (in <code>--out_dir</code>, default <code>reconstruction_vggt/</code>):

- <code>vggt_extracted_cloud.ply</code> — colored point cloud (ASCII PLY, with confidence)
- <code>vggt_extracted_cloud.npz</code> — points, colors, confidence
- <code>camera_poses_output_frame.npy</code> — camera poses in the output frame (GT frame if <code>--camera_source gt</code>, else predicted)
- <code>camera_poses_reconstruction.npy</code> — reconstruction cameras (predicted or aligned)
- <code>intrinsics_predicted.npy</code>, <code>intrinsics_gt_network_geometry.npy</code>
- <code>vggt_extracted_cloud_params.json</code> — parameters used
- <code>confidence_maps/</code> — per-frame confidence visualizations

**CLI arguments**

| Argument | Default | Description |
|----------|---------|-------------|
| <code>--rgb_dir</code> | <code>rgb_images/</code> | Directory of RGB images (<code>image_*.png</code>) |
| <code>--pose_dir</code> | <code>camera_poses/</code> | Directory of camera pose files (<code>pose_*.txt</code>) |
| <code>--camera_params_dir</code> | <code>camera_parameters</code> | Directory with <code>intrinsics.npy</code> and optional <code>camera2ee.npy</code> |
| <code>--out_dir</code> | <code>reconstruction_vggt</code> | Output directory |
| <code>--model_name</code> | <code>facebook/VGGT-1B</code> | Hugging Face model name |
| <code>--image_size</code> | <code>518</code> | VGGT input size (crop mode) |
| <code>--conf_thres</code> | <code>50.0</code> | Confidence percentile to filter out (e.g. 50 = drop bottom 50%) |
| <code>--cloud_source</code> | <code>depth_map</code> | <code>depth_map</code> (unproject depth) or <code>point_map</code> (point head directly) |
| <code>--camera_source</code> | <code>predicted</code> | <code>predicted</code> (VGGT frame) or <code>gt</code> (align to robot/world frame) |
| <code>--alignment_mode</code> | <code>sim3</code> | When <code>--camera_source gt</code>: <code>sim3</code> or <code>se3</code> |
| <code>--mask_black_bg</code> | off | Mask out black background pixels |
| <code>--mask_white_bg</code> | off | Mask out white background pixels |
| <code>--grpc_port</code> | <code>9876</code> | Rerun gRPC port |
| <code>--no_wait</code> | off | Do not wait for user input after logging to Rerun |

**Examples**
```bash
# Defaults (predicted frame, depth-map unprojection)
python vggt_pipeline.py

# Align reconstruction to GT robot frame (Sim3)
python vggt_pipeline.py --camera_source gt

# Use point-map branch, custom output dir, skip interactive wait
python vggt_pipeline.py --cloud_source depth_map --camera_source gt --alignment_mode=sim3 --conf_thres=90.0
```


</details>

<details>
<summary><strong>MASt3R</strong></summary>

<br/>

The script <code>mast3r_pipeline.py</code> runs the <a href="https://github.com/naver/mast3r#mast3r-sfm">MASt3R-SfM</a> pipeline (sparse global alignment) and writes a fused point cloud plus a JSON of the parameters used.

**Basic usage**
```bash
python mast3r_pipeline.py
```

**Outputs** (in <code>--out_dir</code>, default <code>reconstruction/</code>):

- <code>mast3r_output.ply</code> — colored point cloud (ASCII PLY)
- <code>mast3r_params.json</code> — parameters used for that run (for reproducibility)
- <code>cache/</code> — MASt3R-SfM cache (pairs, matches, etc.)

**CLI arguments**

| Argument                | Default         | Description                                                                           |
|-------------------------|-----------------|---------------------------------------------------------------------------------------|
| <code>--rgb_dir</code>          | <code>rgb_images/</code>   | Directory of RGB images (<code>image_*.png</code>)                                     |
| <code>--pose_dir</code>         | <code>camera_poses/</code> | Directory of camera pose files (<code>pose_*.txt</code>)                               |
| <code>--out_dir</code>          | <code>reconstruction</code> | Output directory for PLY and params JSON                                               |
| <code>--image_size</code>       | <code>512</code>           | Image size for MASt3R (longer side)                                                    |
| <code>--scene_graph</code>      | <em>(auto)</em>            | <code>complete</code>, <code>swin-N</code>, <code>logwin-N</code>; empty = auto (complete if &lt;40 images else swin-5) |
| <code>--subsample</code>        | <code>4</code>             | Dense grid step: <code>1</code>, <code>2</code>, <code>4</code>, or <code>8</code>. Smaller = finer/smoother (less 8×8 blockiness), larger = faster |
| <code>--lr1</code>              | <code>0.07</code>          | Coarse alignment learning rate                                                         |
| <code>--niter1</code>           | <code>300</code>           | Coarse alignment iterations                                                            |
| <code>--lr2</code>              | <code>0.01</code>          | Fine refinement learning rate                                                          |
| <code>--niter2</code>           | <code>300</code>           | Fine refinement iterations                                                             |
| <code>--opt_depth</code>        | on                | Optimize depth in alignment (use <code>--no_opt_depth</code> to disable)               |
| <code>--shared_intrinsics</code>| on                | Shared intrinsics across views (use <code>--no_shared_intrinsics</code> to disable)    |
| <code>--matching_conf_thr</code>| <code>5.0</code>           | Matching confidence threshold before fallback                                          |
| <code>--min_conf_thr</code>     | <code>1.5</code>           | Min confidence to keep a point in the output cloud                                     |
| <code>--max_points</code>       | <code>2000000</code>       | Max points to keep (subsample if exceeded)                                             |
| <code>--grpc_port</code>        | <code>9876</code>          | Rerun gRPC port                                                                        |
| <code>--no_wait</code>          | off               | Do not wait for user input after logging to Rerun                                      |

**Examples**
```bash
# Defaults
python mast3r_pipeline.py

# Custom output dir and stronger alignment
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
