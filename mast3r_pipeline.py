import argparse
import json
import torch
import torchvision.io as io
from pathlib import Path
from helpers import *
import numpy as np
from helpers import xyzeuler_to_hmat
from rich import print
import rerun as rr
import rerun.blueprint as rrb


def _parse_args():
    p = argparse.ArgumentParser(description="MASt3R-SfM 3D reconstruction pipeline")
    p.add_argument("--rgb_dir", type=Path, default=Path("rgb_images/"), help="Directory of RGB images")
    p.add_argument("--pose_dir", type=Path, default=Path("camera_poses/"), help="Directory of camera pose files")
    p.add_argument("--out_dir", type=Path, default=Path("reconstruction"), help="Output directory for PLY and params JSON")
    # MASt3R-SfM / load_images
    p.add_argument("--image_size", type=int, default=512, help="Image size for MASt3R (longer side)")
    # make_pairs / scene graph
    p.add_argument("--scene_graph", type=str, default="", help="Scene graph: 'complete', 'swin-N', 'logwin-N', etc. Empty = auto (complete if <40 imgs else swin-5)")
    # Dense grid (smaller = finer, less 8x8 blockiness; larger = faster, more memory)
    p.add_argument("--subsample", type=int, default=1, choices=[1, 2, 4, 8], help="Dense grid step (default 4). 8=blocky 8x8 patches; 4 or 2=smoother; 1=per-pixel (slow)")
    # sparse_global_alignment
    p.add_argument("--lr1", type=float, default=0.07, help="Coarse alignment learning rate")
    p.add_argument("--niter1", type=int, default=300, help="Coarse alignment iterations")
    p.add_argument("--lr2", type=float, default=0.01, help="Fine refinement learning rate")
    p.add_argument("--niter2", type=int, default=300, help="Fine refinement iterations")
    p.add_argument("--opt_depth", action="store_true", default=True, help="Optimize depth in alignment")
    p.add_argument("--no_opt_depth", action="store_false", dest="opt_depth")
    p.add_argument("--shared_intrinsics", action="store_true", default=True, help="Shared intrinsics across views")
    p.add_argument("--no_shared_intrinsics", action="store_false", dest="shared_intrinsics")
    p.add_argument("--matching_conf_thr", type=float, default=5.0, help="Matching confidence threshold before fallback")
    # Dense extraction
    p.add_argument("--min_conf_thr", type=float, default=1.5, help="Min confidence to keep a point in the output cloud")
    p.add_argument("--max_points", type=int, default=2_000_000, help="Max points to keep (subsample if exceeded)")
    # Rerun
    p.add_argument("--grpc_port", type=int, default=9876, help="Rerun gRPC port")
    p.add_argument("--no_wait", action="store_true", help="Do not wait for user input after logging to Rerun")
    return p.parse_args()


args = _parse_args()

# Assume CUDA is available based on user rules
device = torch.device("cuda")
print("Using device:", f"[green]{device}[/green]" if device.type == "cuda" else f"[red]{device}[/red]")

rgb_image_dir = args.rgb_dir
camera_pose_dir = args.pose_dir
image_files = sorted(rgb_image_dir.glob("image_*.png"))
pose_files = sorted(camera_pose_dir.glob("pose_*.txt"))

# Load images as [C, H, W] tensors to GPU
images = [io.read_image(str(p)).to(device) for p in image_files]

# Load poses as [6] tensors to GPU
poses = [
    torch.tensor(
        [float(x) for x in p.read_text().strip().split()],
        dtype=torch.float32,
        device=device,
    )
    for p in pose_files
]

# Batch everything if non-empty
if images and poses:
    images_tensor = torch.stack(images)
    poses_tensor = torch.stack(poses)
    print(f"Loaded {len(images)} pairs.")
    print(f"Batched images shape: {images_tensor.shape}")
    print(f"Batched poses6 shape: {poses_tensor.shape}")
translation_scale = 1
poses_tensor4 = xyzeuler_to_hmat(
    poses_tensor, convention="ROLLPITCHYAW", translation_scale=translation_scale
)
print(f"Batched poses shape: {poses_tensor4.shape}")

# Load intrinsics and cam2ee, expand as batch
intrinsics = np.load("camera_parameters/intrinsics.npy")
intrinsics = (
    torch.tensor(intrinsics)
    .to(device)
    .unsqueeze(0)
    .repeat(len(poses_tensor), 1, 1)
)
cam2ee = np.load("camera_parameters/camera2ee.npy")

# ---- Fix: cam2ee must be (4,4) but was (3,4) ----
# We'll check and expand if needed. 
if cam2ee.shape == (3, 4):
    # Make last row [0, 0, 0, 1], so (4, 4)
    cam2ee_fixed = np.eye(4, dtype=np.float32)
    cam2ee_fixed[:3, :4] = cam2ee
    cam2ee = cam2ee_fixed

cam2ee = torch.tensor(cam2ee).to(device).unsqueeze(0).repeat(len(poses_tensor), 1, 1)
# Now cam2ee is [N, 4, 4] and poses_tensor4 is [N, 4, 4]
poses_tensor_prem = torch.matmul(poses_tensor4, cam2ee)
print(f"Batched intrinsic shape: {intrinsics.shape}")

rr.init("SpectraBreast")
rr.serve_grpc(grpc_port=args.grpc_port) 

import sys
sys.path.append("/home/arota/spectra/mast3r")
import mast3r.utils.path_to_dust3r
from mast3r.model import AsymmetricMASt3R
from mast3r.cloud_opt.sparse_ga import sparse_global_alignment
from mast3r.image_pairs import make_pairs
from dust3r.utils.image import load_images
from dust3r.utils.device import to_numpy

# Load MASt3R model
model_name = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
print(f"Loading {model_name}...")
model = AsymmetricMASt3R.from_pretrained(model_name).to(device)
model.eval()

# Use dust3r load_images for MASt3R-SfM
filelist = [str(p) for p in image_files]
image_size = args.image_size
print("Loading images for MASt3R-SfM...")
imgs_mast3r = load_images(filelist, size=image_size, verbose=True)

print("Creating image pairs...")
scene_graph = args.scene_graph or ('complete' if len(filelist) < 40 else 'swin-5')
pairs = make_pairs(imgs_mast3r, scene_graph=scene_graph, prefilter=None, symmetrize=True)

print("Running sparse global alignment (MASt3R-SfM)...")
cache_dir = args.out_dir / "cache"
cache_dir.mkdir(parents=True, exist_ok=True)
scene = sparse_global_alignment(
    filelist, pairs, str(cache_dir), model,
    subsample=args.subsample,
    lr1=args.lr1, niter1=args.niter1, lr2=args.lr2, niter2=args.niter2,
    device=device, opt_depth=args.opt_depth, shared_intrinsics=args.shared_intrinsics,
    matching_conf_thr=args.matching_conf_thr
)

print("Extracting dense point cloud...")
# Extract 3D points, colors and confidences
pts3d, _, confs = to_numpy(scene.get_dense_pts3d(clean_depth=True, subsample=args.subsample))
rgbimg = to_numpy(scene.imgs)
cams2world = to_numpy(scene.get_im_poses())
focals = to_numpy(scene.get_focals())

# Filter by confidence and concatenate
min_conf_thr = args.min_conf_thr
msk = [c > min_conf_thr for c in confs]

all_pts_world = []
all_colors = []

for i in range(len(rgbimg)):
    pts_i = pts3d[i][msk[i].ravel()]
    col_i = rgbimg[i][msk[i]]
    
    # filter out any non-finite points
    valid = np.isfinite(pts_i.sum(axis=1))
    
    all_pts_world.append(pts_i[valid])
    all_colors.append(col_i[valid])

final_pts = np.concatenate(all_pts_world, axis=0)
final_colors = np.concatenate(all_colors, axis=0)

# Colors from to_numpy(scene.imgs) are usually float [0, 1] or [0, 255]
if final_colors.max() <= 1.0:
    final_colors = (final_colors * 255).astype(np.uint8)
else:
    final_colors = final_colors.astype(np.uint8)

# Subsample if too large for visualization
max_points = args.max_points
if len(final_pts) > max_points:
    indices = np.random.choice(len(final_pts), max_points, replace=False)
    final_pts = final_pts[indices]
    final_colors = final_colors[indices]

# Log to Rerun
print("Logging to Rerun...")
rr.log("/points", rr.Points3D(final_pts, colors=final_colors))

for i in range(len(cams2world)):
    pose = cams2world[i]  # [4, 4] camera to world
    focal = focals[i]
    H, W = rgbimg[i].shape[:2]
    
    # We can reconstruct a basic pinhole intrinsic using the optimized focal
    intrin = np.array([
        [focal.item(), 0, W / 2],
        [0, focal.item(), H / 2],
        [0, 0, 1]
    ])
    
    # Log camera pose
    rr.log(
        f"/cameras/{i}",
        rr.Transform3D(translation=pose[:3, 3], mat3x3=pose[:3, :3])
    )
    # Log pinhole camera intrinsics
    rr.log(
        f"/cameras/{i}/image",
        rr.Pinhole(image_from_camera=intrin, resolution=[W, H])
    )
# Save to disk as ASCII PLY (no Open3D dependency)
out_dir = args.out_dir
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "mast3r_output.ply"


def _save_pointcloud_as_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    """
    Save a point cloud to ASCII PLY.

    points: (N, 3) float32/float64
    colors: (N, 3) uint8 or float in [0, 255]
    """
    assert points.shape[0] == colors.shape[0]
    N = points.shape[0]

    # Ensure float XYZ and uint8 RGB
    pts = points.astype(np.float32)
    cols = colors
    if cols.dtype != np.uint8:
        # assume in [0,255] range
        cols = np.clip(cols, 0, 255).astype(np.uint8)

    header = "\n".join(
        [
            "ply",
            "format ascii 1.0",
            f"element vertex {N}",
            "property float x",
            "property float y",
            "property float z",
            "property uchar red",
            "property uchar green",
            "property uchar blue",
            "end_header",
        ]
    )

    data = np.concatenate(
        [pts, cols.astype(np.float32)], axis=1
    )  # (N, 6) – last 3 will be cast to int in fmt

    with open(path, "w") as f:
        f.write(header + "\n")
        np.savetxt(
            f,
            data,
            fmt="%.6f %.6f %.6f %d %d %d",
        )


print(f"Saving point cloud to {out_path}...")
_save_pointcloud_as_ply(out_path, final_pts, final_colors)

# Save parameters used for this reconstruction
params = {
    "rgb_dir": str(args.rgb_dir),
    "pose_dir": str(args.pose_dir),
    "out_dir": str(args.out_dir),
    "image_size": args.image_size,
    "scene_graph": scene_graph,
    "subsample": args.subsample,
    "lr1": args.lr1,
    "niter1": args.niter1,
    "lr2": args.lr2,
    "niter2": args.niter2,
    "opt_depth": args.opt_depth,
    "shared_intrinsics": args.shared_intrinsics,
    "matching_conf_thr": args.matching_conf_thr,
    "min_conf_thr": args.min_conf_thr,
    "max_points": args.max_points,
    "num_images": len(filelist),
    "num_points_output": int(len(final_pts)),
    "output_ply": str(out_path),
}
params_path = out_dir / "mast3r_params.json"
with open(params_path, "w") as f:
    json.dump(params, f, indent=2)
print(f"Parameters saved to {params_path}")

if not args.no_wait:
    input("Data has been logged to rerun. Open the viewer now and then press enter")

print("[green]Reconstruction complete![/green]")