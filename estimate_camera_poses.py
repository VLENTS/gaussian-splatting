#!/usr/bin/env python3
"""
Estimate camera pose sequence from masks, initial camera, and 3DGS scene.

For each frame t >= 1, optimize a relative transform T_t so that
    render(cam_{t-1} * T_t, scene) matches the target frame outside the
    cumulative mask  M_t = union(mask_0, ..., mask_t).

The L2 loss is computed only on the region OUTSIDE the cumulative mask.

Usage
-----
    python estimate_camera_poses.py \
        --mask_dir   /path/to/masks \
        --camera_json /path/to/camera.json \
        --scene_ply   /path/to/scene.ply \
        [--frames_dir /path/to/video_frames] \
        [--output_dir /path/to/output]

Environment: /media/data8T/xiaoyu/AnimatableGaussians/.mamba
"""

import argparse
import json
import os
import sys
import glob
import math
import numpy as np
import torch
from PIL import Image
from scipy.spatial.transform import Rotation as SciRotation
from scipy.optimize import minimize
from plyfile import PlyData

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from scene.gaussian_model import GaussianModel
from gaussian_renderer import render as gs_render
from scene.cameras import MiniCam
from utils.graphics_utils import getProjectionMatrix


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _Pipe:
    """Minimal pipeline config accepted by the rasteriser."""
    compute_cov3D_python = False
    convert_SHs_python = False
    debug = False
    antialiasing = False


def detect_sh_degree(ply_path):
    plydata = PlyData.read(ply_path)
    n_rest = sum(1 for p in plydata.elements[0].properties
                 if p.name.startswith("f_rest_"))
    if n_rest == 0:
        return 0
    sh_coeffs = n_rest // 3 + 1
    return int(math.sqrt(sh_coeffs)) - 1


def load_camera_json(path):
    with open(path, "r") as f:
        return json.load(f)


def load_masks(mask_dir, target_size=None):
    """Return a list of binary masks (float32, 0/1) sorted by filename."""
    files = sorted(glob.glob(os.path.join(mask_dir, "*.png")))
    if not files:
        files = sorted(glob.glob(os.path.join(mask_dir, "*.jpg")))
    masks = []
    for fpath in files:
        img = Image.open(fpath).convert("L")
        if target_size is not None:
            img = img.resize(target_size, Image.NEAREST)
        arr = np.array(img, dtype=np.float32) / 255.0
        masks.append((arr > 0.5).astype(np.float32))
    return masks


def load_frames(frames_dir, n, target_size=None):
    files = sorted(
        glob.glob(os.path.join(frames_dir, "*.png"))
        + glob.glob(os.path.join(frames_dir, "*.jpg"))
    )
    frames = []
    for fpath in files[:n]:
        img = Image.open(fpath).convert("RGB")
        if target_size is not None:
            img = img.resize(target_size, Image.LANCZOS)
        frames.append(np.array(img, dtype=np.float32) / 255.0)
    return frames


def auto_detect_frames_dir(mask_dir):
    base = mask_dir
    for _ in range(5):
        base = os.path.dirname(base)
        for sub in ("images", "frames", "rgb", "imgs"):
            cand = os.path.join(base, sub)
            if os.path.isdir(cand):
                pngs = glob.glob(os.path.join(cand, "*.png"))
                jpgs = glob.glob(os.path.join(cand, "*.jpg"))
                if pngs or jpgs:
                    return cand
    return None


# ---------------------------------------------------------------------------
# Camera construction
# ---------------------------------------------------------------------------

def make_minicam(w2c_4x4, intrinsics, device="cuda"):
    W, H = intrinsics["width"], intrinsics["height"]
    focal = intrinsics["focal"]
    fovx = 2.0 * math.atan(W / (2.0 * focal))
    fovy = 2.0 * math.atan(H / (2.0 * focal))
    znear, zfar = 0.01, 100.0

    wvt = torch.tensor(w2c_4x4, dtype=torch.float32, device=device).T
    proj = getProjectionMatrix(
        znear=znear, zfar=zfar, fovX=fovx, fovY=fovy
    ).T.to(device)
    full_proj = wvt.unsqueeze(0).bmm(proj.unsqueeze(0)).squeeze(0)
    return MiniCam(W, H, fovy, fovx, znear, zfar, wvt, full_proj)


def render_scene(cam, gaussians, bg):
    with torch.no_grad():
        return gs_render(cam, gaussians, _Pipe(), bg)["render"]


# ---------------------------------------------------------------------------
# Pose helpers
# ---------------------------------------------------------------------------

def build_delta_c2w(params):
    """6-param (axis-angle + translation) -> 4x4 rigid transform."""
    R = SciRotation.from_rotvec(params[:3]).as_matrix().astype(np.float32)
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R
    T[:3, 3] = params[3:].astype(np.float32)
    return T


def c2w_to_readable(c2w):
    R = c2w[:3, :3]
    pos = c2w[:3, 3]
    w2c = np.linalg.inv(c2w).astype(np.float32)
    q = SciRotation.from_matrix(R).as_quat()  # (x, y, z, w)
    return {
        "position_xyz": [float(v) for v in pos],
        "quaternion_wxyz": [float(q[3]), float(q[0]),
                            float(q[1]), float(q[2])],
        "rotation_matrix_3x3": [[float(v) for v in row] for row in R],
        "cam2world_4x4": [[float(v) for v in row] for row in c2w],
        "world2cam_4x4": [[float(v) for v in row] for row in w2c],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Estimate camera pose sequence via 3DGS rendering + L2 outside mask"
    )
    ap.add_argument("--mask_dir", required=True,
                    help="Mask directory (e.g. .../sam2/masks/obj_1)")
    ap.add_argument("--camera_json", required=True,
                    help="Initial camera JSON (readable format)")
    ap.add_argument("--scene_ply", required=True,
                    help="3DGS PLY scene file")
    ap.add_argument("--frames_dir", default=None,
                    help="Video frames dir (auto-detected if omitted)")
    ap.add_argument("--output_dir", default=None,
                    help="Output directory (default: derived from camera_json)")
    ap.add_argument("--sh_degree", type=int, default=None,
                    help="SH degree (auto-detected from PLY if omitted)")
    ap.add_argument("--max_iters", type=int, default=200,
                    help="Max Powell iterations per frame")
    ap.add_argument("--bg_color", type=float, nargs=3, default=[0, 0, 0],
                    help="Background RGB in [0,1]")
    ap.add_argument("--save_debug", action="store_true",
                    help="Save per-frame rendered images for debugging")
    args = ap.parse_args()

    # ---- output dir ----
    if args.output_dir is None:
        args.output_dir = os.path.dirname(os.path.abspath(args.camera_json))

    # ---- auto-detect frames dir ----
    if args.frames_dir is None:
        args.frames_dir = auto_detect_frames_dir(args.mask_dir)

    print(f"Mask dir   : {args.mask_dir}")
    print(f"Camera JSON: {args.camera_json}")
    print(f"Scene PLY  : {args.scene_ply}")
    print(f"Frames dir : {args.frames_dir}")
    print(f"Output dir : {args.output_dir}")

    # ---- load camera ----
    cam_data = load_camera_json(args.camera_json)
    intrinsics = cam_data["intrinsics"]
    c2w_init = np.array(cam_data["pose"]["cam2world_4x4"], dtype=np.float32)
    w2c_init = np.array(cam_data["pose"]["world2cam_4x4"], dtype=np.float32)
    W, H = intrinsics["width"], intrinsics["height"]

    # ---- load Gaussians ----
    sh_deg = args.sh_degree if args.sh_degree is not None else detect_sh_degree(args.scene_ply)
    print(f"SH degree  : {sh_deg}")
    gaussians = GaussianModel(sh_deg)
    gaussians.load_ply(args.scene_ply)
    print(f"Gaussians  : {gaussians.get_xyz.shape[0]}")

    # ---- load masks ----
    masks = load_masks(args.mask_dir, target_size=(W, H))
    N = len(masks)
    print(f"Masks      : {N}")
    if N == 0:
        print("ERROR: no mask files found in", args.mask_dir)
        return

    # ---- load video frames (optional) ----
    frames = None
    if args.frames_dir and os.path.isdir(args.frames_dir):
        frames = load_frames(args.frames_dir, N, target_size=(W, H))
        if frames:
            print(f"Frames     : {len(frames)}")
        else:
            frames = None

    if frames is None:
        print("WARNING: no video frames found — using initial rendering as reference.\n"
              "         The optimisation will be trivial (identity transforms).\n"
              "         Pass --frames_dir to supply the actual video frames.")

    # ---- rendering setup ----
    bg = torch.tensor(args.bg_color, dtype=torch.float32, device="cuda")
    cam0 = make_minicam(w2c_init, intrinsics)
    I_ref = render_scene(cam0, gaussians, bg)  # [3, H, W]

    # optional debug output
    debug_dir = None
    if args.save_debug:
        debug_dir = os.path.join(args.output_dir, "_debug_renders")
        os.makedirs(debug_dir, exist_ok=True)
        _save_tensor_image(I_ref, os.path.join(debug_dir, "frame_000000.png"))

    # ---- sequential optimisation ----
    camera_list = [
        {
            "frame": 0,
            "intrinsics": intrinsics,
            "pose": c2w_to_readable(c2w_init),
        }
    ]

    current_c2w = c2w_init.copy()

    for t in range(1, N):
        print(f"\n=== Frame {t}/{N - 1} ===")

        # cumulative mask M_t = union(mask[0], ..., mask[t])
        cum_mask = np.zeros_like(masks[0])
        for i in range(t + 1):
            np.maximum(cum_mask, masks[i], out=cum_mask)

        mask_t = torch.tensor(
            cum_mask, dtype=torch.float32, device="cuda"
        ).unsqueeze(0)  # [1, H, W]
        outside_count = float((1.0 - mask_t).sum().item())
        coverage = cum_mask.mean() * 100
        print(f"  mask coverage: {coverage:.1f}%  |  outside pixels: {int(outside_count)}")

        if outside_count < 1:
            print("  SKIP: mask covers entire image")
            camera_list.append({
                "frame": t,
                "intrinsics": intrinsics,
                "pose": c2w_to_readable(current_c2w),
                "optimization": {"skipped": True, "reason": "full_mask"},
            })
            continue

        # target image for this frame
        if frames is not None and t < len(frames):
            target = torch.tensor(
                frames[t].transpose(2, 0, 1), dtype=torch.float32, device="cuda"
            )
        else:
            target = I_ref

        # ---- optimise T_t with Powell ----
        prev_c2w = current_c2w.copy()
        eval_count = [0]

        def _loss(params):
            eval_count[0] += 1
            delta = build_delta_c2w(params)
            new_c2w = prev_c2w @ delta
            new_w2c = np.linalg.inv(new_c2w).astype(np.float32)
            cam = make_minicam(new_w2c, intrinsics)
            rendered = render_scene(cam, gaussians, bg)
            diff = (rendered - target) * (1.0 - mask_t)
            return float((diff ** 2).sum() / (outside_count * 3 + 1e-8))

        x0 = np.zeros(6, dtype=np.float64)
        init_loss = _loss(x0)
        print(f"  init loss: {init_loss:.8f}")

        res = minimize(
            _loss, x0, method="Powell",
            options={"maxiter": args.max_iters, "ftol": 1e-10, "xtol": 1e-10},
        )
        print(f"  final loss: {res.fun:.8f}  "
              f"(iters={res.nit}, evals={eval_count[0]})")
        print(f"  delta rot : {res.x[:3]}")
        print(f"  delta trans: {res.x[3:]}")

        delta = build_delta_c2w(res.x)
        current_c2w = (prev_c2w @ delta).astype(np.float32)

        camera_list.append({
            "frame": t,
            "intrinsics": intrinsics,
            "pose": c2w_to_readable(current_c2w),
            "optimization": {
                "loss": float(res.fun),
                "delta_axis_angle": res.x[:3].tolist(),
                "delta_translation": res.x[3:].tolist(),
            },
        })

        if debug_dir is not None:
            w2c = np.linalg.inv(current_c2w).astype(np.float32)
            cam = make_minicam(w2c, intrinsics)
            rendered = render_scene(cam, gaussians, bg)
            _save_tensor_image(
                rendered, os.path.join(debug_dir, f"frame_{t:06d}.png")
            )

    # ---- save output ----
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "camera_sequence_estimated.json")
    output = {
        "description": "Estimated camera pose sequence",
        "num_frames": N,
        "initial_camera": os.path.abspath(args.camera_json),
        "scene_ply": os.path.abspath(args.scene_ply),
        "mask_dir": os.path.abspath(args.mask_dir),
        "cameras": camera_list,
    }
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {N} cameras -> {out_path}")


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _save_tensor_image(tensor, path):
    """Save a [3,H,W] float tensor in [0,1] as PNG."""
    import torchvision
    torchvision.utils.save_image(tensor.clamp(0, 1), path)


if __name__ == "__main__":
    main()
