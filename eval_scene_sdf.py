#!/usr/bin/env python3
"""
Evaluate per-vertex SDF of an SMPLX body sequence against a scene mesh.

Input:
  --smplx_npz   : SMPLX parameter sequence (.npz) containing
                   betas, global_orient, body_pose, transl, etc.
  --scene_mesh  : Scene mesh from SuGaR (.obj / .ply)
  --smplx_model : Path to SMPLX model files

Output:
  --output      : .npz with sdf [T, V] and per-frame statistics

Method: closest-face normal dot product (local sign, works for open scenes).
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import smplx
import torch
import trimesh


def load_smplx_sequence(npz_path):
    """Load SMPLX parameters from npz, return dict of numpy arrays."""
    data = np.load(npz_path, allow_pickle=True)
    out = {}
    for key in data.files:
        out[key] = np.array(data[key], dtype=np.float32)
    return out


def broadcast_to_T(arr, T, name):
    """Ensure array has shape [T, ...], broadcasting from [1, ...] if needed."""
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[0] == 1 and T > 1:
        arr = np.repeat(arr, T, axis=0)
    if arr.shape[0] != T:
        raise ValueError(f"{name}: expected T={T}, got {arr.shape[0]}")
    return arr


def build_smplx_model(model_path, gender, T, device):
    root = model_path
    if os.path.basename(os.path.normpath(root)).lower() == "smplx":
        root = os.path.dirname(os.path.normpath(root))

    model = smplx.create(
        root,
        model_type="smplx",
        gender=gender,
        ext="npz",
        use_pca=False,
        batch_size=T,
    ).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def get_smplx_vertices(model, params, T, device):
    """Run SMPLX forward pass, return vertices [T, V, 3] numpy."""
    def _p(key, dim):
        if key in params:
            arr = broadcast_to_T(params[key], T, key)
            if arr.shape[-1] < dim:
                pad = np.zeros((T, dim - arr.shape[-1]), dtype=np.float32)
                arr = np.concatenate([arr, pad], axis=-1)
            return torch.tensor(arr[:, :dim], dtype=torch.float32, device=device)
        return None

    kwargs = {}

    betas = _p("betas", 10)
    if betas is not None:
        kwargs["betas"] = betas

    for name, dim in [
        ("global_orient", 3),
        ("body_pose", 63),
        ("transl", 3),
        ("left_hand_pose", 45),
        ("right_hand_pose", 45),
        ("jaw_pose", 3),
        ("leye_pose", 3),
        ("reye_pose", 3),
        ("expression", 10),
    ]:
        val = _p(name, dim)
        if val is not None:
            kwargs[name] = val

    with torch.no_grad():
        output = model(**kwargs, return_verts=True)

    return output.vertices.cpu().numpy().astype(np.float64)


def compute_sdf_closest_normal(query_points, mesh_query, face_normals):
    """
    Compute SDF using closest-face normal method.
    
    query_points: [N, 3]
    Returns: sdf [N], negative = penetrating
    """
    closest_pts, dist, face_ids = mesh_query.on_surface(query_points)
    vec = query_points - closest_pts
    normals = face_normals[face_ids]
    dot = np.einsum('ij,ij->i', vec, normals)
    sign = np.sign(dot)
    sign[sign == 0] = -1.0
    return dist * sign


def main():
    ap = argparse.ArgumentParser(
        description="Evaluate per-vertex SDF of SMPLX sequence against scene mesh"
    )
    ap.add_argument("--smplx_npz", required=True,
                    help="SMPLX parameter sequence (.npz)")
    ap.add_argument("--scene_mesh", required=True,
                    help="Scene mesh (.obj or .ply)")
    ap.add_argument("--smplx_model", required=True,
                    help="Path to SMPLX model files")
    ap.add_argument("--gender", default="neutral",
                    help="SMPLX gender (neutral/male/female)")
    ap.add_argument("--scene_transform", type=float, nargs=16, default=None,
                    help="4x4 scene transform matrix (row-major, 16 floats)")
    ap.add_argument("--output", default=None,
                    help="Output .npz path (default: next to smplx_npz)")
    ap.add_argument("--batch_size", type=int, default=0,
                    help="SMPLX batch size, 0=all frames at once")
    args = ap.parse_args()

    t0 = time.time()

    # ---- output path ----
    if args.output is None:
        base = os.path.splitext(args.smplx_npz)[0]
        args.output = base + "_scene_sdf.npz"

    # ---- load scene mesh ----
    print(f"Loading scene mesh: {args.scene_mesh}")
    mesh = trimesh.load(args.scene_mesh, force="mesh", process=False)

    if args.scene_transform is not None:
        T_mat = np.array(args.scene_transform, dtype=np.float64).reshape(4, 4)
        mesh.apply_transform(T_mat)
        print(f"  Applied scene transform")

    components = mesh.split(only_watertight=False)
    if len(components) > 1:
        mesh = max(components, key=lambda c: c.area)
        print(f"  Kept largest component")

    print(f"  Vertices: {len(mesh.vertices)}, Faces: {len(mesh.faces)}")

    mesh_query = trimesh.proximity.ProximityQuery(mesh)
    face_normals = mesh.face_normals.copy()

    # ---- load SMPLX params ----
    print(f"Loading SMPLX params: {args.smplx_npz}")
    params = load_smplx_sequence(args.smplx_npz)

    T_frames = None
    for key in ["global_orient", "body_pose", "transl"]:
        if key in params and params[key].ndim >= 1:
            candidate = params[key].shape[0] if params[key].ndim >= 2 else 1
            if candidate > 1:
                T_frames = candidate
                break
    if T_frames is None:
        T_frames = 1
    print(f"  Sequence length: {T_frames} frames")

    # ---- build SMPLX model and get vertices ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    bs = args.batch_size if args.batch_size > 0 else T_frames
    all_verts = []

    for start in range(0, T_frames, bs):
        end = min(start + bs, T_frames)
        chunk_len = end - start
        chunk_params = {}
        for k, v in params.items():
            if v.ndim >= 2 and v.shape[0] == T_frames:
                chunk_params[k] = v[start:end]
            else:
                chunk_params[k] = v

        model = build_smplx_model(args.smplx_model, args.gender, chunk_len, device)
        verts = get_smplx_vertices(model, chunk_params, chunk_len, device)
        all_verts.append(verts)
        del model
        torch.cuda.empty_cache()

    vertices = np.concatenate(all_verts, axis=0)  # [T, V, 3]
    V = vertices.shape[1]
    print(f"  SMPLX vertices: {V} per frame")

    # ---- compute SDF per frame ----
    print(f"Computing SDF...")
    sdf_all = np.zeros((T_frames, V), dtype=np.float32)

    for t in range(T_frames):
        sdf_all[t] = compute_sdf_closest_normal(
            vertices[t], mesh_query, face_normals
        ).astype(np.float32)

        if (t + 1) % 10 == 0 or t == 0 or t == T_frames - 1:
            penetrating = sdf_all[t] < 0
            n_pen = penetrating.sum()
            if n_pen > 0:
                max_d = np.abs(sdf_all[t][penetrating]).max() * 100
                mean_d = np.abs(sdf_all[t][penetrating]).mean() * 100
            else:
                max_d = mean_d = 0.0
            print(f"  Frame {t+1:4d}/{T_frames}: "
                  f"penetrating={n_pen}/{V} "
                  f"max={max_d:.2f}cm mean={mean_d:.2f}cm")

    # ---- statistics ----
    pen_mask = sdf_all < 0
    pen_depths = np.abs(sdf_all[pen_mask]) if pen_mask.any() else np.array([0.0])

    stats = {
        "T": int(T_frames),
        "V": int(V),
        "penetration_vertex_ratio": float(pen_mask.mean()),
        "penetration_frame_ratio": float((pen_mask.any(axis=1)).mean()),
        "mean_depth_cm": float(pen_depths.mean() * 100),
        "max_depth_cm": float(pen_depths.max() * 100),
        "median_depth_cm": float(np.median(pen_depths) * 100),
        "per_frame_pen_count": pen_mask.sum(axis=1).tolist(),
        "per_frame_max_depth_cm": [
            float(np.abs(sdf_all[t][sdf_all[t] < 0]).max() * 100)
            if (sdf_all[t] < 0).any() else 0.0
            for t in range(T_frames)
        ],
    }

    # ---- save ----
    np.savez(
        args.output,
        sdf=sdf_all,                                       # [T, V]
        vertices=vertices.astype(np.float32),               # [T, V, 3]
    )

    stats_path = os.path.splitext(args.output)[0] + "_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    elapsed = time.time() - t0
    print(f"\nResults saved to: {args.output}")
    print(f"Stats saved to:   {stats_path}")
    print(f"Time: {elapsed:.1f}s")
    print(f"\n=== Summary ===")
    print(f"  Frames:             {T_frames}")
    print(f"  Vertices/frame:     {V}")
    print(f"  Penetrating ratio:  {stats['penetration_vertex_ratio']*100:.2f}% vertices")
    print(f"  Frames with penetration: {stats['penetration_frame_ratio']*100:.1f}%")
    print(f"  Mean depth:         {stats['mean_depth_cm']:.2f} cm")
    print(f"  Max depth:          {stats['max_depth_cm']:.2f} cm")
    print(f"  Median depth:       {stats['median_depth_cm']:.2f} cm")


if __name__ == "__main__":
    main()
