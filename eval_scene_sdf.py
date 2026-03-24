#!/usr/bin/env python3
"""
Compute per-vertex SDF of an SMPLX body sequence against a scene mesh.

Input:
  --smplx_npz   : SMPLX parameter sequence (.npz)
  --scene_mesh  : Scene mesh from SuGaR (.obj / .ply)
  --mlp         : MeshLab project file (.mlp) containing the scene transform
  --smplx_model : Path to SMPLX model files

Output:
  --output      : .npz with sdf [T, V], negative = penetrating

Method: closest-face normal dot product (local sign, works for open scenes).
"""

import argparse
import os
import re
import time
import xml.etree.ElementTree as ET

import numpy as np
import smplx
import torch
import trimesh


# ---------------------------------------------------------------------------
# MeshLab project (.mlp) parsing
# ---------------------------------------------------------------------------

def parse_mlp_transforms(mlp_path):
    """
    Parse a MeshLab .mlp file and return a dict: {mesh_label: 4x4 numpy matrix}.
    """
    tree = ET.parse(mlp_path)
    root = tree.getroot()
    transforms = {}
    for ml_mesh in root.iter("MLMesh"):
        label = ml_mesh.get("label", "")
        mat_elem = ml_mesh.find("MLMatrix44")
        if mat_elem is not None and mat_elem.text:
            vals = [float(x) for x in mat_elem.text.split()]
            if len(vals) == 16:
                transforms[label] = np.array(vals, dtype=np.float64).reshape(4, 4)
    return transforms


def find_scene_transform(mlp_path, scene_mesh_path):
    """
    From an .mlp file, find the transform for the scene mesh.
    Matches by filename (with or without extension differences).
    Falls back to the first non-identity transform.
    """
    transforms = parse_mlp_transforms(mlp_path)
    scene_base = os.path.splitext(os.path.basename(scene_mesh_path))[0].lower()

    for label, mat in transforms.items():
        label_base = os.path.splitext(label)[0].lower()
        if label_base == scene_base:
            return mat

    for label, mat in transforms.items():
        if not np.allclose(mat, np.eye(4), atol=1e-6):
            return mat

    return None


# ---------------------------------------------------------------------------
# SMPLX helpers
# ---------------------------------------------------------------------------

def load_smplx_sequence(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    return {key: np.array(data[key], dtype=np.float32) for key in data.files}


def broadcast_to_T(arr, T, name):
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
        root, model_type="smplx", gender=gender, ext="npz",
        use_pca=False, batch_size=T,
    ).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def get_smplx_vertices(model, params, T, device):
    def _p(key, dim):
        if key in params:
            arr = broadcast_to_T(params[key], T, key)
            if arr.shape[-1] < dim:
                arr = np.concatenate(
                    [arr, np.zeros((T, dim - arr.shape[-1]), dtype=np.float32)], axis=-1
                )
            return torch.tensor(arr[:, :dim], dtype=torch.float32, device=device)
        return None

    kwargs = {}
    betas = _p("betas", 10)
    if betas is not None:
        kwargs["betas"] = betas
    for name, dim in [
        ("global_orient", 3), ("body_pose", 63), ("transl", 3),
        ("left_hand_pose", 45), ("right_hand_pose", 45),
        ("jaw_pose", 3), ("leye_pose", 3), ("reye_pose", 3),
        ("expression", 10),
    ]:
        val = _p(name, dim)
        if val is not None:
            kwargs[name] = val

    with torch.no_grad():
        output = model(**kwargs, return_verts=True)
    return output.vertices.cpu().numpy().astype(np.float64)


# ---------------------------------------------------------------------------
# SDF
# ---------------------------------------------------------------------------

def compute_sdf_closest_normal(query_points, mesh_query, face_normals):
    """
    Signed distance via closest-face normal dot product.
    Negative = penetrating the surface.
    """
    closest_pts, dist, face_ids = mesh_query.on_surface(query_points)
    vec = query_points - closest_pts
    normals = face_normals[face_ids]
    dot = np.einsum('ij,ij->i', vec, normals)
    sign = np.sign(dot)
    sign[sign == 0] = -1.0
    return dist * sign


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Compute per-vertex SDF of SMPLX sequence against scene mesh"
    )
    ap.add_argument("--smplx_npz", required=True,
                    help="SMPLX parameter sequence (.npz)")
    ap.add_argument("--scene_mesh", required=True,
                    help="Scene mesh (.obj / .ply)")
    ap.add_argument("--mlp", default=None,
                    help="MeshLab project file (.mlp) with scene transform")
    ap.add_argument("--smplx_model", required=True,
                    help="Path to SMPLX model files")
    ap.add_argument("--gender", default="neutral")
    ap.add_argument("--output", default=None,
                    help="Output .npz path (default: next to smplx_npz)")
    ap.add_argument("--batch_size", type=int, default=0,
                    help="SMPLX forward batch size, 0=all at once")
    args = ap.parse_args()

    t0 = time.time()

    if args.output is None:
        args.output = os.path.splitext(args.smplx_npz)[0] + "_scene_sdf.npz"

    # ---- load scene mesh ----
    print(f"Loading scene: {args.scene_mesh}")
    mesh = trimesh.load(args.scene_mesh, force="mesh", process=False)

    if args.mlp is not None:
        T_mat = find_scene_transform(args.mlp, args.scene_mesh)
        if T_mat is not None:
            mesh.apply_transform(T_mat)
            print(f"  Applied transform from {os.path.basename(args.mlp)}")
        else:
            print(f"  WARNING: no matching transform found in .mlp, using identity")

    components = mesh.split(only_watertight=False)
    if len(components) > 1:
        mesh = max(components, key=lambda c: c.area)
        print(f"  Kept largest component")

    print(f"  Vertices: {len(mesh.vertices)}, Faces: {len(mesh.faces)}")

    mesh_query = trimesh.proximity.ProximityQuery(mesh)
    face_normals = mesh.face_normals.copy()

    # ---- load SMPLX params ----
    print(f"Loading SMPLX: {args.smplx_npz}")
    params = load_smplx_sequence(args.smplx_npz)

    T_frames = 1
    for key in ["global_orient", "body_pose", "transl"]:
        if key in params and params[key].ndim >= 2 and params[key].shape[0] > 1:
            T_frames = params[key].shape[0]
            break
    print(f"  Frames: {T_frames}")

    # ---- SMPLX forward ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    bs = args.batch_size if args.batch_size > 0 else T_frames
    all_verts = []
    for start in range(0, T_frames, bs):
        end = min(start + bs, T_frames)
        chunk_params = {
            k: (v[start:end] if v.ndim >= 2 and v.shape[0] == T_frames else v)
            for k, v in params.items()
        }
        model = build_smplx_model(args.smplx_model, args.gender, end - start, device)
        all_verts.append(get_smplx_vertices(model, chunk_params, end - start, device))
        del model
        torch.cuda.empty_cache()

    vertices = np.concatenate(all_verts, axis=0)  # [T, V, 3]
    V = vertices.shape[1]
    print(f"  SMPLX vertices: {V}/frame")

    # ---- compute SDF ----
    print(f"Computing SDF ({T_frames} frames x {V} verts)...")
    sdf_all = np.zeros((T_frames, V), dtype=np.float32)

    for t in range(T_frames):
        sdf_all[t] = compute_sdf_closest_normal(
            vertices[t], mesh_query, face_normals
        ).astype(np.float32)

        if (t + 1) % 10 == 0 or t == 0 or t == T_frames - 1:
            n_pen = (sdf_all[t] < 0).sum()
            print(f"  [{t+1:4d}/{T_frames}] penetrating: {n_pen}/{V}")

    # ---- save ----
    np.savez(args.output, sdf=sdf_all, vertices=vertices.astype(np.float32))

    elapsed = time.time() - t0
    print(f"\nSaved: {args.output}")
    print(f"  sdf shape:      [{T_frames}, {V}]  (negative = penetrating)")
    print(f"  vertices shape: [{T_frames}, {V}, 3]")
    print(f"  Time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
