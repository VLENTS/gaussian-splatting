#!/usr/bin/env python3
"""
Compute per-vertex SDF of an SMPLX body sequence against a scene mesh.

Input:
  --smplx_npz    : SMPLX parameter sequence (.npz)
  --scene_mesh   : Scene mesh from SuGaR (.obj / .ply)
  --mlp          : MeshLab project file (.mlp) with similarity transform (sRt)
  --mesh_keyword : Keyword to select mesh entry in .mlp (default: "bg")
  --smplx_model  : Path to SMPLX model files

Output:
  --output       : .npz with sdf [T, V], negative = penetrating

Method: closest-face normal dot product (local sign, works for open scenes).
Transform: sRt similarity from .mlp (scale + rotation + translation).
"""

import argparse
import os
import time
import xml.etree.ElementTree as ET

import numpy as np
import smplx
import torch
import trimesh


# ---------------------------------------------------------------------------
# MeshLab project (.mlp) parsing — sRt similarity transform
# ---------------------------------------------------------------------------

def parse_mlp_matrix(mlp_path, mesh_keyword="bg"):
    """
    Parse .mlp and return 4x4 matrix for the mesh matching *mesh_keyword*.
    Falls back to the first entry if no keyword match.
    """
    root = ET.parse(mlp_path).getroot()
    meshes = root.findall(".//MLMesh")
    if not meshes:
        raise RuntimeError(f"{mlp_path}: no MLMesh found")

    chosen = None
    kw = mesh_keyword.lower().strip()
    if kw:
        for mesh in meshes:
            fn = (mesh.get("filename") or "").lower()
            label = (mesh.get("label") or "").lower()
            if kw in fn or kw in label:
                chosen = mesh
                break
    if chosen is None:
        chosen = meshes[0]

    text = (chosen.findtext("MLMatrix44") or "").strip()
    vals = [float(x) for x in text.split()]
    if len(vals) != 16:
        raise RuntimeError(f"MLMatrix44 must contain 16 floats, got {len(vals)}")

    mat = np.array(vals, dtype=np.float64).reshape(4, 4)
    name = chosen.get("filename") or chosen.get("label") or "<unknown>"
    return mat, name


def decompose_srt(mat):
    """Decompose 4x4 into scale, rotation (3x3), translation (3,)."""
    a = mat[:3, :3].copy()
    t = mat[:3, 3].copy()
    det_a = np.linalg.det(a)
    if abs(det_a) < 1e-12:
        raise RuntimeError("Transform matrix is singular")
    s = np.cbrt(det_a)
    r = a / s
    u, _, vt = np.linalg.svd(r)
    r = u @ vt
    if np.linalg.det(r) < 0:
        u[:, -1] *= -1
        r = u @ vt
    return s, r, t, a


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
                    help="MeshLab project file (.mlp) with sRt similarity transform")
    ap.add_argument("--mesh_keyword", default="bg",
                    help="Keyword to match mesh entry in .mlp (default: bg)")
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
        mat, mlp_mesh_name = parse_mlp_matrix(args.mlp, args.mesh_keyword)
        s, r, t, a = decompose_srt(mat)
        print(f"  .mlp mesh: {mlp_mesh_name}")
        print(f"  scale={s:.6f}, t=[{t[0]:.4f}, {t[1]:.4f}, {t[2]:.4f}]")

        verts = np.asarray(mesh.vertices, dtype=np.float64)
        verts_new = (verts @ a.T) + t
        mesh.vertices = verts_new

        normals = np.asarray(mesh.face_normals, dtype=np.float64)
        normals_new = normals @ r.T
        norms = np.linalg.norm(normals_new, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        mesh.face_normals = normals_new / norms

        print(f"  Applied sRt transform")

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
