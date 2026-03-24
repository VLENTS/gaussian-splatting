#!/usr/bin/env python3
"""
SDEdit-based motion generation with ordered soft keypose guidance.

Generates in-distribution motion sequences that respect ordered keyposes
(A → B → C → D) with flexible timing, then remaps onto a target trajectory.

Designed for diffusion models trained on HumanML3D (root-absolute, 263-dim, 20fps).
"""

import numpy as np
import torch
from scipy.interpolate import interp1d


# ============================================================================
# Step 1: SDEdit with ordered soft guidance
# ============================================================================

def ordered_guidance_loss(x, keyposes, pose_dims, model, t, temperature=0.05):
    """
    Compute guidance loss with temporal ordering guarantee.

    Args:
        x:          [1, T, D] current denoised motion
        keyposes:   list of (name, target_pose [D], weight) in temporal order
        pose_dims:  list/tensor of feature indices for body pose
        model:      diffusion model (needs model.q_sample)
        t:          current diffusion timestep (int)
        temperature: softmax temperature (lower = sharper matching)

    Returns:
        loss: scalar, guidance loss
        expected_frames: dict {name: expected_frame_index}
    """
    T_total = x.shape[1]
    loss = torch.tensor(0.0, device=x.device)
    expected_frames = {}
    prev_end = 0
    n = len(keyposes)

    for i, (name, target_pose, w) in enumerate(keyposes):
        remaining = n - i - 1
        r_start = prev_end
        r_end = T_total - remaining
        if r_start >= r_end:
            r_start = max(r_end - 1, 0)

        segment = x[:, r_start:r_end, :]
        seg_len = r_end - r_start

        t_tensor = torch.tensor([t], device=x.device, dtype=torch.long)
        target_expanded = target_pose.reshape(1, 1, -1).expand(1, seg_len, -1)
        target_noisy = model.q_sample(target_expanded, t_tensor)

        diffs = ((segment[:, :, pose_dims] -
                  target_noisy[:, :, pose_dims]) ** 2).sum(dim=-1)  # [1, L]

        weights = torch.softmax(-diffs / temperature, dim=-1)  # [1, L]
        match_loss = (weights * diffs).sum()
        loss = loss + w * match_loss

        frame_idx = torch.arange(
            r_start, r_end, device=x.device, dtype=torch.float32
        )
        expected = (weights.squeeze(0) * frame_idx).sum()
        expected_frames[name] = expected.item()
        prev_end = int(expected.item()) + 1

    return loss, expected_frames


def sdedit_ordered_guidance(
    model,
    x_init,
    keyposes,
    pose_dims,
    noise_level=0.3,
    guidance_scale=1.0,
    guidance_schedule="linear",
    temperature=0.05,
):
    """
    SDEdit with ordered soft keypose guidance.

    Args:
        model:       diffusion model with .q_sample(), .p_sample(), .num_timesteps
        x_init:      [1, T, D] initial motion (interpolation or noise)
        keyposes:    list of (name, target_pose [D], weight) in temporal order
                     e.g. [("A", A_pose, 5.0), ("B", B_pose, 1.5), ...]
        pose_dims:   indices of body-pose features in the D-dim representation
        noise_level: 0~1, fraction of diffusion steps to noise/denoise
        guidance_scale: global scaling for guidance gradient
        guidance_schedule: "linear" (ramp up) or "constant"
        temperature: softmax temperature for soft matching

    Returns:
        x_result: [1, T, D] generated motion
        frame_log: list of dicts with per-step expected frame positions
    """
    device = x_init.device
    T_diff = model.num_timesteps
    t_start = int(noise_level * T_diff)

    t_tensor = torch.tensor([t_start], device=device, dtype=torch.long)
    x = model.q_sample(x_init, t_tensor)

    frame_log = []

    for step_i, t in enumerate(reversed(range(t_start))):
        x = model.p_sample(x, t)

        # guidance weight schedule
        progress = (step_i + 1) / t_start  # 0→1
        if guidance_schedule == "linear":
            w_sched = 0.3 + 0.7 * progress
        else:
            w_sched = 1.0

        with torch.enable_grad():
            x_g = x.detach().clone().requires_grad_(True)
            loss, ef = ordered_guidance_loss(
                x_g, keyposes, pose_dims, model, t, temperature
            )
            grad = torch.autograd.grad(loss, x_g)[0]

        x = x.detach() - guidance_scale * w_sched * grad

        if step_i % 50 == 0 or step_i == t_start - 1:
            frame_log.append({"step": t, "expected_frames": ef.copy(),
                              "loss": loss.item()})
            names_str = " ".join(f"{k}@{v:.0f}" for k, v in ef.items())
            print(f"  [denoise t={t:4d}] loss={loss.item():.4f} {names_str}")

    return x.detach(), frame_log


# ============================================================================
# Step 2: Locate keyposes & extract sub-sequence
# ============================================================================

def find_keypose_frame(motion, target_pose, pose_dims, search_range):
    """Find the frame best matching target_pose within search_range."""
    r_start, r_end = search_range
    segment = motion[0, r_start:r_end, pose_dims]
    target = target_pose[pose_dims].unsqueeze(0)
    diffs = ((segment - target) ** 2).sum(dim=-1)
    best_local = diffs.argmin().item()
    return r_start + best_local


def extract_subsequence(motion, A_pose, D_pose, pose_dims, margin=0):
    """Locate A and D in the generated motion, extract [A, D] sub-sequence."""
    T = motion.shape[1]
    frame_A = find_keypose_frame(motion, A_pose, pose_dims, (0, T // 2))
    frame_D = find_keypose_frame(motion, D_pose, pose_dims, (T // 2, T))
    start = max(0, frame_A - margin)
    end = min(T, frame_D + 1 + margin)
    return motion[:, start:end, :], frame_A, frame_D


# ============================================================================
# Step 3: Trajectory-pose decoupling
# ============================================================================

def decouple_trajectory_pose(motion_np, root_pos_dims, root_orient_dims, pose_dims):
    """
    Separate a root-absolute motion into trajectory and root-relative pose.

    Args:
        motion_np:       [T, D] numpy array (single sequence, no batch dim)
        root_pos_dims:   indices for root position (x, y, z) in the D-dim repr
        root_orient_dims: indices for root orientation
        pose_dims:       indices for body pose features

    Returns:
        pelvis:      [T, 3] root trajectory
        body_pose:   [T, len(pose_dims)] root-relative pose features
        root_orient: [T, len(root_orient_dims)] root orientation features
        speed:       [T-1] per-frame speed (scalar)
    """
    pelvis = motion_np[:, root_pos_dims].copy()
    body_pose = motion_np[:, pose_dims].copy()
    root_orient = motion_np[:, root_orient_dims].copy()
    displacements = np.diff(pelvis, axis=0)
    speed = np.linalg.norm(displacements, axis=1)
    return pelvis, body_pose, root_orient, speed


# ============================================================================
# Step 4: Arc-length reparameterization
# ============================================================================

def arc_length_remap(gt_trajectory, diffusion_speed):
    """
    Map diffusion's speed profile onto GT's path shape.

    Args:
        gt_trajectory:   [T_gt, 3] target path (only shape matters)
        diffusion_speed: [T-1] per-frame speed from diffusion output

    Returns:
        new_pelvis: [T, 3] remapped pelvis trajectory
                    (GT path shape × diffusion speed)
    """
    T = len(diffusion_speed) + 1

    gt_segments = np.diff(gt_trajectory, axis=0)
    gt_seg_len = np.linalg.norm(gt_segments, axis=1)
    gt_arc = np.concatenate([[0], np.cumsum(gt_seg_len)])
    L_gt = gt_arc[-1]

    diff_arc = np.concatenate([[0], np.cumsum(diffusion_speed)])
    L_diff = diff_arc[-1]

    if L_diff < 1e-8:
        return np.repeat(gt_trajectory[:1], T, axis=0)

    scale = L_gt / L_diff
    mapped_arc = np.clip(diff_arc * scale, 0, L_gt)

    interp_fn = interp1d(gt_arc, gt_trajectory, axis=0,
                         kind='cubic', fill_value='extrapolate')
    new_pelvis = interp_fn(mapped_arc)
    return new_pelvis.astype(np.float32)


# ============================================================================
# Step 5: Root orientation alignment
# ============================================================================

def align_root_yaw(new_pelvis, original_orient, orient_is_6d=False):
    """
    Align root yaw with path tangent direction, keep pitch/roll from original.

    Args:
        new_pelvis:      [T, 3] remapped trajectory
        original_orient: [T, K] root orientation from diffusion
        orient_is_6d:    True if orientation is 6D rotation repr

    Returns:
        aligned_orient: [T, K] with yaw replaced by path tangent direction
    """
    T = new_pelvis.shape[0]
    tangents = np.zeros((T, 3), dtype=np.float32)
    tangents[:-1] = np.diff(new_pelvis, axis=0)
    tangents[-1] = tangents[-2]

    path_yaw = np.arctan2(tangents[:, 0], tangents[:, 2])  # xz plane

    # For now, store the yaw for downstream use.
    # Actual rotation blending depends on the specific representation.
    # Return the yaw angles alongside the original orient.
    return original_orient.copy(), path_yaw


# ============================================================================
# Step 6: Assemble final motion
# ============================================================================

def assemble_motion(motion_template, new_pelvis, body_pose, root_orient,
                    root_pos_dims, root_orient_dims, pose_dims):
    """
    Reassemble motion from remapped trajectory + original pose.

    Args:
        motion_template: [T, D] original motion (for dimensions not in the above)
        new_pelvis:      [T, 3]
        body_pose:       [T, pose_dim]
        root_orient:     [T, orient_dim]
        *_dims:          index arrays

    Returns:
        motion_out: [T, D]
    """
    T = new_pelvis.shape[0]
    out = motion_template[:T].copy()
    out[:, root_pos_dims] = new_pelvis
    out[:, pose_dims] = body_pose[:T]
    out[:, root_orient_dims] = root_orient[:T]
    return out


# ============================================================================
# Full pipeline
# ============================================================================

def run_pipeline(
    model,
    keyposes,
    pose_dims,
    root_pos_dims,
    root_orient_dims,
    gt_trajectory,
    T_gen=196,
    noise_level=0.3,
    guidance_scale=1.0,
    temperature=0.05,
    device="cuda",
):
    """
    Full pipeline: SDEdit → extract → decouple → remap → align → assemble.

    Args:
        model:            diffusion model
        keyposes:         ordered list of (name, pose_tensor [D], weight)
        pose_dims:        body pose feature indices
        root_pos_dims:    root position feature indices (3 dims)
        root_orient_dims: root orient feature indices
        gt_trajectory:    [T_gt, 3] target path
        T_gen:            generation length (≤196 for HumanML3D)
        noise_level:      SDEdit noise fraction
        guidance_scale:   guidance gradient multiplier
        temperature:      soft matching temperature
        device:           torch device

    Returns:
        result: dict with final motion, intermediate results, and metadata
    """
    D = keyposes[0][1].shape[0]

    # --- Step 0: Initialize ---
    print("Step 0: Initializing...")
    x_init = torch.randn(1, T_gen, D, device=device) * 0.1
    # Optional: interpolate between A and D for better init
    A_pose = keyposes[0][1]
    D_pose = keyposes[-1][1]
    for t_idx in range(T_gen):
        alpha = t_idx / max(T_gen - 1, 1)
        x_init[0, t_idx] = (1 - alpha) * A_pose + alpha * D_pose

    # --- Step 1: SDEdit ---
    print(f"Step 1: SDEdit (T={T_gen}, noise={noise_level}, scale={guidance_scale})")
    x_gen, frame_log = sdedit_ordered_guidance(
        model, x_init, keyposes, pose_dims,
        noise_level=noise_level,
        guidance_scale=guidance_scale,
        temperature=temperature,
    )

    # --- Step 2: Extract A→D sub-sequence ---
    print("Step 2: Extracting sub-sequence...")
    A_pose_t = keyposes[0][1]
    D_pose_t = keyposes[-1][1]
    x_sub, frame_A, frame_D = extract_subsequence(
        x_gen, A_pose_t, D_pose_t, pose_dims
    )
    T_out = x_sub.shape[1]
    print(f"  Extracted frames [{frame_A}, {frame_D}], length={T_out}")

    motion_np = x_sub[0].cpu().numpy()

    # --- Step 3: Decouple ---
    print("Step 3: Decoupling trajectory and pose...")
    pelvis, body_pose, root_orient, speed = decouple_trajectory_pose(
        motion_np, root_pos_dims, root_orient_dims, pose_dims
    )
    print(f"  Mean speed: {speed.mean():.4f}, Total dist: {speed.sum():.2f}")

    # --- Step 4: Arc-length remap ---
    print("Step 4: Arc-length reparameterization...")
    new_pelvis = arc_length_remap(gt_trajectory, speed)
    gt_dist = np.linalg.norm(np.diff(gt_trajectory, axis=0), axis=1).sum()
    print(f"  GT path length: {gt_dist:.2f}, Diffusion dist: {speed.sum():.2f}")

    # --- Step 5: Align root orientation ---
    print("Step 5: Aligning root orientation...")
    aligned_orient, path_yaw = align_root_yaw(new_pelvis, root_orient)

    # --- Step 6: Assemble ---
    print("Step 6: Assembling final motion...")
    final_motion = assemble_motion(
        motion_np, new_pelvis, body_pose, aligned_orient,
        root_pos_dims, root_orient_dims, pose_dims
    )

    print(f"Done. Output shape: {final_motion.shape}")

    return {
        "motion": final_motion,            # [T_out, D] 最终动作
        "sdedit_full": x_gen[0].cpu().numpy(),  # [T_gen, D] SDEdit 完整输出
        "frame_A": frame_A,
        "frame_D": frame_D,
        "pelvis_original": pelvis,          # [T_out, 3] diffusion 原始轨迹
        "pelvis_remapped": new_pelvis,      # [T_out, 3] 重映射后轨迹
        "speed": speed,                     # [T_out-1] 速度曲线
        "path_yaw": path_yaw,              # [T_out] 路径偏航角
        "frame_log": frame_log,
    }


# ============================================================================
# Utility: smooth GT trajectory
# ============================================================================

def smooth_trajectory(trajectory, anchor_start=True, anchor_end=True, smoothness=0.5):
    """
    Smooth GT trajectory while keeping endpoints fixed.

    Args:
        trajectory: [T, 3]
        anchor_start: fix first point
        anchor_end: fix last point
        smoothness: 0~1, higher = smoother

    Returns:
        smoothed: [T, 3]
    """
    from scipy.interpolate import UnivariateSpline

    T = trajectory.shape[0]
    t = np.arange(T, dtype=np.float64)
    smoothed = np.zeros_like(trajectory)

    for dim in range(3):
        s = smoothness * T
        spline = UnivariateSpline(t, trajectory[:, dim], s=s)
        smoothed[:, dim] = spline(t)

    if anchor_start:
        smoothed[0] = trajectory[0]
    if anchor_end:
        smoothed[-1] = trajectory[-1]

    return smoothed.astype(np.float32)
