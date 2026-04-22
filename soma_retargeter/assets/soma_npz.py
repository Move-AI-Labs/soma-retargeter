# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os

import numpy as np
import warp as wp
from scipy.spatial.transform import Rotation

import soma_retargeter.assets.bvh as bvh_utils
import soma_retargeter.utils.io_utils as io_utils
from soma_retargeter.animation.animation_buffer import AnimationBuffer, create_animation_buffer_for_skeleton

_GEM_REQUIRED_KEYS = {"global_orient", "body_pose", "transl"}
_POSES_REQUIRED_KEYS = {"poses", "transl"}
_DEFAULT_FPS = 30.0
_EXPECTED_NUM_JOINTS = 78  # Root + 77 SOMA joints

_SOMA_REFERENCE_CACHE: tuple | None = None


def _load_reference_soma_data():
    global _SOMA_REFERENCE_CACHE
    if _SOMA_REFERENCE_CACHE is None:
        bvh_path = io_utils.get_config_file("soma", "soma_zero_frame0.bvh")
        skeleton, anim = bvh_utils.load_bvh(str(bvh_path))
        _SOMA_REFERENCE_CACHE = (skeleton, anim.get_local_transforms(0))

    return _SOMA_REFERENCE_CACHE


def _validate_and_unpack(npz_data, npz_file: str, default_fps: float):
    files = set(npz_data.files)
    supports_gem = _GEM_REQUIRED_KEYS.issubset(files)
    supports_poses = _POSES_REQUIRED_KEYS.issubset(files)

    if supports_gem:
        global_orient = np.asarray(npz_data["global_orient"], dtype=np.float32)
        body_pose = np.asarray(npz_data["body_pose"], dtype=np.float32)
    elif supports_poses:
        poses = np.asarray(npz_data["poses"], dtype=np.float32)
        if poses.ndim != 3 or poses.shape[2] != 3:
            raise ValueError(
                f"[ERROR]: Expected poses shape (L, 77, 3), got {tuple(poses.shape)}"
            )
        if poses.shape[1] != 77:
            raise ValueError(
                f"[ERROR]: Expected poses with 77 joints, got {poses.shape[1]} joints."
            )

        if "rotation_repr" in npz_data.files:
            rotation_repr = str(np.asarray(npz_data["rotation_repr"]).item()).lower()
            if rotation_repr not in ("rotvec", "axis_angle"):
                raise ValueError(
                    f"[ERROR]: Unsupported rotation_repr [{rotation_repr}] in [{npz_file}]. "
                    "Expected 'rotvec' or 'axis_angle'."
                )

        global_orient = poses[:, 0, :]
        body_pose = poses[:, 1:, :]
    else:
        raise ValueError(
            f"[ERROR]: Unsupported SOMA npz schema in [{npz_file}]. "
            f"Expected either keys {sorted(_GEM_REQUIRED_KEYS)} or {sorted(_POSES_REQUIRED_KEYS)}."
        )

    transl = np.asarray(npz_data["transl"], dtype=np.float32)

    if global_orient.ndim != 2 or global_orient.shape[1] != 3:
        raise ValueError(
            f"[ERROR]: Expected global_orient shape (L, 3), got {tuple(global_orient.shape)}"
        )

    num_frames = global_orient.shape[0]

    if body_pose.ndim == 2 and body_pose.shape[1] == 76 * 3:
        body_pose = body_pose.reshape(num_frames, 76, 3)
    elif body_pose.ndim == 3 and body_pose.shape[1:] == (76, 3):
        pass
    else:
        raise ValueError(
            f"[ERROR]: Expected body_pose shape (L, 228) or (L, 76, 3), got {tuple(body_pose.shape)}"
        )

    if body_pose.shape[0] != num_frames:
        raise ValueError(
            f"[ERROR]: body_pose frame count [{body_pose.shape[0]}] does not match global_orient "
            f"frame count [{num_frames}]."
        )

    if transl.ndim != 2 or transl.shape != (num_frames, 3):
        raise ValueError(
            f"[ERROR]: Expected transl shape (L, 3), got {tuple(transl.shape)}"
        )

    if "unit" in npz_data.files:
        unit = str(np.asarray(npz_data["unit"]).item()).lower()
        if unit in ("cm", "centimeter", "centimeters"):
            transl = transl * 0.01
        elif unit not in ("m", "meter", "meters"):
            raise ValueError(
                f"[ERROR]: Unsupported translation unit [{unit}] in [{npz_file}]. "
                "Expected meters or centimeters."
            )

    if "fps" in npz_data.files:
        fps_raw = np.asarray(npz_data["fps"], dtype=np.float32).reshape(-1)
        if fps_raw.size == 0:
            raise ValueError("[ERROR]: fps exists in npz but contains no values.")
        sample_rate = float(fps_raw[0])
    else:
        sample_rate = float(default_fps)

    if sample_rate <= 0.0:
        raise ValueError(f"[ERROR]: fps must be positive, got [{sample_rate}].")

    return global_orient, body_pose, transl, sample_rate


def _compute_rest_world_orientations(npz_data, skeleton, reference_local_transforms):
    if "joint_orient" in npz_data.files:
        joint_orient = np.asarray(npz_data["joint_orient"], dtype=np.float32)
        if joint_orient.shape == (skeleton.num_joints, 3, 3):
            if "joint_names" in npz_data.files:
                joint_names = np.asarray(npz_data["joint_names"]).tolist()
                expected = skeleton.joint_names[1:]
                if len(joint_names) != len(expected) or list(joint_names) != expected:
                    raise ValueError(
                        "[ERROR]: joint_names in npz do not match expected SOMA reference ordering."
                    )
            return list(Rotation.from_matrix(joint_orient))

    world_orient = [Rotation.identity()] * skeleton.num_joints
    for joint_idx, parent_idx in enumerate(skeleton.parent_indices):
        local_rot = Rotation.from_quat(reference_local_transforms[joint_idx, 3:7])
        if parent_idx < 0:
            world_orient[joint_idx] = local_rot
        else:
            world_orient[joint_idx] = world_orient[parent_idx] * local_rot

    return world_orient


def _build_animation_from_soma_params(npz_data, skeleton, reference_local_transforms, global_orient, body_pose, transl, fps):
    num_frames = global_orient.shape[0]
    num_joints = skeleton.num_joints
    if num_joints != _EXPECTED_NUM_JOINTS:
        raise ValueError(
            f"[ERROR]: Expected SOMA skeleton with {_EXPECTED_NUM_JOINTS} joints, got {num_joints}."
        )

    local_transforms = np.zeros((num_frames, num_joints), dtype=wp.transform)
    local_transforms[:] = reference_local_transforms[None, :, :]

    all_rotvecs = np.concatenate([global_orient[:, None, :], body_pose], axis=1)  # (L, 77, 3)
    quats_xyzw = Rotation.from_rotvec(all_rotvecs.reshape(-1, 3)).as_quat().reshape(num_frames, 77, 4)
    quats_xyzw = quats_xyzw.astype(np.float32)

    parent_indices = skeleton.parent_indices
    rest_world_orient = _compute_rest_world_orientations(npz_data, skeleton, reference_local_transforms)

    for joint_idx in range(num_joints):
        parent_idx = parent_indices[joint_idx]
        if parent_idx < 0:
            continue

        body_rot = Rotation.from_quat(quats_xyzw[:, joint_idx - 1])
        local_rot = rest_world_orient[parent_idx].inv() * body_rot * rest_world_orient[joint_idx]
        local_transforms[:, joint_idx, 3:7] = local_rot.as_quat().astype(np.float32)

    hips_idx = skeleton.joint_index("Hips")
    if hips_idx == -1:
        raise ValueError("[ERROR]: Could not find required joint [Hips] in SOMA skeleton.")

    # Keep Root static and drive Hips translation from SOMA transl.
    local_transforms[:, hips_idx, :3] = transl.astype(np.float32)

    return AnimationBuffer(skeleton, num_frames, fps, local_transforms)


def load_soma_npz(npz_file: str, input_skeleton=None, default_fps: float = _DEFAULT_FPS):
    """
    Load a GEM-style SOMA params npz and create ``Skeleton`` and ``AnimationBuffer`` objects.

    Supported schema A (GEM-style):
        - global_orient: (L, 3)
        - body_pose: (L, 228) or (L, 76, 3)
        - transl: (L, 3)

    Supported schema B (motion-data style):
        - poses: (L, 77, 3) rotvec/axis-angle, with joints aligned to SOMA joint_names
        - transl: (L, 3)

    Optional keys:
        - fps: scalar (if missing, ``default_fps`` is used)
        - unit: meters or centimeters (defaults to meters when omitted)
        - joint_orient: (78, 3, 3) rest orientation matrices (used when present)
    """
    if not os.path.exists(npz_file) or os.path.splitext(npz_file)[-1].lower() != ".npz":
        raise ValueError(f"Invalid SOMA npz file path: {npz_file}")

    skeleton, reference_local_transforms = _load_reference_soma_data()
    with np.load(npz_file, allow_pickle=False) as npz_data:
        global_orient, body_pose, transl, sample_rate = _validate_and_unpack(npz_data, npz_file, default_fps)

        animation = _build_animation_from_soma_params(
            npz_data,
            skeleton,
            reference_local_transforms,
            global_orient,
            body_pose,
            transl,
            sample_rate,
        )

    print(
        f"[INFO]: Loaded SOMA npz file [{npz_file}] with "
        f"{animation.num_frames} frames @ {animation.sample_rate} FPS"
    )

    if input_skeleton is not None:
        new_animation = create_animation_buffer_for_skeleton(animation, input_skeleton)
        return input_skeleton, new_animation

    return skeleton, animation
