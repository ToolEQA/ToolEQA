"""Coordinate conversions shared by online tools and RFT verification."""

from __future__ import annotations

from typing import Any

import numpy as np


def depth_boxes_to_habitat_world(
    boxes: Any,
    depth_image: Any,
    camera_pose: Any,
    hfov_degrees: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Lift 2D detections with Habitat depth into metric world coordinates.

    The returned sizes follow ``[length, width, height]``.  Length is the
    robust depth spread inside the central half of the box; width and height
    are obtained by pinhole back-projection at the median object depth.  This
    geometry is used to anchor DetAny3D's monocular estimates to the simulator
    depth sensor instead of trusting monocular absolute depth.
    """
    boxes_array = np.asarray(boxes, dtype=np.float64)
    depth = np.asarray(depth_image, dtype=np.float64)
    pose = np.asarray(camera_pose, dtype=np.float64)
    if boxes_array.size == 0:
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 3), dtype=np.float64)
    if boxes_array.ndim == 1:
        boxes_array = boxes_array.reshape(1, -1)
    if boxes_array.ndim != 2 or boxes_array.shape[1] < 4:
        raise ValueError(f"Expected boxes with shape (N, 4), got {boxes_array.shape}")
    if depth.ndim != 2:
        raise ValueError(f"Expected a 2D depth image, got {depth.shape}")
    if pose.shape != (4, 4):
        raise ValueError(f"Expected a 4x4 camera pose, got {pose.shape}")
    if not 0.0 < float(hfov_degrees) < 180.0:
        raise ValueError(f"Invalid horizontal field of view: {hfov_degrees}")

    height, width = depth.shape
    focal = width / (2.0 * np.tan(np.deg2rad(float(hfov_degrees)) / 2.0))
    principal_x = (width - 1.0) / 2.0
    principal_y = (height - 1.0) / 2.0
    centers: list[np.ndarray] = []
    sizes: list[list[float]] = []

    for raw_box in boxes_array:
        x1 = int(np.clip(np.floor(raw_box[0]), 0, width - 1))
        y1 = int(np.clip(np.floor(raw_box[1]), 0, height - 1))
        x2 = int(np.clip(np.ceil(raw_box[2]), x1 + 1, width))
        y2 = int(np.clip(np.ceil(raw_box[3]), y1 + 1, height))
        inset_x = max((x2 - x1) // 4, 0)
        inset_y = max((y2 - y1) // 4, 0)
        central = depth[y1 + inset_y : y2 - inset_y, x1 + inset_x : x2 - inset_x]
        valid = central[np.isfinite(central) & (central > 0.0)]
        if valid.size == 0:
            continue

        metric_depth = float(np.median(valid))
        # Bounding boxes use half-open pixel edges.  Subtracting one before
        # averaging maps a full-image box to the camera principal point.
        pixel_x = (float(raw_box[0]) + float(raw_box[2]) - 1.0) / 2.0
        pixel_y = (float(raw_box[1]) + float(raw_box[3]) - 1.0) / 2.0
        camera_point = np.array(
            [
                (pixel_x - principal_x) / focal * metric_depth,
                -(pixel_y - principal_y) / focal * metric_depth,
                -metric_depth,
            ],
            dtype=np.float64,
        )
        centers.append(camera_point @ pose[:3, :3].T + pose[:3, 3])

        low, high = np.percentile(valid, [15.0, 85.0])
        object_length = max(float(high - low), 0.05)
        object_width = max(float(raw_box[2] - raw_box[0]) / focal * metric_depth, 0.05)
        object_height = max(float(raw_box[3] - raw_box[1]) / focal * metric_depth, 0.05)
        sizes.append([object_length, object_width, object_height])

    return (
        np.asarray(centers, dtype=np.float64).reshape(-1, 3),
        np.asarray(sizes, dtype=np.float64).reshape(-1, 3),
    )


def detany_camera_to_habitat_world(points: Any, camera_pose: Any) -> np.ndarray:
    """Transform DetAny3D camera-frame points into Habitat world coordinates.

    DetAny3D follows the pinhole/OpenCV convention: +x points right, +y down,
    and +z forward.  Habitat sensors use +x right, +y up, and look along -z.
    ``camera_pose`` is the 4x4 Habitat sensor-to-world transform.
    """
    points_array = np.asarray(points, dtype=np.float64)
    pose_array = np.asarray(camera_pose, dtype=np.float64)

    if points_array.ndim == 1:
        points_array = points_array.reshape(1, -1)
    if points_array.ndim != 2 or points_array.shape[1] != 3:
        raise ValueError(f"Expected points with shape (N, 3), got {points_array.shape}")
    if pose_array.shape != (4, 4):
        raise ValueError(f"Expected a 4x4 camera pose, got {pose_array.shape}")
    if not np.isfinite(points_array).all() or not np.isfinite(pose_array).all():
        raise ValueError("Points and camera pose must contain only finite values")

    habitat_camera_points = points_array * np.array([1.0, -1.0, -1.0])
    return habitat_camera_points @ pose_array[:3, :3].T + pose_array[:3, 3]
