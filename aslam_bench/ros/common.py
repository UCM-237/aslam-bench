"""Utilidades ROS 2 compartidas por los nodos de aslam_bench."""
from __future__ import annotations

import math

import numpy as np

from ..core.grid import OccGrid


def grid_from_msg(msg) -> OccGrid:
    """nav_msgs/OccupancyGrid -> OccGrid (se asume origen sin rotación, como en slam_toolbox)."""
    info = msg.info
    q = info.origin.orientation
    yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
    if abs(yaw) > 1e-3:
        raise ValueError('OccupancyGrid con origen rotado no soportado')
    data = np.asarray(msg.data, dtype=np.int8).reshape(info.height, info.width)
    return OccGrid(data, float(info.resolution), (float(info.origin.position.x), float(info.origin.position.y)))


def yaw_from_quat(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


def quat_from_yaw(yaw):
    return (0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2))


def lookup_pose2d(tf_buffer, target, source, time=None):
    """Pose 2D (x, y, yaw) de `source` en `target` vía TF, o None."""
    from rclpy.time import Time
    try:
        tr = tf_buffer.lookup_transform(target, source, time or Time())
    except Exception:
        return None
    t, q = tr.transform.translation, tr.transform.rotation
    return (t.x, t.y, yaw_from_quat(q))
