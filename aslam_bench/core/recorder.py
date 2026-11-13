"""Registro de episodios en un formato común (simulador 2D y ROS 2/Gazebo).

Estructura de un episodio:
  <ep_dir>/meta.json     método, mapa, pose inicial (mundo), semilla, plataforma, terminación...
  <ep_dir>/poses.csv     t, gt_x, gt_y, gt_yaw (mundo), est_x, est_y, est_yaw (marco 'map' del SLAM)
  <ep_dir>/goals.csv     t, event, x, y, yaw, distance, decision_time, n_frontiers, detail
  <ep_dir>/events.csv    t, kind, detail   (colisiones, fallos de navegación, ...)
  <ep_dir>/maps/*.npz    instantáneas del mapa estimado: t, data, resolution, origin

El marco 'map' del SLAM coincide con la pose inicial del robot (convenio de
slam_toolbox y de la mayoría de Graph SLAM): world = start_pose ⊕ map.
"""
from __future__ import annotations

import csv
import json
import os
import time

import numpy as np


class EpisodeRecorder:
    def __init__(self, out_dir, meta: dict):
        self.dir = out_dir
        os.makedirs(os.path.join(out_dir, 'maps'), exist_ok=True)
        self.meta = dict(meta)
        self.meta.setdefault('created', time.strftime('%Y-%m-%dT%H:%M:%S'))
        self._write_meta()
        self._pose_f = open(os.path.join(out_dir, 'poses.csv'), 'w', newline='')
        self._pose = csv.writer(self._pose_f)
        self._pose.writerow(['t', 'gt_x', 'gt_y', 'gt_yaw', 'est_x', 'est_y', 'est_yaw'])
        self._goal_f = open(os.path.join(out_dir, 'goals.csv'), 'w', newline='')
        self._goal = csv.writer(self._goal_f)
        self._goal.writerow(['t', 'event', 'x', 'y', 'yaw', 'distance', 'decision_time',
                             'n_frontiers', 'detail'])
        self._ev_f = open(os.path.join(out_dir, 'events.csv'), 'w', newline='')
        self._ev = csv.writer(self._ev_f)
        self._ev.writerow(['t', 'kind', 'detail'])
        self.n_snapshots = 0
        self.closed = False

    def _write_meta(self):
        with open(os.path.join(self.dir, 'meta.json'), 'w') as f:
            json.dump(self.meta, f, indent=2, default=float)

    def log_pose(self, t, gt, est):
        self._pose.writerow([f'{t:.3f}', *(f'{v:.4f}' for v in gt), *(f'{v:.4f}' for v in est)])

    def log_goal(self, t, event, goal=None, info=None):
        info = info or {}
        g = (goal.x, goal.y, goal.yaw, goal.distance) if goal is not None else ('', '', '', '')
        self._goal.writerow([f'{t:.3f}', event, *g, info.get('decision_time', ''),
                             info.get('n_frontiers', ''), info.get('prev_goal_end', '') or ''])

    def log_event(self, t, kind, detail=''):
        self._ev.writerow([f'{t:.3f}', kind, detail])

    def snapshot_map(self, t, grid):
        path = os.path.join(self.dir, 'maps', f'{self.n_snapshots:05d}.npz')
        np.savez_compressed(path, t=t, data=grid.data.astype(np.int8),
                            resolution=grid.resolution, origin=np.asarray(grid.origin, float))
        self.n_snapshots += 1

    def finish(self, termination, **extra):
        if self.closed:
            return
        self.meta.update(termination=termination, **extra)
        self._write_meta()
        for f in (self._pose_f, self._goal_f, self._ev_f):
            f.close()
        self.closed = True
