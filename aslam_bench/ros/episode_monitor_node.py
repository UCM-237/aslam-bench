"""Monitor de episodio: registra datos, evalúa la cobertura en línea y termina el episodio.

Escribe exactamente el mismo formato que el simulador 2D (EpisodeRecorder), así
que `analyze` procesa igual los episodios de Gazebo, del simulador 2D y del robot real.

Suscripciones:
  /map                              nav_msgs/OccupancyGrid (mapa del SLAM)
  TF map -> base_link               pose estimada
  gt_topic (tf2_msgs/TFMessage)     pose real desde Gazebo (dynamic_pose/info puenteado)
  /aslam_bench/explorer_events      eventos del explorador (objetivos, 'done', fallos)
  /hazard_detection (opcional)      choques del Create 3 (irobot_create_msgs)

Terminación (igual que en el simulador 2D): cobertura >= objetivo, explorador sin
fronteras, t >= t_max o estancamiento. Al terminar guarda meta.json y sale, y el
launch cierra todo (OnProcessExit -> Shutdown).

En el robot real no hay ground truth: con `map_yaml` vacío solo se registra (la
cobertura se calculará después con un plano medido) y la parada es manual o por t_max.
"""
from __future__ import annotations

import json
import os

import rclpy
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String
from tf2_msgs.msg import TFMessage
from tf2_ros import Buffer, TransformListener

from ..core.config import load_platform
from ..core.grid import load_map_yaml
from ..core.metrics import CoverageEvaluator
from ..core.recorder import EpisodeRecorder
from ..core.strategies import make_strategy
from .common import grid_from_msg, lookup_pose2d, yaw_from_quat
from .explorer_node import EVENTS_TOPIC


class _G:  # objetivo mínimo para EpisodeRecorder.log_goal
    def __init__(self, d):
        self.x, self.y, self.yaw, self.distance = d.get('x'), d.get('y'), d.get('yaw'), d.get('distance')


class EpisodeMonitor(Node):
    def __init__(self):
        super().__init__('aslam_bench_monitor')
        p = self.declare_parameter
        out_dir = p('out_dir', '/tmp/aslam_bench_episode').value
        strategy = p('strategy', 'B1').value
        self.map_name = p('map_name', 'unknown').value
        map_yaml = p('map_yaml', '').value
        self.start = (float(p('start_x', 0.0).value), float(p('start_y', 0.0).value),
                      float(p('start_yaw', 0.0).value))
        platform = p('platform', '').value or None
        self.gt_topic = p('gt_topic', '/aslam_bench/gt_poses').value
        self.robot_model = p('robot_model', 'turtlebot4').value
        self.global_frame = p('global_frame', 'map').value
        self.robot_frame = p('robot_frame', 'base_link').value
        self.cfg = load_platform(platform)
        ep = self.cfg['episode']
        self.t_max = float(p('t_max', ep['t_max']).value)

        self.gt = load_map_yaml(map_yaml) if map_yaml else None
        self.cov = CoverageEvaluator(self.gt, self.start, self.cfg) if self.gt is not None else None
        meta = dict(method=make_strategy(strategy, self.cfg).name, backend=p('backend', 'gazebo').value,
                    loc_mode=p('loc_mode', 'slam').value, map=self.map_name,
                    start_id=int(p('start_id', 0).value), rep=int(p('rep', 0).value),
                    seed=int(p('seed', 0).value), start_pose=list(self.start), t_max=self.t_max,
                    coverage_target=ep['coverage_target'], platform=self.cfg['platform']['name'],
                    reachable_area_m2=self.cov.area_m2 if self.cov else None)
        self.rec = EpisodeRecorder(out_dir, meta)

        self.grid = None
        self.gt_pose = None
        self.t0 = None
        self.cov_hist = []
        self.explorer_done = False
        self.n_collisions = 0
        self.last_bump = -1e9
        self.finished = False

        qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                         reliability=ReliabilityPolicy.RELIABLE)
        self.create_subscription(OccupancyGrid, p('map_topic', '/map').value, self._on_map, qos)
        self.create_subscription(TFMessage, self.gt_topic, self._on_gt, 10)
        self.create_subscription(String, EVENTS_TOPIC, self._on_event, 50)
        try:
            from irobot_create_msgs.msg import HazardDetectionVector
            self.create_subscription(HazardDetectionVector, '/hazard_detection', self._on_hazard, 10)
        except ImportError:
            self.get_logger().warn('irobot_create_msgs no disponible: no se registrarán choques')
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.create_timer(ep['pose_period'], self._log_pose)
        self.create_timer(ep['snapshot_period'], self._snapshot)
        self.get_logger().info(f'Monitor: {meta["method"]} en {self.map_name}, salida {out_dir}')

    # ------------------------------------------------------------------ helpers
    def _t(self):
        now = self.get_clock().now().nanoseconds * 1e-9
        if self.t0 is None:
            return None
        return now - self.t0

    def _on_map(self, msg):
        try:
            self.grid = grid_from_msg(msg)
        except ValueError as e:
            self.get_logger().error(str(e))
            return
        if self.t0 is None:
            self.t0 = self.get_clock().now().nanoseconds * 1e-9
            self.get_logger().info('Primer mapa recibido: empieza el episodio')

    def _on_gt(self, msg):
        for tr in msg.transforms:
            if tr.child_frame_id == self.robot_model or tr.child_frame_id.endswith('/' + self.robot_model):
                t, q = tr.transform.translation, tr.transform.rotation
                self.gt_pose = (t.x, t.y, yaw_from_quat(q))
                return

    def _on_event(self, msg):
        t = self._t()
        if t is None:
            return
        d = json.loads(msg.data)
        if d['event'] == 'goal':
            self.rec.log_goal(t, 'goal', _G(d), d.get('info'))
        elif d['event'] == 'done':
            self.rec.log_goal(t, 'done', None, d.get('info'))
            self.explorer_done = True
        elif d['event'] == 'policy':        # política de velocidad (DRL): tiempo de inferencia
            self.rec.log_goal(t, 'policy', None, d.get('info'))
        elif d['event'] == 'meta':          # datos del agente (ruta y hash de los pesos)
            self.rec.meta.update(d.get('info') or {})
        elif d['event'] == 'nav_failed':
            self.rec.log_event(t, 'nav_failed', json.dumps(d.get('info', {})))

    def _on_hazard(self, msg):
        t = self._t()
        if t is None:
            return
        for h in msg.detections:
            if h.type == 1 and t - self.last_bump > 1.0:   # HazardDetection.BUMP
                self.n_collisions += 1
                self.last_bump = t
                self.rec.log_event(t, 'collision', h.header.frame_id)

    def _log_pose(self):
        t = self._t()
        if t is None or self.finished:
            return
        est = lookup_pose2d(self.tf_buffer, self.global_frame, self.robot_frame)
        if est is None:
            return
        gt = self.gt_pose if self.gt_pose is not None else (float('nan'),) * 3
        self.rec.log_pose(t, gt, est)

    def _snapshot(self):
        t = self._t()
        if t is None or self.grid is None or self.finished:
            return
        self.rec.snapshot_map(t, self.grid)
        ep = self.cfg['episode']
        term = None
        if self.cov is not None:
            c = self.cov.coverage(self.grid)
            self.cov_hist.append((t, c))
            self.get_logger().info(f't={t:.0f}s cobertura={c:.3f}', throttle_duration_sec=30.0)
            past = [cv for (tt, cv) in self.cov_hist if tt <= t - ep['stall_window']]
            if c >= ep['coverage_target']:
                term = 'coverage_reached'
            elif past and c - past[-1] < ep['stall_min_gain']:
                term = 'stalled'
        if self.explorer_done:
            term = term or 'no_frontiers'
        if t >= self.t_max:
            term = term or 'timeout'
        if term:
            self._finish(term, t)

    def _finish(self, term, t):
        self.finished = True
        self.rec.finish(term, duration=t, n_collisions=self.n_collisions)
        self.get_logger().info(f'Episodio terminado: {term} (t={t:.0f}s)')
        raise SystemExit


def main(args=None):
    rclpy.init(args=args)
    node = EpisodeMonitor()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        if not node.finished:
            node.rec.finish('interrupted', duration=node._t() or 0.0, n_collisions=node.n_collisions)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
