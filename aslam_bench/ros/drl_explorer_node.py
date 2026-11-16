"""Nodo de exploración DRL: la política (TD3 de c3-aslamdrl-platform) mueve el robot.

A diferencia de `explorer` (B0, B1...), no envía objetivos a Nav2: publica velocidades
a `drl.step_period` (10 Hz, como en el entrenamiento). Usa la misma `ExplorerLogic` solo
para el criterio común de fin ("sin fronteras alcanzables") y los mismos eventos para
el monitor, así que métricas y análisis no cambian.

Suscripciones : /map (nav_msgs/OccupancyGrid), /scan (sensor_msgs/LaserScan),
                TF map -> base_link y base_link -> <frame del LiDAR>
Publica       : /cmd_vel (geometry_msgs/Twist o TwistStamped), /aslam_bench/explorer_events

Parámetros: strategy (DRL | DRL:<alias|ruta.pt>), drl_weights (ruta al .pt; tiene prioridad),
            seed, platform, map_topic, scan_topic, cmd_vel_topic, cmd_vel_stamped,
            global_frame, robot_frame.
"""
from __future__ import annotations

import json

import numpy as np
import rclpy
from geometry_msgs.msg import Twist, TwistStamped
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener

from ..core.config import load_platform
from ..core.explorer import NAV_IDLE, ExplorerLogic
from .common import grid_from_msg, lookup_pose2d

EVENTS_TOPIC = '/aslam_bench/explorer_events'


class DrlExplorerNode(Node):
    def __init__(self):
        super().__init__('aslam_bench_explorer')
        p = self.declare_parameter
        self.strategy = p('strategy', 'DRL').value
        seed = int(p('seed', 0).value)
        platform = p('platform', '').value or None
        weights = p('drl_weights', '').value or None
        self.map_topic = p('map_topic', '/map').value
        scan_topic = p('scan_topic', '/scan').value
        cmd_topic = p('cmd_vel_topic', '/cmd_vel').value
        self.stamped = bool(p('cmd_vel_stamped', False).value)
        self.global_frame = p('global_frame', 'map').value
        self.robot_frame = p('robot_frame', 'base_link').value

        self.cfg = load_platform(platform)
        kw = {'weights': weights} if weights else {}
        self.logic = ExplorerLogic(self.strategy, self.cfg, seed=seed, **kw)
        if not self.logic.velocity_mode:
            raise SystemExit(f'drl_explorer solo admite estrategias de velocidad (DRL...), no {self.strategy}')
        self.policy = self.logic.strategy
        self.policy.load()                     # falla aquí, no a mitad del episodio, si no hay pesos
        self.policy.reset()

        self.grid = None
        self.scan = None                        # (ranges, angles en el marco del robot)
        self.scan_yaw = {}                      # frame del LiDAR -> yaw respecto a robot_frame
        self.dec_times = []
        self.stopped = False

        qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                         reliability=ReliabilityPolicy.RELIABLE)
        self.create_subscription(OccupancyGrid, self.map_topic, self._on_map, qos)
        self.create_subscription(LaserScan, scan_topic, self._on_scan, qos_profile_sensor_data)
        self.cmd = self.create_publisher(TwistStamped if self.stamped else Twist, cmd_topic, 10)
        self.events = self.create_publisher(String, EVENTS_TOPIC, 10)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.create_timer(self.policy.step_period, self._step)
        self.create_timer(self.cfg['exploration']['replan_period'], self._tick)
        self.meta_sent = False
        self.get_logger().info(f'Explorador {self.policy.name} listo: {self.policy.policy.weights} '
                               f'(sha256 {self.policy.policy.weights_sha})')

    # ------------------------------------------------------------------ callbacks
    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _publish(self, kind, info=None):
        d = dict(t=self._now(), event=kind, info=info or {})
        self.events.publish(String(data=json.dumps(d, default=float)))

    def _on_map(self, msg):
        try:
            self.grid = grid_from_msg(msg)
        except ValueError as e:
            self.get_logger().error(str(e))

    def _on_scan(self, msg):
        frame = msg.header.frame_id
        if frame not in self.scan_yaw:
            pose = lookup_pose2d(self.tf_buffer, self.robot_frame, frame)
            if pose is None:
                self.get_logger().warn(f'Sin TF {self.robot_frame}->{frame}', throttle_duration_sec=5.0)
                return
            self.scan_yaw[frame] = pose[2]
        n = len(msg.ranges)
        angles = msg.angle_min + np.arange(n) * msg.angle_increment + self.scan_yaw[frame]
        r = np.asarray(msg.ranges, float)
        r = np.where(r < msg.range_min, np.nan, r)
        r = np.where(r > msg.range_max, np.inf, r)
        self.scan = (r, angles)

    def _send(self, v, w):
        tw = Twist()
        tw.linear.x, tw.angular.z = float(v), float(w)
        if self.stamped:
            m = TwistStamped()
            m.header.stamp = self.get_clock().now().to_msg()
            m.header.frame_id = self.robot_frame
            m.twist = tw
            self.cmd.publish(m)
        else:
            self.cmd.publish(tw)

    def _step(self):
        if self.stopped or self.grid is None or self.scan is None:
            return
        pose = lookup_pose2d(self.tf_buffer, self.global_frame, self.robot_frame)
        if pose is None:
            self.get_logger().warn('Sin TF map->base_link', throttle_duration_sec=5.0)
            return
        v, w, info = self.policy.act(*self.scan, self.grid, pose)
        self.dec_times.append(info['decision_time'])
        self._send(v, w)

    def _tick(self):
        if self.stopped or self.grid is None:
            return
        if not self.meta_sent:
            self._publish('meta', self.policy.describe())
            self.meta_sent = True
        pose = lookup_pose2d(self.tf_buffer, self.global_frame, self.robot_frame)
        if pose is None:
            return
        act = self.logic.update(self._now(), self.grid, pose[:2], NAV_IDLE)
        if self.dec_times:
            self._publish('policy', dict(act.info, decision_time=float(np.mean(self.dec_times))))
            self.dec_times = []
        if act.kind == 'done':
            self.stop()
            self._publish('done', act.info)
            self.get_logger().info('Sin fronteras alcanzables: exploración terminada')

    def stop(self):
        self.stopped = True
        for _ in range(3):
            self._send(0.0, 0.0)


def main(args=None):
    rclpy.init(args=args)
    node = DrlExplorerNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        try:
            node.stop()
        except Exception:
            pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
