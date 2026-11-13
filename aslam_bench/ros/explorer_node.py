"""Nodo de exploración: envuelve ExplorerLogic (B0, B1, ...) y envía objetivos a Nav2.

Suscripciones : /map (nav_msgs/OccupancyGrid), TF map -> base_link
Acción        : navigate_to_pose (nav2_msgs/action/NavigateToPose)
Publica       : /aslam_bench/explorer_events (std_msgs/String, JSON) para el monitor

Parámetros: strategy (B0|B1|...), seed, platform (ruta a platform.yaml),
            map_topic, global_frame, robot_frame, nav_action.
"""
from __future__ import annotations

import json

import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import OccupancyGrid
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener

from ..core.config import load_platform
from ..core.explorer import NAV_ACTIVE, NAV_FAILED, NAV_IDLE, NAV_SUCCEEDED, ExplorerLogic
from .common import grid_from_msg, lookup_pose2d, quat_from_yaw

EVENTS_TOPIC = '/aslam_bench/explorer_events'


class ExplorerNode(Node):
    def __init__(self):
        super().__init__('aslam_bench_explorer')
        p = self.declare_parameter
        self.strategy = p('strategy', 'B1').value
        seed = int(p('seed', 0).value)
        platform = p('platform', '').value or None
        self.map_topic = p('map_topic', '/map').value
        self.global_frame = p('global_frame', 'map').value
        self.robot_frame = p('robot_frame', 'base_link').value
        nav_action = p('nav_action', 'navigate_to_pose').value

        self.cfg = load_platform(platform)
        self.logic = ExplorerLogic(self.strategy, self.cfg, seed=seed)
        self.grid = None
        self.nav_status = NAV_IDLE
        self.goal_handle = None
        self.goal_seq = 0

        qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                         reliability=ReliabilityPolicy.RELIABLE)
        self.create_subscription(OccupancyGrid, self.map_topic, self._on_map, qos)
        self.events = self.create_publisher(String, EVENTS_TOPIC, 10)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.nav = ActionClient(self, NavigateToPose, nav_action)
        self.create_timer(self.cfg['exploration']['replan_period'], self._tick)
        self.get_logger().info(f'Explorador {self.logic.strategy.name} listo (seed={seed})')

    # ------------------------------------------------------------------ callbacks
    def _on_map(self, msg):
        try:
            self.grid = grid_from_msg(msg)
        except ValueError as e:
            self.get_logger().error(str(e))

    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _publish(self, kind, goal=None, info=None):
        d = dict(t=self._now(), event=kind, info=info or {})
        if goal is not None:
            d.update(x=goal.x, y=goal.y, yaw=goal.yaw, distance=goal.distance)
        self.events.publish(String(data=json.dumps(d, default=float)))

    def _tick(self):
        if self.logic.done or self.grid is None:
            return
        if not self.nav.server_is_ready():
            self.get_logger().info('Esperando a Nav2...', throttle_duration_sec=10.0)
            return
        pose = lookup_pose2d(self.tf_buffer, self.global_frame, self.robot_frame)
        if pose is None:
            self.get_logger().warn('Sin TF map->base_link', throttle_duration_sec=5.0)
            return
        act = self.logic.update(self._now(), self.grid, pose[:2], self.nav_status)
        if act.kind == 'goal':
            self._send_goal(act.goal)
            self._publish('goal', act.goal, act.info)
        elif act.kind == 'done':
            self._cancel()
            self._publish('done', None, act.info)
            self.get_logger().info('Sin fronteras alcanzables: exploración terminada')

    # ------------------------------------------------------------------ Nav2
    def _cancel(self):
        if self.goal_handle is not None:
            self.goal_handle.cancel_goal_async()
            self.goal_handle = None

    def _send_goal(self, goal):
        self._cancel()
        self.goal_seq += 1
        seq = self.goal_seq
        msg = NavigateToPose.Goal()
        ps = PoseStamped()
        ps.header.frame_id = self.global_frame
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.pose.position.x, ps.pose.position.y = goal.x, goal.y
        qx, qy, qz, qw = quat_from_yaw(goal.yaw)
        ps.pose.orientation.x, ps.pose.orientation.y = qx, qy
        ps.pose.orientation.z, ps.pose.orientation.w = qz, qw
        msg.pose = ps
        self.nav_status = NAV_ACTIVE
        fut = self.nav.send_goal_async(msg)
        fut.add_done_callback(lambda f: self._on_accepted(f, seq))

    def _on_accepted(self, fut, seq):
        if seq != self.goal_seq:
            return
        handle = fut.result()
        if not handle.accepted:
            self.nav_status = NAV_FAILED
            self._publish('nav_failed', info=dict(reason='rejected'))
            return
        self.goal_handle = handle
        handle.get_result_async().add_done_callback(lambda f: self._on_result(f, seq))

    def _on_result(self, fut, seq):
        if seq != self.goal_seq:
            return   # resultado de un objetivo ya sustituido
        status = fut.result().status
        self.goal_handle = None
        if status == GoalStatus.STATUS_SUCCEEDED:
            self.nav_status = NAV_SUCCEEDED
        else:
            self.nav_status = NAV_FAILED
            self._publish('nav_failed', info=dict(status=int(status)))


def main(args=None):
    rclpy.init(args=args)
    node = ExplorerNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
