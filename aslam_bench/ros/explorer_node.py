"""Nodo de exploración: envuelve ExplorerLogic (B0..B8) y envía objetivos a Nav2.

Suscripciones : /map (nav_msgs/OccupancyGrid), TF map -> base_link
                semantic_topic (std_msgs/String, JSON) mapa semántico para B8: lista de
                landmarks {id, x, y, probs|alpha, state, n_obs} o {classes, landmarks}
                en el marco global; si nadie lo publica, B8 trabaja sin semántica
Acción        : navigate_to_pose (nav2_msgs/action/NavigateToPose)
Publica       : /aslam_bench/explorer_events (std_msgs/String, JSON) para el monitor
                /aslam_bench/markers (visualization_msgs/MarkerArray) fronteras y objetivo, para RViz

Parámetros: strategy (B0..B8, B7:<pesos.npz>), seed, platform (ruta a platform.yaml),
            map_topic, global_frame, robot_frame, nav_action, semantic_topic.

B4, B5 y B8 usan el grafo de poses sustituto de core/posegraph.py, construido con la
trayectoria map -> base_link (no el grafo interno del SLAM).
"""
from __future__ import annotations

import json

import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import Point, PoseStamped
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import OccupancyGrid
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import ColorRGBA, String
from tf2_ros import Buffer, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

from ..core.config import load_platform
from ..core.explorer import NAV_ACTIVE, NAV_FAILED, NAV_IDLE, NAV_SUCCEEDED, ExplorerLogic
from .common import grid_from_msg, lookup_pose2d, quat_from_yaw

EVENTS_TOPIC = '/aslam_bench/explorer_events'
MARKERS_TOPIC = '/aslam_bench/markers'


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
        semantic_topic = p('semantic_topic', '/aslam_bench/semantic_map').value

        self.cfg = load_platform(platform)
        self.logic = ExplorerLogic(self.strategy, self.cfg, seed=seed)
        self.grid = None
        self.nav_status = NAV_IDLE
        self.goal_handle = None
        self.goal_seq = 0
        self.meta_sent = False

        qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                         reliability=ReliabilityPolicy.RELIABLE)
        self.create_subscription(OccupancyGrid, self.map_topic, self._on_map, qos)
        self.create_subscription(String, semantic_topic, self._on_semantic, qos)
        self.events = self.create_publisher(String, EVENTS_TOPIC, 10)
        self.markers = self.create_publisher(MarkerArray, MARKERS_TOPIC, qos)
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

    def _on_semantic(self, msg):
        try:
            self.logic.semantic.from_json(msg.data)
        except (ValueError, KeyError, TypeError) as e:
            self.get_logger().error(f'Mapa semántico no válido: {e}', throttle_duration_sec=10.0)

    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _publish(self, kind, goal=None, info=None):
        d = dict(t=self._now(), event=kind, info=info or {})
        if goal is not None:
            d.update(x=goal.x, y=goal.y, yaw=goal.yaw, distance=goal.distance, goal_info=goal.info)
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
        if not self.meta_sent:
            self._publish('meta', dict(strategy_params=self.logic.strategy.params()))
            self.meta_sent = True
        act = self.logic.update(self._now(), self.grid, pose, self.nav_status)
        if act.kind == 'goal':
            self._send_goal(act.goal)
            self._publish('goal', act.goal, act.info)
            self._publish_markers(act.goal)
        elif act.kind == 'done':
            self._publish_markers(None)
            self._cancel()
            self._publish('meta', dict(self.logic.stats(), **{
                f'sem_{k}': v for k, v in self.logic.semantic.summary().items()}))
            self._publish('done', None, act.info)
            self.get_logger().info('Sin fronteras alcanzables: exploración terminada')

    # ------------------------------------------------------------------ RViz
    def _publish_markers(self, goal):
        """Fronteras abiertas (puntos naranja = celdas, esferas azules = objetivos candidatos)
        y objetivo elegido (flecha verde). Solo para visualizar; no afecta a las métricas."""
        stamp = self.get_clock().now().to_msg()

        def marker(mid, mtype, scale, rgba):
            m = Marker()
            m.header.frame_id, m.header.stamp = self.global_frame, stamp
            m.ns, m.id, m.type, m.action = 'aslam_bench', mid, mtype, Marker.ADD
            m.pose.orientation.w = 1.0
            m.scale.x, m.scale.y, m.scale.z = scale
            m.color = ColorRGBA(r=rgba[0], g=rgba[1], b=rgba[2], a=rgba[3])
            return m

        def pt(x, y, z=0.05):
            return Point(x=float(x), y=float(y), z=float(z))

        res, (ox, oy) = self.grid.resolution, self.grid.origin
        cells = marker(0, Marker.POINTS, (res, res, 0.0), (1.0, 0.5, 0.0, 0.9))
        cands = marker(1, Marker.SPHERE_LIST, (0.15, 0.15, 0.15), (0.2, 0.4, 1.0, 0.8))
        for f in self.logic.last_frontiers:
            cells.points += [pt(ox + (j + 0.5) * res, oy + (i + 0.5) * res) for i, j in f.cells]
            cands.points.append(pt(*f.goal))
        arrow = marker(2, Marker.ARROW, (0.5, 0.08, 0.08), (0.1, 0.9, 0.1, 1.0))
        if goal is None:
            arrow.action = Marker.DELETE
        else:
            arrow.pose.position = pt(goal.x, goal.y, 0.1)
            qx, qy, qz, qw = quat_from_yaw(goal.yaw)
            arrow.pose.orientation.x, arrow.pose.orientation.y = qx, qy
            arrow.pose.orientation.z, arrow.pose.orientation.w = qz, qw
        # datos propios de la estrategia (B3: árboles RRT y puntos frontera; B6: árbol NBV)
        extra = self.logic.strategy.viz() or {}
        pts = marker(3, Marker.POINTS, (0.08, 0.08, 0.0), (0.85, 0.2, 0.85, 0.9))
        pts.points = [pt(x, y, 0.08) for x, y in extra.get('points', [])]
        segs = marker(4, Marker.LINE_LIST, (0.01, 0.0, 0.0), (0.4, 0.4, 0.4, 0.6))
        for (x0, y0), (x1, y1) in extra.get('segments', []):
            segs.points += [pt(x0, y0, 0.02), pt(x1, y1, 0.02)]
        for m in (pts, segs):
            if not m.points:
                m.action = Marker.DELETE
        self.markers.publish(MarkerArray(markers=[cells, cands, arrow, pts, segs]))

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
