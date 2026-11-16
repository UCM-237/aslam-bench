"""Un episodio completo en Gazebo: simulación + SLAM + Nav2 + explorador + monitor.

  ros2 launch aslam_bench episode.launch.py world:=rooms_small \
      map_yaml:=$(ros2 pkg prefix aslam_bench)/share/aslam_bench/maps/rooms_small.yaml \
      x:=1.5 y:=5.0 yaw:=0.0 strategy:=B1 seed:=1 out_dir:=/tmp/ep_test

El episodio termina solo (el monitor sale y el launch se apaga). Argumentos para
enchufar tus propios componentes:
  sim_launch   launch de simulación (por defecto sim_tb4.launch.py)
  slam_launch  launch de SLAM; vacío = slam_toolbox de turtlebot4_navigation.
               Para tu Graph SLAM: un launch que publique /map y TF map->odom.
  nav2_launch  launch de Nav2; vacío = turtlebot4_navigation/nav2.launch.py

Agente DRL (strategy:=DRL o DRL:<alias|ruta.pt>): se lanza `drl_explorer` en lugar de
`explorer` y NO se lanza Nav2 (la política publica /cmd_vel directamente).
  drl_weights      ruta al actor .pt (si se omite: drl.weights de platform.yaml)
  cmd_vel_stamped  true si la base espera geometry_msgs/TwistStamped (TB4 en Jazzy)

Para verlo en tiempo real (por defecto todo va sin ventanas, como en las campañas):
  headless     false = abre la GUI de Gazebo (por defecto true)
  rviz         true = abre RViz con mapa, scan, robot, plan de Nav2, fronteras y objetivo
  rviz_config  fichero .rviz (por defecto config/aslam_bench.rviz)
"""
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, EmitEvent, IncludeLaunchDescription,
                            RegisterEventHandler, TimerAction)
from launch.conditions import IfCondition, UnlessCondition
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (AndSubstitution, EqualsSubstitution, LaunchConfiguration,
                                  NotSubstitution, PathJoinSubstitution, PythonExpression)
from launch_ros.actions import Node


def _arg(name, default):
    return DeclareLaunchArgument(name, default_value=str(default))


def generate_launch_description():
    bench = get_package_share_directory('aslam_bench')
    tb4_nav = get_package_share_directory('turtlebot4_navigation')
    L = LaunchConfiguration
    args = [_arg('world', 'rooms_small'),
            DeclareLaunchArgument('map_yaml', default_value=PathJoinSubstitution(
                [bench, 'maps', [LaunchConfiguration('world'), '.yaml']])), _arg('x', 1.5), _arg('y', 5.0),
            _arg('yaw', 0.0), _arg('strategy', 'B1'), _arg('seed', 0), _arg('out_dir', '/tmp/aslam_bench_ep'),
            _arg('start_id', 0), _arg('rep', 0), _arg('loc_mode', 'slam'), _arg('platform', ''),
            _arg('t_max', 900.0), _arg('robot_model', 'turtlebot4'),
            _arg('sim_launch', bench + '/launch/sim_tb4.launch.py'),
            _arg('slam_launch', ''), _arg('nav2_launch', ''), _arg('explorer_delay', 15.0),
            _arg('drl_weights', ''), _arg('cmd_vel_stamped', 'false'),
            _arg('headless', 'true'), _arg('rviz', 'false'),
            _arg('rviz_config', bench + '/config/aslam_bench.rviz')]
    is_drl = PythonExpression(["'", LaunchConfiguration('strategy'), "'.upper().startswith('DRL')"])

    sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(L('sim_launch')),
        launch_arguments={'world': L('world'), 'x': L('x'), 'y': L('y'), 'yaw': L('yaw'),
                          'headless': L('headless')}.items())

    no_slam = EqualsSubstitution(L('slam_launch'), '')
    slam_default = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(tb4_nav + '/launch/slam.launch.py'),
        launch_arguments={'use_sim_time': 'true'}.items(), condition=IfCondition(no_slam))
    slam_custom = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(L('slam_launch')),
        launch_arguments={'use_sim_time': 'true'}.items(), condition=UnlessCondition(no_slam))

    # Nav2 solo para los métodos que eligen objetivos (el DRL emite velocidades)
    no_nav = EqualsSubstitution(L('nav2_launch'), '')
    nav_default = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(tb4_nav + '/launch/nav2.launch.py'),
        launch_arguments={'use_sim_time': 'true'}.items(),
        condition=IfCondition(AndSubstitution(no_nav, NotSubstitution(is_drl))))
    nav_custom = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(L('nav2_launch')),
        launch_arguments={'use_sim_time': 'true'}.items(),
        condition=IfCondition(AndSubstitution(NotSubstitution(no_nav), NotSubstitution(is_drl))))

    # Pose real de todos los modelos dinámicos (la publica el SceneBroadcaster de gz)
    gt_bridge = Node(package='ros_gz_bridge', executable='parameter_bridge', name='gt_bridge',
                     arguments=[['/world/', L('world'), '/dynamic_pose/info@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V']],
                     remappings=[(['/world/', L('world'), '/dynamic_pose/info'], '/aslam_bench/gt_poses')],
                     parameters=[{'use_sim_time': True}])

    common = {'use_sim_time': True, 'platform': L('platform'), 'strategy': L('strategy')}
    monitor = Node(package='aslam_bench', executable='episode_monitor', name='aslam_bench_monitor', output='screen',
                   parameters=[common, {'out_dir': L('out_dir'), 'map_name': L('world'), 'map_yaml': L('map_yaml'),
                                        'start_x': L('x'), 'start_y': L('y'), 'start_yaw': L('yaw'),
                                        'start_id': L('start_id'), 'rep': L('rep'), 'seed': L('seed'),
                                        'loc_mode': L('loc_mode'), 't_max': L('t_max'),
                                        'robot_model': L('robot_model'), 'backend': 'gazebo'}])
    explorer = Node(package='aslam_bench', executable='explorer', name='aslam_bench_explorer', output='screen',
                    parameters=[common, {'seed': L('seed')}], condition=UnlessCondition(is_drl))
    drl_explorer = Node(package='aslam_bench', executable='drl_explorer', name='aslam_bench_explorer',
                        output='screen', condition=IfCondition(is_drl),
                        parameters=[common, {'seed': L('seed'), 'drl_weights': L('drl_weights'),
                                             'cmd_vel_stamped': L('cmd_vel_stamped')}])

    rviz = Node(package='rviz2', executable='rviz2', name='rviz2', output='log',
                arguments=['-d', L('rviz_config')], parameters=[{'use_sim_time': True}],
                condition=IfCondition(L('rviz')))

    return LaunchDescription(args + [
        sim, slam_default, slam_custom, nav_default, nav_custom, gt_bridge, monitor, rviz,
        TimerAction(period=L('explorer_delay'), actions=[explorer, drl_explorer]),
        RegisterEventHandler(OnProcessExit(target_action=monitor, on_exit=[EmitEvent(event=Shutdown())])),
    ])
