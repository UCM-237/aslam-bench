"""Simulación por defecto: gz sim (Harmonic) con un mundo de aslam_bench + TurtleBot 4 Lite.

TurtleBot 4 Lite = iRobot Create 3 + RPLIDAR A1 + cámara OAK-D Lite, es decir, la
plataforma de referencia. La cámara se simula y publica, pero la iteración 1 no la usa.

Probado solo en papel (no había ROS en el entorno de desarrollo). Supone ROS 2 Jazzy +
turtlebot4_simulator (paquete turtlebot4_gz_bringup). En Humble el paquete se llama
turtlebot4_ignition_bringup: cambia `spawn_pkg` abajo. Si ya tienes tu propia
simulación del Create 3, sustituye este fichero con el argumento `sim_launch` de
episode.launch.py: solo tiene que aceptar world, x, y, yaw (y opcionalmente headless) y
publicar /scan, /tf y /clock.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (AppendEnvironmentVariable, DeclareLaunchArgument,
                            IncludeLaunchDescription)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node

spawn_pkg = 'turtlebot4_gz_bringup'


def generate_launch_description():
    bench = get_package_share_directory('aslam_bench')
    worlds = os.path.join(bench, 'worlds')
    args = [DeclareLaunchArgument('world', default_value='rooms_small'),
            DeclareLaunchArgument('model', default_value='lite'),
            DeclareLaunchArgument('x', default_value='1.5'),
            DeclareLaunchArgument('y', default_value='5.0'),
            DeclareLaunchArgument('yaw', default_value='0.0'),
            DeclareLaunchArgument('headless', default_value='true')]
    world = LaunchConfiguration('world')

    env = [AppendEnvironmentVariable('GZ_SIM_RESOURCE_PATH', worlds)]
    for pkg in ('turtlebot4_description', 'irobot_create_description'):
        try:
            env.append(AppendEnvironmentVariable(
                'GZ_SIM_RESOURCE_PATH', os.path.dirname(get_package_share_directory(pkg))))
        except Exception:
            pass

    gz_args = PythonExpression([
        "'", PathJoinSubstitution([worlds, world]), ".sdf -r' + (' -s --headless-rendering' if '",
        LaunchConfiguration('headless'), "' == 'true' else '')"])
    gz = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution(
            [get_package_share_directory('ros_gz_sim'), 'launch', 'gz_sim.launch.py'])),
        launch_arguments={'gz_args': gz_args}.items())

    clock = Node(package='ros_gz_bridge', executable='parameter_bridge', name='clock_bridge',
                 arguments=['/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock'], output='screen')

    spawn = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution(
            [get_package_share_directory(spawn_pkg), 'launch', 'turtlebot4_spawn.launch.py'])),
        launch_arguments={'world': world, 'model': LaunchConfiguration('model'),
                          'x': LaunchConfiguration('x'), 'y': LaunchConfiguration('y'),
                          'z': '0.0', 'yaw': LaunchConfiguration('yaw')}.items())
    return LaunchDescription(args + env + [gz, clock, spawn])
