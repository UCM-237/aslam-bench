import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'aslam_bench'


def files(pattern):
    return [f for f in glob(pattern) if os.path.isfile(f)]


setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['tests']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', files('launch/*.py')),
        ('share/' + package_name + '/config', files('config/*.yaml')),
        ('share/' + package_name + '/config/campaigns', files('config/campaigns/*.yaml')),
        ('share/' + package_name + '/maps', files('maps/*')),
        ('share/' + package_name + '/worlds', files('worlds/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Jesús',
    description='Banco de pruebas de exploración activa (Active SLAM) para Create 3 + LiDAR 2D',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'explorer = aslam_bench.ros.explorer_node:main',
            'episode_monitor = aslam_bench.ros.episode_monitor_node:main',
            'run_campaign = aslam_bench.tools.run_campaign:main',
            'analyze = aslam_bench.tools.analyze:main',
            'make_maps = aslam_bench.tools.make_maps:main',
            'map_to_world = aslam_bench.tools.map_to_world:main',
        ],
    },
)
