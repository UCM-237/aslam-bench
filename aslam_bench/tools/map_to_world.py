"""Convierte un mapa de ocupación (PGM + YAML) en un mundo SDF para Gazebo (gz sim, Harmonic).

Cada celda ocupada se extruye a una caja de altura `--height`; las celdas se
fusionan en rectángulos para que el mundo tenga pocas entidades. Así el mapa
real (ground truth) del mundo es exactamente el YAML de partida.

Uso:  python -m aslam_bench.tools.map_to_world maps/rooms_small.yaml --out worlds/
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from ..core.grid import load_map_yaml


def merge_rectangles(occ):
    """Descompone una máscara booleana en rectángulos (i0, j0, i1, j1) exclusivos por arriba."""
    occ = occ.copy()
    h, w = occ.shape
    rects = []
    for i in range(h):
        j = 0
        while j < w:
            if not occ[i, j]:
                j += 1
                continue
            j1 = j
            while j1 < w and occ[i, j1]:
                j1 += 1
            i1 = i + 1
            while i1 < h and occ[i1, j:j1].all():
                i1 += 1
            occ[i:i1, j:j1] = False
            rects.append((i, j, i1, j1))
            j = j1
    return rects


WORLD_HEADER = """<?xml version="1.0" ?>
<sdf version="1.8">
  <world name="{name}">
    <physics name="1ms" type="ignored">
      <max_step_size>0.003</max_step_size>
      <real_time_factor>1.0</real_time_factor>
    </physics>
    <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"/>
    <plugin filename="gz-sim-user-commands-system" name="gz::sim::systems::UserCommands"/>
    <plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"/>
    <plugin filename="gz-sim-contact-system" name="gz::sim::systems::Contact"/>
    <plugin filename="gz-sim-imu-system" name="gz::sim::systems::Imu"/>
    <plugin filename="gz-sim-sensors-system" name="gz::sim::systems::Sensors">
      <render_engine>ogre2</render_engine>
    </plugin>
    <gui fullscreen="0"/>
    <light type="directional" name="sun">
      <cast_shadows>true</cast_shadows>
      <pose>0 0 10 0 0 0</pose>
      <diffuse>0.8 0.8 0.8 1</diffuse>
      <specular>0.2 0.2 0.2 1</specular>
      <direction>-0.5 0.1 -0.9</direction>
    </light>
    <model name="ground_plane">
      <static>true</static>
      <link name="link">
        <collision name="collision"><geometry><plane><normal>0 0 1</normal><size>200 200</size></plane></geometry></collision>
        <visual name="visual"><geometry><plane><normal>0 0 1</normal><size>200 200</size></plane></geometry>
          <material><ambient>0.8 0.8 0.8 1</ambient><diffuse>0.8 0.8 0.8 1</diffuse></material></visual>
      </link>
    </model>
    <model name="{name}_walls">
      <static>true</static>
      <link name="link">
"""

WORLD_FOOTER = """      </link>
    </model>
  </world>
</sdf>
"""

BOX = """        <collision name="c{k}"><pose>{x:.3f} {y:.3f} {z:.3f} 0 0 0</pose><geometry><box><size>{sx:.3f} {sy:.3f} {sz:.3f}</size></box></geometry></collision>
        <visual name="v{k}"><pose>{x:.3f} {y:.3f} {z:.3f} 0 0 0</pose><geometry><box><size>{sx:.3f} {sy:.3f} {sz:.3f}</size></box></geometry>
          <material><ambient>0.6 0.6 0.65 1</ambient><diffuse>0.6 0.6 0.65 1</diffuse></material></visual>
"""


def map_to_sdf(yaml_path, name=None, height=1.0):
    g = load_map_yaml(yaml_path)
    name = name or os.path.splitext(os.path.basename(yaml_path))[0]
    res, (ox, oy) = g.resolution, g.origin
    parts = [WORLD_HEADER.format(name=name)]
    for k, (i0, j0, i1, j1) in enumerate(merge_rectangles(g.occ_mask())):
        parts.append(BOX.format(k=k, x=ox + (j0 + j1) / 2 * res, y=oy + (i0 + i1) / 2 * res,
                                z=height / 2, sx=(j1 - j0) * res, sy=(i1 - i0) * res, sz=height))
    parts.append(WORLD_FOOTER)
    return ''.join(parts)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('maps', nargs='+')
    ap.add_argument('--out', default='worlds')
    ap.add_argument('--height', type=float, default=1.0)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    for m in a.maps:
        name = os.path.splitext(os.path.basename(m))[0]
        sdf = map_to_sdf(m, name, a.height)
        with open(os.path.join(a.out, name + '.sdf'), 'w') as f:
            f.write(sdf)
        print(f'{name}.sdf: {sdf.count("<collision")} cajas')


if __name__ == '__main__':
    main()
