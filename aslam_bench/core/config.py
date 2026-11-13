"""Carga de la configuración de la plataforma de referencia."""
from __future__ import annotations

import copy
import os

import yaml

PKG_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_PLATFORM = os.path.join(PKG_ROOT, 'config', 'platform.yaml')


def _deep_update(base, upd):
    for k, v in (upd or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_platform(path=None, overrides=None):
    path = path or os.environ.get('ASLAM_BENCH_PLATFORM') or DEFAULT_PLATFORM
    if not os.path.exists(path):
        # instalación con colcon: config dentro de share/aslam_bench
        try:
            from ament_index_python.packages import get_package_share_directory
            path = os.path.join(get_package_share_directory('aslam_bench'), 'config', 'platform.yaml')
        except Exception:  # pragma: no cover
            pass
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return _deep_update(copy.deepcopy(cfg), overrides)


def robot_clearance(cfg):
    """Radio usado para planificar: radio del robot + margen de inflado."""
    return cfg['platform']['base']['radius'] + cfg['navigation']['inflation_margin']


def resource_dir(name):
    """Directorio de recursos ('maps', 'worlds', 'config'): árbol de fuentes o share/ de colcon."""
    d = os.path.join(PKG_ROOT, name)
    if os.path.isdir(d):
        return d
    try:
        from ament_index_python.packages import get_package_share_directory
        return os.path.join(get_package_share_directory('aslam_bench'), name)
    except Exception:  # pragma: no cover
        return d
