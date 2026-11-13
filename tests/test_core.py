"""Pruebas del núcleo (no requieren ROS):  pytest -q tests/"""
import math
import os

import numpy as np
import pytest

from aslam_bench.core import stats as st
from aslam_bench.core.config import load_platform
from aslam_bench.core.explorer import NAV_IDLE, NAV_SUCCEEDED, ExplorerLogic
from aslam_bench.core.frontiers import build_context, frontier_mask
from aslam_bench.core.grid import OccGrid, geodesic, load_map_yaml, save_map_yaml, se2_apply, se2_inverse_apply
from aslam_bench.core.metrics import CoverageEvaluator, map_quality

CFG = load_platform()


def room(w=60, h=40, unknown_right=True):
    """Habitación 3x2 m (res 0.05) con paredes; mitad derecha desconocida si se pide."""
    d = np.zeros((h, w), np.int8)
    d[0, :] = d[-1, :] = d[:, 0] = d[:, -1] = 100
    if unknown_right:
        d[1:-1, w // 2:-1] = -1
    return OccGrid(d, 0.05, (0.0, 0.0))


def test_world_cell_roundtrip():
    g = room()
    i, j = g.world_to_cell(1.23, 0.77)
    x, y = g.cell_to_world(i, j)
    assert abs(x - 1.23) <= 0.025 and abs(y - 0.77) <= 0.025


def test_se2_inverse():
    pose = (1.0, -2.0, 0.7)
    x, y = se2_apply(pose, 0.3, 0.4)
    xb, yb = se2_inverse_apply(pose, x, y)
    assert np.allclose([xb, yb], [0.3, 0.4])


def test_map_yaml_roundtrip(tmp_path):
    g = room()
    save_map_yaml(g, str(tmp_path / 'm.yaml'))
    g2 = load_map_yaml(str(tmp_path / 'm.yaml'))
    assert (g2.data == g.data).all() and g2.resolution == g.resolution


def test_geodesic_straight_line():
    passable = np.ones((10, 30), bool)
    d, path = geodesic(passable, (5, 0), 0.05, return_path_to=(5, 20))
    assert d[5, 20] == pytest.approx(1.0)
    assert path[0] == (5, 0) and path[-1] == (5, 20)


def test_frontier_detected_at_unknown_boundary():
    g = room()
    fm = frontier_mask(g)
    cols = np.unique(np.nonzero(fm)[1])
    assert set(cols) == {29}             # columna libre justo antes de lo desconocido


def test_thin_unknown_strip_is_not_frontier():
    g = room(unknown_right=False)
    g.data[10:30, 20] = -1               # 'interior de pared' de 1 celda de ancho
    assert not frontier_mask(g).any()


def test_nearest_frontier_goal():
    g = room()
    ctx = build_context(g, (0.5, 1.0), 0.221, 0.3)
    assert len(ctx.frontiers) == 1
    f = ctx.frontiers[0]
    assert 1.1 < f.goal[0] < 1.5 and np.isfinite(f.distance)


def test_explorer_logic_cycle():
    g = room()
    logic = ExplorerLogic('B1', CFG, seed=0)
    a = logic.update(0.0, g, (0.5, 1.0), NAV_IDLE)
    assert a.kind == 'goal'
    g.data[1:-1, 30:-1] = 0              # se explora todo
    a = logic.update(1.0, g, (a.goal.x, a.goal.y), NAV_SUCCEEDED)
    assert a.kind == 'done'


def test_random_strategy_gives_reachable_goal():
    g = room(unknown_right=False)
    g.data[1:-1, 45:-1] = -1
    logic = ExplorerLogic('B0', CFG, seed=3)
    a = logic.update(0.0, g, (0.5, 1.0), NAV_IDLE)
    assert a.kind == 'goal'
    assert g.free_mask()[g.world_to_cell(a.goal.x, a.goal.y)]


def test_coverage_and_quality_perfect_map():
    gt = room(unknown_right=False)
    start = (0.5, 1.0, 0.3)
    # mapa estimado = mapa real expresado en el marco de la pose inicial
    est = OccGrid(np.full((160, 160), -1, np.int8), 0.05, (-4.0, -4.0))
    ii, jj = np.indices(est.shape)
    xm, ym = est.cell_to_world(ii, jj)
    xw, yw = se2_apply(start, xm, ym)
    est.data[:] = gt.lookup(xw, yw)
    ev = CoverageEvaluator(gt, start, CFG)
    assert ev.coverage(est) > 0.97
    q = map_quality(est, gt, ev.region, start, CFG)
    assert q['occ_recall'] > 0.9 and q['free_iou'] > 0.9 and q['adnn'] < 0.05


def test_stats_helpers():
    assert st.cliffs_delta([1, 2, 3], [4, 5, 6]) == -1.0
    lo, hi = st.wilson_ci(8, 10)
    assert 0.4 < lo < 0.8 < hi <= 1.0
    adj = st.holm([0.01, 0.04, 0.03])
    assert np.allclose(adj, [0.03, 0.06, 0.06])
    xs, ys = st.kaplan_meier([10, 20, 30, 40], [True, True, False, True])
    assert ys[-1] == 0.0 and ys[1] == 0.75


def test_sim2d_short_episode(tmp_path):
    from aslam_bench.core.metrics import episode_metrics
    from aslam_bench.sim2d.simulator import run_episode
    gt = room(unknown_right=False)
    cfg = load_platform(overrides={'episode': {'t_max': 30.0}})
    meta = run_episode(gt, (0.5, 1.0, 0.0), 'B1', cfg, str(tmp_path / 'ep'),
                       dict(map='room', start_id=0), seed=0)
    assert meta['termination'] in ('coverage_reached', 'no_frontiers', 'timeout')
    row, curve = episode_metrics(str(tmp_path / 'ep'), gt, cfg)
    assert row['coverage_final'] > 0.9 and len(curve) >= 1
