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


# ------------------------------------------------------------------ agente DRL
from aslam_bench.core import drl_policy as dp  # noqa: E402
from aslam_bench.core.strategies import make_strategy  # noqa: E402


def test_lidar_is_rplidar_c1():
    lc = CFG['platform']['lidar']
    assert lc['model'] == 'RPLIDAR C1' and lc['n_beams'] == dp.N_SCAN and lc['range_min'] == 0.05


def test_resample_scan_identity_and_angles():
    a = dp.training_scan_angles()
    r = np.linspace(0.1, 5.9, dp.N_SCAN)
    out = dp.resample_scan(r, a)
    # el último haz (2π) mira hacia delante, igual que el primero (Gazebo repite la dirección)
    assert out.shape == (dp.N_SCAN,) and np.allclose(out[:-1], r[:-1] / dp.SCAN_CAP, atol=1e-6)
    assert out[-1] == out[0]
    # LiDAR en [-pi, pi) (simulador 2D): el haz frontal debe acabar en el índice 0
    a2 = np.linspace(-math.pi, math.pi, 360, endpoint=False)
    r2 = np.full(360, np.inf)
    r2[180] = 1.2                     # ángulo 0 = delante
    r2[0] = np.nan                    # por debajo de range_min
    out2 = dp.resample_scan(r2, a2)
    assert out2[0] == pytest.approx(0.2) and out2[250] == 0.0 and out2[125] == 1.0


def test_lem_gem_match_reference_loops():
    """La versión vectorizada coincide con los bucles de MapProcessor (copiados aquí)."""
    rng = np.random.default_rng(0)
    m = rng.choice([-1, 0, 100], size=(120, 150), p=[0.5, 0.4, 0.1]).astype(int)
    m[:10, :] = -1
    rc = (30, 40)
    H = W = 24
    hb = wb = int(round(24 * 3.5))
    r0, c0 = rc[0] - hb // 2, rc[1] - wb // 2
    canvas = np.full((hb, wb), -1, int)
    r0c, c0c = max(0, r0), max(0, c0)
    r1c, c1c = min(m.shape[0] - 1, r0 + hb - 1), min(m.shape[1] - 1, c0 + wb - 1)
    sub = m[r0c:r1c + 1, c0c:c1c + 1]
    canvas[r0c - r0:r0c - r0 + sub.shape[0], c0c - c0:c0c - c0 + sub.shape[1]] = sub
    ref = np.zeros((H, W), np.uint8)
    for i in range(H):
        for j in range(W):
            v = canvas[int(i * hb / H):int((i + 1) * hb / H), int(j * wb / W):int((j + 1) * wb / W)].max()
            ref[i, j] = 255 if v == 100 else (0 if v == 0 else 128)
    assert np.array_equal(dp.compute_lem(m, rc), ref)
    gem = dp.compute_gem(m, rc)
    assert gem.shape == (24, 24) and set(np.unique(gem)) <= {0, 128, 255} and (gem == 128).sum() >= 4


def test_observation_layout_and_ternary_map():
    g = room()
    g.data[5:10, 5:10] = 40          # probabilidad intermedia (log-odds) -> desconocido
    t = dp.ternary_map(g)
    assert set(np.unique(t)) <= {-1, 0, 100} and (t[5:10, 5:10] == -1).all()
    obs = dp.build_observation(np.ones(dp.N_SCAN), g, (0.75, 1.0, math.pi / 2), (0.3, -0.2))
    assert obs.shape == (dp.STATE_SIZE,) == (1656,)
    assert np.allclose(obs[-4:], [0.3, -0.2, 1.0, 0.0], atol=1e-6)
    assert obs[dp.N_SCAN:dp.N_SCAN + 1152].max() <= 1.0


def test_action_to_cmd():
    assert dp.action_to_cmd([-1, 0]) == (0.0, 0.0)
    v, w = dp.action_to_cmd([1, -1])
    assert v == pytest.approx(0.3) and w == pytest.approx(-1.9)


def test_drl_strategy_naming_without_torch():
    cfg = load_platform(overrides={'drl': {'weights': '/x/a.pt', 'agents': {'e500': '/x/e500.pt'}}})
    assert make_strategy('DRL', cfg).name == 'DRL' and make_strategy('DRL', cfg).weights == '/x/a.pt'
    s = make_strategy('DRL:e500', cfg)
    assert s.name == 'DRL_e500' and s.weights == '/x/e500.pt' and s.outputs == 'velocity'
    assert make_strategy('DRL:/m/actor_stage0_episode7.pt', cfg).name == 'DRL_actor_stage0_episode7'
    assert make_strategy('DRL', cfg, weights='/y.pt').weights == '/y.pt'
    assert make_strategy('B1', cfg).outputs == 'goal'


def _save_random_actor(path):
    torch = pytest.importorskip('torch')
    torch.manual_seed(0)
    torch.save(dp.make_actor().state_dict(), path)


def test_drl_policy_loads_pt_and_acts(tmp_path):
    w = tmp_path / 'actor_stage0_episode1.pt'
    _save_random_actor(w)
    pol = dp.DrlPolicy(str(w), v_max=0.26, w_max=1.0)
    a = dp.training_scan_angles()
    v, wz, info = pol.act(np.full(dp.N_SCAN, 2.0), a, room(), (0.5, 0.5, 0.0))
    assert 0.0 <= v <= 0.26 and -1.0 <= wz <= 1.0 and info['decision_time'] > 0
    assert pol.last_obs.shape == (1656,) and len(pol.weights_sha) == 12


def test_sim2d_drl_episode(tmp_path):
    w = tmp_path / 'actor_stage0_episode1.pt'
    _save_random_actor(w)
    from aslam_bench.sim2d.simulator import run_episode
    gt = load_map_yaml(os.path.join(os.path.dirname(__file__), '..', 'maps', 'rooms_small.yaml'))
    cfg = load_platform(overrides={'episode': {'t_max': 20.0}})
    meta = run_episode(gt, (1.5, 5.0, 0.0), f'DRL:{w}', cfg, str(tmp_path / 'ep'), seed=1)
    assert meta['method'] == 'DRL_actor_stage0_episode1' and meta['termination'] == 'timeout'
    assert meta['drl_weights'] == str(w) and 'drl_weights_sha256' in meta
    import pandas as pd
    goals = pd.read_csv(tmp_path / 'ep' / 'goals.csv')
    assert (goals['event'] == 'policy').sum() >= 10
