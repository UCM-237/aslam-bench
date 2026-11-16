"""Pruebas de las estrategias B2-B8 y de sus piezas (sin ROS):  pytest -q tests/"""
import math

import numpy as np
import pytest

from aslam_bench.core.config import load_platform
from aslam_bench.core.explorer import NAV_IDLE, NAV_SUCCEEDED, ExplorerLogic
from aslam_bench.core.frontiers import build_context
from aslam_bench.core.grid import OccGrid
from aslam_bench.core.infogain import line_free, unknown_in_disk, visible_entropy, visible_unknown
from aslam_bench.core.posegraph import PoseGraph
from aslam_bench.core.semantics import SemanticMap, SemanticSim, ig_sem
from aslam_bench.core.strategies import STRATEGIES, LearnedFrontier, make_strategy, save_mlp

CFG = load_platform()
GOAL_METHODS = ['B2', 'B3', 'B4', 'B5', 'B6', 'B8']


def room(w=60, h=40, unknown_right=True):
    """Habitación 3x2 m (res 0.05) con paredes; mitad derecha desconocida si se pide."""
    d = np.zeros((h, w), np.int8)
    d[0, :] = d[-1, :] = d[:, 0] = d[:, -1] = 100
    if unknown_right:
        d[1:-1, w // 2:-1] = -1
    return OccGrid(d, 0.05, (0.0, 0.0))


def corridor_world():
    """Pasillo de 8 m: los 3 primeros metros conocidos, el resto desconocido."""
    d = np.full((30, 170), -1, np.int8)
    d[0, :] = d[-1, :] = d[:, 0] = d[:, -1] = 100
    d[1:-1, 1:60] = 0
    return OccGrid(d, 0.05, (0.0, 0.0))


# ------------------------------------------------------------------ registro

@pytest.mark.parametrize('key', ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8'])
def test_registry_names(key):
    st = make_strategy(key, CFG)
    assert st.name.startswith(key + '_') and STRATEGIES[st.name] is type(st)
    assert st.params()          # parámetros efectivos para meta.json


# ------------------------------------------------------------------ IG

def test_visible_unknown_blocked_by_wall():
    g = room()
    a = visible_unknown(g, 1.2, 1.0, radius=3.0)
    assert a > 0.5                                # ve la mitad desconocida
    g.data[1:-1, 28] = 100                        # pared que la tapa
    assert visible_unknown(g, 1.2, 1.0, radius=3.0) == 0.0


def test_visible_entropy_counts_unknown_as_one_bit():
    g = room()
    ent = visible_entropy(g, 1.2, 1.0, radius=3.0)
    assert ent == pytest.approx(visible_unknown(g, 1.2, 1.0, radius=3.0), rel=0.05)


def test_unknown_in_disk_and_line_free():
    g = room()
    assert unknown_in_disk(g, 0.5, 1.0, 0.3) == 0.0 and unknown_in_disk(g, 2.2, 1.0, 0.3) > 0
    assert line_free(g, (0.3, 0.3), (2.5, 1.5))
    g.data[:, 20] = 100
    assert not line_free(g, (0.3, 0.3), (2.5, 1.5))


# ------------------------------------------------------------------ grafo de poses

def walk(pg, pts, step=0.1):
    for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
        n = max(1, int(math.hypot(x1 - x0, y1 - y0) / step))
        for k in range(1, n + 1):
            pg.update((x0 + (x1 - x0) * k / n, y0 + (y1 - y0) * k / n))


def test_posegraph_chain_variance_and_dopt():
    pg = PoseGraph(kf_spacing=0.5, sigma_odom=0.1, lc_radius=1.0, lc_min_sep=10, lc_cooldown=5)
    walk(pg, [(0, 0), (5, 0)])
    n, ld, var = pg.state()
    assert n in (10, 11) and pg.n_loop_closures() == 0
    assert var == pytest.approx(0.1 ** 2 * 5, rel=0.05)           # crece con la distancia
    # seguir recto: sin cierres, la D-optimalidad 'útil' no cambia y la varianza crece
    pr = pg.predict([(5 + 0.05 * k, 0) for k in range(1, 41)])
    assert pr['n_lc'] == 0 and abs(pr['dopt_gain']) < 1e-6 and pr['var'] > var


def test_posegraph_loop_closure_reduces_uncertainty():
    pg = PoseGraph(kf_spacing=0.5, sigma_odom=0.1, sigma_lc=0.1, lc_radius=1.0, lc_min_sep=10, lc_cooldown=5)
    walk(pg, [(0, 0), (4, 0), (4, 3), (0.8, 3)])
    _, _, var0 = pg.state()
    # volver al inicio cierra el bucle; alejarse no
    back = pg.predict([(0.8, 3 - 0.05 * k) for k in range(1, 61)])
    away = pg.predict([(0.8 - 0.0, 3 + 0.05 * k) for k in range(1, 61)])
    assert back['n_lc'] >= 1 and back['dopt_gain'] > 0.5 and back['var'] < var0
    assert away['n_lc'] == 0 and away['var'] > var0
    walk(pg, [(0.8, 3), (0.8, 0)])
    assert pg.n_loop_closures() >= 1


def test_posegraph_cooldown_limits_closures():
    pg = PoseGraph(kf_spacing=0.5, lc_radius=1.0, lc_min_sep=10, lc_cooldown=6)
    walk(pg, [(0, 0), (10, 0), (10, 0.5), (0, 0.5)])               # ida y vuelta pegado a la ida
    assert 1 <= pg.n_loop_closures() <= 4


# ------------------------------------------------------------------ estrategias de objetivo

@pytest.mark.parametrize('method', GOAL_METHODS)
def test_goal_strategies_pick_reachable_goal(method):
    g = corridor_world()
    logic = ExplorerLogic(method, CFG, seed=0)
    a = logic.update(0.0, g, (0.5, 0.7, 0.0), NAV_IDLE)
    assert a.kind == 'goal'
    i, j = g.world_to_cell(a.goal.x, a.goal.y)
    assert g.free_mask()[i, j] and a.goal.x > 1.0               # hacia lo desconocido
    assert np.isfinite(a.goal.distance)


@pytest.mark.parametrize('method', GOAL_METHODS)
def test_goal_strategies_done_when_explored(method):
    g = room()
    logic = ExplorerLogic(method, CFG, seed=0)
    a = logic.update(0.0, g, (0.5, 1.0, 0.0), NAV_IDLE)
    assert a.kind == 'goal'
    g.data[1:-1, 30:-1] = 0
    a = logic.update(1.0, g, (a.goal.x, a.goal.y, 0.0), NAV_SUCCEEDED)
    assert a.kind == 'done'


def test_b2_prefers_informative_frontier():
    # dos fronteras a la misma distancia: una da a una sala grande, otra a un hueco pequeño
    d = np.full((60, 120), -1, np.int8)
    d[0, :] = d[-1, :] = d[:, 0] = d[:, -1] = 100
    d[25:35, 50:70] = 0                                # sala central conocida
    d[25:35, 49] = d[25:35, 70] = 0
    d[20:40, 30:49] = 100                              # izquierda: armario cerrado...
    d[26:34, 40:49] = -1                               # ...con un hueco desconocido pequeño
    g = OccGrid(d, 0.05, (0.0, 0.0))
    ctx = build_context(g, (3.0, 1.5), 0.221, 0.3, min_unknown_cells=10)
    st = make_strategy('B2', CFG)
    goal = st.choose(ctx, np.random.default_rng(0), [])
    assert goal.x > 3.0                                # elige la derecha (mucho por descubrir)


def test_b3_rrt_detects_frontier_points():
    g = corridor_world()
    st = make_strategy('B3', CFG)
    for k in range(5):
        st.observe(float(k), g, (0.5, 0.7, 0.0))
    assert len(st.points) > 0
    assert all(p[0] > 2.0 for p in st.points)         # en el borde con lo desconocido
    v = st.viz()
    assert v['segments'] and v['points']


def test_b4_b5_value_loop_closure_when_uncertain():
    """Tras una vuelta larga, B4 y B5 dan valor positivo a volver al inicio."""
    d = np.zeros((80, 100), np.int8)
    d[0, :] = d[-1, :] = d[:, 0] = d[:, -1] = 100
    d[15:65, 15:85] = 100                              # bloque central: circuito alrededor
    d[1:15, 90:99] = -1                                # algo sin explorar en una esquina
    g = OccGrid(d, 0.05, (0.0, 0.0))
    for method, key in (('B4', 'dH_pose'), ('B5', 'dopt_gain')):
        logic = ExplorerLogic(method, CFG, seed=0)
        pts = [(0.4, 0.4), (4.6, 0.4), (4.6, 3.6), (0.4, 3.6), (0.4, 1.6)]
        for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
            for k in range(1, 41):
                logic.pose_graph.update((x0 + (x1 - x0) * k / 40, y0 + (y1 - y0) * k / 40), g)
        ctx = logic._context(g, (0.4, 1.6))
        st = logic.strategy
        lcs = st.lc_candidates(ctx, [])
        assert lcs, method
        vals = [st.predict(ctx, c) for c in lcs]
        assert any(v['n_lc'] > 0 for v in vals), method
        goal = st.choose(ctx, np.random.default_rng(0), [])
        assert key in goal.info


def test_b6_nbv_executes_first_step_only():
    g = corridor_world()
    st = make_strategy('B6', CFG)
    ctx = build_context(g, (0.5, 0.7), 0.221, 0.3)
    ctx.robot_yaw = 0.0
    goal = st.choose(ctx, np.random.default_rng(0), [])
    assert goal.info['kind'] == 'nbv'
    assert math.hypot(goal.x - 0.5, goal.y - 0.7) <= 2 * CFG['strategies']['B6']['step'] + 0.1
    assert st.viz()['segments']


def test_b7_requires_weights_and_runs_with_them(tmp_path):
    g = corridor_world()
    ctx = build_context(g, (0.5, 0.7), 0.221, 0.3)
    with pytest.raises(RuntimeError):
        make_strategy('B7', CFG).choose(ctx, np.random.default_rng(0), [])
    rng = np.random.default_rng(0)
    n = len(LearnedFrontier.FEATURES)
    path = str(tmp_path / 'w.npz')
    save_mlp(path, [(rng.normal(size=(n, 8)), np.zeros(8)), (rng.normal(size=(8, 1)), np.zeros(1))])
    st = make_strategy(f'B7:{path}', CFG)
    assert st.name == 'B7_w'
    goal = st.choose(ctx, rng, [])
    assert goal is not None and 'score' in goal.info


# ------------------------------------------------------------------ semántica y B8

def test_semantic_map_confirms_and_reduces_entropy():
    sm = SemanticMap(['a', 'b', 'c'], confirm_obs=3, confirm_views=2)
    h0 = None
    for k in range(4):
        sm.integrate([dict(id=0, x=1.0, y=1.0, cls=1, score=0.9)], (0.0 + 0.6 * k, 0.0))
        h0 = h0 if h0 is not None else sm.landmarks[0].entropy
    lm = sm.landmarks[0]
    assert lm.state == 'confirmed' and lm.entropy < h0 and int(np.argmax(lm.probs)) == 1
    sm2 = SemanticMap()
    sm2.from_json('{"landmarks": [{"id": 3, "x": 1, "y": 2, "probs": [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], '
                  '"state": "confirmed", "n_obs": 5}]}')
    assert sm2.landmarks[3].state == 'confirmed'


def test_ig_sem_depends_on_orientation():
    g = room(unknown_right=False)
    sm = SemanticMap(['a', 'b'])
    sm.integrate([dict(id=0, x=2.5, y=1.0, cls=0, score=0.6)], (0.5, 1.0))
    hf = math.radians(69)
    assert ig_sem(sm, g, 1.0, 1.0, 0.0, hf, 4.0) > 0              # mirando al objeto
    assert ig_sem(sm, g, 1.0, 1.0, math.pi, hf, 4.0) == 0          # de espaldas


def test_semantic_sim_places_and_detects_objects():
    g = room(w=120, h=80, unknown_right=False)
    sim = SemanticSim(g, (3.0, 2.0, 0.0), CFG, seed=0)
    assert len(sim.objects) >= 3
    seen = set()
    for yaw in np.linspace(-math.pi, math.pi, 12, endpoint=False):
        for _ in range(5):
            seen |= {o['id'] for o in sim.observe((3.0, 2.0, yaw), (0.0, 0.0, yaw))}
    assert seen


def test_b8_turns_camera_to_uncertain_landmark():
    g = corridor_world()
    logic = ExplorerLogic('B8', CFG, seed=0)
    # landmark dudoso junto a la pared, lejos del frente de exploración
    logic.semantic.integrate([dict(id=0, x=1.5, y=0.2, cls=0, score=0.5)], (0.5, 0.7))
    a = logic.update(0.0, g, (0.5, 0.7, 0.0), NAV_IDLE)
    assert a.kind == 'goal' and a.goal.info.get('orient')
    assert a.goal.info['ig_sem'] > 0


# ------------------------------------------------------------------ episodios cortos en sim2d

@pytest.mark.parametrize('method', GOAL_METHODS)
def test_sim2d_short_episode_all_methods(tmp_path, method):
    import json
    import os

    from aslam_bench.sim2d.simulator import run_episode
    gt = room(w=100, h=60, unknown_right=False)
    gt.data[:, 50] = 100                                 # tabique con puerta: hay que moverse
    gt.data[25:35, 50] = 0
    gt.data[:, 75] = 100
    gt.data[5:15, 75] = 0
    cfg = load_platform(overrides={'episode': {'t_max': 40.0}})
    meta = run_episode(gt, (0.5, 1.0, 0.0), method, cfg, str(tmp_path / 'ep'), dict(map='room'), seed=0)
    assert meta['termination'] in ('coverage_reached', 'no_frontiers', 'timeout', 'stalled')
    assert meta['strategy_params'] and 'sem_n_objects' in meta
    assert os.path.isfile(tmp_path / 'ep' / 'semantic.json')
    import pandas as pd
    goals = pd.read_csv(tmp_path / 'ep' / 'goals.csv')
    infos = [json.loads(s) for s in goals['info'].dropna()]
    assert infos and all('kind' in i for i in infos)
