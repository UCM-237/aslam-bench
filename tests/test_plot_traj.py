"""Pruebas de plot_traj: marcos de referencia y generación de figuras (sin ROS)."""
import os

import numpy as np

from aslam_bench.core.grid import OccGrid, save_map_yaml, se2_apply, se2_inverse_apply
from aslam_bench.core.recorder import EpisodeRecorder
from aslam_bench.tools import plot_traj

START = (2.0, 1.0, 0.9)


class _Goal:
    def __init__(self, x, y):
        self.x, self.y, self.yaw, self.distance = x, y, 0.0, 1.0


def _fake_episode(ep_dir, method='B1', start_id=0, est_offset=0.0):
    """Trayectoria real conocida en el mundo; la estimada se guarda en el marco 'map'."""
    rec = EpisodeRecorder(ep_dir, dict(method=method, map='box', start_id=start_id, rep=0,
                                       loc_mode='oracle', backend='sim2d', start_pose=list(START)))
    t = np.linspace(0, 10, 21)
    gx, gy, gyaw = 2.0 + 0.2 * t, 1.0 + 0.1 * t, np.full_like(t, 0.3)
    mx, my = se2_inverse_apply(START, gx, gy)
    for k in range(len(t)):
        rec.log_pose(t[k], (gx[k], gy[k], gyaw[k]), (mx[k] + est_offset, my[k], gyaw[k] - START[2]))
    gm = se2_inverse_apply(START, 3.0, 1.5)
    rec.log_goal(0.0, 'goal', _Goal(*gm), dict(decision_time=0.01, n_frontiers=3))
    rec.log_goal(9.0, 'done', None, {})
    rec.log_event(5.0, 'collision')
    grid = OccGrid(np.zeros((40, 40), np.int8), 0.1, (-2.0, -2.0))
    grid.data[10, 20] = 100
    rec.snapshot_map(10.0, grid)
    rec.finish('coverage_reached', duration=10.0)
    return (gx, gy, gyaw), grid


def test_estimate_goals_and_collisions_in_world(tmp_path):
    ep = str(tmp_path / 'ep')
    (gx, gy, gyaw), grid = _fake_episode(ep)
    tr = plot_traj.load_trajectory(ep)
    assert np.allclose(tr['est'][:, 0], gx, atol=1e-3) and np.allclose(tr['est'][:, 1], gy, atol=1e-3)
    assert np.allclose(tr['est'][:, 2], gyaw, atol=1e-3)
    assert tr['goals'].shape == (1, 2) and np.allclose(tr['goals'][0], (3.0, 1.5), atol=1e-3)
    assert tr['collisions'].shape == (1, 2) and np.allclose(tr['collisions'][0], (3.0, 1.5), atol=1e-3)
    occ = plot_traj.est_map_occupied_world(tr['last_map'], tr['start'])
    xm, ym = grid.cell_to_world(10, 20)
    assert np.allclose(occ, [se2_apply(START, xm, ym)])


def test_collisions_fall_back_to_estimate_without_ground_truth(tmp_path):
    ep = str(tmp_path / 'ep')
    rec = EpisodeRecorder(ep, dict(method='B1', map='box', start_pose=list(START)))
    nan = float('nan')
    rec.log_pose(0.0, (nan, nan, nan), (0.0, 0.0, 0.0))
    rec.log_pose(2.0, (nan, nan, nan), (1.0, 0.0, 0.0))
    rec.log_event(1.0, 'collision')
    rec.finish('interrupted')
    tr = plot_traj.load_trajectory(ep)
    assert np.allclose(tr['collisions'][0], se2_apply(START, 0.5, 0.0))


def test_cli_campaign_writes_episode_and_grid_figures(tmp_path):
    maps = tmp_path / 'maps'
    maps.mkdir()
    d = np.zeros((60, 80), np.int8)
    d[0, :] = d[-1, :] = d[:, 0] = d[:, -1] = 100
    save_map_yaml(OccGrid(d, 0.1, (0.0, 0.0)), str(maps / 'box.yaml'))
    camp = tmp_path / 'camp'
    eps = []
    for sid in (0, 1):
        for meth in ('B0', 'B1'):
            ep = camp / 'episodes' / 'oracle' / 'box' / meth / f's{sid:02d}_r0'
            _fake_episode(str(ep), meth, sid, est_offset=0.05 * sid)
            eps.append(ep)
    out = tmp_path / 'figs'
    trs = plot_traj.main([str(camp), '--maps', str(maps), '--grid', '--est-map', '--out', str(out)])
    assert len(trs) == 4
    assert (out / 'fig_traj_oracle_box.png').exists()
    assert (out / 'oracle__box__B1__s01_r0.png').exists()
    # sin --out: traj.png dentro de cada episodio; los filtros se respetan
    trs = plot_traj.main([str(camp), '--maps', str(maps), '--method', 'B1'])
    assert len(trs) == 2
    assert all(os.path.exists(os.path.join(tr['dir'], 'traj.png')) for tr in trs)
    assert not (eps[0] / 'traj.png').exists()
