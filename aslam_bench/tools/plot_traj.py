"""Dibuja las trayectorias de los episodios sobre el mapa real.

  python -m aslam_bench.tools.plot_traj results/iter1_sim2d                 # campaña entera
  python -m aslam_bench.tools.plot_traj results/iter1_sim2d/episodes/oracle/rooms_small/B1/s00_r0
  ros2 run aslam_bench plot_traj results/iter1_gazebo --grid --map rooms_small

Para cada episodio dibuja, en el marco del mundo (el del YAML del mapa):
  * el mapa real (PGM + YAML de `--maps`, según `map` de meta.json),
  * la trayectoria real (gt_*) y la estimada por el SLAM (est_*), que se pasa del
    marco 'map' al mundo con la pose inicial de meta.json: world = start_pose ⊕ map,
  * los objetivos de goals.csv (están en el marco 'map', igual que la estimación),
  * las colisiones de events.csv, situadas donde estaba el robot en ese instante,
  * opcionalmente (`--est-map`) las celdas ocupadas del último mapa estimado.

Salida: `<episodio>/traj.png` (o en `--out`) y, con `--grid`, una figura por
(modo de localización, mapa) con una fila por pose inicial/repetición y una
columna por método, en `<campaña>/analysis/` (o en `--out`).
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from ..core.config import resource_dir  # noqa: E402
from ..core.grid import OccGrid, load_map_yaml, se2_apply, wrap_angle  # noqa: E402

# Mismos colores que analyze (paleta categórica validada)
C_GT, C_EST, C_GOAL, C_COLL, C_EMAP = '#2a78d6', '#eb6834', '#1baf7a', '#e34948', '#6250d6'
INK, INK2 = '#0b0b0b', '#52514e'
MAP_CMAP = ListedColormap(['#d9d8d4', '#fcfcfb', '#3d3c3a'])   # desconocido, libre, ocupado


# ----------------------------------------------------------------------------- datos

def find_episodes(path):
    """Directorios de episodio bajo `path` (o el propio `path` si es un episodio)."""
    if os.path.isfile(os.path.join(path, 'meta.json')):
        return [path]
    metas = glob.glob(os.path.join(path, '**', 'meta.json'), recursive=True)
    return sorted(os.path.dirname(m) for m in metas)


def _read_csv(path, columns):
    try:
        df = pd.read_csv(path)
    except (OSError, pd.errors.EmptyDataError):
        df = pd.DataFrame(columns=columns)
    for c in columns:
        if c != 'kind' and c != 'event' and c != 'detail' and c in df:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return df


def load_trajectory(ep_dir):
    """Lee un episodio y lo devuelve todo en el marco del mundo.

    Devuelve un dict con meta, t, gt (N x 3), est (N x 3, ya en el mundo),
    goals (M x 2, mundo), collisions (K x 2, mundo) y last_map (ruta .npz o None).
    """
    with open(os.path.join(ep_dir, 'meta.json')) as f:
        meta = json.load(f)
    start = tuple(float(v) for v in meta.get('start_pose', (0.0, 0.0, 0.0)))
    poses = _read_csv(os.path.join(ep_dir, 'poses.csv'),
                      ['t', 'gt_x', 'gt_y', 'gt_yaw', 'est_x', 'est_y', 'est_yaw'])
    goals = _read_csv(os.path.join(ep_dir, 'goals.csv'), ['t', 'event', 'x', 'y', 'yaw'])
    events = _read_csv(os.path.join(ep_dir, 'events.csv'), ['t', 'kind', 'detail'])

    t = poses['t'].to_numpy(float)
    gt = poses[['gt_x', 'gt_y', 'gt_yaw']].to_numpy(float)
    ex, ey = se2_apply(start, poses['est_x'].to_numpy(float), poses['est_y'].to_numpy(float))
    eyaw = wrap_angle(poses['est_yaw'].to_numpy(float) + start[2])
    est = np.column_stack([ex, ey, eyaw]) if len(t) else np.zeros((0, 3))

    g = goals[goals['event'] == 'goal'].dropna(subset=['x', 'y'])
    gx, gy = se2_apply(start, g['x'].to_numpy(float), g['y'].to_numpy(float))
    goals_w = np.column_stack([gx, gy]) if len(g) else np.zeros((0, 2))

    # colisiones: posición del robot en ese instante (real si hay ground truth, si no la estimada)
    tc = events.loc[events['kind'] == 'collision', 't'].dropna().to_numpy(float)
    ref = gt if len(gt) and np.isfinite(gt[:, :2]).all() else est
    if len(tc) and len(t):
        coll = np.column_stack([np.interp(tc, t, ref[:, 0]), np.interp(tc, t, ref[:, 1])])
    else:
        coll = np.zeros((0, 2))

    snaps = sorted(glob.glob(os.path.join(ep_dir, 'maps', '*.npz')))
    return dict(meta=meta, start=start, t=t, gt=gt, est=est, goals=goals_w, collisions=coll,
                last_map=snaps[-1] if snaps else None, dir=ep_dir)


def est_map_occupied_world(npz_path, start, occ_thresh=65):
    """Centros de las celdas ocupadas del mapa estimado, pasados al mundo."""
    with np.load(npz_path) as z:
        grid = OccGrid(z['data'], float(z['resolution']), tuple(np.asarray(z['origin'], float)[:2]))
    i, j = np.nonzero(grid.data >= occ_thresh)
    xm, ym = grid.cell_to_world(i, j)
    return np.column_stack(se2_apply(start, xm, ym)) if len(i) else np.zeros((0, 2))


class MapCache:
    def __init__(self, maps_dir):
        self.dir, self._c = maps_dir, {}

    def get(self, name):
        if not name:
            return None
        if name not in self._c:
            path = os.path.join(self.dir, name + '.yaml')
            self._c[name] = load_map_yaml(path) if os.path.exists(path) else None
            if self._c[name] is None:
                print(f'[plot_traj] aviso: no encuentro {path}; se dibuja sin mapa de fondo')
        return self._c[name]


# ----------------------------------------------------------------------------- dibujo

def draw_map(ax, gt: OccGrid):
    img = np.ones(gt.shape, int)               # libre
    img[gt.unknown_mask()] = 0
    img[gt.occ_mask()] = 2
    ax.imshow(img, cmap=MAP_CMAP, vmin=0, vmax=2, origin='lower', extent=gt.extent(),
              interpolation='nearest', zorder=0)


def _arrow(ax, pose, color, size):
    x, y, th = pose
    ax.annotate('', xy=(x + size * math.cos(th), y + size * math.sin(th)), xytext=(x, y),
                arrowprops=dict(arrowstyle='-|>', color=color, lw=1.5), zorder=6)


def draw_episode(ax, tr, gt_map=None, est_map=False, lw=1.6, title=True):
    if gt_map is not None:
        draw_map(ax, gt_map)
        x0, x1, y0, y1 = gt_map.extent()
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
    if est_map and tr['last_map']:
        occ = est_map_occupied_world(tr['last_map'], tr['start'])
        ax.scatter(occ[:, 0], occ[:, 1], s=0.6, c=C_EMAP, marker='s', linewidths=0, zorder=1)
    gt, est = tr['gt'], tr['est']
    has_gt = len(gt) and np.isfinite(gt[:, :2]).any()
    if has_gt:
        ax.plot(gt[:, 0], gt[:, 1], color=C_GT, lw=lw, zorder=3, solid_capstyle='round')
    if len(est):
        ax.plot(est[:, 0], est[:, 1], color=C_EST, lw=lw * 0.8, ls=(0, (3, 2)), zorder=4)
    if len(tr['goals']):
        ax.scatter(tr['goals'][:, 0], tr['goals'][:, 1], marker='x', s=22, c=C_GOAL, linewidths=1.4, zorder=5)
    if len(tr['collisions']):
        ax.scatter(tr['collisions'][:, 0], tr['collisions'][:, 1], marker='o', s=30, facecolors='none',
                   edgecolors=C_COLL, linewidths=1.4, zorder=7)
    size = 0.5
    _arrow(ax, tr['start'], INK, size)
    ax.plot(*tr['start'][:2], 'o', color=INK, ms=4, zorder=6)
    end = gt[-1] if has_gt and np.isfinite(gt[-1, :2]).all() else (est[-1] if len(est) else None)
    if end is not None:
        ax.plot(end[0], end[1], 's', color=INK, ms=4, zorder=6)
    ax.set_aspect('equal')
    ax.tick_params(labelsize=7, colors=INK2)
    if title:
        m = tr['meta']
        ep = os.path.basename(os.path.normpath(tr['dir']))
        t1 = f"{m.get('method', '?')} · {m.get('map', '?')} · {ep} · {m.get('loc_mode', '?')}"
        t2 = f"{m.get('termination', '?')}, {m.get('duration', float('nan')):.0f} s"
        if has_gt and len(est):
            err = np.hypot(est[:, 0] - gt[:, 0], est[:, 1] - gt[:, 1])
            err = err[np.isfinite(err)]
            if len(err):
                t2 += f', ATE {math.sqrt(np.mean(err ** 2)):.2f} m'
        ax.set_title(f'{t1}\n{t2}', fontsize=8, color=INK)


def legend_handles(est_map=False):
    h = [Line2D([], [], color=C_GT, lw=1.6, label='real (ground truth)'),
         Line2D([], [], color=C_EST, lw=1.3, ls=(0, (3, 2)), label='estimada (SLAM → mundo)'),
         Line2D([], [], color=C_GOAL, marker='x', ls='', label='objetivos'),
         Line2D([], [], color=C_COLL, marker='o', mfc='none', ls='', label='colisiones'),
         Line2D([], [], color=INK, marker='o', ls='', ms=4, label='inicio'),
         Line2D([], [], color=INK, marker='s', ls='', ms=4, label='fin')]
    if est_map:
        h.append(Line2D([], [], color=C_EMAP, marker='s', ls='', ms=3, label='mapa estimado (ocupado)'))
    return h


def plot_episode(tr, gt_map, out_png, est_map=False):
    fig, ax = plt.subplots(figsize=(6.4, 5.6), facecolor='#fcfcfb')
    draw_episode(ax, tr, gt_map, est_map)
    ax.legend(handles=legend_handles(est_map), fontsize=7, loc='upper left', bbox_to_anchor=(1.01, 1.0),
              frameon=False)
    ax.set_xlabel('x (m)', fontsize=8)
    ax.set_ylabel('y (m)', fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160, bbox_inches='tight')
    plt.close(fig)


def plot_grid(trs, gt_map, out_png, title, est_map=False):
    """Filas = (pose inicial, repetición), columnas = método."""
    methods = sorted({tr['meta'].get('method', '?') for tr in trs})
    rows = sorted({(tr['meta'].get('start_id', 0), tr['meta'].get('rep', 0)) for tr in trs})
    cell = {((tr['meta'].get('start_id', 0), tr['meta'].get('rep', 0)), tr['meta'].get('method', '?')): tr
            for tr in trs}
    w = 3.4
    if gt_map is not None:
        x0, x1, y0, y1 = gt_map.extent()
        h = max(1.8, min(5.0, w * (y1 - y0) / max(x1 - x0, 1e-6)))
    else:
        h = w
    fig, axes = plt.subplots(len(rows), len(methods), figsize=(w * len(methods), (h + 0.5) * len(rows) + 0.8),
                             squeeze=False, facecolor='#fcfcfb')
    for r, key in enumerate(rows):
        for c, meth in enumerate(methods):
            ax = axes[r][c]
            tr = cell.get((key, meth))
            if tr is None:
                ax.axis('off')
                continue
            draw_episode(ax, tr, gt_map, est_map, lw=1.2, title=False)
            m = tr['meta']
            ax.set_title(f"{meth} · s{key[0]:02d}_r{key[1]} · {m.get('termination', '?')}", fontsize=8, color=INK)
    fig.suptitle(title, fontsize=10, color=INK)
    fig.tight_layout()
    fig.legend(handles=legend_handles(est_map), loc='upper center', bbox_to_anchor=(0.5, 0.0),
               ncol=4, fontsize=7, frameon=False)
    fig.savefig(out_png, dpi=130, bbox_inches='tight')
    plt.close(fig)


# ----------------------------------------------------------------------------- CLI

def _out_name(root, ep_dir):
    rel = os.path.relpath(ep_dir, root)
    if rel == '.':
        rel = os.path.basename(os.path.normpath(ep_dir))
    rel = rel.replace(os.sep, '__')
    if rel.startswith('episodes__'):
        rel = rel[len('episodes__'):]
    return rel + '.png'


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('path', help='directorio de un episodio o de una campaña')
    ap.add_argument('--maps', default=None, help='directorio de mapas (por defecto <paquete>/maps)')
    ap.add_argument('--out', default=None,
                    help='directorio de salida (por defecto traj.png en cada episodio y analysis/ para --grid)')
    ap.add_argument('--grid', action='store_true', help='además, una figura por (modo, mapa): pose x método')
    ap.add_argument('--no-episodes', action='store_true', help='no generar la figura de cada episodio')
    ap.add_argument('--est-map', action='store_true', help='superponer las celdas ocupadas del último mapa estimado')
    ap.add_argument('--map', dest='f_map', action='append', help='filtrar por mapa (repetible)')
    ap.add_argument('--method', dest='f_method', action='append', help='filtrar por método (repetible)')
    ap.add_argument('--loc-mode', dest='f_loc', action='append', help='filtrar por modo de localización (repetible)')
    a = ap.parse_args(argv)

    root = os.path.abspath(a.path)
    eps = find_episodes(root)
    if not eps:
        raise SystemExit(f'No hay episodios (meta.json) en {root}')
    maps = MapCache(a.maps or resource_dir('maps'))
    if a.out:
        os.makedirs(a.out, exist_ok=True)

    trs = []
    for ep in eps:
        try:
            tr = load_trajectory(ep)
        except Exception as e:  # episodio corrupto: se avisa y se sigue
            print(f'[plot_traj] ERROR en {ep}: {e}')
            continue
        m = tr['meta']
        if (a.f_map and m.get('map') not in a.f_map) or (a.f_method and m.get('method') not in a.f_method) \
                or (a.f_loc and m.get('loc_mode') not in a.f_loc):
            continue
        trs.append(tr)
        if not a.no_episodes:
            out = os.path.join(a.out, _out_name(root, ep)) if a.out else os.path.join(ep, 'traj.png')
            plot_episode(tr, maps.get(m.get('map')), out, a.est_map)
    if not a.no_episodes:
        print(f'[plot_traj] {len(trs)} figuras de episodio' + (f' en {a.out}' if a.out else ' (traj.png en cada episodio)'))

    if a.grid and len(trs) > 1:
        out_dir = a.out or os.path.join(root, 'analysis')
        os.makedirs(out_dir, exist_ok=True)
        groups = {}
        for tr in trs:
            groups.setdefault((tr['meta'].get('loc_mode', '?'), tr['meta'].get('map', '?')), []).append(tr)
        for (lm, mp), g in sorted(groups.items()):
            out = os.path.join(out_dir, f'fig_traj_{lm}_{mp}.png')
            plot_grid(g, maps.get(mp), out, f'Trayectorias · {mp} · {lm}', a.est_map)
            print(f'[plot_traj] {out}')
    return trs


if __name__ == '__main__':
    main()
