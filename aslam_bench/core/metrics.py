"""Métricas por episodio, calculadas offline a partir de lo que guarda EpisodeRecorder.

Familias (ver plan de evaluación, sección 4):
  * Eficiencia: curva cobertura-tiempo/distancia, T90/T95, D90/D95, AUC, cobertura final,
    longitud de trayectoria, rotación total, eficiencia (m² por m).
  * Calidad del mapa: precisión/recall de celdas ocupadas con tolerancia, IoU de libre, ADNN.
  * SLAM: ATE (RMSE), error final y máximo de posición.
  * Robustez: éxito, terminación, colisiones, objetivos fallidos.
  * Coste: tiempo de decisión (media, p95).

Cobertura = fracción del ÁREA ALCANZABLE (celdas libres del mapa real accesibles
para el robot desde la pose inicial, ampliadas por el radio + margen) que aparece
como conocida (libre u ocupada) en el mapa estimado.
"""
from __future__ import annotations

import glob
import json
import math
import os

import numpy as np
import pandas as pd
from scipy import ndimage

from .config import robot_clearance
from .grid import OccGrid, disk, geodesic, inflate, se2_apply, se2_inverse_apply, wrap_angle

COVERAGE_LEVELS = (0.5, 0.75, 0.9, 0.95)


# ----------------------------------------------------------------------------- área alcanzable

def reachable_region(gt: OccGrid, start_xy, cfg):
    """Máscara (en la rejilla real) de celdas libres que el robot puede llegar a observar de cerca."""
    clearance = robot_clearance(cfg)
    free = gt.free_mask()
    center_ok = free & ~inflate(gt.occ_mask() | gt.unknown_mask(), clearance, gt.resolution)
    si, sj = (int(v) for v in gt.world_to_cell(*start_xy))
    dist, _ = geodesic(center_ok, (si, sj), gt.resolution)
    reach_centers = np.isfinite(dist)
    # celdas entre las posiciones alcanzables del centro y la pared (+ margen)
    region = inflate(reach_centers, clearance + cfg['episode']['reachable_margin'], gt.resolution) & free
    return region


def _gt_cells_in_map_frame(gt: OccGrid, mask, start_pose):
    i, j = np.nonzero(mask)
    xw, yw = gt.cell_to_world(i, j)
    return se2_inverse_apply(start_pose, xw, yw)


class CoverageEvaluator:
    """Precalcula el área alcanzable para evaluar muchas instantáneas rápidamente."""

    def __init__(self, gt: OccGrid, start_pose, cfg):
        self.gt, self.start_pose, self.cfg = gt, tuple(start_pose), cfg
        self.region = reachable_region(gt, start_pose[:2], cfg)
        self.area_m2 = float(self.region.sum()) * gt.resolution ** 2
        self._xm, self._ym = _gt_cells_in_map_frame(gt, self.region, self.start_pose)

    def coverage(self, est: OccGrid):
        if len(self._xm) == 0:
            return 0.0
        v = est.lookup(self._xm, self._ym)
        known = ((v >= 0) & (v < self.cfg['mapping']['free_thresh'])) | (v >= self.cfg['mapping']['occ_thresh'])
        return float(known.mean())


# ----------------------------------------------------------------------------- calidad del mapa

def map_quality(est: OccGrid, gt: OccGrid, region, start_pose, cfg, tol=0.10):
    res = gt.resolution
    ft, ot = cfg['mapping']['free_thresh'], cfg['mapping']['occ_thresh']
    # 1) estimado -> rejilla real: celdas ocupadas estimadas en coordenadas del mundo
    ei, ej = np.nonzero(est.data >= ot)
    ex, ey = est.cell_to_world(ei, ej)
    wx, wy = se2_apply(start_pose, ex, ey)
    gi, gj = gt.world_to_cell(wx, wy)
    inside = gt.in_bounds(gi, gj)
    est_occ_w = np.zeros(gt.shape, bool)
    est_occ_w[gi[inside], gj[inside]] = True

    gt_occ = gt.occ_mask()
    eval_region = ndimage.binary_dilation(region, structure=disk(3))
    gt_occ_eval = gt_occ & eval_region
    dt_gt = ndimage.distance_transform_edt(~gt_occ) * res      # distancia a pared real
    dt_est = ndimage.distance_transform_edt(~est_occ_w) * res  # distancia a pared estimada

    n_est = len(ei)
    if n_est:
        d_est = np.full(n_est, np.inf)
        d_est[inside] = dt_gt[gi[inside], gj[inside]]
        occ_precision = float((d_est <= tol).mean())
        adnn = float(np.mean(np.minimum(d_est, 1.0)))  # celdas fuera del mapa real cuentan 1 m
    else:
        occ_precision, adnn = float('nan'), float('nan')
    occ_recall = float((dt_est[gt_occ_eval] <= tol).mean()) if gt_occ_eval.any() else float('nan')

    # 2) rejilla real -> estimado: IoU de espacio libre en la región de evaluación
    ri, rj = np.nonzero(eval_region)
    xw, yw = gt.cell_to_world(ri, rj)
    xm, ym = se2_inverse_apply(start_pose, xw, yw)
    v = est.lookup(xm, ym)
    est_free = (v >= 0) & (v < ft)
    gt_free = gt.free_mask()[ri, rj]
    union = (est_free | gt_free).sum()
    free_iou = float((est_free & gt_free).sum() / union) if union else float('nan')
    return dict(occ_precision=occ_precision, occ_recall=occ_recall, free_iou=free_iou, adnn=adnn)


# ----------------------------------------------------------------------------- carga de episodios

def load_snapshot(path):
    z = np.load(path)
    return float(z['t']), OccGrid(z['data'], float(z['resolution']), tuple(z['origin'].tolist()))


def load_episode(ep_dir):
    with open(os.path.join(ep_dir, 'meta.json')) as f:
        meta = json.load(f)
    poses = pd.read_csv(os.path.join(ep_dir, 'poses.csv'))
    goals = pd.read_csv(os.path.join(ep_dir, 'goals.csv'))
    events = pd.read_csv(os.path.join(ep_dir, 'events.csv'))
    snaps = sorted(glob.glob(os.path.join(ep_dir, 'maps', '*.npz')))
    return meta, poses, goals, events, snaps


# ----------------------------------------------------------------------------- métricas de episodio

def _first_time(t, c, level):
    idx = np.flatnonzero(c >= level)
    return float(t[idx[0]]) if len(idx) else float('nan')


def episode_metrics(ep_dir, gt: OccGrid, cfg):
    """Devuelve (fila de métricas: dict, curva: DataFrame t/coverage/distance)."""
    meta, poses, goals, events, snaps = load_episode(ep_dir)
    start = tuple(meta['start_pose'])
    t_max = float(meta.get('t_max', cfg['episode']['t_max']))
    ev = CoverageEvaluator(gt, start, cfg)

    # trayectoria real
    t_p = poses['t'].to_numpy()
    gx, gy, gyaw = (poses[c].to_numpy() for c in ('gt_x', 'gt_y', 'gt_yaw'))
    seg = np.hypot(np.diff(gx), np.diff(gy)) if len(gx) > 1 else np.zeros(0)
    cumdist = np.concatenate([[0.0], np.cumsum(seg)])
    rot = float(np.abs(wrap_angle(np.diff(gyaw))).sum()) if len(gyaw) > 1 else 0.0

    # curva de cobertura
    ts, covs, last_grid = [], [], None
    for p in snaps:
        t, grid = load_snapshot(p)
        ts.append(t)
        covs.append(ev.coverage(grid))
        last_grid = grid
    ts, covs = np.asarray(ts), np.asarray(covs)
    order = np.argsort(ts)
    ts, covs = ts[order], covs[order]
    dist_at = np.interp(ts, t_p, cumdist) if len(t_p) else np.zeros_like(ts)
    curve = pd.DataFrame(dict(t=ts, coverage=covs, distance=dist_at))

    row = dict(method=meta['method'], map=meta['map'], start_id=meta['start_id'],
               rep=meta.get('rep', 0), seed=meta.get('seed', 0),
               loc_mode=meta.get('loc_mode', 'slam'), backend=meta.get('backend', '?'),
               termination=meta.get('termination', 'unknown'),
               duration=float(meta.get('duration', ts[-1] if len(ts) else 0.0)),
               reachable_area_m2=ev.area_m2, t_max=t_max)
    final_cov = float(covs[-1]) if len(covs) else 0.0
    target = cfg['episode']['coverage_target']
    row['coverage_final'] = final_cov
    row['success'] = bool(final_cov >= target - 1e-9)
    for lv in COVERAGE_LEVELS:
        tag = int(round(lv * 100))
        tl = _first_time(ts, covs, lv)
        row[f'T{tag}'] = tl
        row[f'D{tag}'] = float(np.interp(tl, t_p, cumdist)) if np.isfinite(tl) and len(t_p) else float('nan')
        row[f'T{tag}_censored'] = not np.isfinite(tl)
    # AUC de la cobertura en [0, t_max], manteniendo el último valor (escalón)
    if len(ts):
        tt = np.concatenate([[0.0], ts, [t_max]])
        cc = np.concatenate([[0.0], covs, [covs[-1]]])
        tt = np.clip(tt, 0, t_max)
        row['auc'] = float(np.sum(np.diff(tt) * cc[:-1]) / t_max)
    else:
        row['auc'] = 0.0
    row['path_length'] = float(cumdist[-1]) if len(cumdist) else 0.0
    row['rotation_total'] = rot
    row['efficiency_m2_per_m'] = final_cov * ev.area_m2 / row['path_length'] if row['path_length'] > 0 else float('nan')

    # calidad del mapa final
    if last_grid is not None:
        row.update(map_quality(last_grid, gt, ev.region, start, cfg))

    # error de localización (marco del mapa -> mundo usando la pose inicial)
    if len(t_p):
        ewx, ewy = se2_apply(start, poses['est_x'].to_numpy(), poses['est_y'].to_numpy())
        err = np.hypot(ewx - gx, ewy - gy)
        row['ate_rmse'] = float(np.sqrt(np.mean(err ** 2)))
        row['pos_err_final'] = float(err[-1])
        row['pos_err_max'] = float(err.max())

    # navegación / decisión
    dec = pd.to_numeric(goals.loc[goals['event'] == 'goal', 'decision_time'], errors='coerce').dropna()
    row['n_goals'] = int((goals['event'] == 'goal').sum())
    row['decision_time_mean'] = float(dec.mean()) if len(dec) else float('nan')
    row['decision_time_p95'] = float(np.percentile(dec, 95)) if len(dec) else float('nan')
    row['n_collisions'] = int((events['kind'] == 'collision').sum()) if len(events) else 0
    row['n_nav_failures'] = int((events['kind'] == 'nav_failed').sum()) if len(events) else 0
    for k in ('n_goals_failed', 'n_goals_timeout', 'n_goals_invalidated'):
        if k in meta:
            row[k] = meta[k]
    return row, curve


def compute_campaign_metrics(campaign_dir, cfg, maps_dir, out_csv=None, curves_csv=None, verbose=True):
    """Recorre <campaign_dir>/episodes/**/meta.json y calcula métricas de todos los episodios."""
    from .grid import load_map_yaml
    metas = sorted(glob.glob(os.path.join(campaign_dir, 'episodes', '**', 'meta.json'), recursive=True))
    gts, rows, curves = {}, [], []
    for m in metas:
        ep = os.path.dirname(m)
        with open(m) as f:
            name = json.load(f)['map']
        if name not in gts:
            gts[name] = load_map_yaml(os.path.join(maps_dir, name + '.yaml'))
        try:
            row, curve = episode_metrics(ep, gts[name], cfg)
        except Exception as e:  # episodio corrupto: se registra, no se oculta
            if verbose:
                print(f'[metrics] ERROR en {ep}: {e}')
            continue
        row['episode'] = os.path.relpath(ep, campaign_dir)
        rows.append(row)
        for k in ('method', 'map', 'start_id', 'rep', 'loc_mode'):
            curve[k] = row[k]
        curves.append(curve)
    df = pd.DataFrame(rows)
    cv = pd.concat(curves, ignore_index=True) if curves else pd.DataFrame()
    if out_csv:
        df.to_csv(out_csv, index=False)
    if curves_csv:
        cv.to_csv(curves_csv, index=False)
    return df, cv
