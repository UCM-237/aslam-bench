"""Simulador 2D rápido para validar el pipeline y prototipar métodos sin Gazebo.

NO sustituye a Gazebo: modela un robot diferencial circular, un LiDAR 2D por
trazado de rayos sobre el mapa real, un mapeo por log-odds y un navegador
sencillo (Dijkstra sobre el mapa estimado + seguidor de trayectoria). La
localización es la odometría, con ruido opcional (`odom_noise`); con ruido 0 es
el modo 'pose oráculo' del plan de evaluación.

Usa la misma `ExplorerLogic` y el mismo `EpisodeRecorder` que la versión ROS 2,
así que los resultados se procesan con exactamente las mismas herramientas.
"""
from __future__ import annotations

import math
import time

import numpy as np
from scipy import ndimage

from ..core.config import robot_clearance
from ..core.explorer import (NAV_ACTIVE, NAV_FAILED, NAV_IDLE, NAV_SUCCEEDED, ExplorerLogic)
from ..core.grid import OccGrid, geodesic, inflate, se2_apply, se2_inverse_apply, wrap_angle
from ..core.metrics import CoverageEvaluator
from ..core.recorder import EpisodeRecorder
from ..core.semantics import SemanticSim

L_FREE, L_OCC, L_MIN, L_MAX = -1.4, 2.0, -4.0, 4.0   # una observación basta para libre(<25)/ocupado(>=65)


class Lidar2D:
    def __init__(self, gt: OccGrid, cfg, rng):
        lc = cfg['platform']['lidar']
        self.gt = gt
        self.blocked = gt.occ_mask() | gt.unknown_mask()  # fuera del edificio = opaco
        self.n = int(lc['n_beams'])
        self.rmin, self.rmax = float(lc['range_min']), float(lc['range_max'])
        self.noise = float(lc['noise_std'])
        self.angles = np.linspace(-math.pi, math.pi, self.n, endpoint=False)
        self.samples = np.arange(0.0, self.rmax + gt.resolution, gt.resolution * 0.5)
        self.rng = rng

    def scan(self, pose):
        x, y, th = pose
        a = self.angles + th
        px = x + np.cos(a)[:, None] * self.samples[None, :]
        py = y + np.sin(a)[:, None] * self.samples[None, :]
        i, j = self.gt.world_to_cell(px, py)
        ok = self.gt.in_bounds(i, j)
        hit = np.ones_like(ok)
        hit[ok] = self.blocked[i[ok], j[ok]]
        first = np.where(hit.any(axis=1), hit.argmax(axis=1), -1)
        r = np.where(first >= 0, self.samples[np.maximum(first, 0)], np.inf)
        finite = np.isfinite(r)
        r[finite] += self.rng.normal(0, self.noise, finite.sum())
        r[finite & (r > self.rmax)] = np.inf
        r[finite & (r < self.rmin)] = np.nan
        return r


class LogOddsMapper:
    """Rejilla en el marco 'map' (= pose inicial), con tamaño fijo que cubre el mapa real."""

    def __init__(self, gt: OccGrid, start_pose, cfg, margin=1.5):
        res = cfg['mapping']['resolution']
        x0, x1, y0, y1 = gt.extent()
        cx, cy = se2_inverse_apply(start_pose, [x0, x1, x1, x0], [y0, y0, y1, y1])
        ox, oy = math.floor((min(cx) - margin) / res) * res, math.floor((min(cy) - margin) / res) * res
        w = int(math.ceil((max(cx) + margin - ox) / res))
        h = int(math.ceil((max(cy) + margin - oy) / res))
        self.res, self.origin = res, (ox, oy)
        self.L = np.zeros((h, w), np.float32)
        self.seen = np.zeros((h, w), bool)
        self.rmax = cfg['platform']['lidar']['range_max']
        self._grid = OccGrid(np.full((h, w), -1, np.int8), res, self.origin)

    def integrate(self, est_pose, ranges, angles):
        x, y, th = est_pose
        valid = ~np.isnan(ranges)
        hit = np.isfinite(ranges) & valid
        r = np.where(hit, ranges, self.rmax)
        r = np.where(valid, r, 0.0)
        a = angles + th
        step = self.res * 0.5
        s = np.arange(0.0, self.rmax, step)
        mask = s[None, :] < (r[:, None] - self.res)
        px = x + np.cos(a)[:, None] * s[None, :]
        py = y + np.sin(a)[:, None] * s[None, :]
        g = self._grid
        fi, fj = g.world_to_cell(px[mask], py[mask])
        hx, hy = x + np.cos(a[hit]) * ranges[hit], y + np.sin(a[hit]) * ranges[hit]
        oi, oj = g.world_to_cell(hx, hy)
        h, w = self.L.shape
        okf = (fi >= 0) & (fi < h) & (fj >= 0) & (fj < w)
        oko = (oi >= 0) & (oi < h) & (oj >= 0) & (oj < w)
        free_lin = np.unique(fi[okf] * w + fj[okf])
        occ_lin = np.unique(oi[oko] * w + oj[oko])
        free_lin = np.setdiff1d(free_lin, occ_lin, assume_unique=True)
        Lf = self.L.reshape(-1)
        Lf[free_lin] += L_FREE
        Lf[occ_lin] += L_OCC
        np.clip(self.L, L_MIN, L_MAX, out=self.L)
        sf = self.seen.reshape(-1)
        sf[free_lin] = True
        sf[occ_lin] = True

    def grid(self):
        p = 1.0 / (1.0 + np.exp(-self.L))
        d = np.where(self.seen, np.round(p * 100), -1).astype(np.int8)
        self._grid = OccGrid(d, self.res, self.origin)
        return self._grid


class SimpleNavigator:
    """Sustituto mínimo de Nav2: Dijkstra en el mapa estimado y seguidor con giro en el sitio."""

    def __init__(self, cfg):
        self.cfg = cfg
        b = cfg['platform']['base']
        self.vmax, self.wmax = b['max_lin_vel'], b['max_ang_vel']
        self.tol = cfg['navigation']['goal_tolerance']
        self.clear = robot_clearance(cfg)
        self.goal = None
        self.goal_yaw = None
        self.path = None
        self.status = NAV_IDLE
        self.t_plan = -1e9
        self.t_progress = 0.0
        self.best_d = math.inf

    def set_goal(self, t, goal_xy, yaw=None):
        """Con `yaw`, al llegar gira en el sitio hasta esa orientación (como Nav2 con su
        tolerancia de yaw); sin él, basta la posición (comportamiento de la iteración 1)."""
        self.goal, self.path, self.status = goal_xy, None, NAV_ACTIVE
        self.goal_yaw = yaw
        self.t_plan, self.t_progress, self.best_d = -1e9, t, math.inf
        self.recover_until, self.n_coll = -1e9, 0

    def on_collision(self, t):
        """Recuperación tipo Nav2 (BackUp): retroceder 1 s y replanificar; 3 choques -> fallo."""
        self.n_coll += 1
        self.path = None
        if self.n_coll >= 3:
            self.status = NAV_FAILED
        else:
            self.recover_until = t + 1.0

    def cancel(self):
        self.goal, self.path, self.status = None, None, NAV_IDLE

    def _plan(self, grid: OccGrid, pose):
        # Primero con holgura extra (como el coste de inflado de Nav2, que aleja de las
        # paredes); si no hay camino, con la holgura mínima.
        for clear in (self.clear + 0.10, self.clear):
            occ = inflate(grid.occ_mask(), clear, grid.resolution)
            passable = ~occ            # desconocido transitable (allow_unknown de Nav2)
            si, sj = (int(v) for v in grid.world_to_cell(pose[0], pose[1]))
            r = int(math.ceil(clear / grid.resolution)) + 1   # permitir salir si está pegado a la pared
            sl = (slice(max(0, si - r), si + r + 1), slice(max(0, sj - r), sj + r + 1))
            passable[sl] |= ~grid.occ_mask()[sl]
            gi, gj = (int(v) for v in grid.world_to_cell(*self.goal))
            h, w = passable.shape
            if not (0 <= gi < h and 0 <= gj < w):
                return None
            passable[gi, gj] = True
            _, path = geodesic(passable, (si, sj), grid.resolution, return_path_to=(gi, gj))
            if path is not None:
                break
        if path is None:
            return None
        xs, ys = grid.cell_to_world(np.array([p[0] for p in path]), np.array([p[1] for p in path]))
        return np.stack([xs, ys], axis=1)

    def _blocked(self, grid, pose):
        if self.path is None or len(self.path) == 0:
            return False
        d = np.hypot(self.path[:, 0] - pose[0], self.path[:, 1] - pose[1])
        ahead = self.path[(d < 2.0)]
        if len(ahead) == 0:
            return False
        v = grid.lookup(ahead[:, 0], ahead[:, 1])
        return bool((v >= self.cfg['mapping']['occ_thresh']).any())

    def command(self, t, grid, pose):
        """Devuelve (v, w) y actualiza self.status."""
        if self.status != NAV_ACTIVE:
            return 0.0, 0.0
        if t < self.recover_until:
            return -0.1, 0.0
        dg = math.hypot(self.goal[0] - pose[0], self.goal[1] - pose[1])
        if dg < self.tol:
            if self.goal_yaw is not None:
                e = float(wrap_angle(self.goal_yaw - pose[2]))
                if abs(e) > 0.15:
                    self.t_progress = t
                    return 0.0, math.copysign(min(self.wmax, max(0.3, 2.0 * abs(e))), e)
            self.status = NAV_SUCCEEDED
            return 0.0, 0.0
        if dg < self.best_d - 0.05:
            self.best_d, self.t_progress = dg, t
        if t - self.t_progress > 20.0:
            self.status = NAV_FAILED
            return 0.0, 0.0
        if self.path is None or t - self.t_plan > 10.0 or self._blocked(grid, pose):
            self.path, self.t_plan = self._plan(grid, pose), t
            if self.path is None:
                self.status = NAV_FAILED
                return 0.0, 0.0
        # punto de anticipación
        d = np.hypot(self.path[:, 0] - pose[0], self.path[:, 1] - pose[1])
        k = int(np.argmin(d))
        self.path = self.path[k:]
        d = d[k:]
        la = np.flatnonzero(d > 0.20)          # anticipación corta: no recortar esquinas
        tx, ty = self.path[la[0]] if len(la) else self.goal
        e = float(wrap_angle(math.atan2(ty - pose[1], tx - pose[0]) - pose[2]))
        if abs(e) > 0.4:
            return 0.0, math.copysign(min(self.wmax, 2.5 * abs(e)), e)
        v = self.vmax * max(0.2, 1.0 - abs(e) / 0.4) * min(1.0, dg / 0.5 + 0.3)
        return v, float(np.clip(2.5 * e, -self.wmax, self.wmax))


def run_episode(gt: OccGrid, start_pose, method, cfg, out_dir, meta_extra=None, seed=0,
                odom_noise=(0.0, 0.0), dt=0.1):
    """Ejecuta un episodio completo y lo guarda en out_dir. Devuelve meta (dict)."""
    rng = np.random.default_rng(seed)
    ep = cfg['episode']
    lidar = Lidar2D(gt, cfg, rng)
    mapper = LogOddsMapper(gt, start_pose, cfg)
    nav = SimpleNavigator(cfg)
    explorer = ExplorerLogic(method, cfg, seed=seed)
    sem_cfg = cfg.get('semantics') or {}
    sem = SemanticSim(gt, start_pose, cfg, seed=seed) if sem_cfg.get('sim_enabled', True) else None
    t_cam = -1e9
    cov_eval = CoverageEvaluator(gt, start_pose, cfg)
    clear_gt = ndimage.distance_transform_edt(~(gt.occ_mask() | gt.unknown_mask())) * gt.resolution
    radius = cfg['platform']['base']['radius']
    loc_mode = 'oracle' if odom_noise == (0.0, 0.0) or not any(odom_noise) else 'odom_noise'
    meta = dict(method=explorer.strategy.name, backend='sim2d', loc_mode=loc_mode,
                start_pose=list(map(float, start_pose)), seed=int(seed), t_max=ep['t_max'],
                coverage_target=ep['coverage_target'], platform=cfg['platform']['name'],
                odom_noise=list(odom_noise), strategy_params=explorer.strategy.params())
    meta.update(meta_extra or {})
    vel_mode = explorer.velocity_mode            # política que emite (v, w): DRL
    if vel_mode:
        explorer.strategy.reset()
        meta.update(explorer.strategy.describe())  # carga los pesos y guarda ruta + hash
        step_period = explorer.strategy.step_period
    rec = EpisodeRecorder(out_dir, meta)

    true = np.array(start_pose, float)
    est = np.zeros(3)                       # marco map = pose inicial
    t, wall0 = 0.0, time.perf_counter()
    lidar_period = 1.0 / cfg['platform']['lidar']['rate_hz']
    t_lidar = t_dec = t_snap = t_pose = -1e9
    cov_hist = []                           # (t, cov) para detectar estancamiento
    termination, collisions = None, 0
    grid = mapper.grid()
    last_scan = None
    t_act, v_cmd, w_cmd, dec_times, in_contact = -1e9, 0.0, 0.0, [], False
    while termination is None:
        if t - t_lidar >= lidar_period - 1e-9:
            last_scan = lidar.scan(true)
            mapper.integrate(est, last_scan, lidar.angles)
            grid = mapper.grid()
            t_lidar = t
        if sem is not None and t - t_cam >= sem.period - 1e-9:
            explorer.semantic.integrate(sem.observe(true, est), est[:2])
            t_cam = t
        if t - t_dec >= cfg['exploration']['replan_period'] - 1e-9:
            act = explorer.update(t, grid, est, nav.status)
            if vel_mode and dec_times:
                # una fila por periodo de decisión con el tiempo medio de inferencia
                rec.log_goal(t, 'policy', None, dict(act.info, decision_time=float(np.mean(dec_times))))
                dec_times = []
            if act.kind == 'goal':
                nav.set_goal(t, (act.goal.x, act.goal.y),
                             act.goal.yaw if act.goal.info.get('orient') else None)
                rec.log_goal(t, 'goal', act.goal, act.info)
            elif act.kind == 'done':
                rec.log_goal(t, 'done', None, act.info)
                termination = 'no_frontiers'
            if nav.status == NAV_FAILED:
                rec.log_event(t, 'nav_failed')
                nav.cancel()
            t_dec = t
        if t - t_snap >= ep['snapshot_period'] - 1e-9 or termination:
            rec.snapshot_map(t, grid)
            c = cov_eval.coverage(grid)
            cov_hist.append((t, c))
            t_snap = t
            if c >= ep['coverage_target']:
                termination = termination or 'coverage_reached'
            past = [cv for (tt, cv) in cov_hist if tt <= t - ep['stall_window']]
            if past and c - past[-1] < ep['stall_min_gain'] and not termination:
                termination = 'stalled'
        if t - t_pose >= ep['pose_period'] - 1e-9 or termination:
            rec.log_pose(t, true, est)
            t_pose = t
        if termination:
            break
        if t >= ep['t_max']:
            termination = 'timeout'
            rec.snapshot_map(t, grid)
            rec.log_pose(t, true, est)
            break

        if vel_mode:
            if t - t_act >= step_period - 1e-9:
                v_cmd, w_cmd, pinfo = explorer.strategy.act(last_scan, lidar.angles, grid, est)
                dec_times.append(pinfo['decision_time'])
                t_act = t
            v, w = v_cmd, w_cmd
        else:
            v, w = nav.command(t, grid, est)
        # movimiento real
        nx = true[0] + v * dt * math.cos(true[2] + w * dt / 2)
        ny = true[1] + v * dt * math.sin(true[2] + w * dt / 2)
        i, j = gt.world_to_cell(nx, ny)
        if v == 0.0 or (gt.in_bounds(i, j) and clear_gt[i, j] > radius):
            dx, dy = nx - true[0], ny - true[1]
            true[:2] = nx, ny
            moved_v = v
            in_contact = False
        else:
            if vel_mode:
                # la política no tiene recuperación y sigue empujando: se cuenta cada contacto
                # una sola vez (como un bumper), no cada paso bloqueado
                if not in_contact:
                    collisions += 1
                    rec.log_event(t, 'collision')
                in_contact = True
            else:
                collisions += 1
                rec.log_event(t, 'collision')
            if vel_mode:
                pass
            elif v > 0:
                nav.on_collision(t)
            else:
                nav.recover_until = -1e9     # no puede retroceder: replanifica desde aquí
            dx = dy = 0.0
            moved_v = 0.0
        true[2] = float(wrap_angle(true[2] + w * dt))
        # odometría (con ruido proporcional al movimiento)
        sv, sw = odom_noise
        vo = moved_v * (1 + rng.normal(0, sv)) if sv else moved_v
        wo = w + rng.normal(0, sw) * (abs(w) + 0.1 * abs(moved_v)) if sw else w
        est[0] += vo * dt * math.cos(est[2] + wo * dt / 2)
        est[1] += vo * dt * math.sin(est[2] + wo * dt / 2)
        est[2] = float(wrap_angle(est[2] + wo * dt))
        t = round(t + dt, 6)

    extra = dict(explorer.stats())
    if sem is not None:
        extra.update(sem.evaluate(explorer.semantic))
        rec.write_json('semantic.json', sem.dump(explorer.semantic))
    rec.finish(termination, duration=t, wall_time=time.perf_counter() - wall0,
               n_collisions=collisions, **extra)
    return rec.meta
