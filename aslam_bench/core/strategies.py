"""Estrategias de selección de objetivos (la única pieza que cambia entre métodos).

Hay dos tipos de estrategia, según `outputs`:
  * 'goal'     : `choose(ctx, rng, blacklist) -> Goal | None` y `still_valid(grid, goal)`;
                 el objetivo lo alcanza Nav2 (ROS) o SimpleNavigator (sim2d). B0, B1...
  * 'velocity' : `act(ranges, angles, grid, pose) -> (v, w, info)` a `step_period`; la
                 política mueve el robot directamente (agente DRL end-to-end).
Añadir un método nuevo es añadir una clase aquí y registrarla en STRATEGIES (o, para
variantes con parámetro como 'DRL:<agente>', en `make_strategy`).

Métodos (plan de evaluación, §3): B0 aleatorio, B1 frontera más cercana, B2 utilidad
IG − λ·coste, B3 RRT-exploration, B4 IG con incertidumbre de pose, B5 Active Graph-SLAM
por D-optimalidad, B6 NBV por muestreo, B7 selector de fronteras aprendido, B8 utilidad
semántica, DRL agente end-to-end. Los parámetros de B2–B8 están en el bloque
`strategies:` de config/platform.yaml (y se guardan en meta.json de cada episodio).
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field

import numpy as np

from .frontiers import MapContext, frontier_mask, is_still_frontier
from .grid import wrap_angle
from .infogain import line_free, unknown_in_disk, visible_entropy, visible_unknown


@dataclass
class Goal:
    x: float
    y: float
    yaw: float
    distance: float                      # distancia geodésica estimada (m)
    target: tuple | None = None          # punto de interés (p. ej. centroide de la frontera)
    info: dict = field(default_factory=dict)


def _heading(ctx, x, y):
    return math.atan2(y - ctx.robot_xy[1], x - ctx.robot_xy[0])


def _blacklisted(x, y, blacklist, radius):
    return any((x - bx) ** 2 + (y - by) ** 2 < radius ** 2 for bx, by in blacklist)


class Strategy:
    name = 'base'
    outputs = 'goal'                  # 'goal' | 'velocity'
    uses_frontier_validity = False    # True si still_valid() comprueba que la frontera persiste

    uses_pose_graph = False           # True si necesita ctx.pose_graph (B4, B5, B8)
    defaults: dict = {}               # parámetros por defecto (se mezclan con cfg['strategies'][key])
    key = None                        # clave en cfg['strategies']

    def __init__(self, cfg):
        self.cfg = cfg
        sc = cfg.get('strategies') or {}
        self.common = dict(ig_radius=4.0, ig_rays=120)
        self.common.update({k: v for k, v in sc.items() if not isinstance(v, dict)})
        self.p = dict(self.defaults)
        if self.key:
            self.p.update(sc.get(self.key) or {})

    def params(self):
        """Parámetros efectivos (se guardan en meta.json)."""
        return dict(self.common, **self.p) if self.key else {}

    def observe(self, t, grid, pose):
        """Se llama en cada ciclo de decisión, haya o no que elegir objetivo (B3)."""

    def viz(self):
        """Datos extra para RViz: dict(points=[(x, y)...], segments=[((x0, y0), (x1, y1))...])."""
        return None

    def choose(self, ctx: MapContext, rng: np.random.Generator, blacklist) -> Goal | None:
        raise NotImplementedError

    def still_valid(self, grid, goal: Goal) -> bool:
        return True

    # utilidades comunes
    def ig(self, grid, x, y):
        return visible_unknown(grid, x, y, self.common['ig_radius'], self.common['ig_rays'])

    def frontier_still_valid(self, grid, goal):
        if goal.info.get('kind', 'frontier') != 'frontier':
            return True
        return is_still_frontier(grid, goal.target or (goal.x, goal.y), 0.15,
                                 self.cfg['exploration']['min_unknown_cells'])


class RandomGoal(Strategy):
    """B0 - Cota inferior: objetivo aleatorio en espacio libre conocido y alcanzable.

    Se muestrea uniformemente entre las celdas transitables cuya distancia
    geodésica está en `random_goal_dist`; si no hay ninguna, entre todas las
    alcanzables a más de 0,5 m. No usa información de fronteras para decidir.
    """
    name = 'B0_random'

    def choose(self, ctx, rng, blacklist):
        dmin, dmax = self.cfg['exploration']['random_goal_dist']
        rad = self.cfg['exploration']['blacklist_radius']
        tol = self.cfg['navigation']['goal_tolerance']
        d = ctx.dist
        for lo, hi in ((dmin, dmax), (0.5, np.inf)):
            cand = np.argwhere(ctx.passable & np.isfinite(d) & (d >= max(lo, tol)) & (d <= hi))
            if len(cand) == 0:
                continue
            for _ in range(20):
                i, j = cand[rng.integers(len(cand))]
                x, y = ctx.grid.cell_to_world(i, j)
                x, y = float(x), float(y)
                if not _blacklisted(x, y, blacklist, rad):
                    return Goal(x, y, _heading(ctx, x, y), float(d[i, j]))
        return None


class NearestFrontier(Strategy):
    """B1 - Frontera más cercana (Yamauchi, 1997) por distancia geodésica."""
    name = 'B1_nearest_frontier'
    uses_frontier_validity = True

    def choose(self, ctx, rng, blacklist):
        rad = self.cfg['exploration']['blacklist_radius']
        cands = [f for f in ctx.frontiers
                 if np.isfinite(f.distance) and not _blacklisted(*f.goal, blacklist, rad)]
        if not cands:
            return None
        # desempate: frontera más grande
        f = min(cands, key=lambda f: (round(f.distance, 2), -f.size_m))
        gx, gy = f.goal
        yaw = math.atan2(f.centroid[1] - gy, f.centroid[0] - gx) if (gx, gy) != f.centroid \
            else _heading(ctx, gx, gy)
        return Goal(gx, gy, yaw, f.distance, target=f.target,
                    info=dict(frontier_size=f.size_m))

    def still_valid(self, grid, goal):
        # si la frontera ya se ha observado, replanificar (comportamiento clásico)
        r = 0.15   # m alrededor de la celda frontera objetivo
        return is_still_frontier(grid, goal.target or (goal.x, goal.y), r,
                                 self.cfg['exploration']['min_unknown_cells'])


def _frontier_goal(ctx, f, **info):
    gx, gy = f.goal
    yaw = math.atan2(f.centroid[1] - gy, f.centroid[0] - gx) if (gx, gy) != f.centroid \
        else _heading(ctx, gx, gy)
    return Goal(gx, gy, yaw, f.distance, target=f.target,
                info=dict(kind='frontier', frontier_size=f.size_m, **info))


class FrontierUtility(Strategy):
    """B2 - Frontera con utilidad U = IG − λ·coste (González-Baños & Latombe, 2002; explore_lite).

    IG = área desconocida (m²) visible desde el objetivo de la frontera (trazado de rayos
    hasta `ig_radius`); coste = distancia geodésica (m). Se elige la frontera de máxima U.
    """
    name = 'B2_frontier_utility'
    key = 'B2'
    uses_frontier_validity = True
    defaults = dict(lambda_cost=1.0)

    def choose(self, ctx, rng, blacklist):
        best, bu = None, -np.inf
        for f in ctx.frontiers:
            ig = self.ig(ctx.grid, *f.goal)
            u = ig - self.p['lambda_cost'] * f.distance
            if u > bu:
                best, bu = _frontier_goal(ctx, f, ig=ig, utility=u), u
        return best

    def still_valid(self, grid, goal):
        return self.frontier_still_valid(grid, goal)


class RRTExploration(Strategy):
    """B3 - RRT-exploration (Umari & Mukhopadhyay, IROS 2017).

    Detectores: un RRT global (persistente, raíz en la pose inicial) y uno local (se
    reinicia en la pose del robot cada vez que encuentra una frontera) crecen en cada ciclo
    sobre el mapa. Cuando una arista nueva entra en lo desconocido, su extremo es un punto
    frontera; si atraviesa un obstáculo se descarta. Filtro: se quitan los puntos que ya no
    tocan lo desconocido o con poca ganancia, y se agrupan (radio `cluster_radius`).
    Asignador: revenue = info_multiplier · IG − coste, con IG = área desconocida en un disco
    de `info_radius` y, como en el original, IG × hysteresis_gain si el punto está a menos
    de hysteresis_radius del robot. Diferencia con el original: coste geodésico (no euclídeo).
    """
    name = 'B3_rrt_exploration'
    key = 'B3'
    uses_frontier_validity = True
    defaults = dict(eta_global=1.0, eta_local=0.5, iters_global=60, iters_local=60,
                    info_radius=1.0, info_multiplier=3.0, hysteresis_radius=3.0, hysteresis_gain=2.0,
                    cluster_radius=0.5, min_gain=0.05, max_points=300, max_nodes=4000, seed=0)

    def __init__(self, cfg):
        super().__init__(cfg)
        self.rng = np.random.default_rng(int(self.p['seed']))
        self.gtree = None            # (N, 2) nodos, padres
        self.gpar = None
        self.ltree = None
        self.lpar = None
        self.points = []             # puntos frontera detectados (x, y)

    def _bbox(self, grid):
        known = grid.data >= 0
        ii, jj = np.nonzero(known)
        if len(ii) == 0:
            return None
        x0, y0 = grid.cell_to_world(ii.min(), jj.min())
        x1, y1 = grid.cell_to_world(ii.max(), jj.max())
        return float(x0) - 1.0, float(x1) + 1.0, float(y0) - 1.0, float(y1) + 1.0

    def _edge(self, grid, p0, p1):
        """1 libre, 0 obstáculo, -1 entra en lo desconocido (como ObstacleFree de rrt_exploration)."""
        d = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
        n = max(2, int(math.ceil(d / (grid.resolution * 0.5))) + 1)
        i, j = grid.world_to_cell(np.linspace(p0[0], p1[0], n), np.linspace(p0[1], p1[1], n))
        ok = grid.in_bounds(i, j)
        if not ok.all():
            return 0
        v = grid.data[i, j]
        occ = v >= 65
        unk = v < 0
        first_occ = int(np.argmax(occ)) if occ.any() else n
        first_unk = int(np.argmax(unk)) if unk.any() else n
        if first_occ < first_unk:
            return 0
        if first_unk < n:
            return -1
        return 1

    def _grow(self, grid, nodes, par, iters, eta, bbox, local, robot_xy):
        x0, x1, y0, y1 = bbox
        nodes, par = list(map(tuple, nodes)), list(par)
        for _ in range(int(iters)):
            xr = (self.rng.uniform(x0, x1), self.rng.uniform(y0, y1))
            arr = np.asarray(nodes)
            k = int(np.argmin((arr[:, 0] - xr[0]) ** 2 + (arr[:, 1] - xr[1]) ** 2))
            xn = nodes[k]
            d = math.hypot(xr[0] - xn[0], xr[1] - xn[1])
            if d < 1e-6:
                continue
            s = min(1.0, eta / d)
            xnew = (xn[0] + s * (xr[0] - xn[0]), xn[1] + s * (xr[1] - xn[1]))
            r = self._edge(grid, xn, xnew)
            if r == -1:
                self.points.append(xnew)
                if local:
                    nodes, par = [tuple(robot_xy)], [-1]
            elif r == 1 and len(nodes) < self.p['max_nodes']:
                nodes.append(xnew)
                par.append(k)
        return np.asarray(nodes), par

    def observe(self, t, grid, pose):
        bbox = self._bbox(grid)
        if bbox is None:
            return
        xy = (float(pose[0]), float(pose[1]))
        if self.gtree is None:
            self.gtree, self.gpar = np.asarray([xy]), [-1]
        if self.ltree is None:
            self.ltree, self.lpar = np.asarray([xy]), [-1]
        self.gtree, self.gpar = self._grow(grid, self.gtree, self.gpar, self.p['iters_global'],
                                           self.p['eta_global'], bbox, False, xy)
        self.ltree, self.lpar = self._grow(grid, self.ltree, self.lpar, self.p['iters_local'],
                                           self.p['eta_local'], bbox, True, xy)
        if len(self.points) > self.p['max_points']:
            self.points = self.points[-int(self.p['max_points']):]

    def _filter(self, grid):
        """Quita puntos ya explorados y agrupa (equivale al filtro con mean-shift del original).

        'Explorado' con la misma máscara de fronteras que el resto de métodos (sin huecos
        pequeños ni interiores de pared), para que still_valid y el filtro coincidan.
        """
        fm = frontier_mask(grid, self.cfg['exploration']['min_unknown_cells'])
        r = max(1, int(math.ceil(0.3 / grid.resolution)))
        h, w = grid.shape

        def near_frontier(p):
            i, j = (int(v) for v in grid.world_to_cell(p[0], p[1]))
            return fm[max(0, i - r):min(h, i + r + 1), max(0, j - r):min(w, j + r + 1)].any()

        keep = [p for p in self.points
                if near_frontier(p)
                and unknown_in_disk(grid, p[0], p[1], self.p['info_radius']) >= self.p['min_gain']]
        self.points = keep
        clusters = []
        for p in keep:
            for c in clusters:
                if math.hypot(p[0] - c[0][0], p[1] - c[0][1]) < self.p['cluster_radius']:
                    c.append(p)
                    break
            else:
                clusters.append([p])
        reps = []
        for c in clusters:
            a = np.asarray(c)
            m = a.mean(axis=0)
            reps.append(tuple(a[int(np.argmin(((a - m) ** 2).sum(axis=1)))]))
        return reps

    def choose(self, ctx, rng, blacklist):
        if self.gtree is None:
            self.observe(0.0, ctx.grid, ctx.robot_xy)
        rad = self.cfg['exploration']['blacklist_radius']
        reach = self.cfg['platform']['base']['radius'] + self.cfg['navigation']['inflation_margin'] + 0.4
        best, bu = None, -np.inf
        rx, ry = ctx.robot_xy
        for p in self._filter(ctx.grid):
            cell = ctx.goal_near(p[0], p[1], reach)
            if cell is None:
                continue
            gx, gy = (float(v) for v in ctx.grid.cell_to_world(*cell))
            if _blacklisted(gx, gy, blacklist, rad):
                continue
            dist = float(ctx.dist[cell])
            if dist <= 2 * self.cfg['navigation']['goal_tolerance']:
                continue
            ig = unknown_in_disk(ctx.grid, p[0], p[1], self.p['info_radius'])
            if math.hypot(p[0] - rx, p[1] - ry) <= self.p['hysteresis_radius']:
                ig *= self.p['hysteresis_gain']
            u = self.p['info_multiplier'] * ig - dist
            if u > bu:
                yaw = math.atan2(p[1] - gy, p[0] - gx) if math.hypot(p[0] - gx, p[1] - gy) > 0.05 \
                    else _heading(ctx, gx, gy)
                best, bu = Goal(gx, gy, yaw, dist, target=p,
                                info=dict(kind='frontier', ig=ig, utility=u, n_points=len(self.points))), u
        if best is None:
            # sin puntos RRT útiles todavía: se recurre a la frontera más cercana (por mapa)
            g = NearestFrontier(self.cfg).choose(ctx, rng, blacklist)
            if g is not None:
                g.info['kind'] = 'frontier'
                g.info['rrt_fallback'] = True
            return g
        return best

    def still_valid(self, grid, goal):
        return is_still_frontier(grid, goal.target or (goal.x, goal.y), 0.3,
                                 self.cfg['exploration']['min_unknown_cells'])

    def viz(self):
        segs = []
        for nodes, par in ((self.gtree, self.gpar), (self.ltree, self.lpar)):
            if nodes is None:
                continue
            segs += [(tuple(nodes[k]), tuple(nodes[p])) for k, p in enumerate(par) if p >= 0]
        return dict(points=list(self.points), segments=segs)


class _GraphUtility(Strategy):
    """Base de B4/B5/B8: candidatos = fronteras + keyframes antiguos (revisita / cierre de bucle)."""
    uses_pose_graph = True
    uses_frontier_validity = True

    def lc_candidates(self, ctx, blacklist):
        pg = ctx.pose_graph
        if pg is None or len(pg.nodes) <= pg.lc_min_sep + 1:
            return []
        rad = self.cfg['exploration']['blacklist_radius']
        tol = self.cfg['navigation']['goal_tolerance']
        n = len(pg.nodes)
        out, taken = [], []
        for k in range(0, n - pg.lc_min_sep):
            x, y = pg.nodes[k]
            if any(math.hypot(x - a, y - b) < 1.0 for a, b in taken):
                continue
            cell = ctx.goal_near(x, y, 0.3)
            if cell is None:
                continue
            d = float(ctx.dist[cell])
            if not (max(tol, 1.0) < d <= self.p.get('lc_max_dist', 10.0)):
                continue
            gx, gy = (float(v) for v in ctx.grid.cell_to_world(*cell))
            if _blacklisted(gx, gy, blacklist, rad):
                continue
            taken.append((x, y))
            # cuanto mayor el ciclo que se cerraría respecto al recorrido, más prometedor
            ratio = (n - k) * pg.kf_spacing / max(d, 0.5)
            out.append((ratio, Goal(gx, gy, _heading(ctx, gx, gy), d, target=(x, y),
                                    info=dict(kind='loop_closure', keyframe=k))))
        out.sort(key=lambda r: -r[0])
        return [g for _, g in out[:int(self.p.get('max_lc_candidates', 8))]]

    def candidates(self, ctx, blacklist):
        return [_frontier_goal(ctx, f) for f in ctx.frontiers] + self.lc_candidates(ctx, blacklist)

    def predict(self, ctx, goal):
        cell = tuple(int(v) for v in ctx.grid.world_to_cell(goal.x, goal.y))
        path = ctx.path_xy(cell)
        return ctx.pose_graph.predict(path or [(goal.x, goal.y)], ctx.grid)

    def still_valid(self, grid, goal):
        return self.frontier_still_valid(grid, goal)


class PoseUncertaintyIG(_GraphUtility):
    """B4 - Ganancia de información con incertidumbre de pose (Stachniss, Grisetti & Burgard, RSS 2005).

    U(a) = IG_mapa(a) + w_pose · ΔH_pose(a) − λ · coste(a)
      * IG_mapa : entropía (bits·m²) de las celdas visibles desde el destino (desconocida =
                  1 bit, conocida ≈ 0), es decir, la reducción esperada de entropía del mapa;
      * ΔH_pose : H(pose actual) − H(pose al llegar), en bits; la pose se predice recorriendo
                  el camino sobre el grafo sustituto (`PoseGraph`): explorar la hace crecer
                  (ΔH < 0) y volver a un keyframe antiguo la reduce si cierra un bucle (ΔH > 0).
    Acciones: fronteras (exploración) y keyframes antiguos (cierre de bucle), como en el original;
    un keyframe solo cuenta como acción si el camino hasta él cierra algún bucle en el grafo.
    """
    name = 'B4_pose_uncertainty_ig'
    key = 'B4'
    defaults = dict(lambda_cost=1.0, w_pose=1.0, max_lc_candidates=8, lc_max_dist=10.0)

    def choose(self, ctx, rng, blacklist):
        pg = ctx.pose_graph
        _, _, var0 = pg.state()
        best, bu = None, -np.inf
        for g in self.candidates(ctx, blacklist):
            ig = visible_entropy(ctx.grid, g.x, g.y, self.common['ig_radius'], self.common['ig_rays'])
            pr = self.predict(ctx, g)
            if g.info['kind'] == 'loop_closure' and pr['n_lc'] == 0:
                continue          # revisitar sin cerrar un bucle no es una acción de cierre
            dh = math.log2(max(var0, 1e-9) / pr['var']) if var0 > 0 else 0.0
            u = ig + self.p['w_pose'] * dh - self.p['lambda_cost'] * g.distance
            if u > bu:
                g.info.update(ig=ig, dH_pose=dh, var_now=var0, var_goal=pr['var'], n_lc=pr['n_lc'], utility=u)
                best, bu = g, u
        return best


class DOptGraphSLAM(_GraphUtility):
    """B5 - Active Graph-SLAM por D-optimalidad (Placed & Castellanos, RA-L 2022-2023).

    U(a) = IG_geom(a) + w_dopt · ΔD(a) − λ · coste(a)
      * IG_geom : área desconocida visible desde el destino (m²), como B2;
      * ΔD      : cambio de la D-optimalidad del grafo de poses estimada con el Laplaciano
                  ponderado (log del nº ponderado de árboles generadores), normalizada por
                  nodo (ver `PoseGraph.predict`). Recorrer terreno nuevo apenas la cambia;
                  los caminos que cierran bucles la aumentan.
    """
    name = 'B5_dopt_graph_slam'
    key = 'B5'
    defaults = dict(lambda_cost=1.0, w_dopt=1.0, max_lc_candidates=8, lc_max_dist=10.0)

    def choose(self, ctx, rng, blacklist):
        best, bu = None, -np.inf
        for g in self.candidates(ctx, blacklist):
            ig = self.ig(ctx.grid, g.x, g.y)
            pr = self.predict(ctx, g)
            if g.info['kind'] == 'loop_closure' and pr['n_lc'] == 0:
                continue          # revisitar sin cerrar un bucle no es una acción de cierre
            u = ig + self.p['w_dopt'] * pr['dopt_gain'] - self.p['lambda_cost'] * g.distance
            if u > bu:
                g.info.update(ig=ig, dopt_gain=pr['dopt_gain'], n_lc=pr['n_lc'], utility=u)
                best, bu = g, u
        return best


class NBVPlanner(Strategy):
    """B6 - Next-Best-View por muestreo con horizonte deslizante (Bircher et al., ICRA 2016), en 2D.

    En cada decisión crece un RRT en el espacio libre conocido (transitable) alrededor del
    robot; cada nodo acumula Gain(n) = Gain(padre) + IG(n) · exp(−λ · c(n)), con IG = área
    desconocida visible (LiDAR 360°) y c(n) la longitud del camino desde la raíz. Se ejecuta
    solo el primer tramo de la mejor rama (horizonte deslizante) y el resto de la rama se
    reinserta en el árbol siguiente. Si tras `max_iters` ningún nodo supera `min_gain` (zona
    explorada lejos de las fronteras), se va a la frontera más cercana: el NBVP original
    terminaría aquí, lo que en 2D deja fronteras lejanas sin visitar.
    """
    name = 'B6_nbv_planner'
    key = 'B6'
    uses_frontier_validity = True     # solo afecta a los objetivos de respaldo (frontera más cercana)
    defaults = dict(iters=80, max_iters=300, max_nodes=150, step=1.0, lam=0.5, min_gain=0.5,
                    min_step=0.6, box=5.0, ig_rays=60, seed=0)

    def __init__(self, cfg):
        super().__init__(cfg)
        self.rng = np.random.default_rng(int(self.p['seed']))
        self.prev_branch = []
        self.tree = None

    def choose(self, ctx, rng, blacklist):
        grid = ctx.grid
        rx, ry = ctx.robot_xy
        R, rays = self.common['ig_radius'], int(self.p['ig_rays'])
        nodes, par, cost, gain = [(rx, ry)], [-1], [0.0], [0.0]

        def add(p, k):
            c = cost[k] + math.hypot(p[0] - nodes[k][0], p[1] - nodes[k][1])
            ig = visible_unknown(grid, p[0], p[1], R, rays)
            nodes.append(p)
            par.append(k)
            cost.append(c)
            gain.append(gain[k] + ig * math.exp(-self.p['lam'] * c))

        def free(p0, p1):
            return line_free(grid, p0, p1, ctx.passable)

        # reinsertar la mejor rama anterior
        k = 0
        for p in self.prev_branch:
            if not free(nodes[k], p):
                break
            add(p, k)
            k = len(nodes) - 1
        box = self.p['box']
        it = 0
        while it < self.p['max_iters'] and len(nodes) < self.p['max_nodes']:
            it += 1
            if it > self.p['iters'] and max(gain) >= self.p['min_gain']:
                break
            xr = (rx + self.rng.uniform(-box, box), ry + self.rng.uniform(-box, box))
            arr = np.asarray(nodes)
            k = int(np.argmin((arr[:, 0] - xr[0]) ** 2 + (arr[:, 1] - xr[1]) ** 2))
            d = math.hypot(xr[0] - nodes[k][0], xr[1] - nodes[k][1])
            if d < 0.2:
                continue
            s = min(1.0, self.p['step'] / d)
            p = (nodes[k][0] + s * (xr[0] - nodes[k][0]), nodes[k][1] + s * (xr[1] - nodes[k][1]))
            i, j = (int(v) for v in grid.world_to_cell(*p))
            if not (grid.in_bounds(i, j) and ctx.passable[i, j] and np.isfinite(ctx.dist[i, j])):
                continue
            if free(nodes[k], p):
                add(p, k)
        self.tree = (nodes, par)
        b = int(np.argmax(gain))
        if gain[b] < self.p['min_gain'] or b == 0:
            self.prev_branch = []
            g = NearestFrontier(self.cfg).choose(ctx, rng, blacklist)
            if g is not None:
                g.info.update(kind='frontier', nbv_fallback=True)
            return g
        branch = []
        while b > 0:
            branch.append(b)
            b = par[b]
        branch.reverse()
        step = 0
        while step < len(branch) - 1 and cost[branch[step]] < self.p['min_step']:
            step += 1
        gx, gy = nodes[branch[step]]
        self.prev_branch = [nodes[q] for q in branch[step + 1:]]
        cell = tuple(int(v) for v in grid.world_to_cell(gx, gy))
        prev = nodes[par[branch[step]]]
        yaw = math.atan2(gy - prev[1], gx - prev[0])
        return Goal(gx, gy, yaw, float(ctx.dist[cell]),
                    info=dict(kind='nbv', best_gain=float(max(gain)), n_nodes=len(nodes),
                              branch_len=len(branch)))

    def still_valid(self, grid, goal):
        return self.frontier_still_valid(grid, goal)

    def viz(self):
        if self.tree is None:
            return None
        nodes, par = self.tree
        return dict(points=list(self.prev_branch),
                    segments=[(nodes[k], nodes[p]) for k, p in enumerate(par) if p >= 0])


class LearnedFrontier(Strategy):
    """B7 - Selector de fronteras aprendido (en la línea de Niroui et al., RA-L 2019).

    Una red (MLP) puntúa cada frontera candidata a partir de rasgos del mapa y se elige la
    de mayor puntuación; la navegación sigue siendo Nav2 / SimpleNavigator. Los pesos son un
    .npz con W0, b0, W1, b1, ... (tanh en las capas ocultas, salida lineal escalar) sobre
    los rasgos de `features()`; ver `save_mlp`. Qué pesos se usan:
      'B7:<ruta.npz>' en `methods:`, o `strategies.B7.weights`, o ASLAM_BENCH_B7_WEIGHTS.
    NO incluye pesos entrenados: sin ellos la estrategia falla al decidir (no recurre en
    silencio a otro método).
    """
    name = 'B7_learned_frontier'
    key = 'B7'
    uses_frontier_validity = True
    defaults = dict(weights='')
    FEATURES = ('dist', 'euclid', 'ig', 'size', 'turn', 'n_frontiers', 'rank_dist')

    def __init__(self, cfg, weights=None):
        super().__init__(cfg)
        self.weights = weights or self.p.get('weights') or os.environ.get('ASLAM_BENCH_B7_WEIGHTS') or ''
        if weights:
            self.name = f'B7_{os.path.splitext(os.path.basename(weights))[0]}'
        self.layers = None

    def params(self):
        return dict(super().params(), weights=self.weights)

    def load(self):
        if self.layers is None:
            if not self.weights or not os.path.isfile(self.weights):
                raise RuntimeError(f'B7 necesita pesos entrenados (.npz); no encuentro "{self.weights}". '
                                   'Usa B7:<ruta.npz>, strategies.B7.weights o ASLAM_BENCH_B7_WEIGHTS.')
            z = np.load(self.weights)
            n = len([k for k in z.files if k.startswith('W')])
            self.layers = [(z[f'W{i}'], z[f'b{i}']) for i in range(n)]
            if self.layers[0][0].shape[0] != len(self.FEATURES):
                raise ValueError(f'B7: la red espera {self.layers[0][0].shape[0]} rasgos y hay '
                                 f'{len(self.FEATURES)} ({self.FEATURES})')
        return self.layers

    def features(self, ctx, frontiers, yaw):
        rx, ry = ctx.robot_xy
        d = np.array([f.distance for f in frontiers])
        rank = np.argsort(np.argsort(d))
        X = []
        for f, r in zip(frontiers, rank):
            gx, gy = f.goal
            turn = abs(float(wrap_angle(math.atan2(gy - ry, gx - rx) - yaw)))
            X.append([f.distance / 10.0, math.hypot(gx - rx, gy - ry) / 10.0, self.ig(ctx.grid, gx, gy) / 20.0,
                      f.size_m / 5.0, turn / math.pi, len(frontiers) / 20.0, r / max(1, len(frontiers) - 1)])
        return np.asarray(X, float)

    def scores(self, X):
        h = X
        layers = self.load()
        for k, (W, b) in enumerate(layers):
            h = h @ W + b
            if k < len(layers) - 1:
                h = np.tanh(h)
        return h.reshape(-1)

    def choose(self, ctx, rng, blacklist):
        fr = list(ctx.frontiers)
        if not fr:
            return None
        s = self.scores(self.features(ctx, fr, ctx.robot_yaw))
        k = int(np.argmax(s))
        return _frontier_goal(ctx, fr[k], score=float(s[k]))

    def still_valid(self, grid, goal):
        return self.frontier_still_valid(grid, goal)


def save_mlp(path, layers):
    """Guarda una red [(W0, b0), (W1, b1), ...] en el formato que lee B7."""
    np.savez(path, **{f'W{i}': W for i, (W, _) in enumerate(layers)},
             **{f'b{i}': b for i, (_, b) in enumerate(layers)})


class SemanticUtility(_GraphUtility):
    """B8 - Utilidad semántica del diseño de la capa semántica (§7):

        U(g) = w_e · IG_geom(g) + w_s · IG_sem(g) + w_l · LC(g) − w_c · coste(g)

      * IG_geom : área desconocida visible desde g (LiDAR 360°, m²), como B2;
      * IG_sem  : Σ entropía de clase H(p) (bits) de los landmarks que entrarían en el FOV de
                  la cámara con la orientación final elegida, + un extra por cada landmark
                  'tentative' por confirmar;
      * LC      : si la varianza de la pose supera `var_threshold`, la reducción de entropía
                  de la pose (bits, grafo sustituto) al llegar a g, solo si desde g se ve
                  algún landmark confirmado (cierre de bucle verificable con semántica);
      * coste   : distancia geodésica (m).
    Candidatos: fronteras, puntos de vista de landmarks dudosos y, con incertidumbre alta,
    puntos de vista de landmarks confirmados. La orientación final se elige entre `n_yaws`
    para maximizar w_s·IG_sem + w_l·LC (la cámara tiene FOV estrecho), y se pide a Nav2.
    Sin mapa semántico (p. ej. en ROS sin el tópico), IG_sem = LC = 0 y se reduce a B2.
    """
    name = 'B8_semantic_utility'
    key = 'B8'
    defaults = dict(w_e=1.0, w_s=3.0, w_l=2.0, w_c=1.0, var_threshold=0.05, n_yaws=8,
                    view_dist=1.5, entropy_threshold=1.0, tentative_bonus=0.5, max_visits=2,
                    max_lc_candidates=6, lc_max_dist=10.0)

    def __init__(self, cfg):
        super().__init__(cfg)
        cam = cfg['platform'].get('camera') or {}
        s = cfg.get('semantics') or {}
        self.hfov = math.radians(float(cam.get('hfov_deg', 69)))
        self.cam_range = float(s.get('cam_range', 4.0))
        self.visits = {}

    def _viewpoints(self, ctx, lm, blacklist, kind):
        rad = self.cfg['exploration']['blacklist_radius']
        out = []
        for a in np.linspace(0, 2 * math.pi, 8, endpoint=False):
            x = lm.xy[0] + self.p['view_dist'] * math.cos(a)
            y = lm.xy[1] + self.p['view_dist'] * math.sin(a)
            cell = ctx.goal_near(x, y, 0.4)
            if cell is None:
                continue
            gx, gy = (float(v) for v in ctx.grid.cell_to_world(*cell))
            d = float(ctx.dist[cell])
            if d <= self.cfg['navigation']['goal_tolerance'] or _blacklisted(gx, gy, blacklist, rad):
                continue
            if not line_free(ctx.grid, (gx, gy), tuple(lm.xy)):
                continue
            out.append(Goal(gx, gy, math.atan2(lm.xy[1] - gy, lm.xy[0] - gx), d, target=tuple(lm.xy),
                            info=dict(kind=kind, landmark=int(lm.id))))
        out.sort(key=lambda g: g.distance)
        return out[:2]

    def candidates(self, ctx, blacklist):
        cands = [_frontier_goal(ctx, f) for f in ctx.frontiers]
        sm = ctx.semantic
        if sm is None or not sm.landmarks:
            return cands
        _, _, var0 = ctx.pose_graph.state() if ctx.pose_graph is not None else (0, 0, 0.0)
        for lm in sm.landmarks.values():
            if self.visits.get(lm.id, 0) >= self.p['max_visits']:
                continue
            if lm.state == 'tentative' or lm.entropy > self.p['entropy_threshold']:
                cands += self._viewpoints(ctx, lm, blacklist, 'semantic')
            elif var0 > self.p['var_threshold']:
                cands += self._viewpoints(ctx, lm, blacklist, 'loop_closure')
        return cands

    def choose(self, ctx, rng, blacklist):
        from .semantics import confirmed_in_view, ig_sem
        sm = ctx.semantic
        pg = ctx.pose_graph
        _, _, var0 = pg.state() if pg is not None else (0, 0, 0.0)
        lc_on = pg is not None and var0 > self.p['var_threshold']
        yaws = list(np.linspace(-math.pi, math.pi, int(self.p['n_yaws']), endpoint=False))
        best, bu = None, -np.inf
        for g in self.candidates(ctx, blacklist):
            ig_g = self.ig(ctx.grid, g.x, g.y)
            dh = 0.0
            if lc_on:
                pr = self.predict(ctx, g)
                dh = max(0.0, math.log2(var0 / pr['var']))
            best_yaw, best_sem, best_lc, best_v = g.yaw, 0.0, 0.0, -np.inf
            for yaw in [g.yaw] + yaws:
                sem = ig_sem(sm, ctx.grid, g.x, g.y, yaw, self.hfov, self.cam_range,
                             self.p['tentative_bonus']) if sm is not None and sm.landmarks else 0.0
                lc = dh if dh > 0 and sm is not None and confirmed_in_view(
                    sm, ctx.grid, g.x, g.y, yaw, self.hfov, self.cam_range) else 0.0
                v = self.p['w_s'] * sem + self.p['w_l'] * lc
                if v > best_v + 1e-9:
                    best_yaw, best_sem, best_lc, best_v = yaw, sem, lc, v
            u = self.p['w_e'] * ig_g + best_v - self.p['w_c'] * g.distance
            if u > bu:
                g.yaw = float(best_yaw)
                g.info.update(ig=ig_g, ig_sem=best_sem, lc=best_lc, utility=u, orient=True)
                best, bu = g, u
        if best is not None and best.info.get('landmark') is not None:
            lid = best.info['landmark']
            self.visits[lid] = self.visits.get(lid, 0) + 1
        return best


class DrlVelocity(Strategy):
    """DRL - Agente TD3 de c3-aslamdrl-platform (rama drl): emite (v, w) a 10 Hz.

    Qué agente (.pt) se usa, por orden de prioridad:
      1. 'DRL:<ruta>.pt' en `methods:` (o en el parámetro `strategy` del nodo);
      2. 'DRL:<alias>' con `drl.agents[<alias>]` en la configuración / campaña;
      3. `drl.weights`, o la variable de entorno ASLAM_BENCH_DRL_WEIGHTS para 'DRL' a secas.
    El nombre del método en los resultados es 'DRL' o 'DRL_<alias|nombre del .pt>'.
    Los pesos se cargan la primera vez que se llama a `act()` (así `run_campaign` y el
    monitor pueden instanciar la estrategia sin torch ni pesos).
    """
    name = 'DRL'
    outputs = 'velocity'

    def __init__(self, cfg, agent=None, weights=None):
        super().__init__(cfg)
        dcfg = cfg.get('drl', {}) or {}
        self.agent = agent
        if agent and agent.endswith('.pt'):
            stem = os.path.basename(agent)[:-3]
            self.name = f'DRL_{stem}'
            self.weights = weights or agent
        elif agent:
            self.name = f'DRL_{agent}'
            self.weights = weights or (dcfg.get('agents') or {}).get(agent)
        else:
            self.weights = weights or dcfg.get('weights') or None
        self.weights = self.weights or os.environ.get('ASLAM_BENCH_DRL_WEIGHTS') or None
        self.step_period = float(dcfg.get('step_period', 0.1))
        self.policy = None

    def load(self):
        if self.policy is None:
            from .drl_policy import DrlPolicy
            b = self.cfg['platform']['base']
            dcfg = self.cfg.get('drl', {}) or {}
            self.policy = DrlPolicy(self.weights, v_max=b['max_lin_vel'], w_max=b['max_ang_vel'],
                                    clip_to_platform=bool(dcfg.get('clip_to_platform', True)),
                                    device=dcfg.get('device', 'cpu'),
                                    torch_threads=dcfg.get('torch_threads', 1))
        return self.policy

    def reset(self):
        if self.policy is not None:
            self.policy.reset()

    def act(self, ranges, angles, grid, pose):
        return self.load().act(ranges, angles, grid, pose)

    def describe(self):
        p = self.load()
        return dict(drl_weights=p.weights, drl_weights_sha256=p.weights_sha,
                    drl_clip_to_platform=p.clip, drl_step_period=self.step_period)

    def choose(self, ctx, rng, blacklist):
        raise RuntimeError('DrlVelocity no elige objetivos: usa act()')


STRATEGIES = {
    'B0': RandomGoal, 'random': RandomGoal, RandomGoal.name: RandomGoal,
    'B1': NearestFrontier, 'nearest_frontier': NearestFrontier, NearestFrontier.name: NearestFrontier,
    'DRL': DrlVelocity, 'drl': DrlVelocity,
}
for _cls, _alias in ((FrontierUtility, 'frontier_utility'), (RRTExploration, 'rrt_exploration'),
                     (PoseUncertaintyIG, 'pose_uncertainty_ig'), (DOptGraphSLAM, 'dopt_graph_slam'),
                     (NBVPlanner, 'nbv_planner'), (LearnedFrontier, 'learned_frontier'),
                     (SemanticUtility, 'semantic_utility')):
    STRATEGIES[_cls.key] = STRATEGIES[_alias] = STRATEGIES[_cls.name] = _cls


def make_strategy(name, cfg, **kw) -> Strategy:
    base, _, arg = str(name).partition(':')
    if base in ('DRL', 'drl') and arg:
        return DrlVelocity(cfg, agent=arg, **kw)
    if base.startswith('DRL_') and base[4:] in ((cfg.get('drl') or {}).get('agents') or {}):
        return DrlVelocity(cfg, agent=base[4:], **kw)   # nombre ya resuelto ('DRL_<alias>')
    if base in ('B7', 'learned_frontier') and arg:
        return LearnedFrontier(cfg, weights=arg)
    try:
        cls = STRATEGIES[name]
    except KeyError:
        raise ValueError(f'Estrategia desconocida: {name}. Opciones: {sorted(STRATEGIES)} '
                         f'(o DRL:<alias|ruta.pt>, B7:<ruta.npz>)') from None
    return cls(cfg, **kw) if cls is DrlVelocity else cls(cfg)
