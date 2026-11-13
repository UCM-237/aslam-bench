"""Estrategias de selección de objetivos (la única pieza que cambia entre métodos).

Hay dos tipos de estrategia, según `outputs`:
  * 'goal'     : `choose(ctx, rng, blacklist) -> Goal | None` y `still_valid(grid, goal)`;
                 el objetivo lo alcanza Nav2 (ROS) o SimpleNavigator (sim2d). B0, B1...
  * 'velocity' : `act(ranges, angles, grid, pose) -> (v, w, info)` a `step_period`; la
                 política mueve el robot directamente (agente DRL end-to-end).
Añadir un método nuevo es añadir una clase aquí y registrarla en STRATEGIES (o, para
variantes con parámetro como 'DRL:<agente>', en `make_strategy`).
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field

import numpy as np

from .frontiers import MapContext, is_still_frontier


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

    def __init__(self, cfg):
        self.cfg = cfg

    def choose(self, ctx: MapContext, rng: np.random.Generator, blacklist) -> Goal | None:
        raise NotImplementedError

    def still_valid(self, grid, goal: Goal) -> bool:
        return True


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
                                    device=dcfg.get('device', 'cpu'))
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


def make_strategy(name, cfg, **kw) -> Strategy:
    base, _, arg = str(name).partition(':')
    if base in ('DRL', 'drl') and arg:
        return DrlVelocity(cfg, agent=arg, **kw)
    if base.startswith('DRL_') and base[4:] in ((cfg.get('drl') or {}).get('agents') or {}):
        return DrlVelocity(cfg, agent=base[4:], **kw)   # nombre ya resuelto ('DRL_<alias>')
    try:
        cls = STRATEGIES[name]
    except KeyError:
        raise ValueError(f'Estrategia desconocida: {name}. Opciones: {sorted(STRATEGIES)} '
                         f'(o DRL:<alias|ruta.pt>)') from None
    return cls(cfg, **kw) if cls is DrlVelocity else cls(cfg)
