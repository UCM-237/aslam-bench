"""Estrategias de selección de objetivos (la única pieza que cambia entre métodos).

Todas las estrategias implementan `choose(ctx, rng, blacklist) -> Goal | None` y
`still_valid(grid, goal) -> bool`. Añadir un método nuevo (B2..B5 o el DRL) es
añadir una clase aquí y registrarla en STRATEGIES.
"""
from __future__ import annotations

import math
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


STRATEGIES = {
    'B0': RandomGoal, 'random': RandomGoal, RandomGoal.name: RandomGoal,
    'B1': NearestFrontier, 'nearest_frontier': NearestFrontier, NearestFrontier.name: NearestFrontier,
}


def make_strategy(name, cfg) -> Strategy:
    try:
        return STRATEGIES[name](cfg)
    except KeyError:
        raise ValueError(f'Estrategia desconocida: {name}. Opciones: {sorted(STRATEGIES)}') from None
