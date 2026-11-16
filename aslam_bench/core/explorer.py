"""Lógica de exploración común, independiente de ROS.

La usan tanto el nodo ROS 2 (`explorer_node`) como el simulador 2D, de modo que
el comportamiento de cada método es idéntico en ambos entornos.

Ciclo (llamar a `update` a `replan_period`):
  * Si el objetivo actual se alcanzó, falló o caducó -> se decide uno nuevo
    (los fallidos/caducados se vetan en `blacklist_radius`).
  * Si la estrategia dice que el objetivo ya no tiene sentido (p. ej. la frontera
    se ha observado) -> se decide uno nuevo.
  * Si no quedan fronteras alcanzables -> DONE (criterio común para todos los métodos).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .config import robot_clearance
from .frontiers import build_context
from .strategies import Goal, make_strategy

# Estados de navegación que el entorno (Nav2 o el simulador) comunica a la lógica
NAV_IDLE, NAV_ACTIVE, NAV_SUCCEEDED, NAV_FAILED = 'idle', 'active', 'succeeded', 'failed'


@dataclass
class Action:
    kind: str                       # 'none' | 'goal' | 'done'
    goal: Goal | None = None
    info: dict = field(default_factory=dict)


class ExplorerLogic:
    def __init__(self, strategy_name, cfg, seed=0):
        self.cfg = cfg
        self.strategy = make_strategy(strategy_name, cfg)
        self.rng = np.random.default_rng(seed)
        self.goal: Goal | None = None
        self.goal_t0 = 0.0
        self.goal_timeout = 0.0
        self.blacklist: list[tuple] = []
        self.n_goals = 0
        self.n_failed = 0
        self.n_timeouts = 0
        self.n_replans = 0
        self.done = False
        self.last_frontiers = []        # fronteras abiertas de la última decisión (para visualizar)

    def _timeout_for(self, goal: Goal):
        ex = self.cfg['exploration']
        v = self.cfg['platform']['base']['max_lin_vel']
        return max(ex['goal_timeout_min'], ex['goal_timeout_factor'] * goal.distance / v)

    def update(self, t, grid, robot_xy, nav_status) -> Action:
        if self.done:
            return Action('none')
        reason = None
        if self.goal is not None:
            if nav_status == NAV_FAILED:
                self.blacklist.append((self.goal.x, self.goal.y))
                self.n_failed += 1
                reason = 'failed'
            elif nav_status == NAV_SUCCEEDED:
                reason = 'reached'
                # Llegó y la frontera sigue ahí -> no es observable (interior de un mueble,
                # artefacto del mapeo...): se veta para no volver en bucle.
                if self.strategy.still_valid(grid, self.goal) and self.strategy.uses_frontier_validity:
                    self.blacklist.append((self.goal.x, self.goal.y))
                    reason = 'reached_unobservable'
            elif nav_status == NAV_IDLE:
                reason = 'idle'
            elif t - self.goal_t0 > self.goal_timeout:
                self.blacklist.append((self.goal.x, self.goal.y))
                self.n_timeouts += 1
                reason = 'timeout'
            elif not self.strategy.still_valid(grid, self.goal):
                self.n_replans += 1
                reason = 'invalidated'
            else:
                return Action('none')
            self.goal = None

        t0 = time.perf_counter()
        ex = self.cfg['exploration']
        ctx = build_context(grid, robot_xy, robot_clearance(self.cfg), ex['min_frontier_size'],
                            ex['min_unknown_cells'])
        rad = ex['blacklist_radius']
        tol = self.cfg['navigation']['goal_tolerance']
        # fronteras abiertas: alcanzables, no vetadas y no 'ya alcanzadas' (dentro de la tolerancia)
        ctx.frontiers = [f for f in ctx.frontiers if np.isfinite(f.distance) and f.distance > tol and not any(
            (f.goal[0] - bx) ** 2 + (f.goal[1] - by) ** 2 < rad ** 2 for bx, by in self.blacklist)]
        open_frontiers = ctx.frontiers
        self.last_frontiers = open_frontiers
        goal = self.strategy.choose(ctx, self.rng, self.blacklist) if open_frontiers else None
        dt = time.perf_counter() - t0
        info = dict(decision_time=dt, n_frontiers=len(open_frontiers), prev_goal_end=reason)
        if goal is None:
            self.done = True
            return Action('done', info=info)
        self.goal, self.goal_t0 = goal, t
        self.goal_timeout = self._timeout_for(goal)
        self.n_goals += 1
        return Action('goal', goal, info)

    def stats(self):
        return dict(n_goals=self.n_goals, n_goals_failed=self.n_failed,
                    n_goals_timeout=self.n_timeouts, n_goals_invalidated=self.n_replans)
