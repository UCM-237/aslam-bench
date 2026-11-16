"""Ganancia de información geométrica de un punto de vista sobre la rejilla de ocupación.

Lo comparten B2–B8 para que todas las estrategias midan la IG igual:
  * `visible_unknown`  : área (m²) de celdas desconocidas visibles desde (x, y) dentro de
                         `radius`, por trazado de rayos; lo desconocido es transparente y lo
                         ocupado opaco. Con `yaw` y `fov` se limita a un sector (cámara).
  * `visible_entropy`  : igual, pero suma la entropía binaria (bits) de cada celda visible:
                         desconocida = 1 bit, libre/ocupada segura ≈ 0 (B4).
  * `unknown_in_disk`  : área desconocida en un disco, sin visibilidad (RRT-exploration, B3).
"""
from __future__ import annotations

import math

import numpy as np

from .grid import OCC_THRESH, OccGrid


def _rays(grid: OccGrid, x, y, radius, n_rays, yaw=None, fov=None):
    if fov is not None and yaw is not None and fov < 2 * math.pi - 1e-6:
        n = max(3, int(round(n_rays * fov / (2 * math.pi))))
        a = yaw + np.linspace(-fov / 2, fov / 2, n)
    else:
        a = np.linspace(-math.pi, math.pi, n_rays, endpoint=False)
    s = np.arange(grid.resolution * 0.5, radius, grid.resolution * 0.7)
    px = x + np.cos(a)[:, None] * s[None, :]
    py = y + np.sin(a)[:, None] * s[None, :]
    i, j = grid.world_to_cell(px, py)
    ok = grid.in_bounds(i, j)
    val = np.full(i.shape, 100, np.int16)          # fuera del mapa = opaco
    val[ok] = grid.data[i[ok], j[ok]]
    blocked = val >= OCC_THRESH
    # visible hasta el primer obstáculo (incluido: el obstáculo también se 've')
    first = np.where(blocked.any(axis=1), blocked.argmax(axis=1), blocked.shape[1])
    vis = np.arange(blocked.shape[1])[None, :] <= first[:, None]
    vis &= ok
    h, w = grid.shape
    lin = np.where(vis, i * w + j, -1)
    return lin, val, vis


def visible_unknown(grid: OccGrid, x, y, radius=3.0, n_rays=90, yaw=None, fov=None):
    """Área desconocida (m²) visible desde (x, y)."""
    lin, val, vis = _rays(grid, x, y, radius, n_rays, yaw, fov)
    cells = np.unique(lin[vis & (val < 0)])
    return float(len(cells)) * grid.resolution ** 2


def _h(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def visible_entropy(grid: OccGrid, x, y, radius=3.0, n_rays=90):
    """Entropía (bits·m²) de las celdas visibles desde (x, y): desconocida = 1 bit/celda."""
    lin, val, vis = _rays(grid, x, y, radius, n_rays)
    lin_v = lin[vis]
    cells, first = np.unique(lin_v, return_index=True)
    v = val[vis][first].astype(float)
    ent = np.where(v < 0, 1.0, _h(v / 100.0))
    return float(ent.sum()) * grid.resolution ** 2


def unknown_in_disk(grid: OccGrid, x, y, radius):
    """Área desconocida (m²) dentro de un disco (sin comprobar visibilidad)."""
    i, j = (int(v) for v in grid.world_to_cell(x, y))
    r = max(1, int(math.ceil(radius / grid.resolution)))
    h, w = grid.shape
    i0, i1, j0, j1 = max(0, i - r), min(h, i + r + 1), max(0, j - r), min(w, j + r + 1)
    if i0 >= i1 or j0 >= j1:
        return 0.0
    ii, jj = np.ogrid[i0:i1, j0:j1]
    disk = (ii - i) ** 2 + (jj - j) ** 2 <= r * r
    return float(((grid.data[i0:i1, j0:j1] < 0) & disk).sum()) * grid.resolution ** 2


def map_entropy(grid: OccGrid):
    """Entropía total del mapa (bits): útil como métrica y para depurar B4."""
    v = grid.data.astype(float)
    return float(np.where(v < 0, 1.0, _h(v / 100.0)).sum())


def line_free(grid: OccGrid, p0, p1, mask=None):
    """¿El segmento p0-p1 no atraviesa celdas ocupadas (o, con `mask`, sale de mask)?"""
    d = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
    n = max(2, int(math.ceil(d / (grid.resolution * 0.5))) + 1)
    xs = np.linspace(p0[0], p1[0], n)
    ys = np.linspace(p0[1], p1[1], n)
    i, j = grid.world_to_cell(xs, ys)
    if not grid.in_bounds(i, j).all():
        return False
    if mask is not None:
        return bool(mask[i, j].all())
    return bool((grid.data[i, j] < OCC_THRESH).all())
