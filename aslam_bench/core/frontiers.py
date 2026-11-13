"""Detección de fronteras (Yamauchi, 1997) sobre una OccupancyGrid.

Una celda frontera es una celda libre conocida con al menos un vecino (8-conexo)
desconocido. Las celdas frontera se agrupan en clusters 8-conexos y, para cada
cluster, se busca la celda transitable alcanzable más cercana (distancia
geodésica desde el robot) dentro de un radio igual al de seguridad del robot:
esa celda es el objetivo de navegación del cluster.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

from .grid import OccGrid, geodesic, inflate

_EIGHT = np.ones((3, 3), dtype=bool)


@dataclass
class Frontier:
    cells: np.ndarray            # (N, 2) índices (i, j) de las celdas frontera
    size_m: float                # longitud aproximada (n_celdas * resolución)
    centroid: tuple              # (x, y) en el marco del mapa
    goal: tuple                  # (x, y) objetivo transitable asociado
    goal_cell: tuple             # (i, j)
    distance: float              # distancia geodésica robot -> objetivo (m)
    target: tuple = None         # (x, y) celda frontera más próxima al objetivo


@dataclass
class MapContext:
    """Todo lo que una estrategia necesita para decidir (se calcula una vez por decisión)."""
    grid: OccGrid
    robot_xy: tuple
    robot_cell: tuple
    passable: np.ndarray
    dist: np.ndarray
    frontier_mask: np.ndarray
    frontiers: list = field(default_factory=list)


def frontier_mask(grid: OccGrid, min_unknown_cells: int = 0):
    """Celdas libres con algún vecino desconocido.

    Los 'agujeros' desconocidos pequeños (< min_unknown_cells), típicos de la
    separación angular entre haces a larga distancia, y las franjas desconocidas
    estrechas (interior de paredes) no generan fronteras.
    """
    free = grid.free_mask()
    unknown = grid.unknown_mask()
    # Apertura morfológica 3x3: elimina franjas desconocidas de 1-2 celdas de ancho
    # (interior de paredes y muebles, nunca observable) que generarían fronteras fantasma.
    unknown = ndimage.binary_opening(unknown, structure=_EIGHT)
    if min_unknown_cells > 1 and unknown.any():
        lab, n = ndimage.label(unknown, structure=_EIGHT)
        sizes = np.bincount(lab.ravel())
        keep = sizes >= min_unknown_cells
        keep[0] = False
        unknown = keep[lab]
    return free & ndimage.binary_dilation(unknown, structure=_EIGHT)


def passable_mask(grid: OccGrid, clearance_m: float, robot_cell=None):
    """Celdas libres a más de `clearance_m` de un obstáculo.

    Si el robot está dentro de la zona inflada (pegado a una pared), se permiten
    también las celdas libres de su entorno inmediato para que pueda salir,
    como hacen los costmaps de Nav2.
    """
    passable = grid.free_mask() & ~inflate(grid.occ_mask(), clearance_m, grid.resolution)
    if robot_cell is not None:
        free = grid.free_mask()
        r = int(math.ceil(clearance_m / grid.resolution)) + 1
        i, j = robot_cell
        h, w = passable.shape
        i0, i1, j0, j1 = max(0, i - r), min(h, i + r + 1), max(0, j - r), min(w, j + r + 1)
        if i0 < i1 and j0 < j1:
            passable[i0:i1, j0:j1] |= free[i0:i1, j0:j1]
    return passable


def build_context(grid: OccGrid, robot_xy, clearance_m, min_frontier_size_m,
                  min_unknown_cells=10) -> MapContext:
    res = grid.resolution
    ri, rj = (int(v) for v in grid.world_to_cell(robot_xy[0], robot_xy[1]))
    passable = passable_mask(grid, clearance_m, (ri, rj))
    dist, _ = geodesic(passable, (ri, rj), res)
    fm = frontier_mask(grid, min_unknown_cells)
    ctx = MapContext(grid, tuple(robot_xy), (ri, rj), passable, dist, fm)

    labels, n = ndimage.label(fm, structure=_EIGHT)
    if n == 0:
        return ctx
    min_cells = max(1, int(math.ceil(min_frontier_size_m / res)))
    reach_r = max(1, int(math.ceil(clearance_m / res)))
    objects = ndimage.find_objects(labels)
    h, w = fm.shape
    for k, sl in enumerate(objects, start=1):
        if sl is None:
            continue
        local = labels[sl] == k
        ncells = int(local.sum())
        if ncells < min_cells:
            continue
        # ventana ampliada para buscar celdas transitables cerca del cluster
        i0, i1 = max(0, sl[0].start - reach_r), min(h, sl[0].stop + reach_r)
        j0, j1 = max(0, sl[1].start - reach_r), min(w, sl[1].stop + reach_r)
        mask = np.zeros((i1 - i0, j1 - j0), dtype=bool)
        mask[sl[0].start - i0:sl[0].stop - i0, sl[1].start - j0:sl[1].stop - j0] = local
        near = ndimage.binary_dilation(mask, iterations=reach_r, structure=_EIGHT)
        d = np.where(near, dist[i0:i1, j0:j1], np.inf)
        if not np.isfinite(d).any():
            continue
        gi, gj = np.unravel_index(int(np.argmin(d)), d.shape)
        gi, gj = gi + i0, gj + j0
        ci, cj = np.nonzero(local)
        ci, cj = ci + sl[0].start, cj + sl[1].start
        cx, cy = grid.cell_to_world(ci, cj)
        gx, gy = grid.cell_to_world(gi, gj)
        kt = int(np.argmin((ci - gi) ** 2 + (cj - gj) ** 2))
        ctx.frontiers.append(Frontier(
            cells=np.stack([ci, cj], axis=1), size_m=ncells * res,
            centroid=(float(cx.mean()), float(cy.mean())), goal=(float(gx), float(gy)),
            goal_cell=(int(gi), int(gj)), distance=float(dist[gi, gj]),
            target=(float(cx[kt]), float(cy[kt]))))
    return ctx


def is_still_frontier(grid: OccGrid, xy, radius_m, min_unknown_cells=10):
    """¿Queda alguna celda frontera a menos de radius_m de xy?

    La máscara de fronteras se calcula sobre el mapa completo: recortar antes
    partiría las regiones desconocidas grandes y el filtro de 'agujeros' las
    descartaría por error.
    """
    res = grid.resolution
    i, j = (int(v) for v in grid.world_to_cell(xy[0], xy[1]))
    r = max(1, int(math.ceil(radius_m / res)))
    h, w = grid.shape
    i0, i1, j0, j1 = max(0, i - r), min(h, i + r + 1), max(0, j - r), min(w, j + r + 1)
    if i0 >= i1 or j0 >= j1:
        return False
    fm = frontier_mask(grid, min_unknown_cells)
    return bool(fm[i0:i1, j0:j1].any())
