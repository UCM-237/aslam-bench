"""Genera los mapas de referencia (ground truth) en formato map_server (PGM + YAML).

Los mismos mapas se usan en el simulador 2D y, convertidos a mundos SDF con
`map_to_world`, en Gazebo, de modo que el ground truth es exacto por construcción.

Uso:  python -m aslam_bench.tools.make_maps --out maps/
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from ..core.grid import OccGrid, UNKNOWN, save_map_yaml

RES = 0.05
WALL = 0.15


class Canvas:
    def __init__(self, w, h, res=RES):
        self.res = res
        self.data = np.full((int(round(h / res)), int(round(w / res))), UNKNOWN, np.int8)

    def _sl(self, x0, y0, x1, y1):
        r = self.res
        i0, i1 = int(round(min(y0, y1) / r)), int(round(max(y0, y1) / r))
        j0, j1 = int(round(min(x0, x1) / r)), int(round(max(x0, x1) / r))
        return slice(max(i0, 0), i1), slice(max(j0, 0), j1)

    def free(self, x0, y0, x1, y1):
        self.data[self._sl(x0, y0, x1, y1)] = 0

    def box(self, x0, y0, x1, y1):
        self.data[self._sl(x0, y0, x1, y1)] = 100

    def hwall(self, y, x0, x1, t=WALL):
        self.box(x0, y - t / 2, x1, y + t / 2)

    def vwall(self, x, y0, y1, t=WALL):
        self.box(x - t / 2, y0, x + t / 2, y1)

    def door_h(self, y, xc, width=0.9, t=WALL):
        self.free(xc - width / 2, y - t / 2 - 0.01, xc + width / 2, y + t / 2 + 0.01)

    def door_v(self, x, yc, width=0.9, t=WALL):
        self.free(x - t / 2 - 0.01, yc - width / 2, x + t / 2 + 0.01, yc + width / 2)

    def outer(self, x0, y0, x1, y1, t=WALL):
        self.free(x0, y0, x1, y1)
        self.hwall(y0, x0 - t / 2, x1 + t / 2)
        self.hwall(y1, x0 - t / 2, x1 + t / 2)
        self.vwall(x0, y0 - t / 2, y1 + t / 2)
        self.vwall(x1, y0 - t / 2, y1 + t / 2)

    def grid(self):
        return OccGrid(self.data.copy(), self.res, (0.0, 0.0))


def rooms_small():
    """Vivienda pequeña (~12 x 9 m): pasillo central y cinco habitaciones con muebles."""
    c = Canvas(13, 10)
    c.outer(0.5, 0.5, 12.5, 9.5)
    c.hwall(4.0, 0.5, 12.5)          # fila inferior de habitaciones
    c.hwall(6.0, 0.5, 12.5)          # fila superior; pasillo 4..6
    c.vwall(6.5, 0.5, 4.0)
    c.vwall(5.0, 6.0, 9.5)
    c.vwall(9.0, 6.0, 9.5)
    for xc in (3.0, 9.5):
        c.door_h(4.0, xc)
    for xc in (2.5, 7.0, 11.0):
        c.door_h(6.0, xc)
    # mobiliario
    c.box(1.2, 1.2, 2.6, 2.0)
    c.box(8.0, 1.0, 9.0, 2.5)
    c.box(10.8, 7.8, 12.0, 9.0)
    c.box(6.0, 8.5, 7.5, 9.2)
    return c.grid()


def office_corridor():
    """Oficinas (~22 x 10 m): pasillo largo de 2 m con cuatro despachos a cada lado."""
    c = Canvas(23, 11)
    c.outer(0.5, 0.5, 22.5, 10.5)
    c.hwall(4.5, 0.5, 22.5)
    c.hwall(6.5, 0.5, 22.5)
    for x in (6.0, 11.5, 17.0):
        c.vwall(x, 0.5, 4.5)
        c.vwall(x, 6.5, 10.5)
    for xc in (3.0, 9.0, 14.0, 20.0):
        c.door_h(4.5, xc)
        c.door_h(6.5, xc + 0.8)
    # mesas en algunos despachos
    for (x0, y0) in ((1.5, 1.5), (12.5, 2.0), (7.0, 8.0), (18.5, 8.2)):
        c.box(x0, y0, x0 + 1.6, y0 + 0.8)
    return c.grid()


def cluttered_hall():
    """Nave diáfana (~14 x 12 m) con columnas, cajas y tabiques parciales."""
    c = Canvas(15, 13)
    c.outer(0.5, 0.5, 14.5, 12.5)
    rng = np.random.default_rng(7)
    for x in (4.0, 8.0, 11.5):
        for y in (4.0, 8.5):
            c.box(x - 0.2, y - 0.2, x + 0.2, y + 0.2)
    c.vwall(6.0, 9.5, 12.5)
    c.hwall(6.5, 9.5, 14.5)
    c.vwall(2.5, 0.5, 3.0)
    placed = 0
    while placed < 10:
        x, y = rng.uniform(1.5, 13.5), rng.uniform(1.5, 11.5)
        w, h = rng.uniform(0.4, 1.2), rng.uniform(0.4, 1.0)
        if 1.0 < x < 2.5 and 1.0 < y < 2.5:   # deja libre la esquina de salida habitual
            continue
        c.box(x, y, x + w, y + h)
        placed += 1
    return c.grid()


MAPS = dict(rooms_small=rooms_small, office_corridor=office_corridor, cluttered_hall=cluttered_hall)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', default='maps')
    ap.add_argument('--only', nargs='*')
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    for name, fn in MAPS.items():
        if a.only and name not in a.only:
            continue
        g = fn()
        save_map_yaml(g, os.path.join(a.out, name + '.yaml'))
        free = (g.data == 0).sum() * g.resolution ** 2
        print(f'{name}: {g.shape[1]}x{g.shape[0]} celdas, {free:.1f} m² libres')


if __name__ == '__main__':
    main()
