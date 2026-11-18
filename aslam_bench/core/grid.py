"""Rejillas de ocupación, E/S en formato map_server y utilidades geométricas.

Convenio (igual que nav_msgs/OccupancyGrid):
  * data[i, j] con i = fila (crece con y) y j = columna (crece con x).
  * origin = (x, y) de la esquina de la celda (0, 0); se asume yaw de origen 0.
  * valores: -1 desconocido, 0..100 probabilidad de ocupación.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass

import numpy as np
import yaml
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

UNKNOWN = -1
FREE_THRESH = 25
OCC_THRESH = 65


@dataclass
class OccGrid:
    data: np.ndarray            # int8 (H, W)
    resolution: float
    origin: tuple[float, float]

    @property
    def shape(self):
        return self.data.shape

    def world_to_cell(self, x, y):
        j = np.floor((np.asarray(x) - self.origin[0]) / self.resolution).astype(int)
        i = np.floor((np.asarray(y) - self.origin[1]) / self.resolution).astype(int)
        return i, j

    def cell_to_world(self, i, j):
        x = self.origin[0] + (np.asarray(j) + 0.5) * self.resolution
        y = self.origin[1] + (np.asarray(i) + 0.5) * self.resolution
        return x, y

    def in_bounds(self, i, j):
        h, w = self.shape
        return (i >= 0) & (i < h) & (j >= 0) & (j < w)

    def free_mask(self, free_thresh=FREE_THRESH):
        return (self.data >= 0) & (self.data < free_thresh)

    def occ_mask(self, occ_thresh=OCC_THRESH):
        return self.data >= occ_thresh

    def known_mask(self, free_thresh=FREE_THRESH, occ_thresh=OCC_THRESH):
        return self.free_mask(free_thresh) | self.occ_mask(occ_thresh)

    def unknown_mask(self, free_thresh=FREE_THRESH, occ_thresh=OCC_THRESH):
        return ~self.known_mask(free_thresh, occ_thresh)

    def lookup(self, x, y, default=UNKNOWN):
        """Valor de la rejilla en puntos (x, y); `default` fuera de límites."""
        i, j = self.world_to_cell(x, y)
        ok = self.in_bounds(i, j)
        out = np.full(np.shape(i), default, dtype=np.int16)
        out[ok] = self.data[i[ok], j[ok]]
        return out

    def extent(self):
        h, w = self.shape
        return (self.origin[0], self.origin[0] + w * self.resolution,
                self.origin[1], self.origin[1] + h * self.resolution)


# ----------------------------------------------------------------------------- E/S PGM + YAML

def read_pgm(path):
    with open(path, 'rb') as f:
        raw = f.read()
    # Cabecera: magic, ancho, alto, maxval (con posibles comentarios '#')
    tokens, pos = [], 0
    while len(tokens) < 4:
        while raw[pos:pos + 1].isspace():
            pos += 1
        if raw[pos:pos + 1] == b'#':
            while raw[pos:pos + 1] not in (b'\n', b''):
                pos += 1
            continue
        start = pos
        while not raw[pos:pos + 1].isspace():
            pos += 1
        tokens.append(raw[start:pos])
    magic, w, h, maxval = tokens[0], int(tokens[1]), int(tokens[2]), int(tokens[3])
    pos += 1
    if magic == b'P5':
        dtype = np.uint8 if maxval < 256 else np.dtype('>u2')
        img = np.frombuffer(raw, dtype=dtype, count=w * h, offset=pos).reshape(h, w)
    elif magic == b'P2':
        img = np.array(raw[pos:].split(), dtype=int)[: w * h].reshape(h, w)
    else:
        raise ValueError(f'PGM no soportado: {magic!r}')
    return img.astype(np.int32), maxval


def write_pgm(path, img):
    img = np.asarray(img, dtype=np.uint8)
    h, w = img.shape
    with open(path, 'wb') as f:
        f.write(f'P5\n{w} {h}\n255\n'.encode())
        f.write(img.tobytes())


def load_map_yaml(path) -> OccGrid:
    with open(path) as f:
        meta = yaml.safe_load(f)
    img_path = meta['image']
    if not os.path.isabs(img_path):
        img_path = os.path.join(os.path.dirname(os.path.abspath(path)), img_path)
    img, maxval = read_pgm(img_path)
    p = img / float(maxval)
    if not meta.get('negate', 0):
        p = 1.0 - p
    data = np.full(img.shape, UNKNOWN, dtype=np.int8)
    data[p > meta.get('occupied_thresh', 0.65)] = 100
    data[p < meta.get('free_thresh', 0.196)] = 0
    data = np.flipud(data).copy()          # fila 0 de la imagen = y máxima
    ox, oy = meta['origin'][0], meta['origin'][1]
    return OccGrid(data, float(meta['resolution']), (float(ox), float(oy)))


def save_map_yaml(grid: OccGrid, yaml_path):
    base = os.path.splitext(os.path.basename(yaml_path))[0]
    pgm = os.path.join(os.path.dirname(os.path.abspath(yaml_path)), base + '.pgm')
    img = np.full(grid.shape, 205, dtype=np.uint8)
    img[grid.free_mask()] = 254
    img[grid.occ_mask()] = 0
    write_pgm(pgm, np.flipud(img))
    meta = dict(image=base + '.pgm', mode='trinary', resolution=float(grid.resolution),
                origin=[float(grid.origin[0]), float(grid.origin[1]), 0.0],
                negate=0, occupied_thresh=0.65, free_thresh=0.196)
    with open(yaml_path, 'w') as f:
        yaml.safe_dump(meta, f, sort_keys=False)


# ----------------------------------------------------------------------------- morfología y grafos

def disk(radius_cells: int):
    r = int(max(0, radius_cells))
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return (x * x + y * y) <= r * r + 1e-9


def inflate(mask, radius_m, resolution):
    r = int(math.ceil(radius_m / resolution))
    if r <= 0 or not mask.any():
        return mask.copy()
    return ndimage.binary_dilation(mask, structure=disk(r))


_NEIGH = [(0, 1, 1.0), (1, 0, 1.0), (1, 1, math.sqrt(2)), (1, -1, math.sqrt(2))]


def grid_graph(passable, resolution):
    """Grafo 8-conexo (no dirigido) de las celdas transitables. Devuelve (csr, idx)."""
    h, w = passable.shape
    idx = -np.ones((h, w), dtype=np.int64)
    nodes = np.flatnonzero(passable)
    idx.flat[nodes] = np.arange(len(nodes))
    rows, cols, wts = [], [], []
    for di, dj, c in _NEIGH:
        i0, i1 = 0, h - di
        j0, j1 = max(0, -dj), w - max(0, dj)
        src = passable[i0:i1, j0:j1] & passable[i0 + di:i1 + di, j0 + dj:j1 + dj]
        rows.append(idx[i0:i1, j0:j1][src])
        cols.append(idx[i0 + di:i1 + di, j0 + dj:j1 + dj][src])
        wts.append(np.full(int(src.sum()), c * resolution))
    rows, cols, wts = (np.concatenate(a) for a in (rows, cols, wts))
    g = coo_matrix((wts, (rows, cols)), shape=(len(nodes), len(nodes))).tocsr()
    return g, idx


def geodesic(passable, start_ij, resolution, return_path_to=None):
    """Distancia geodésica (m) desde start_ij a todas las celdas transitables.

    Devuelve (dist, path) donde dist es (H, W) con inf en lo no alcanzable y
    path es la lista de celdas hasta `return_path_to` (o None).
    """
    passable = passable.copy()
    si, sj = start_ij
    h, w = passable.shape
    dist = np.full((h, w), np.inf)
    if not (0 <= si < h and 0 <= sj < w):
        return dist, None
    passable[si, sj] = True
    g, idx = grid_graph(passable, resolution)
    d, pred = dijkstra(g, directed=False, indices=idx[si, sj], return_predecessors=True)
    dist[passable] = d[idx[passable]]
    path = None
    if return_path_to is not None:
        gi, gj = return_path_to
        if 0 <= gi < h and 0 <= gj < w and np.isfinite(dist[gi, gj]):
            node, path = idx[gi, gj], []
            flat_nodes = np.flatnonzero(passable)
            while node >= 0:
                path.append(divmod(int(flat_nodes[node]), w))
                node = pred[node]
            path.reverse()
    return dist, path


class GeodesicTree:
    """Como `geodesic`, pero conserva el árbol de caminos mínimos para extraer varios caminos."""

    def __init__(self, passable, start_ij, resolution):
        passable = passable.copy()
        si, sj = start_ij
        h, w = passable.shape
        self.shape = (h, w)
        self.dist = np.full((h, w), np.inf)
        self.pred = None
        if not (0 <= si < h and 0 <= sj < w):
            return
        passable[si, sj] = True
        g, self.idx = grid_graph(passable, resolution)
        d, self.pred = dijkstra(g, directed=False, indices=self.idx[si, sj], return_predecessors=True)
        self.dist[passable] = d[self.idx[passable]]
        self.flat_nodes = np.flatnonzero(passable)

    def path(self, goal_ij):
        """Celdas (i, j) desde el inicio hasta goal_ij, o None si no es alcanzable."""
        gi, gj = goal_ij
        h, w = self.shape
        if self.pred is None or not (0 <= gi < h and 0 <= gj < w) or not np.isfinite(self.dist[gi, gj]):
            return None
        node, out = self.idx[gi, gj], []
        while node >= 0:
            out.append(divmod(int(self.flat_nodes[node]), w))
            node = self.pred[node]
        out.reverse()
        return out


# ----------------------------------------------------------------------------- transformaciones SE(2)

def se2_apply(pose, x, y):
    """Aplica la pose (x0, y0, yaw0) a puntos expresados en su marco local."""
    x0, y0, th = pose
    c, s = math.cos(th), math.sin(th)
    x, y = np.asarray(x, float), np.asarray(y, float)
    return x0 + c * x - s * y, y0 + s * x + c * y


def se2_inverse_apply(pose, x, y):
    """Expresa puntos del mundo en el marco local de `pose`."""
    x0, y0, th = pose
    c, s = math.cos(th), math.sin(th)
    dx, dy = np.asarray(x, float) - x0, np.asarray(y, float) - y0
    return c * dx + s * dy, -s * dx + c * dy


def wrap_angle(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi
