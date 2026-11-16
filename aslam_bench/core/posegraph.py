"""Grafo de poses sustituto para los criterios de incertidumbre (B4, B5, B8).

El simulador 2D no tiene SLAM y el Graph SLAM del robot aún no publica su grafo, así
que las estrategias que razonan sobre la incertidumbre de pose mantienen su propio
grafo a partir de la trayectoria estimada (la misma en sim2d y en ROS):

  * nodos  : keyframes cada `kf_spacing` m de recorrido;
  * aristas: odometría entre keyframes consecutivos, con información
             w = 1 / (sigma_odom² · d)   (el ruido crece con la distancia), y
             cierres de bucle cuando el robot vuelve a menos de `lc_radius` de un
             keyframe con al menos `lc_min_sep` nodos de separación y línea de visión
             libre en el mapa, con w = 1 / sigma_lc². Tras un cierre no se acepta otro
             hasta `lc_cooldown` keyframes después (al recorrer un tramo ya conocido, el
             SLAM no añade un cierre por keyframe; si no, volver sobre los propios pasos
             parecería muy informativo).

Con información escalar e isótropa en (x, y), la matriz de información del grafo es
L_w ⊗ I₂ (L_w = Laplaciano ponderado, anclado en el nodo 0). De ahí:
  * D-optimalidad: log det(L̃_w) = log del nº ponderado de árboles generadores
    (Placed & Castellanos, RA-L 2022; Khosoussi et al., IJRR 2019);
  * varianza marginal de un nodo = resistencia efectiva entre el ancla y el nodo
    (por eje), que da la entropía de la pose actual (Stachniss et al., RSS 2005).

`predict(path)` simula recorrer un camino (añade keyframes y cierres de bucle) sin
modificar el grafo, y devuelve ambos criterios tras el recorrido.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu

from .infogain import line_free


class PoseGraph:
    def __init__(self, kf_spacing=0.5, sigma_odom=0.1, sigma_lc=0.1, lc_radius=1.0, lc_min_sep=20,
                 lc_cooldown=10):
        self.kf_spacing = float(kf_spacing)
        self.sigma_odom = float(sigma_odom)
        self.sigma_lc = float(sigma_lc)
        self.lc_radius = float(lc_radius)
        self.lc_min_sep = int(lc_min_sep)
        self.lc_cooldown = int(lc_cooldown)
        self._last_lc = -10 ** 9           # índice del último keyframe con cierre de bucle
        self.nodes: list[tuple] = []
        self.edges: list[tuple] = []          # (i, j, w, kind)
        self._acc = 0.0
        self._last_xy = None
        self._cache = None

    @classmethod
    def from_cfg(cls, cfg):
        p = dict(cfg.get('pose_graph') or {})
        return cls(**p)

    # ------------------------------------------------------------------ construcción
    def w_odom(self, d):
        return 1.0 / (self.sigma_odom ** 2 * max(d, 1e-3))

    @property
    def w_lc(self):
        return 1.0 / self.sigma_lc ** 2

    def n_loop_closures(self):
        return sum(1 for e in self.edges if e[3] == 'lc')

    def _lc_targets(self, nodes, k, xy, grid, last_lc):
        """Keyframe más cercano (separado al menos lc_min_sep) con el que cerrar bucle."""
        if k - self.lc_min_sep <= 0 or k - last_lc < self.lc_cooldown:
            return None
        old = np.asarray(nodes[:k - self.lc_min_sep + 1])
        d = np.hypot(old[:, 0] - xy[0], old[:, 1] - xy[1])
        m = int(np.argmin(d))
        if d[m] > self.lc_radius:
            return None
        if grid is not None and not line_free(grid, xy, tuple(old[m])):
            return None
        return m

    def update(self, xy, grid=None):
        """Añade la pose actual si se ha recorrido kf_spacing. Devuelve True si hay nodo nuevo."""
        xy = (float(xy[0]), float(xy[1]))
        if self._last_xy is None:
            self.nodes.append(xy)
            self._last_xy = xy
            self._cache = None
            return True
        self._acc += math.hypot(xy[0] - self._last_xy[0], xy[1] - self._last_xy[1])
        self._last_xy = xy
        if self._acc < self.kf_spacing:
            return False
        k = len(self.nodes)
        self.nodes.append(xy)
        self.edges.append((k - 1, k, self.w_odom(self._acc), 'odom'))
        self._acc = 0.0
        m = self._lc_targets(self.nodes, k, xy, grid, self._last_lc)
        if m is not None:
            self.edges.append((m, k, self.w_lc, 'lc'))
            self._last_lc = k
        self._cache = None
        return True

    # ------------------------------------------------------------------ álgebra
    @staticmethod
    def _reduced_laplacian(n, edges):
        if n <= 1:
            return None
        rows, cols, vals = [], [], []
        for i, j, w, _ in edges:
            rows += [i, j, i, j]
            cols += [i, j, j, i]
            vals += [w, w, -w, -w]
        L = coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsc()
        return L[1:, 1:].tocsc()            # anclado en el nodo 0 (marco 'map')

    @classmethod
    def _analyse(cls, n, edges, query=None):
        """(logdet(L̃), varianza marginal del último nodo, o de `query`)."""
        Lr = cls._reduced_laplacian(n, edges)
        if Lr is None:
            return 0.0, 0.0
        lu = splu(Lr)
        logdet = float(np.sum(np.log(np.abs(lu.U.diagonal()))))
        q = (n - 1) if query is None else query
        if q == 0:
            return logdet, 0.0
        e = np.zeros(n - 1)
        e[q - 1] = 1.0
        var = float(lu.solve(e)[q - 1])
        return logdet, var

    def state(self):
        """(n, logdet, var_actual) del grafo actual (en caché hasta el siguiente nodo)."""
        if self._cache is None:
            ld, var = self._analyse(len(self.nodes), self.edges)
            # varianza de la pose actual = la del último keyframe + el recorrido desde él
            self._cache = (len(self.nodes), ld, var)
        n, ld, var = self._cache
        return n, ld, var + (1.0 / self.w_odom(self._acc) if self._acc > 0 else 0.0)

    def predict(self, path_xy, grid=None):
        """Recorre `path_xy` (lista de (x, y) desde la pose actual) sin modificar el grafo.

        Devuelve dict(n, logdet, var, n_lc, dopt_gain):
          * var       : varianza marginal (m², por eje) de la pose al final del camino;
          * dopt_gain : aumento de log t_w(G) (D-optimalidad del grafo, nats) descontando el
                        que se debe solo a añadir nodos: Δlogdet − Σ ln w(aristas de odometría
                        nuevas). Añadir un nodo colgante multiplica t_w(G) por el peso de su
                        arista, así que un camino sin cierres da exactamente 0 y cada cierre
                        de bucle suma ln(1 + w_lc·R_eff) (R_eff: resistencia efectiva entre
                        sus extremos). Es la forma de comparar grafos de distinto tamaño que
                        usamos en lugar de la normalización por nodo de Placed & Castellanos,
                        que con keyframes de longitud desigual premia o castiga recorridos
                        sin cierres según su discretización.
        """
        nodes = list(self.nodes)
        edges = list(self.edges)
        n0, ld0, var0 = self.state()
        if not nodes:
            return dict(n=0, logdet=0.0, var=0.0, n_lc=0, dopt_gain=0.0)
        acc, last = self._acc, (self._last_xy or nodes[-1])
        n_lc, last_lc, tree_ld = 0, self._last_lc, 0.0
        for xy in path_xy:
            acc += math.hypot(xy[0] - last[0], xy[1] - last[1])
            last = xy
            if acc < self.kf_spacing:
                continue
            k = len(nodes)
            nodes.append((float(xy[0]), float(xy[1])))
            edges.append((k - 1, k, self.w_odom(acc), 'odom'))
            tree_ld += math.log(self.w_odom(acc))
            acc = 0.0
            m = self._lc_targets(nodes, k, xy, grid, last_lc)
            if m is not None:
                edges.append((m, k, self.w_lc, 'lc'))
                n_lc += 1
                last_lc = k
        n1 = len(nodes)
        if n1 == n0:
            self.state()
            ld1, var1 = ld0, self._cache[2]
        else:
            ld1, var1 = self._analyse(n1, edges)
        var1 += 1.0 / self.w_odom(acc) if acc > 0 else 0.0
        gain = ld1 - ld0 - tree_ld if n0 > 1 else 0.0
        return dict(n=n1, logdet=ld1, var=max(var1, 1e-9), n_lc=n_lc, dopt_gain=gain)
