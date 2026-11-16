"""Capa semántica mínima para B8 (diseño en semantica/diseno_capa_semantica.md, §3 y §7).

  * `SemanticMap`  : landmarks de objeto en el marco 'map' con la clase como distribución
                     de Dirichlet (α_c += score), nº de observaciones, poses desde las que
                     se han visto y estado 'tentative' | 'confirmed' (§3.1). La entropía de
                     clase H(p) es la incertidumbre semántica que usa IG_sem.
  * `ig_sem`       : Σ H(p) de los landmarks que entrarían en el FOV de la cámara desde
                     (x, y, yaw), con un extra por cada landmark 'tentative' (§7).
  * `SemanticSim`  : solo sim2d. Objetos sintéticos junto a las paredes del mapa real y
                     una cámara simulada (FOV, alcance, línea de visión, detección y
                     confusión de clase). La asociación de datos es ideal (por id): aquí se
                     evalúa la decisión, no el front-end semántico.

En ROS el mapa semántico llega por un tópico (JSON en std_msgs/String, ver
`SemanticMap.from_json`), hasta que exista el mensaje propio del diseño.
"""
from __future__ import annotations

import json
import math

import numpy as np
from scipy import ndimage

from .grid import OccGrid, se2_apply, se2_inverse_apply, wrap_angle
from .infogain import line_free

DEFAULT_CLASSES = ['chair', 'couch', 'bed', 'dining table', 'toilet', 'sink', 'refrigerator',
                   'oven', 'tv', 'potted plant']


class Landmark:
    __slots__ = ('id', 'xy', 'alpha', 'n_obs', 'views', 'state')

    def __init__(self, lid, xy, n_classes, prior=0.5):
        self.id = lid
        self.xy = np.asarray(xy, float)
        self.alpha = np.full(n_classes, prior)
        self.n_obs = 0
        self.views: list[tuple] = []        # poses (x, y) desde las que se ha observado
        self.state = 'tentative'

    @property
    def probs(self):
        return self.alpha / self.alpha.sum()

    @property
    def entropy(self):
        p = self.probs
        return float(-(p * np.log2(np.clip(p, 1e-12, 1))).sum())

    def to_dict(self, classes):
        p = self.probs
        return dict(id=int(self.id), x=float(self.xy[0]), y=float(self.xy[1]), state=self.state,
                    n_obs=int(self.n_obs), entropy=self.entropy, label=classes[int(np.argmax(p))],
                    probs=[float(v) for v in p])


class SemanticMap:
    def __init__(self, classes=None, confirm_obs=3, confirm_views=2, view_sep=0.5, prior=0.5):
        self.classes = list(classes or DEFAULT_CLASSES)
        self.confirm_obs, self.confirm_views = int(confirm_obs), int(confirm_views)
        self.view_sep, self.prior = float(view_sep), float(prior)
        self.landmarks: dict[int, Landmark] = {}

    @staticmethod
    def from_cfg(cfg):
        s = cfg.get('semantics') or {}
        return SemanticMap(s.get('classes'), s.get('confirm_obs', 3), s.get('confirm_views', 2),
                           s.get('view_sep', 0.5), s.get('prior', 0.5))

    def integrate(self, obs, robot_xy):
        """obs: lista de dict(id, x, y, cls, score) en el marco 'map'."""
        for o in obs:
            lm = self.landmarks.get(o['id'])
            if lm is None:
                lm = self.landmarks[o['id']] = Landmark(o['id'], (o['x'], o['y']), len(self.classes),
                                                        self.prior)
            # posición: media incremental de las observaciones
            lm.xy = (lm.xy * lm.n_obs + np.array([o['x'], o['y']])) / (lm.n_obs + 1)
            lm.n_obs += 1
            lm.alpha[o['cls']] += o['score']
            if all(math.hypot(robot_xy[0] - v[0], robot_xy[1] - v[1]) > self.view_sep for v in lm.views):
                lm.views.append((float(robot_xy[0]), float(robot_xy[1])))
            if lm.n_obs >= self.confirm_obs and len(lm.views) >= self.confirm_views:
                lm.state = 'confirmed'

    def set_from_list(self, items):
        """Sustituye el mapa por una lista de dicts (ROS): id, x, y, probs|alpha, state, n_obs."""
        self.landmarks = {}
        for d in items:
            lm = Landmark(int(d['id']), (d['x'], d['y']), len(self.classes), self.prior)
            if 'alpha' in d:
                lm.alpha = np.asarray(d['alpha'], float)
            elif 'probs' in d:
                lm.alpha = np.asarray(d['probs'], float) * max(1.0, float(d.get('n_obs', 1)))
            lm.n_obs = int(d.get('n_obs', 1))
            lm.state = d.get('state', 'tentative')
            self.landmarks[lm.id] = lm

    def from_json(self, text):
        d = json.loads(text)
        if isinstance(d, dict):
            if d.get('classes'):
                self.classes = list(d['classes'])
            d = d.get('landmarks', [])
        self.set_from_list(d)

    def summary(self):
        lms = list(self.landmarks.values())
        return dict(n_landmarks=len(lms),
                    n_confirmed=sum(1 for lm in lms if lm.state == 'confirmed'),
                    mean_class_entropy=float(np.mean([lm.entropy for lm in lms])) if lms else float('nan'))


def in_fov(px, py, x, y, yaw, hfov, rng):
    dx, dy = px - x, py - y
    d = math.hypot(dx, dy)
    if d > rng or d < 0.2:
        return False
    return abs(float(wrap_angle(math.atan2(dy, dx) - yaw))) <= hfov / 2


def ig_sem(smap: SemanticMap, grid: OccGrid, x, y, yaw, hfov, cam_range, tentative_bonus=0.5):
    """IG_sem desde (x, y, yaw): Σ H(p) (bits) + bonus por landmark 'tentative' en el FOV."""
    g = 0.0
    for lm in smap.landmarks.values():
        if in_fov(lm.xy[0], lm.xy[1], x, y, yaw, hfov, cam_range) and line_free(grid, (x, y), tuple(lm.xy)):
            g += lm.entropy + (tentative_bonus if lm.state == 'tentative' else 0.0)
    return g


def confirmed_in_view(smap: SemanticMap, grid, x, y, yaw, hfov, cam_range):
    return sum(1 for lm in smap.landmarks.values()
               if lm.state == 'confirmed' and in_fov(lm.xy[0], lm.xy[1], x, y, yaw, hfov, cam_range)
               and line_free(grid, (x, y), tuple(lm.xy)))


# ----------------------------------------------------------------------------- simulación (sim2d)

class SemanticSim:
    """Objetos sintéticos y cámara monocular simulada para el simulador 2D."""

    def __init__(self, gt: OccGrid, start_pose, cfg, seed=0):
        s = cfg.get('semantics') or {}
        cam = cfg['platform'].get('camera') or {}
        self.classes = list(s.get('classes') or DEFAULT_CLASSES)
        self.hfov = math.radians(float(cam.get('hfov_deg', 69)))
        self.range = float(s.get('cam_range', 4.0))
        self.p_det = float(s.get('p_detect', 0.8))
        self.p_correct = float(s.get('p_correct', 0.7))
        self.sigma = float(s.get('sigma_pos', 0.05))
        self.period = float(s.get('cam_period', 0.5))
        self.gt, self.start = gt, tuple(start_pose)
        rng = np.random.default_rng(int(s.get('objects_seed', 0)) + 7919)   # objetos: solo del mapa
        self.rng = np.random.default_rng(seed + 104729)                      # detecciones: del episodio
        self.objects = self._place(gt, float(s.get('objects_per_m2', 0.06)), rng)

    def _place(self, gt, density, rng):
        free = gt.free_mask()
        dist = ndimage.distance_transform_edt(~gt.occ_mask()) * gt.resolution
        band = free & (dist > 0.20) & (dist < 0.40)          # junto a las paredes
        cand = np.argwhere(band)
        n = max(3, int(round(free.sum() * gt.resolution ** 2 * density)))
        objs, min_sep = [], 1.0
        for k in rng.permutation(len(cand)):
            x, y = (float(v) for v in gt.cell_to_world(*cand[k]))
            if all(math.hypot(x - o['x'], y - o['y']) >= min_sep for o in objs):
                objs.append(dict(id=len(objs), x=x, y=y, cls=int(rng.integers(len(self.classes)))))
            if len(objs) >= n:
                break
        return objs

    def observe(self, true_pose, est_pose):
        """Detecciones desde la pose real, expresadas en el marco 'map' con la pose estimada."""
        x, y, yaw = true_pose
        out = []
        for o in self.objects:
            if not in_fov(o['x'], o['y'], x, y, yaw, self.hfov, self.range):
                continue
            if not line_free(self.gt, (x, y), (o['x'], o['y'])):
                continue
            if self.rng.random() > self.p_det:
                continue
            lx, ly = se2_inverse_apply(true_pose, o['x'], o['y'])           # relativo al robot
            lx, ly = float(lx) + self.rng.normal(0, self.sigma), float(ly) + self.rng.normal(0, self.sigma)
            mx, my = se2_apply(est_pose, lx, ly)
            if self.rng.random() < self.p_correct:
                c = o['cls']
            else:
                c = int(self.rng.choice([k for k in range(len(self.classes)) if k != o['cls']]))
            out.append(dict(id=o['id'], x=float(mx), y=float(my), cls=c,
                            score=float(self.rng.uniform(0.5, 0.95))))
        return out

    def evaluate(self, smap: SemanticMap):
        """Métricas del mapa semántico frente a los objetos reales (ids conocidos)."""
        n = len(self.objects)
        lms = smap.landmarks
        conf = [lms[o['id']] for o in self.objects if o['id'] in lms and lms[o['id']].state == 'confirmed']
        correct = sum(1 for o in self.objects if o['id'] in lms
                      and int(np.argmax(lms[o['id']].probs)) == o['cls'])
        return dict(sem_n_objects=n, sem_n_seen=len(lms), sem_n_confirmed=len(conf),
                    sem_recall_seen=len(lms) / n if n else float('nan'),
                    sem_recall_confirmed=len(conf) / n if n else float('nan'),
                    sem_class_accuracy=correct / n if n else float('nan'),
                    sem_mean_entropy=smap.summary()['mean_class_entropy'])

    def dump(self, smap: SemanticMap):
        """Objetos (mundo) y landmarks finales (marco 'map') para plot_traj."""
        return dict(classes=self.classes, objects=self.objects,
                    landmarks=[lm.to_dict(self.classes) for lm in smap.landmarks.values()])
