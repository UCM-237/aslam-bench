"""Política DRL de exploración (TD3 de c3-aslamdrl-platform, rama `drl`) sin ROS.

Reproduce la observación y la red *actor* de `applications/drl_active_slam` para poder
evaluar un agente entrenado (`actor_stage{S}_episode{N}.pt`) dentro de aslam_bench, tanto
en el simulador 2D como en el nodo ROS 2 (`drl_explorer`). Solo hace inferencia: no
entrena, no añade ruido de exploración y no necesita el paquete `drl_active_slam`.

Observación (1656 valores, igual que `DRLEnvironment.get_state`):
  * 500 rangos LiDAR en [0, 1] (rango / 6 m), haz k a k·2π/499 rad del eje x del robot
    (sensor de Gazebo Classic: min_angle=0, max_angle=2π, 500 muestras);
  * LEM 24x24: recorte de 3,5·24 celdas centrado en el robot, alineado con el marco
    'map' (sin rotar), max-pooling: ocupado=255, libre=0, desconocido=128;
  * GEM 24x24: caja explorada del mapa reducida por vecino más próximo
    (conocido=255, desconocido=0) con el robot marcado (3x3 celdas = 128);
  * acción anterior normalizada (a_lin, a_ang) y sin/cos del yaw en el marco 'map'.
Acción: tanh en [-1, 1]^2 -> v = (a0 + 1)/2 · 0,3 m/s (sin marcha atrás), w = a1 · 1,9 rad/s.

`torch` se importa de forma perezosa: B0/B1 y el análisis no lo necesitan.
"""
from __future__ import annotations

import hashlib
import math
import os
import time

import numpy as np

from .grid import FREE_THRESH, OCC_THRESH, OccGrid

# --- constantes del entrenamiento (common/settings.py y drl_environment.py) ---
N_SCAN = 500
SCAN_CAP = 6.0                    # LIDAR_DISTANCE_CAP (m)
SCAN_ANGLE_SPAN = 2 * math.pi     # min_angle=0, max_angle=2π en el sensor de entrenamiento
MAP_HW = 24
LEM_SCALE = 3.5                   # processor.lem_scale en DRLEnvironment
GEM_MARKER_SIZE = 3
GEM_MARKER_VALUE = 128
N_SCALARS = 4                     # acción previa (2) + sin/cos yaw (2)
STATE_SIZE = N_SCAN + 2 * MAP_HW * MAP_HW + N_SCALARS   # 1656
ACTION_SIZE = 2
HIDDEN_SIZE = 512
TRAIN_V_MAX = 0.3                 # SPEED_LINEAR_MAX (m/s)
TRAIN_W_MAX = 1.9                 # SPEED_ANGULAR_MAX (rad/s)
ENABLE_BACKWARD = False


def training_scan_angles(n=N_SCAN):
    """Ángulos (rad, marco del robot) de los haces que vio el agente al entrenar."""
    return np.arange(n) * (SCAN_ANGLE_SPAN / (n - 1))


# ---------------------------------------------------------------------------------- observación
def resample_scan(ranges, angles, n=N_SCAN, cap=SCAN_CAP):
    """Rangos de cualquier LiDAR 2D -> los `n` haces del entrenamiento, normalizados a [0, 1].

    `angles` son los ángulos de cada haz en el marco del robot (rad). Se toma el haz más
    próximo en ángulo (circular). inf / sin retorno -> 1 (como Gazebo con clip a 6 m);
    NaN (por debajo de range_min) -> 0, que el agente interpreta como obstáculo pegado.
    """
    r = np.asarray(ranges, float)
    a = np.mod(np.asarray(angles, float), 2 * math.pi)
    r = np.where(np.isnan(r), 0.0, r)
    r = np.where(np.isinf(r), cap, r)
    r = np.clip(r / cap, 0.0, 1.0)
    order = np.argsort(a)
    a, r = a[order], r[order]
    q = np.mod(training_scan_angles(n), 2 * math.pi)
    # vecino más próximo circular
    a_ext = np.concatenate([a[-1:] - 2 * math.pi, a, a[:1] + 2 * math.pi])
    r_ext = np.concatenate([r[-1:], r, r[:1]])
    k = np.searchsorted(a_ext, q)
    k = np.clip(k, 1, len(a_ext) - 1)
    left_closer = (q - a_ext[k - 1]) <= (a_ext[k] - q)
    return np.where(left_closer, r_ext[k - 1], r_ext[k]).astype(np.float32)


def ternary_map(grid: OccGrid, free_thresh=FREE_THRESH, occ_thresh=OCC_THRESH):
    """Rejilla -> valores exactos de slam_toolbox (-1 / 0 / 100), que es lo que espera la LEM.

    El mapper log-odds del simulador 2D publica probabilidades 0..100; sin este paso casi
    ninguna celda vale exactamente 0 o 100 y la LEM sale entera 'desconocida'.
    """
    d = grid.data.astype(np.int16)
    out = np.full(d.shape, -1, np.int16)
    out[(d >= 0) & (d < free_thresh)] = 0
    out[d >= occ_thresh] = 100
    return out


def _robot_cell(grid: OccGrid, x, y):
    # igual que MapProcessor.get_robot_cell (truncado, no floor)
    col = int((x - grid.origin[0]) / grid.resolution)
    row = int((y - grid.origin[1]) / grid.resolution)
    h, w = grid.shape
    return (row, col) if (0 <= row < h and 0 <= col < w) else None


def compute_lem(m, rc, hw=MAP_HW, scale=LEM_SCALE):
    """Local Egocentric Map (port vectorizado de MapProcessor.generate_lem)."""
    if rc is None:
        return None
    H = W = hw
    hb = wb = int(round(hw * scale))
    r0, c0 = rc[0] - hb // 2, rc[1] - wb // 2
    canvas = np.full((hb, wb), -1, np.int16)
    rr0, cc0 = max(0, r0), max(0, c0)
    rr1, cc1 = min(m.shape[0], r0 + hb), min(m.shape[1], c0 + wb)
    if rr1 > rr0 and cc1 > cc0:
        canvas[rr0 - r0:rr1 - r0, cc0 - c0:cc1 - c0] = m[rr0:rr1, cc0:cc1]
    rs = np.array([int(i * hb / H) for i in range(H)])
    cs = np.array([int(j * wb / W) for j in range(W)])
    pooled = np.maximum.reduceat(np.maximum.reduceat(canvas, rs, axis=0), cs, axis=1)
    lem = np.full((H, W), 128, np.uint8)
    lem[pooled == 100] = 255
    lem[pooled == 0] = 0
    return lem


def compute_gem(m, rc, hw=MAP_HW, marker=GEM_MARKER_SIZE, marker_value=GEM_MARKER_VALUE):
    """Global Exploration Map (port vectorizado de MapProcessor.compute_gem)."""
    known = np.argwhere(m != -1)
    if known.size == 0:
        return None
    min_r, min_c = known.min(axis=0)
    max_r, max_c = known.max(axis=0)
    sub = m[min_r:max_r + 1, min_c:max_c + 1]
    src = np.where(sub != -1, 255, 0).astype(np.uint8)
    h_in, w_in = src.shape
    ri = np.clip(((np.arange(hw) + 0.5) * h_in / hw).astype(int), 0, h_in - 1)
    ci = np.clip(((np.arange(hw) + 0.5) * w_in / hw).astype(int), 0, w_in - 1)
    gem = src[np.ix_(ri, ci)].copy()
    if rc is not None:
        r_rel = min(max(rc[0] - min_r, 0), h_in - 1)
        c_rel = min(max(rc[1] - min_c, 0), w_in - 1)
        i_red = min(max(int(round((r_rel + 0.5) * hw / h_in - 0.5)), 0), hw - 1)
        j_red = min(max(int(round((c_rel + 0.5) * hw / w_in - 0.5)), 0), hw - 1)
        d = max(1, int(marker))
        d = (d - 1 if d % 2 == 0 else d) // 2
        gem[max(0, i_red - d):min(hw - 1, i_red + d) + 1,
            max(0, j_red - d):min(hw - 1, j_red + d) + 1] = marker_value
    return gem


def build_observation(scan01, grid: OccGrid | None, pose, prev_action):
    """Vector de estado (STATE_SIZE,) en el mismo orden que DRLEnvironment.get_state."""
    maps = np.zeros(2 * MAP_HW * MAP_HW, np.float32)
    if grid is not None:
        m = ternary_map(grid)
        rc = _robot_cell(grid, pose[0], pose[1])
        lem, gem = compute_lem(m, rc), compute_gem(m, rc)
        if lem is not None and gem is not None:   # el original pone ceros si falta alguno
            maps = np.concatenate([lem.ravel(), gem.ravel()]).astype(np.float32) / 255.0
    scal = np.array([prev_action[0], prev_action[1], math.sin(pose[2]), math.cos(pose[2])], np.float32)
    return np.concatenate([np.asarray(scan01, np.float32), maps, scal])


def action_to_cmd(action):
    """Acción normalizada [-1, 1]^2 -> (v, w) con los límites del entrenamiento."""
    a0, a1 = float(action[0]), float(action[1])
    v = a0 * TRAIN_V_MAX if ENABLE_BACKWARD else (a0 + 1.0) / 2.0 * TRAIN_V_MAX
    return v, a1 * TRAIN_W_MAX


# ---------------------------------------------------------------------------------- red
def make_actor(state_size=STATE_SIZE, action_size=ACTION_SIZE, hidden_size=HIDDEN_SIZE):
    """Misma arquitectura (y nombres de parámetros) que `Actor` en drl_agent/td3.py."""
    import torch
    from torch import nn

    n_maps = 2 * MAP_HW * MAP_HW

    class Actor(nn.Module):
        def __init__(self):
            super().__init__()
            out_dimension = 20
            self.cnn_extract_laser = nn.Sequential(
                nn.Conv1d(1, 16, kernel_size=7, stride=3), nn.ReLU(),
                nn.Conv1d(16, 1, kernel_size=6, stride=1, padding=1), nn.ReLU(),
                nn.AdaptiveMaxPool1d(out_dimension), nn.Flatten())
            self.cnn_extract_map = nn.Sequential(
                nn.Conv2d(2, 16, kernel_size=3, stride=2, padding=1), nn.ReLU(),
                nn.Conv2d(16, 2, kernel_size=2, stride=2, padding=1), nn.ReLU(),
                nn.AdaptiveMaxPool2d((5, 5)), nn.Flatten())
            self.fa1 = nn.Linear(out_dimension + 50 + (state_size - N_SCAN - n_maps), hidden_size)
            self.fa2 = nn.Linear(hidden_size, hidden_size)
            self.fa3 = nn.Linear(hidden_size, action_size)

        def forward(self, states):
            if states.dim() == 1:
                states = states.unsqueeze(0)
            scan = states[:, :N_SCAN].unsqueeze(1)
            maps = states[:, N_SCAN:N_SCAN + n_maps].reshape(-1, 2, MAP_HW, MAP_HW)
            scalars = states[:, N_SCAN + n_maps:]
            x = torch.cat([self.cnn_extract_laser(scan), self.cnn_extract_map(maps), scalars], dim=1)
            x = torch.relu(self.fa1(x))
            x = torch.relu(self.fa2(x))
            return torch.tanh(self.fa3(x))

    return Actor()


def _sha256(path, n=12):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()[:n]


class DrlPolicy:
    """Carga un actor TD3 (.pt) y convierte (scan, mapa, pose) en (v, w).

    weights        : ruta a `actor_stage{S}_episode{N}.pt` (state_dict del actor).
    v_max, w_max   : límites de la plataforma; si `clip_to_platform`, la orden se recorta
                     a ellos después de des-normalizar con los límites del entrenamiento.
    """

    def __init__(self, weights, v_max=None, w_max=None, clip_to_platform=True, device='cpu',
                 torch_threads=1):
        import torch
        if torch_threads:
            # la red es pequeña (~1 ms por paso): con varios episodios en paralelo (-j N) los
            # hilos de torch por proceso saturan la CPU y el episodio va ~10 veces más lento
            torch.set_num_threads(int(torch_threads))
        if not weights:
            raise ValueError('DRL: no se ha indicado el fichero de pesos (.pt). Usa drl.weights, '
                             'drl.agents, "DRL:<ruta.pt>" o la variable ASLAM_BENCH_DRL_WEIGHTS.')
        weights = os.path.abspath(os.path.expanduser(weights))
        if not os.path.isfile(weights):
            raise FileNotFoundError(f'DRL: no existe el fichero de pesos {weights}')
        self.torch = torch
        self.device = device
        self.actor = make_actor().to(device)
        state = torch.load(weights, map_location=device)
        if isinstance(state, dict) and 'state_dict' in state:
            state = state['state_dict']
        self.actor.load_state_dict(state)
        self.actor.eval()
        self.weights = weights
        self.weights_sha = _sha256(weights)
        self.v_max, self.w_max = v_max, w_max
        self.clip = clip_to_platform
        self.reset()

    def reset(self):
        self.prev_action = (0.0, 0.0)
        self.last_obs = None

    def act(self, ranges, angles, grid, pose):
        """Devuelve (v, w, info). `pose` = (x, y, yaw) en el marco del mapa."""
        t0 = time.perf_counter()
        obs = build_observation(resample_scan(ranges, angles), grid, pose, self.prev_action)
        with self.torch.no_grad():
            a = self.actor(self.torch.from_numpy(obs).to(self.device)).squeeze(0).cpu().numpy()
        self.prev_action = (float(a[0]), float(a[1]))
        self.last_obs = obs
        v, w = action_to_cmd(a)
        if self.clip:
            if self.v_max is not None:
                v = float(np.clip(v, -self.v_max, self.v_max))
            if self.w_max is not None:
                w = float(np.clip(w, -self.w_max, self.w_max))
        return v, w, dict(decision_time=time.perf_counter() - t0, a_lin=float(a[0]), a_ang=float(a[1]))
