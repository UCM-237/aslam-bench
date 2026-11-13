"""Ejecuta una campaña de experimentos (matriz método x mapa x pose inicial x repetición x modo).

  python -m aslam_bench.tools.run_campaign config/campaigns/iter1_sim2d.yaml --out results/ -j 4

Es reanudable: los episodios ya terminados (meta.json con 'termination') se saltan.
Las poses iniciales se muestrean una sola vez por mapa y se guardan en
<out>/<name>/starts.yaml, de modo que todos los métodos comparten condiciones
(diseño pareado). Backends:
  * sim2d  : simulador 2D en proceso (paralelizable con -j).
  * gazebo : un `ros2 launch aslam_bench episode.launch.py ...` por episodio, en serie.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import yaml

from ..core.config import load_platform, robot_clearance
from ..core.grid import geodesic, inflate, load_map_yaml
from ..core.strategies import make_strategy


def sample_starts(gt, n, seed, clearance_m, cfg):
    """Poses aleatorias en el componente libre principal, lejos de obstáculos."""
    rng = np.random.default_rng(seed)
    free = gt.free_mask()
    ok = free & ~inflate(gt.occ_mask() | gt.unknown_mask(), max(clearance_m, robot_clearance(cfg)), gt.resolution)
    # componente más grande (evita empezar dentro de un hueco cerrado)
    from scipy import ndimage
    lab, k = ndimage.label(ok, structure=np.ones((3, 3)))
    if k == 0:
        raise RuntimeError('No hay celdas válidas para poses iniciales')
    big = np.argmax(np.bincount(lab.ravel())[1:]) + 1
    cells = np.argwhere(lab == big)
    sel = cells[rng.choice(len(cells), size=n, replace=False)]
    out = []
    for (i, j) in sel:
        x, y = gt.cell_to_world(i, j)
        out.append([round(float(x), 3), round(float(y), 3), round(float(rng.uniform(-math.pi, math.pi)), 3)])
    return out


def episode_seed(*parts):
    return zlib.crc32('/'.join(map(str, parts)).encode()) & 0x7FFFFFFF


def build_jobs(camp, cfg, maps_dir, out_root):
    starts_path = os.path.join(out_root, 'starts.yaml')
    if os.path.exists(starts_path):
        with open(starts_path) as f:
            starts = yaml.safe_load(f)
    else:
        starts = {}
    for m in camp['maps']:
        if m not in starts:
            gt = load_map_yaml(os.path.join(maps_dir, m + '.yaml'))
            starts[m] = sample_starts(gt, camp['n_starts'], camp['start_seed'] + episode_seed(m) % 1000,
                                      camp.get('start_clearance', 0.5), cfg)
    with open(starts_path, 'w') as f:
        yaml.safe_dump(starts, f)
    jobs = []
    for lm, lm_cfg in camp['loc_modes'].items():
        for m in camp['maps']:
            for sid, sp in enumerate(starts[m][:camp['n_starts']]):
                for rep in range(camp.get('reps', 1)):
                    seed = episode_seed(lm, m, sid, rep)          # mismo para todos los métodos
                    for meth in camp['methods']:
                        name = make_strategy(meth, cfg).name
                        ep_dir = os.path.join(out_root, 'episodes', lm, m, name, f's{sid:02d}_r{rep}')
                        jobs.append(dict(loc_mode=lm, loc_cfg=lm_cfg or {}, map=m, start_id=sid,
                                         start=sp, rep=rep, seed=seed, method=meth, ep_dir=ep_dir))
    return jobs


def resolve_drl(cfg, camp, camp_dir):
    """Fusiona el bloque `drl:` de la campaña y pasa a absolutas las rutas de los .pt.

    Las rutas relativas se buscan primero respecto al fichero de campaña y, si no existen,
    respecto al directorio de trabajo.
    """
    d = cfg.setdefault('drl', {})
    for k, v in (camp.get('drl') or {}).items():
        if k == 'agents':
            d['agents'] = {**(d.get('agents') or {}), **(v or {})}
        else:
            d[k] = v

    def absp(p):
        if not p:
            return p
        p = os.path.expanduser(str(p))
        if os.path.isabs(p):
            return p
        c = os.path.abspath(os.path.join(camp_dir, p))
        return os.path.abspath(p) if (not os.path.exists(c) and os.path.exists(p)) else c

    d['weights'] = absp(d.get('weights'))
    d['agents'] = {k: absp(v) for k, v in (d.get('agents') or {}).items()}
    camp['methods'] = [(f'DRL:{absp(m[4:])}' if str(m).startswith(('DRL:', 'drl:')) and str(m).endswith('.pt')
                        else m) for m in camp['methods']]
    for m in camp['methods']:
        st = make_strategy(m, cfg)
        if getattr(st, 'outputs', 'goal') == 'velocity':
            if not st.weights:
                sys.exit(f'{m}: falta el fichero de pesos (.pt); ver el bloque drl: de config/platform.yaml')
            if not os.path.isfile(st.weights):
                sys.exit(f'{m}: no existe {st.weights}')


def _done(ep_dir):
    try:
        with open(os.path.join(ep_dir, 'meta.json')) as f:
            return 'termination' in json.load(f)
    except (OSError, ValueError):
        return False


def run_sim2d_job(job, cfg, maps_dir):
    from ..sim2d.simulator import run_episode
    gt = load_map_yaml(os.path.join(maps_dir, job['map'] + '.yaml'))
    if os.path.isdir(job['ep_dir']):
        shutil.rmtree(job['ep_dir'])
    meta = run_episode(gt, tuple(job['start']), job['method'], cfg, job['ep_dir'],
                       meta_extra=dict(map=job['map'], start_id=job['start_id'], rep=job['rep'],
                                       loc_mode=job['loc_mode']),
                       seed=job['seed'], odom_noise=tuple(job['loc_cfg'].get('odom_noise', (0.0, 0.0))))
    return job, meta


def run_gazebo_job(job, camp, cfg, maps_dir, platform_path):
    g = camp.get('gazebo', {})
    if os.path.isdir(job['ep_dir']):
        shutil.rmtree(job['ep_dir'])
    os.makedirs(job['ep_dir'], exist_ok=True)
    x, y, yaw = job['start']
    args = dict(world=job['map'], map_yaml=os.path.abspath(os.path.join(maps_dir, job['map'] + '.yaml')),
                x=x, y=y, yaw=yaw, strategy=job['method'], seed=job['seed'], out_dir=os.path.abspath(job['ep_dir']),
                start_id=job['start_id'], rep=job['rep'], loc_mode=job['loc_mode'],
                platform=os.path.abspath(platform_path), **g.get('extra_args', {}))
    st = make_strategy(job['method'], cfg)
    if getattr(st, 'outputs', 'goal') == 'velocity':
        args['drl_weights'] = st.weights     # el nodo no ve el bloque drl: de la campaña
    cmd = ['ros2', 'launch', g.get('launch_package', 'aslam_bench'), g.get('launch_file', 'episode.launch.py')]
    cmd += [f'{k}:={v}' for k, v in args.items()]
    timeout = cfg['episode']['t_max'] * 1.5 + g.get('startup_grace', 60) + 60
    log = open(os.path.join(job['ep_dir'], 'launch.log'), 'w')
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        pass
    finally:
        # asegurar que no quedan Gazebo/Nav2 vivos entre episodios
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(proc.pid, sig)
                proc.wait(timeout=15)
                break
            except (ProcessLookupError, subprocess.TimeoutExpired):
                continue
        log.close()
    meta = {}
    try:
        with open(os.path.join(job['ep_dir'], 'meta.json')) as f:
            meta = json.load(f)
    except (OSError, ValueError):
        meta = dict(termination='launch_failed')
    return job, meta


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('campaign')
    ap.add_argument('--out', default='results')
    ap.add_argument('--maps', default=None, help='directorio de mapas (por defecto <paquete>/maps)')
    ap.add_argument('--platform', default=None)
    ap.add_argument('-j', '--jobs', type=int, default=1, help='procesos en paralelo (solo sim2d)')
    ap.add_argument('--limit', type=int, default=None, help='ejecutar solo los N primeros episodios pendientes')
    ap.add_argument('--force', action='store_true', help='repetir episodios ya terminados')
    a = ap.parse_args(argv)

    with open(a.campaign) as f:
        camp = yaml.safe_load(f)
    from ..core.config import DEFAULT_PLATFORM, resource_dir
    platform_path = a.platform or DEFAULT_PLATFORM
    cfg = load_platform(platform_path, camp.get('platform_overrides'))
    resolve_drl(cfg, camp, os.path.dirname(os.path.abspath(a.campaign)))
    maps_dir = a.maps or resource_dir('maps')
    out_root = os.path.join(a.out, camp['name'])
    os.makedirs(out_root, exist_ok=True)
    with open(os.path.join(out_root, 'campaign.yaml'), 'w') as f:
        yaml.safe_dump(dict(campaign=camp, platform=cfg), f, sort_keys=False, allow_unicode=True)

    jobs = build_jobs(camp, cfg, maps_dir, out_root)
    pending = [j for j in jobs if a.force or not _done(j['ep_dir'])]
    if a.limit:
        pending = pending[:a.limit]
    print(f'[{camp["name"]}] {len(jobs)} episodios, {len(pending)} pendientes', flush=True)
    t0 = time.time()

    def report(i, job, meta):
        el = time.time() - t0
        print(f'  [{i}/{len(pending)}] {job["loc_mode"]}/{job["map"]}/{job["method"]}/s{job["start_id"]}_r{job["rep"]}'
              f' -> {meta.get("termination")} t={meta.get("duration", float("nan")):.0f}s  ({el / 60:.1f} min)', flush=True)

    if camp['backend'] == 'sim2d':
        if a.jobs > 1:
            with ProcessPoolExecutor(a.jobs) as ex:
                futs = [ex.submit(run_sim2d_job, j, cfg, maps_dir) for j in pending]
                for i, fu in enumerate(as_completed(futs), 1):
                    report(i, *fu.result())
        else:
            for i, j in enumerate(pending, 1):
                report(i, *run_sim2d_job(j, cfg, maps_dir))
    elif camp['backend'] == 'gazebo':
        for i, j in enumerate(pending, 1):
            report(i, *run_gazebo_job(j, camp, cfg, maps_dir, platform_path))
    else:
        sys.exit(f'backend desconocido: {camp["backend"]}')
    print(f'Hecho en {(time.time() - t0) / 60:.1f} min. Resultados en {out_root}')


if __name__ == '__main__':
    main()
