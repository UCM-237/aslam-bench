# aslam_bench: banco de pruebas de exploración activa

Paquete ROS 2 (ament_python) y herramientas Python para comparar métodos de exploración activa / Active SLAM en un **iRobot Create 3 con LiDAR 2D**. El objetivo final es comparar un método DRL con baselines clásicos. Es la **iteración 1** del [plan de evaluación](docs/plan_evaluacion_drl_vs_aslam.md):

- Plataforma de referencia definida: [`docs/plataforma_referencia.md`](docs/plataforma_referencia.md) y [`config/platform.yaml`](config/platform.yaml). La cámara está en el modelo, pero ningún método la usa todavía.
- Baselines **B0** (objetivo aleatorio, cota inferior) y **B1** (frontera más cercana, Yamauchi 1997).
- Entorno para lanzar episodios, registrar datos y calcular métricas, con dos backends que escriben el mismo formato:
  - **sim2d**: simulador 2D propio, rápido y sin ROS. Sirve para validar el pipeline y prototipar (unos 15 s por episodio).
  - **gazebo**: ROS 2 + Gazebo + TurtleBot 4 Lite simulado + SLAM + Nav2.
- Análisis estadístico pareado (Wilcoxon + Holm, Cliff's δ, McNemar, Kaplan-Meier, Friedman/Nemenyi cuando haya ≥ 3 métodos).

```
aslam_bench/
├── config/platform.yaml            ← plataforma de referencia (única fuente de parámetros)
├── config/campaigns/*.yaml         ← matrices de experimentos
├── maps/  worlds/                  ← mapas reales (PGM+YAML) y mundos SDF generados a partir de ellos
├── launch/episode.launch.py        ← un episodio en Gazebo (sim + SLAM + Nav2 + explorador + monitor)
├── launch/sim_tb4.launch.py        ← simulación por defecto (TurtleBot 4 Lite en gz sim)
├── aslam_bench/core/               ← sin ROS: rejillas, fronteras, estrategias, lógica, métricas, estadística
├── aslam_bench/sim2d/              ← simulador 2D
├── aslam_bench/ros/                ← nodos explorer y episode_monitor
├── aslam_bench/tools/              ← make_maps, map_to_world, run_campaign, analyze, plot_traj
└── tests/                          ← pytest del núcleo
```

## 1. Uso rápido sin ROS (simulador 2D)

```bash
pip install -r requirements.txt
python -m pytest -q tests/                                     # 15 pruebas
python -m aslam_bench.tools.run_campaign config/campaigns/iter1_sim2d.yaml --out results -j 4
python -m aslam_bench.tools.analyze results/iter1_sim2d        # → results/iter1_sim2d/analysis/report.md
python -m aslam_bench.tools.plot_traj results/iter1_sim2d --grid # → traj.png por episodio y analysis/fig_traj_*.png
```

La campaña es reanudable: si se interrumpe, al relanzarla salta los episodios terminados. Las poses iniciales se muestrean una vez y se guardan en `starts.yaml`, y todos los métodos usan las mismas, porque el diseño es pareado.

## 2. Gazebo (ROS 2 Jazzy + Gazebo Harmonic)

> **Sin probar todavía en Gazebo.** En el entorno donde se desarrolló no había ROS. La lógica que comparten los nodos (`core/`) está probada en el simulador 2D y con pytest. Los nodos y los launch solo están revisados y compilados. Lo previsible es tener que ajustar nombres de paquetes, de topics o del modelo según tu instalación. Abajo está lo que conviene comprobar primero.

```bash
# dependencias
sudo apt install ros-jazzy-turtlebot4-simulator ros-jazzy-turtlebot4-navigation ros-jazzy-ros-gz
# compilar
cd ~/ws/src && cp -r /ruta/aslam_bench . && cd ~/ws && colcon build --packages-select aslam_bench && source install/setup.bash

# un episodio suelto (termina solo)
ros2 launch aslam_bench episode.launch.py world:=rooms_small x:=1.5 y:=5.0 yaw:=0.0 strategy:=B1 seed:=1 out_dir:=/tmp/ep1

# campaña completa en Gazebo (episodios en serie) y análisis
ros2 run aslam_bench run_campaign $(ros2 pkg prefix aslam_bench)/share/aslam_bench/config/campaigns/iter1_gazebo.yaml --out results
ros2 run aslam_bench analyze results/iter1_gazebo
ros2 run aslam_bench plot_traj results/iter1_gazebo --grid
```

Con tu propio Graph SLAM: `slam_launch:=/ruta/mi_slam.launch.py`. Debe publicar `/map` y la TF `map→odom`, y el marco `map` debe empezar en la pose inicial del robot. Con tu propia simulación del Create 3: `sim_launch:=...`, que debe aceptar `world x y yaw` y publicar `/scan`, `/tf` y `/clock`. Las dos opciones se pueden fijar en `gazebo.extra_args` del YAML de campaña.

### Ver un episodio en tiempo real

Por defecto todo arranca sin ventanas (lo que interesa en una campaña). Para un episodio suelto:

```bash
# Gazebo con interfaz gráfica + RViz
ros2 launch aslam_bench episode.launch.py world:=rooms_small x:=1.5 y:=5.0 yaw:=0.0 \
    strategy:=B1 seed:=1 out_dir:=/tmp/ep1 headless:=false rviz:=true
```

- `headless:=false` abre la GUI de Gazebo (el mundo y el robot moviéndose).
- `rviz:=true` abre RViz con `config/aslam_bench.rviz`: marco fijo `map`, el mapa de SLAM (`/map`), el scan (`/scan`), el robot y las TF, el plan de Nav2 (`/plan`) y los marcadores del explorador en `/aslam_bench/markers` (celdas frontera en naranja, objetivos candidatos en azul y el objetivo elegido como flecha verde). El costmap global está incluido pero desactivado; se activa en el panel *Displays*. Con `rviz_config:=/ruta/mia.rviz` se usa otra configuración.
- Se pueden usar por separado: `rviz:=true` solo (con Gazebo sin ventana) es lo más ligero y suele bastar para seguir la exploración.

Para verlo durante una campaña, añádelos en el YAML: `gazebo.extra_args: {headless: false, rviz: true}`. Se abrirá y cerrará una ventana por episodio, así que conviene solo para depurar con pocos episodios. Mostrar la GUI consume CPU/GPU y puede ralentizar la simulación respecto a las campañas sin ventanas, así que no mezcles resultados de ambos modos.

Si ya tienes una sesión en marcha, también puedes abrir RViz aparte: `rviz2 -d $(ros2 pkg prefix aslam_bench)/share/aslam_bench/config/aslam_bench.rviz --ros-args -p use_sim_time:=true`.

**Comprobaciones al primer arranque:**
1. `ros2 topic echo /aslam_bench/gt_poses --once` debe contener un `child_frame_id` igual a `robot_model` (por defecto `turtlebot4`). Si no aparece, mira el nombre real con `gz topic -e -t /world/rooms_small/dynamic_pose/info` y pásalo como `robot_model:=`.
2. El mundo carga (`GZ_SIM_RESOURCE_PATH` incluye `share/aslam_bench/worlds`) y el robot aparece en la pose pedida.
3. `ros2 run tf2_ros tf2_echo map base_link` responde, y el explorador deja de escribir "Esperando a Nav2...".
4. Al terminar, `/tmp/ep1/` contiene `meta.json` con `termination`, `poses.csv`, `goals.csv` y `maps/*.npz`.

## 3. Formato de un episodio

Es el mismo para sim2d, Gazebo y el robot real (ver `core/recorder.py`):

| Fichero | Contenido |
|---|---|
| `meta.json` | método, mapa, pose inicial (mundo), semilla, modo de localización, terminación, duración |
| `poses.csv` | `t, gt_x, gt_y, gt_yaw` (mundo) y `est_x, est_y, est_yaw` (marco del SLAM) |
| `goals.csv` | objetivos enviados, tiempo de decisión, nº de fronteras, cómo terminó el objetivo anterior |
| `events.csv` | colisiones, fallos de navegación |
| `maps/*.npz` | instantáneas del mapa estimado cada 5 s |

### Marcos de referencia

- `gt_*` y la pose inicial (`start_pose` en `meta.json`) están en el marco del **mundo**, que es el del YAML del mapa (los mundos SDF se generan con `map_to_world` a partir de él, así que Gazebo usa las mismas coordenadas).
- `est_*`, los objetivos de `goals.csv` y los mapas de `maps/*.npz` están en el marco **`map` del SLAM**, que empieza en la pose inicial del robot. Para pasarlos al mundo: `world = start_pose ⊕ map`, es decir, `se2_apply(start_pose, x, y)` de `core/grid.py`, y al yaw se le suma el de la pose inicial.
- `events.csv` no guarda posición. `plot_traj` coloca cada colisión donde estaba el robot en ese instante, interpolando la trayectoria real (o la estimada si no hay ground truth).

### Dibujar trayectorias (`plot_traj`)

```bash
# un episodio → <episodio>/traj.png
python -m aslam_bench.tools.plot_traj results/iter1_sim2d/episodes/oracle/rooms_small/B1_nearest_frontier/s00_r0
# una campaña entera, más una figura por (modo, mapa) con filas = pose inicial y columnas = método
python -m aslam_bench.tools.plot_traj results/iter1_sim2d --grid
# solo la comparativa de un mapa, con el último mapa estimado superpuesto, en otra carpeta
python -m aslam_bench.tools.plot_traj results/iter1_sim2d --grid --no-episodes --map rooms_small --est-map --out figs/
```

Cada figura dibuja sobre el mapa real la trayectoria real (azul), la estimada por el SLAM ya pasada al mundo (naranja discontinua), los objetivos (×), las colisiones (○), el inicio con su orientación y el final. En el título aparecen método, mapa, episodio, modo de localización, motivo de terminación, duración y ATE. Con `--est-map` se añaden las celdas ocupadas del último mapa estimado, también pasadas al mundo; si la localización deriva, se ven las paredes desplazadas respecto al mapa real.

| Opción | Efecto |
|---|---|
| `--maps DIR` | directorio de mapas (por defecto `maps/` del paquete). Si no encuentra el mapa del episodio, dibuja sin fondo |
| `--out DIR` | guarda todas las figuras ahí (`<modo>__<mapa>__<método>__<episodio>.png`) en lugar de `traj.png` en cada episodio |
| `--grid` | figura comparativa `fig_traj_<modo>_<mapa>.png` en `<campaña>/analysis/` (o en `--out`) |
| `--no-episodes` | no genera las figuras individuales |
| `--est-map` | superpone el último mapa estimado |
| `--map`, `--method`, `--loc-mode` | filtros (se pueden repetir) |

En el robot real, sin ground truth, `gt_*` es NaN: se dibuja solo la trayectoria estimada.

## 4. Métricas por episodio (`core/metrics.py`)

| Familia | Métricas |
|---|---|
| Eficiencia | curva de cobertura frente a tiempo y distancia, T50/T75/T90/T95, D50…D95, AUC normalizada, cobertura final, longitud de trayectoria, rotación total, m² explorados por metro |
| Mapa | precisión y recall de celdas ocupadas (tolerancia 0,10 m), IoU del espacio libre, ADNN |
| Localización | ATE (RMSE), error final y máximo |
| Robustez | éxito (≥ 95 %), motivo de terminación, colisiones, objetivos fallidos o caducados |
| Coste | tiempo de decisión (media, p95) |

## 5. Añadir un método (B2…B5, tu DRL)

```python
# aslam_bench/core/strategies.py
class MiDRL(Strategy):
    name = 'DRL'
    def __init__(self, cfg):
        super().__init__(cfg)
        self.policy = cargar_politica(...)
    def choose(self, ctx, rng, blacklist):
        # ctx.grid (OccGrid), ctx.robot_xy, ctx.frontiers (con distancia geodésica), ctx.dist, ctx.passable
        x, y = self.policy(ctx)
        return Goal(x, y, yaw=0.0, distance=float(ctx.dist[ctx.grid.world_to_cell(x, y)]))
STRATEGIES['DRL'] = MiDRL
```

Después basta con añadir `DRL` a `methods:` en la campaña. Si tu política genera velocidades en lugar de objetivos, el monitor y las métricas sirven igual. Solo hay que sustituir el nodo `explorer` por el tuyo y publicar los eventos en `/aslam_bench/explorer_events`.
