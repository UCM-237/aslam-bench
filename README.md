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
├── aslam_bench/ros/                ← nodos explorer, drl_explorer y episode_monitor
├── aslam_bench/tools/              ← make_maps, map_to_world, run_campaign, analyze
└── tests/                          ← pytest del núcleo
```

## 1. Uso rápido sin ROS (simulador 2D)

```bash
pip install -r requirements.txt
python -m pytest -q tests/                                     # 20 pruebas (2 necesitan torch)
python -m aslam_bench.tools.run_campaign config/campaigns/iter1_sim2d.yaml --out results -j 4
python -m aslam_bench.tools.analyze results/iter1_sim2d        # → results/iter1_sim2d/analysis/report.md
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
```

Con tu propio Graph SLAM: `slam_launch:=/ruta/mi_slam.launch.py`. Debe publicar `/map` y la TF `map→odom`, y el marco `map` debe empezar en la pose inicial del robot. Con tu propia simulación del Create 3: `sim_launch:=...`, que debe aceptar `world x y yaw` y publicar `/scan`, `/tf` y `/clock`. Las dos opciones se pueden fijar en `gazebo.extra_args` del YAML de campaña.

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

## 4. Métricas por episodio (`core/metrics.py`)

| Familia | Métricas |
|---|---|
| Eficiencia | curva de cobertura frente a tiempo y distancia, T50/T75/T90/T95, D50…D95, AUC normalizada, cobertura final, longitud de trayectoria, rotación total, m² explorados por metro |
| Mapa | precisión y recall de celdas ocupadas (tolerancia 0,10 m), IoU del espacio libre, ADNN |
| Localización | ATE (RMSE), error final y máximo |
| Robustez | éxito (≥ 95 %), motivo de terminación, colisiones, objetivos fallidos o caducados |
| Coste | tiempo de decisión (media, p95) |

## 5. Agente DRL (TD3 de c3-aslamdrl-platform, rama `drl`)

El agente de `applications/drl_active_slam` está integrado como método **`DRL`**. A diferencia de B0/B1 **no elige objetivos: emite (v, ω) a 10 Hz**, así que no pasa por Nav2 (Gazebo) ni por `SimpleNavigator` (sim2d). La lógica común solo aplica el mismo criterio de fin que a los demás (sin fronteras alcanzables), además de cobertura, estancamiento y `t_max`.

- `core/drl_policy.py`: la red `Actor` de `td3.py` (mismos nombres de parámetros) y la observación de `DRLEnvironment.get_state` (1656 valores: 500 haces LiDAR / 6 m, LEM y GEM de 24×24, acción previa, sin/cos del yaw). No importa `drl_active_slam` ni ROS. Solo hace inferencia y no añade ruido de exploración.
- Adaptadores: cualquier LiDAR se remuestrea a los 500 haces del entrenamiento (0…2π desde el eje x del robot). El mapa se umbraliza a −1/0/100 (los valores exactos de slam_toolbox que espera la LEM) y la pose es la estimada en el marco `map`.
- Acción: v = (a₀+1)/2·0,3 m/s y ω = a₁·1,9 rad/s, que son los límites del entrenamiento. Con `drl.clip_to_platform: true` (por defecto) se recorta a `platform.base` (0,26 m/s y 1,0 rad/s), igual que las baselines.
- Colisiones: el episodio no termina; cada contacto se cuenta una vez.
- `torch` solo hace falta para el DRL (`pip install torch --index-url https://download.pytorch.org/whl/cpu`).

### Elegir el agente (.pt)

Solo hace falta el `actor_stage{S}_episode{N}.pt` (el `state_dict` del actor). Cópialo, por ejemplo, a `models/` (ignorado por Git). Formas de elegirlo, de más a menos prioritaria:

| Dónde | Ejemplo | Método en los resultados |
|---|---|---|
| `methods:` de la campaña con ruta | `DRL:models/actor_stage0_episode1000.pt` | `DRL_actor_stage0_episode1000` |
| `methods:` con alias + `drl.agents` | `DRL:e1000` y `drl: {agents: {e1000: models/...pt}}` | `DRL_e1000` |
| `drl.weights` (campaña o `platform.yaml`) | `DRL` | `DRL` |
| Launch de Gazebo | `strategy:=DRL drl_weights:=/ruta/actor.pt` | `DRL` |
| Variable de entorno | `ASLAM_BENCH_DRL_WEIGHTS=/ruta/actor.pt` | `DRL` |

Varios alias en la misma campaña permiten comparar semillas o episodios de entrenamiento. `meta.json` guarda la ruta y el SHA-256 de los pesos usados.

```bash
python -m aslam_bench.tools.run_campaign config/campaigns/drl_sim2d.yaml --out results -j 4
python -m aslam_bench.tools.analyze results/drl_sim2d
# Gazebo (sin Nav2 para el DRL; en Jazzy añade cmd_vel_stamped:=true si el TB4 espera TwistStamped)
ros2 launch aslam_bench episode.launch.py world:=rooms_small strategy:=DRL \
    drl_weights:=$PWD/models/actor_stage0_episode1000.pt out_dir:=/tmp/ep_drl
```

### LiDAR: RPLIDAR C1

`config/platform.yaml` describe ahora el **RPLIDAR C1** (500 haces, 10 Hz, 0,05–12 m), que es el del robot de c3-aslamdrl-platform y el que vio el agente al entrenar. El simulador 2D lo usa directamente. En Gazebo, el modelo del TurtleBot 4 trae su propio sensor (A1); el nodo DRL remuestrea lo que llegue, pero para simular de verdad el C1 hay que lanzar un modelo con ese sensor mediante `sim_launch`. Los resultados de la iteración 1 se obtuvieron con el A1M8 (360 haces, 5,5 Hz): para compararlos con B0/B1 hay que repetir esas baselines con el C1.

### Salto de dominio (a tener en cuenta al interpretar)

El agente se entrenó en Gazebo Classic, en una arena de 4,2×4,2 m, con episodios de 40 s y su propia función de éxito (GEM conocida > 95 %). Aquí se evalúa en mapas mayores y con episodios de hasta 900 s.

## 6. Añadir otro método

```python
# aslam_bench/core/strategies.py
class MiMetodo(Strategy):
    name = 'B2_mi_metodo'
    outputs = 'goal'            # o 'velocity' + act(ranges, angles, grid, pose) -> (v, w, info)
    def choose(self, ctx, rng, blacklist):
        # ctx.grid (OccGrid), ctx.robot_xy, ctx.frontiers (con distancia geodésica), ctx.dist, ctx.passable
        ...
STRATEGIES['B2'] = MiMetodo
```
