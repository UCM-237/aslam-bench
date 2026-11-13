# Plataforma de referencia (iteración 1)

Todos los métodos (B0, B1 y, más adelante, B2–B5 y tu DRL) se evalúan sobre **la misma plataforma**. Solo cambia la pieza que decide a dónde ir. Los parámetros numéricos están en un único fichero, [`config/platform.yaml`](../config/platform.yaml), que leen el simulador 2D, los nodos ROS 2 y el cálculo de métricas.

> He tomado como referencia concreta el **TurtleBot 4 Lite**, que es exactamente "Create 3 + LiDAR 2D + cámara": tiene simulador oficial en Gazebo y usa los mismos drivers que el robot real. Si tu LiDAR, tu cámara o tu ordenador de a bordo son otros, cambia `platform.yaml`. Nada más depende de esos valores.

## 1. Hardware

| Componente | Referencia | Parámetros usados | Notas |
|---|---|---|---|
| Base | iRobot Create 3 | radio 0,171 m; v_max 0,26 m/s; ω_max 1,0 rad/s | El Create 3 admite 0,306 m/s. 0,26 es el valor por defecto de Nav2 en el TurtleBot 4 y es el límite común para todos los métodos. |
| LiDAR 2D | RPLIDAR A1M8 | 360°, 360 haces, 0,15–12 m, 5,5 Hz, σ = 1 cm | `/scan`, frame `rplidar_link` |
| Cámara | OAK-D Lite (RGB usada como monocular) | HFOV 69°, 320×240 | **Presente y publicando, pero desactivada** (`camera.enabled: false`). Ningún método la consume en la iteración 1. Queda lista para YOLOv8 y la capa semántica. |
| Cómputo | Raspberry Pi 4 (a bordo) / PC (simulación) | — | El tiempo de decisión se mide siempre; conviene repetirlo en la Pi antes de ir al robot real. |
| Bumpers | Create 3 `/hazard_detection` | — | Se cuentan como colisiones. |

## 2. Software

| Capa | Referencia | Cómo cambiarla |
|---|---|---|
| ROS 2 / simulador | Jazzy + Gazebo Harmonic + `turtlebot4_simulator` | En Humble el paquete de spawn es `turtlebot4_ignition_bringup` (ver `launch/sim_tb4.launch.py`). |
| SLAM | `slam_toolbox` (vía `turtlebot4_navigation`) | Argumento `slam_launch:=` en `episode.launch.py` → tu Graph SLAM (odometría + ICP). Requisitos: publicar `/map` (OccupancyGrid) y la TF `map → odom`. |
| Navegación | Nav2 con la configuración de `turtlebot4_navigation` | Argumento `nav2_launch:=`. Igual para todos los métodos. |
| Exploración | `aslam_bench/explorer` (B0, B1) | Estrategia nueva = una clase en `core/strategies.py`. |
| Registro y métricas | `aslam_bench/episode_monitor` + `analyze` | — |

### Interfaz común de los métodos

```
/map (OccupancyGrid)  +  TF map→base_link
        │
        ▼
 ExplorerLogic.update(t, mapa, pose, estado_nav)   ← igual en ROS 2 y en el simulador 2D
        │ Strategy.choose(...)  (B0 | B1 | ... | DRL)
        ▼
 objetivo (x, y, yaw) → Nav2 NavigateToPose
```

Reglas comunes, fijadas en `platform.yaml`:
- Fronteras: celdas libres junto a una región desconocida. Se agrupan en clusters de al menos 0,30 m. Se ignoran los huecos desconocidos de menos de 10 celdas y las franjas de 1–2 celdas de ancho, como el interior de las paredes.
- Un objetivo fallido o caducado se veta en un radio de 0,5 m. El tiempo límite de un objetivo es `max(30 s, 3·d/v_max)`.
- El explorador termina cuando no quedan fronteras alcanzables. Este criterio se aplica igual a todos los métodos, incluido B0.

## 3. Criterio de episodio

| Parámetro | Valor |
|---|---|
| Objetivo de cobertura | 95 % del **área alcanzable** |
| t_max | 900 s simulados |
| Estancamiento | menos de 1 % de cobertura ganada en 300 s → fallo |
| Instantáneas del mapa | cada 5 s |
| Muestras de pose | cada 0,2 s |

**Área alcanzable**: celdas libres del mapa real a las que puede llegar el centro del robot desde la pose inicial (con su radio + margen), ampliadas hasta la pared. La cobertura es la fracción de esas celdas que aparecen como conocidas en el mapa estimado. Se expresa en el marco del SLAM usando la pose inicial: `world = start ⊕ map`.

## 4. Modos de localización

| Modo | Dónde | Para qué |
|---|---|---|
| `oracle` | simulador 2D | Pose perfecta: mide solo la calidad de la decisión. |
| `odom_noise` | simulador 2D | Odometría con deriva y sin corrección (σ_v = 3 %, σ_ω = 5 %). Es una cota pesimista y deja ver qué métodos sufren más con la deriva. |
| `slam` | Gazebo / robot real | SLAM real (slam_toolbox o tu Graph SLAM). |

## 5. Mundos

Hay tres mapas de referencia generados por código (`tools/make_maps.py`). Se convierten a mundos SDF extruyendo las celdas ocupadas (`tools/map_to_world.py`), así que **el ground truth de Gazebo es exactamente el mapa YAML**.

| Mapa | Tamaño | Rasgo |
|---|---|---|
| `rooms_small` | 12 × 9 m, 94 m² libres | vivienda: pasillo + 5 habitaciones, muebles |
| `office_corridor` | 22 × 10 m, 199 m² libres | pasillo largo de 2 m, 8 despachos |
| `cluttered_hall` | 14 × 12 m, 156 m² libres | nave abierta con columnas, cajas y tabiques parciales |

Siguientes pasos naturales: mapas de HouseExpo para el entrenamiento y test del DRL, y los mundos AWS (*small house*, *bookstore*) como mapas de otra distribución.
