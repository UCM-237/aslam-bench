# Plan de evaluación: exploración activa con DRL frente a A-SLAM clásico (fronteras y ganancia de información)

Plataforma de referencia: iRobot Create 3 + LiDAR 2D + cámara monocular (YOLOv8), ROS 2, Graph SLAM (odometría + ICP), Gazebo y robot real.

> Supuestos (no he visto tu código DRL): la política decide **a dónde ir** (objetivo/frontera) o **cómo moverse** (velocidades). El plan cubre ambos casos; donde cambia algo lo indico. Las referencias están citadas de memoria; conviene verificar año/venue antes de incluirlas en un artículo.

---

## 0. Resumen en una página

1. **Aislar la capa de decisión.** Todos los métodos comparten SLAM, planificador/controlador (Nav2), sensores, límites de velocidad y criterio de parada. Solo cambia el *selector de objetivos* (o la política).
2. **Baselines (mínimo recomendado, 5 + 1 cota inferior):** aleatorio/seguimiento de pared, frontera más cercana, frontera con utilidad (ganancia − coste), RRT-exploration, ganancia de información con incertidumbre de pose (entropía conjunta / MI), y Active SLAM por optimalidad D sobre el grafo.
3. **Métricas en cinco familias:** eficiencia de exploración (cobertura vs tiempo/distancia, T90/T95, AUC), calidad del mapa (IoU/precisión vs mapa real), calidad del SLAM (ATE/RPE, D-opt), robustez (éxito, colisiones, atascos) y coste computacional. Opcional: métricas semánticas.
4. **Dos modos de localización:** *pose oráculo* (ground truth de Gazebo) para medir solo la calidad de la decisión, y *SLAM completo* para medir el efecto acoplado. Es la forma más limpia de demostrar que el DRL aporta algo más que "explorar rápido".
5. **Mapas separados train / validación / test**, con test nunca vistos por la política DRL, clasificados por complejidad.
6. **Estadística seria:** ≥20 ejecuciones por (método, mapa) en simulación, ≥3–5 semillas de entrenamiento del DRL, pruebas pareadas no paramétricas con corrección múltiple y tamaños del efecto.
7. **Validación real reducida** (2–3 entornos, 3–5 ejecuciones por método) con los 2–3 métodos más relevantes.

---

## 1. Preguntas de investigación

Formularlas explícitamente ayuda a decidir qué medir y qué no:

- **RQ1 – Eficiencia:** ¿explora el DRL más área por unidad de tiempo/distancia que los métodos clásicos?
- **RQ2 – Calidad:** ¿obtiene mapas y trayectorias más precisos (menos deriva, mejores cierres de bucle) o solo más rápidos?
- **RQ3 – Generalización:** ¿se mantiene la ventaja en entornos con topologías no vistas en entrenamiento (tamaño, número de habitaciones, pasillos largos, simetrías)?
- **RQ4 – Robustez y transferencia:** ¿cuánto se degrada con ruido de sensores/odometría y al pasar a robot real (sim-to-real gap) comparado con los clásicos?
- **RQ5 – Coste:** ¿cuál es el coste computacional en ejecución y el coste de entrenamiento?
- (Opcional) **RQ6 – Semántica:** si la política usa la capa semántica, ¿mejora la detección y localización de objetos/habitaciones?

---

## 2. Diseño experimental para una comparación justa

### 2.1 Arquitectura común

```
 Gazebo / robot ──► Graph SLAM (común) ──► /map, /tf, covarianzas del grafo
                                   │
                                   ▼
               ┌──────── selector de objetivos (lo que se compara) ────────┐
               │  DRL | nearest frontier | utility frontier | RRT | IG | D-opt │
               └───────────────────────────────────────────────────────────┘
                                   │ goal (x, y, θ)
                                   ▼
                         Nav2 (planner + controller comunes)
```

- Implementa cada baseline como un nodo con la **misma interfaz** (p. ej. una acción `SelectGoal` o publicar `NavigateToPose`). Así el lanzador de experimentos solo cambia un parámetro.
- **Si tu DRL emite velocidades directamente** (end-to-end), no pasa por Nav2. Entonces: (a) usa los mismos límites de velocidad/aceleración del Create 3 (≈0,3 m/s) en todos, (b) reporta colisiones y contactos de los bumpers por separado, y (c) considera una variante "DRL elige objetivo + Nav2" como ablación para separar *qué decide* de *cómo se mueve*.

### 2.2 Controlar los factores de confusión

| Factor | Cómo controlarlo |
|---|---|
| SLAM | Mismo back-end y parámetros. Ejecutar además con **pose oráculo**. |
| Navegación local | Mismo Nav2 (costmaps, inflado, controlador, recoveries). |
| Sensores | Mismo alcance LiDAR, ruido y FOV de cámara. |
| Criterio de parada | Común (ver 2.3). |
| Ajuste de hiperparámetros | Ajustar baselines en mapas de **validación** con un presupuesto comparable al esfuerzo de ajuste del DRL; documentarlo. |
| Pose inicial | Misma lista de poses iniciales (semillas) para todos → permite pruebas **pareadas**. |
| Aleatoriedad | Fijar semillas de Gazebo, ruido y planificadores; registrar todo. |

### 2.3 Criterio de terminación (idéntico para todos)

Termina el episodio cuando ocurra lo primero de:
1. Cobertura ≥ 95 % del **área alcanzable** (calculada sobre el mapa real con flood-fill desde la pose inicial, inflando por el radio del robot).
2. El método declara que no quedan objetivos (sin fronteras válidas). Registrar si lo hizo de forma prematura.
3. Presupuesto máximo de tiempo `T_max` (p. ej. 2–3 × el tiempo que tarda el mejor baseline en el mapa) o de distancia.
4. Fallo: atasco > 60 s, colisión grave o divergencia del SLAM (ATE > umbral). Contar como **fracaso**, no descartar.

---

## 3. Técnicas de base (baselines)

Recomiendo un conjunto que cubra el espectro *heurístico → geométrico → informativo → orientado a incertidumbre*, porque tu DRL probablemente se compare de forma distinta con cada familia.

| ID | Técnica | Qué representa | Referencia | Implementación práctica |
|---|---|---|---|---|
| **B0** | Paseo aleatorio / seguimiento de pared | Cota inferior | — | Trivial; útil para normalizar resultados |
| **B1** | Frontera más cercana | Exploración clásica por fronteras | Yamauchi, CIRA 1997 | Fácil sobre `/map` |
| **B2** | Frontera con utilidad `U = IG − λ·coste` | Fronteras + ganancia de información geométrica | González-Baños & Latombe, IJRR 2002; `explore_lite` (m-explore) | Hay ports a ROS 2 de `m-explore` (`explore_lite`) |
| **B3** | RRT-exploration | Detección de fronteras por muestreo; robusta en mapas grandes | Umari & Mukhopadhyay, IROS 2017 | `rrt_exploration` (existen ports ROS 2) |
| **B4** | Ganancia de información con incertidumbre de pose | IG "de verdad" del A-SLAM: reducción esperada de entropía de mapa **y** trayectoria | Stachniss, Grisetti & Burgard, RSS 2005 | Adaptar a Graph SLAM: entropía del mapa + término de covarianza de pose; opcional MI rápida tipo CSQMI (Charrow et al., RSS 2015) |
| **B5** | Active Graph-SLAM por optimalidad D | Criterio de optimalidad sobre el grafo de poses (equilibrio exploración / revisita) | Placed & Castellanos (RA-L 2022–2023, conectividad del grafo ≈ D-opt); survey Placed et al., T-RO 2023 | Utilidad = IG geométrica + ΔD-opt estimada vía Laplaciano ponderado del grafo |

Opcionales según dónde quieras publicar:

- **B6 – NBV por muestreo (Next-Best-View)**, Bircher et al., ICRA 2016 (NBVP), adaptado a 2D. Es un baseline muy citado en exploración.
- **B7 – Otro método aprendido**, si hay código disponible: p. ej. Niroui et al. (RA-L 2019, DRL + fronteras) o Li et al. (DRL para exploración automática en mapas 2D, con dataset HouseExpo), o Active Neural SLAM (Chaplot et al., ICLR 2020) como referencia conceptual. Un revisor suele preguntar "¿y contra otro DRL?".
- **B8 – Tu propia función de utilidad semántica** (`U = w_e·IG_geom + w_s·IG_sem + w_l·LC − w_c·coste`, del diseño de la capa semántica) como baseline clásico "fuerte" con la misma información que ve el DRL. Es la comparación más justa si tu DRL usa semántica.

**Mínimo viable:** B0, B1, B2, B4, B5. Si vas justo de tiempo, B3 puede sustituir a B2 (ambos son geométricos).

---

## 4. Métricas

### 4.1 Eficiencia de exploración (RQ1)

| Métrica | Definición | Notas |
|---|---|---|
| Curva cobertura–tiempo y cobertura–distancia | % área alcanzable conocida (libre + ocupada correctamente) vs t y vs d | **La figura principal.** Mostrar mediana + banda IQR o IC95 |
| T₉₀, T₉₅ | Tiempo hasta 90 / 95 % de cobertura | Censurado si no se alcanza (tratar como fracaso / `T_max`) |
| D₉₀, D₉₅ | Distancia recorrida hasta 90 / 95 % | Menos sensible a la velocidad del controlador |
| AUC normalizada | Área bajo la curva cobertura–tiempo / `T_max` | Resume toda la curva en un número |
| Cobertura final | % al terminar | |
| Eficiencia | m² explorados por metro recorrido | |
| Rotación total | Σ|Δθ| | Relevante con cámara de FOV estrecho y para el coste energético |
| Revisitas / oscilaciones | Nº de veces que se vuelve a una celda ya visitada; cambios de objetivo bruscos | Las políticas DRL a veces oscilan |

### 4.2 Calidad del mapa (RQ2)

Comparando con el mapa de ocupación real (generado desde el mundo Gazebo, o plano medido en real):

- **IoU** de celdas ocupadas y de celdas libres; precisión/recall de ocupadas.
- **ADNN** (distancia media al vecino más cercano entre celdas ocupadas estimadas y reales): sensible a paredes dobles por deriva.
- **Entropía del mapa** final y su evolución.
- Detección de "mapas corruptos" (paredes duplicadas), registrar como fallo cualitativo.

### 4.3 Calidad del SLAM / incertidumbre (RQ2)

- **ATE** (RMSE) y **RPE** frente a ground truth (Gazebo: plugin de pose o `/model_states`; real: motion capture si tienes acceso, o marcadores/posiciones conocidas en el suelo). Herramienta: `evo`.
- **D-optimalidad** (o log-det de la covarianza marginal) de la pose a lo largo del tiempo, y valor final.
- Nº de cierres de bucle aceptados, nº de cierres falsos (si tienes GT se pueden detectar).
- Error máximo de pose durante el episodio (no solo el final).

### 4.4 Robustez y seguridad (RQ4)

- **Tasa de éxito** (llegó a 95 % sin fallo) con IC de Wilson.
- Colisiones (en Create 3: `hazard_detection` / bumpers; en Gazebo: contactos).
- Recuperaciones de Nav2, atascos, tiempo parado.
- Terminación prematura ("cree que ha terminado" con cobertura < 90 %).

### 4.5 Coste computacional (RQ5)

- Tiempo de decisión por paso (media, p95), frecuencia de decisión.
- CPU / RAM / GPU en el ordenador que vaya en el robot.
- Coste de entrenamiento del DRL: pasos, horas de reloj, hardware. Ponerlo en la tabla, aunque no se compare directamente.

### 4.6 Métricas semánticas (opcional, RQ6)

Si tu política o el sistema usa la capa semántica:
- **Recall de objetos**: % de objetos reales detectados y localizados a < 0,5 m (con clase correcta).
- Error medio de posición de landmarks; precisión de clase.
- Tiempo hasta encontrar un objeto/habitación objetivo (si la tarea incluye navegación semántica).
- Precisión del etiquetado de habitaciones.

### 4.7 Cómo agregar entre mapas

Los valores absolutos dependen mucho del tamaño del mapa. Recomiendo:
- Normalizar T₉₀/D₉₀ por el del mejor baseline de ese mapa (o por B1), y reportar **ratios**.
- **Perfiles de rendimiento** (Dolan & Moré, 2002): fracción de problemas en los que cada método está dentro de un factor τ del mejor. Muy útil con muchos mapas.

---

## 5. Escenarios

### 5.1 Simulación (Gazebo)

- **Partición estricta**: mapas de entrenamiento (DRL), validación (ajuste de hiperparámetros de todos) y **test** (nunca vistos).
- **Fuentes de mapas**:
  - Planos 2D de **HouseExpo** (≈35 k planos de viviendas) extruidos a mundos Gazebo: diversidad enorme y fácil de partir.
  - Mundos estándar: AWS RoboMaker *small house*, *bookstore*, *hospital*; mundos del TurtleBot 3/4. Útiles porque otros autores los usan.
  - 2–3 mundos generados a medida para casos "difíciles" (ver abajo).
- **Clasificar los mapas de test por complejidad** y reportar por categoría:
  - Tamaño (pequeño < 50 m², medio, grande > 200 m²).
  - Topología: pocas habitaciones / muchas habitaciones / pasillos largos.
  - Desorden (obstáculos, muebles con patas que el LiDAR ve mal).
  - **Aliasing perceptual** (simetrías, pasillos repetitivos): castiga a métodos que no gestionan la incertidumbre y es donde B4/B5 deberían brillar.
  - Espacios abiertos mayores que el alcance del LiDAR.
- **Perturbaciones** (para RQ4): ruido de odometría ×1, ×2, ×4 (deslizamiento), ruido LiDAR, obstáculos dinámicos (personas), puertas cerradas.
- **Poses iniciales**: ≥ 10 por mapa, muestreadas una vez y compartidas por todos los métodos.

### 5.2 Robot real

- 2–3 entornos: laboratorio con muebles, pasillo largo, y si es posible un entorno tipo vivienda.
- Métodos: DRL + los 2 baselines más fuertes en simulación (+ B1 como referencia barata).
- 3–5 ejecuciones por método y entorno, con las mismas poses iniciales (marcar en el suelo).
- Ground truth: motion capture si existe; si no, plano medido + puntos de control (marcadores/AprilTags en posiciones medidas) para estimar ATE en esos puntos.
- Registrar todo en rosbag2 para recalcular métricas offline.

---

## 6. Protocolo y automatización

1. **Fichero de experimento YAML**: `{método, mapa, pose_inicial, semilla, nivel_ruido, modo_localización}`. Un script genera la matriz completa.
2. **Lanzador headless** (Gazebo sin GUI) que ejecuta cada configuración, aplica el criterio de parada y guarda: `/map` periódico (p. ej. cada 5 s), `/tf`, pose GT, odometría, objetivos elegidos, eventos de Nav2, uso de CPU, y la salida del grafo (covarianzas).
3. **Cálculo de métricas offline** a partir de los bags (Python): así puedes añadir métricas sin repetir experimentos.
4. **Tamaño de la campaña** (orientativo): 6 métodos × 15 mapas test × 10 poses × 2 modos de localización ≈ 1800 episodios. A ~10 min por episodio en tiempo real son ~300 h → paralelizar varias instancias de Gazebo (contenedores Docker con `ROS_DOMAIN_ID` distinto) y usar `real_time_factor` > 1 si la simulación lo aguanta. Empieza con un subconjunto piloto.
5. **Piloto**: 2 mapas, 3 métodos, 5 ejecuciones. Sirve para validar el pipeline y estimar la varianza → decidir N definitivo con un análisis de potencia sencillo.

---

## 7. Análisis estadístico

- Reportar **mediana e IQR** (las distribuciones de tiempos suelen ser asimétricas y con colas), además de media ± desviación si la comunidad lo espera.
- **Comparación DRL vs cada baseline**: prueba de **Wilcoxon de rangos con signo pareada** por (mapa, pose inicial). Corrección de **Holm** por comparaciones múltiples.
- **Comparación global entre todos los métodos**: **Friedman + post-hoc de Nemenyi** sobre los rangos por mapa, con *critical difference diagram* (Demšar, JMLR 2006).
- **Tamaño del efecto**: Cliff's δ o Vargha–Delaney A₁₂ (más informativo que el p-valor con N grande).
- **Tasas de éxito**: IC de Wilson; comparar con prueba de McNemar (pareada).
- **Datos censurados** (no alcanzan T₉₅): o bien tratarlos como `T_max` y decirlo, o usar curvas de supervivencia tipo Kaplan–Meier "probabilidad de no haber alcanzado el 95 % en t". Lo segundo es elegante y poco usado.
- **Variabilidad del DRL**: entrenar **≥ 3–5 semillas** y evaluar todas; reportar la variación entre semillas (Henderson et al., AAAI 2018 "Deep RL that matters"; Agarwal et al., NeurIPS 2021, *rliable*, IQM con IC por bootstrap). Elegir "la mejor semilla" sin decirlo es la crítica más común de los revisores.

Herramientas: `scipy.stats`, `scikit-posthocs`, `rliable`, `evo`, `lifelines` (Kaplan–Meier).

---

## 8. Ablaciones del método DRL

Para explicar **por qué** funciona (y no solo que funciona):

- Sin el término de incertidumbre / cierre de bucle en la recompensa (si lo tiene).
- Variaciones en la observación: solo mapa local vs mapa global; con y sin covarianza de pose; con y sin semántica.
- Espacio de acción: elegir frontera (discreto) vs objetivo continuo vs velocidades.
- Entrenado con pose oráculo vs con SLAM en el bucle (afecta mucho a la transferencia).
- Curriculum / tamaño del conjunto de entrenamiento: rendimiento en test en función de nº de mapas de entrenamiento (curva de generalización).
- Robustez a cambios de hardware: alcance del LiDAR, velocidad máxima.

---

## 9. Figuras y tablas sugeridas

1. Curvas cobertura–tiempo (y –distancia) por categoría de mapa, con bandas.
2. Boxplots de T₉₅, D₉₅, ATE y IoU por método.
3. Diagrama de diferencia crítica (Friedman/Nemenyi).
4. Perfil de rendimiento (Dolan–Moré) sobre todos los mapas de test.
5. Tabla resumen: método × {éxito, T₉₅, D₉₅, AUC, ATE, IoU, t_decisión}.
6. Mapas finales y trayectorias de ejemplo (mismo mapa y pose inicial) superpuestos al mapa real.
7. Evolución de D-opt / entropía de pose en el tiempo (muestra el compromiso exploración–explotación).
8. Barra sim vs real por método (sim-to-real gap).

---

## 10. Plan de actuación por fases

| Fase | Duración orientativa | Entregable |
|---|---|---|
| **F0 – Infraestructura** | 2 semanas | Interfaz común de selector de objetivos, lanzador headless, criterio de parada, generación de mapas GT y área alcanzable, logging |
| **F1 – Baselines** | 2–3 semanas | B0–B5 funcionando sobre tu SLAM y Nav2, ajustados en validación |
| **F2 – Piloto** | 1 semana | 2 mapas × 3 métodos × 5 ejecuciones; pipeline de métricas validado; estimación de varianza y N |
| **F3 – Campaña en simulación** | 2–4 semanas (de cómputo, en paralelo con F4) | Todos los métodos × mapas test × poses × modos de localización; perturbaciones |
| **F4 – Ablaciones DRL** | 1–2 semanas | Variantes de la sección 8 |
| **F5 – Robot real** | 2 semanas | 2–3 entornos, 3–5 ejecuciones por método |
| **F6 – Análisis y redacción** | 2 semanas | Estadística, figuras, discusión de limitaciones |

---

## 11. Riesgos y recomendaciones

- **El SLAM puede dominar los resultados.** Si el Graph SLAM falla en pasillos simétricos, todos los métodos salen mal y la comparación pierde poder. Por eso el modo con **pose oráculo** es importante: separa "decide bien" de "el SLAM aguantó". Reporta los fallos de SLAM por separado.
- **Sobreajuste a la distribución de mapas.** Si entrenaste en HouseExpo, incluye en test mapas de otro origen (AWS, laboratorio real). Si el DRL solo gana dentro de su distribución, es un resultado honesto pero hay que decirlo.
- **Ajuste desigual.** Los baselines mal parametrizados son la crítica más frecuente. Documenta el ajuste y, si puedes, publica las configuraciones.
- **Velocidad del Create 3.** Es lenta (≈0,3 m/s); las diferencias en tiempo pueden quedar dominadas por rotaciones in situ. Reporta también distancia y rotación total.
- **FOV de la cámara.** Si la semántica entra en juego, el orientado final del objetivo importa; asegúrate de que los baselines con semántica (B8) también eligen orientación.
- **Sim-to-real.** Añade ruido realista en simulación (deslizamiento en giros, reflejos/huecos en el LiDAR, latencia de YOLO de 30–100 ms) antes de concluir nada sobre transferencia.
- **Reproducibilidad.** Versiona mapas, semillas, configuraciones y pesos de la política; publica el benchmark si es posible: aumenta el impacto del trabajo.

---

## 12. Bibliografía de partida (verificar antes de citar)

- B. Yamauchi, "A frontier-based approach for autonomous exploration", CIRA 1997.
- H. González-Baños, J.-C. Latombe, "Navigation strategies for exploring indoor environments", IJRR 2002.
- C. Stachniss, G. Grisetti, W. Burgard, "Information gain-based exploration using Rao-Blackwellized particle filters", RSS 2005.
- H. Umari, S. Mukhopadhyay, "Autonomous robotic exploration based on multiple rapidly-exploring randomized trees", IROS 2017.
- B. Charrow et al., "Information-theoretic planning with trajectory optimization for dense 3D mapping" (CSQMI), RSS 2015.
- A. Bircher et al., "Receding horizon next-best-view planner for 3D exploration", ICRA 2016.
- J. A. Placed, J. A. Castellanos, trabajos sobre optimalidad D y conectividad del grafo en Active Graph-SLAM (RA-L, 2022–2023).
- J. A. Placed et al., "A survey on Active Simultaneous Localization and Mapping: State of the art and new frontiers", IEEE T-RO 2023.
- F. Niroui et al., "Deep reinforcement learning robotic exploration in cluttered urban environments", RA-L 2019.
- H. Li et al., "Deep reinforcement learning-based automatic exploration for navigation in unknown environment" (HouseExpo), IEEE TNNLS 2019.
- D. S. Chaplot et al., "Learning to explore using Active Neural SLAM", ICLR 2020.
- J. Demšar, "Statistical comparisons of classifiers over multiple data sets", JMLR 2006.
- P. Henderson et al., "Deep reinforcement learning that matters", AAAI 2018.
- R. Agarwal et al., "Deep reinforcement learning at the edge of the statistical precipice", NeurIPS 2021.
- E. Dolan, J. Moré, "Benchmarking optimization software with performance profiles", Math. Programming 2002.
