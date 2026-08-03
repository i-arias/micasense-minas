# Detección de minas antipersona con imágenes multiespectrales

Sistema de visión artificial para la **detección remota de minas antipersona (MAP)
tipo quiebrapatas enterradas**, a partir de imágenes multiespectrales de una cámara
**MicaSense RedEdge-MX** (5 bandas). En lugar de detectar el artefacto directamente,
el sistema detecta la **anomalía espectral que la mina induce en el suelo y la
vegetación** (estrés vegetal crónico y perturbación del suelo).

> **Trabajo de grado — Ingeniería Electrónica, Universidad del Valle.**
> Grupo de investigación PSI (Percepción y Sistemas Inteligentes).

---

## Resultado principal

Modelo final: **EasyEnsemble x10 XGBoost** con las **70 características** de mayor
ganancia, submuestreo 1:1 y validación **Leave-One-Session-Out (LOSO)** sobre 6
sesiones de campo:

| Métrica | Valor |
|---|---|
| AUC (LOSO) | **0.947 ± 0.007** |
| Sensibilidad (umbral 0.5) | **82.3 %** |
| Especificidad (umbral 0.5) | **88.4 %** |
| Falsos positivos por mina | **~4.2** |
| Generalización a sitios no vistos (LOZO) | AUC **0.933** |

La validación es **honesta**: LOSO evita la fuga de datos entre sesiones (una
partición aleatoria inflaba el AUC a 0.999), y un experimento *leave-one-zone-out*
confirma que el modelo generaliza a **emplazamientos físicos nuevos**, no memoriza
los sitios.

---

## La idea

La mina está enterrada: no se ve. Lo que sí es observable es el **efecto** que
produce en superficie a lo largo del tiempo —estrés en la vegetación y perturbación
del suelo—, que altera la firma espectral captada por la cámara. Por eso el análisis
es **por ventanas** (estadísticas del área), no píxel a píxel: la oclusión puntual
(una hoja, una piedra) no impide la detección porque la señal proviene del entorno
perturbado completo.

---

## Estructura del repositorio

```
.
├── micasense_preview_capture.py   # captura desde la cámara (HTTP API)
├── micasense_utils.py             # utilidades de captura/descarga
├── process/preprocesamiento.py    # calibración radiométrica + alineación de bandas (SIFT)
├── thesisV2/
│   ├── scripts/                   # experimentos finales (comun.py, extraccion_v2.py, p1..p8)
│   ├── resultados/                # CSV con las métricas de cada experimento
│   └── REGISTRO.md                # bitácora canónica (todos los resultados)
├── docs/tesis.pdf                 # documento de tesis (PDF)
├── Doc_Final/tesis_isaac_v6final2/# fuente LaTeX del documento
└── requirements.txt
```

---

## Reproducibilidad

```bash
python -m venv .venv && .venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

El flujo va de las imágenes crudas al modelo final:

```bash
# 1) Calibración radiométrica + alineación de bandas → TIF calibrado por zona
python process/preprocesamiento.py --ruta "FINAL_DATASET/SESION_N"

# 2) Extracción de características (ventana deslizante) → CSV
python thesisV2/scripts/extraccion_v2.py

# 3) Modelo final: EasyEnsemble x10 XGBoost, top-70, validación LOSO
python thesisV2/scripts/p5_modelo_final.py
```

Los experimentos del documento están en `thesisV2/scripts/` (p1–p8) y escriben sus
métricas a `thesisV2/resultados/*.csv`.

> **Nota sobre los datos.** Las imágenes multiespectrales crudas (`FINAL_DATASET/`,
> ~5 GB de GeoTIFF) y las matrices de características extraídas (69 MB c/u) **no se
> incluyen** en el repositorio por su tamaño. Se pueden regenerar con
> `thesisV2/scripts/extraccion_v2.py`, o solicitar al autor. Los CSV con las métricas
> de cada experimento sí están incluidos en `thesisV2/resultados/`.

---

## Metodología en breve

1. **Captura** a 1 m de altura, en posición nadir, con panel de reflectancia calibrada.
2. **Preprocesamiento**: calibración radiométrica + alineación de bandas (SIFT+RANSAC).
3. **Extracción**: ventana deslizante 128×128 px (50 % solapamiento) → 42 índices
   espectrales × 9 estadísticas = **378 características** por ventana.
4. **Etiquetado** por *bounding box*: una ventana es positiva si su centro cae dentro
   del recuadro de la MAP (487 positivas vs 14 414 negativas, desbalance real 30:1).
5. **Clasificación**: comparación de RF / SVM / XGBoost / EasyEnsemble; selección de
   características por ganancia de XGBoost; validación LOSO (+ LOZO).

---

## Documento

El documento completo de la tesis está en **[`docs/tesis.pdf`](docs/tesis.pdf)**
(fuente LaTeX en `Doc_Final/`).

## Autor y créditos

- **Autor:** Isaac Arias Marín
- **Directores:** Sandra E. Nope Rodríguez (Ph.D.), Hermes A. Tenorio Tamayo (M.Sc.)
- **Institución:** Universidad del Valle — Ingeniería Electrónica — Grupo PSI

## Licencia

Código bajo licencia [MIT](LICENSE). El documento de tesis es material académico del autor.
