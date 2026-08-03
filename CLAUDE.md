# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Thesis project by Isaac Arias Marín: detection of buried anti-personnel mines using multispectral imagery from a MicaSense RedEdge-MX camera. Detection is based on vegetation stress and soil anomalies caused by buried objects. **Dataset is closed: S2–S7 (6 sessions, 14,901 patches). Final classifier: EasyEnsemble x10 XGBoost, top-70 features (XGBoost-selected), LOSO — AUC=0.947, Sens(0.5)=0.823, ~4.2 FP/mina.** The thesis document (`Doc_Final/tesis_isaac_v6final2/`, the "thesisV2" rewrite) presents ONLY the bounding-box labeling; the old "Enfoque A" coarse-labeling baseline was removed at the advisors' request. Canonical results, figures and scripts live in `thesisV2/` (`thesisV2/REGISTRO.md` has all P1–P9 numbers). The Enfoque A/B sections further below are historical.

## Estado actual (canónico)

**Documento entregado; en espera de las correcciones de los evaluadores.** Las correcciones
que tocan el documento se acumulan en `correcciones_futuras.md` (se aplican en batch, no una
por una). La sustentación se prepara en `Expo/` (`Sustentacion_v2.pptx` + diagramas). El
documento vive en `Doc_Final/tesis_isaac_v6final2/` (compila con `latexmk -pdf main.tex`).

**Números canónicos (fuente de verdad = CSV en `thesisV2/resultados/`):**
- Dataset: S2–S7, 14 901 ventanas, 487 positivas / 14 414 negativas (30:1).
- Modelo final: EasyEnsemble x10 XGBoost, top-70 (XGBoost-gain), submuestreo 1:1, LOSO.
- **AUC = 0.947 ± 0.007 · Sens(0.5) = 0.823 · Spec(0.5) = 0.884 · FP/mina = 4.16 (~4.2).**
- Confusión (LOSO, top-70): TP 401 · FN 86 · FP 1670 · TN 12744.
- AUC por fold: S2 0.936 · S3 0.946 · S4 0.942 · S5 0.950 · S6 0.958 · S7 0.948.
- **LOZO** (deja un emplazamiento físico fuera, top-70): AUC **0.933** → generaliza a sitios no vistos.
- Clasificadores (378 feat, LOSO): RF 0.915 · SVM 0.929 · XGBoost 0.941 · EasyEnsemble 0.945.
- Selección de k: criterio 99.5 % → k=60; se elige **k=70** (Pareto). Barrido en `p4_barrido_k.csv`.
- Profundidad: Sens 87.5 % (1 cm) → 70.7 % (7 cm); AUC estable ~0.91–0.94.
- Falsos positivos: 93 % sobre fondo (de 1363 FP en zonas de mina); objetos solo 6.8 %.
- Importancia (§4.3, RF-Gini top-20, reproducible en `p2_top20_importancia.csv`): 12 de 20
  involucran NIR/Red Edge; top-5 = Red_mean, SI_std, GRVI_p25, RENDVI_std, VARI_p25.
- FP/mina se define como FP/TP (por mina **detectada**), consistente con el documento (§3.6.4).

## Commands

```bash
# Activate virtual environment
.venv\Scripts\activate

# Install processing dependencies
pip install -r process/requirements.txt

# Run full pipeline (preprocessing → feature extraction → classification)
python process/pipeline_completo.py --ruta_datos "FINAL_DATASET/SESION_N"

# Run a single pipeline step (1=preproc, 2=features, 3=clasif)
python process/pipeline_completo.py --ruta_datos "FINAL_DATASET/SESION_N" --paso 1

# Run individual steps directly
python process/preprocesamiento.py --ruta "FINAL_DATASET/SESION_N"
python process/extraccion_features.py --ruta "FINAL_DATASET/SESION_N/PROCESADAS"
python process/clasificador_rf.py --dataset "FINAL_DATASET/features_combinado.csv"

# Combine sessions and evaluate (honest LOSO validation)
python tests/combinar_datasets.py
python tests/evaluar_loso.py

# Test scripts (single image, quick verification)
python tests/test_preprocesamiento.py
python tests/test_extraccion.py
python tests/ver_calibrado.py "FINAL_DATASET/SESION_N/PROCESADAS/ZONA_CONTROL_calibrado.tif"

# Capture (requires camera WiFi or mock server)
python tests/micasense_mock_server.py                                     # Terminal 1 (simula camara sin hardware)
python micasense_preview_capture.py --base http://localhost:8080 --session test  # Terminal 2
python micasense_preview_capture.py --session SESION_4                   # campo (camara real)
```

Pipeline outputs land in `SESION_N/PROCESADAS/`: `*_calibrado.tif` (one per zone), `features_dataset.csv`. Combined dataset: `FINAL_DATASET/features_combinado.csv`. Model: `FINAL_DATASET/modelo_rf_loso.joblib`.

There are no automated tests. Validation is done by running the pipeline on real data and checking LOSO metrics.

## Analysis Scripts (all completed — dataset closed S2–S7)

These scripts in `tests/clasificadores/` were run once on the complete dataset:

| Script | Output | Purpose |
|---|---|---|
| `extraccion_segmentada.py` | `features_combinado_segmentado.csv` | Enfoque B re-labeling with Roboflow bboxes |
| `analisis_falsos_positivos.py` | `FINAL_DATASET/analisis_fp/` | FP location vs interference objects |
| `analisis_por_profundidad.py` | `FINAL_DATASET/figuras/auc_vs_profundidad.png` | AUC by burial depth (1/3/5/7 cm) |
| `seleccion_features_enfoqueB.py` | `tests/clasificadores/resultados_seleccion_features_enfoqueB.csv` | Optimal feature count with RF (historical, k=100) |
| `seleccion_features_xgb_easy.py` | `FINAL_DATASET/figuras/seleccion_features_xgb.png` | Optimal k with XGBoost+EasyEnsemble — **historical (k=50); thesisV2 final: k=70** |
| `comparacion_clasificadores.py` | `tests/clasificadores/comparacion_clasificadores.csv` | RF vs XGBoost vs SVM |
| `validacion_splits.py` | `FINAL_DATASET/figuras/comparacion_splits.png` | LOSO leakage proof: compares 4 validation schemes (random/LOSO/zone/temporal) |
| `generar_figuras_tesis.py` | `FINAL_DATASET/figuras/auc_evolucion_enfoqueB.png`, `comparacion_enfoques.png` | Thesis-ready figures |
| `generar_pipeline_enfoque_b.py` | `FINAL_DATASET/pipeline_enfoque_b.png` | Pipeline diagram with segmentation stage |

**Roboflow V3 annotations** (4 classes: MAP id=3, Botella id=1, Lata id=2, Piedra id=4) at `FINAL_DATASET/referencia_roboflow/minas-groundtruth.v3i.coco/`. 56 images (S1–S7), 224 annotations.

## Feature Selection Experiment

**Final decision (thesisV2): k=70, XGBoost selects its own features, random 1:1 undersampling per fold (no object quota).** Using RF importances to feed a different classifier (XGBoost) was flagged as methodologically inconsistent. The pipeline uses XGBoost gain-based importance for selection. The k=50 table below is from the original run; thesisV2 re-ran the sweep without the object quota and chose **k=70** as the Pareto-optimal point on the plateau (dominates k=60 in AUC/Sens/FP-mina), prioritizing sensitivity.

Script: `tests/clasificadores/seleccion_features_xgb_easy.py` — barrido k=10/20/50/100/150/200/all with EasyEnsemble x10 LOSO.

| k | AUC | Sens(0.5) | FPR | FP/mina |
|---|---|---|---|---|
| 10 | 0.900 | 0.807 | 0.156 | 5.7 |
| 20 | 0.921 | 0.842 | 0.147 | 5.2 |
| **50** | **0.930** | **0.840** | **0.127** | **4.5** |
| 100 | 0.935 | 0.840 | 0.121 | 4.3 |
| 200 | 0.934 | 0.803 | 0.116 | 4.3 |
| all (378) | 0.930 | 0.815 | 0.118 | 4.3 |

**Historical (original run): k=50 chosen.** AUC plateaus between k=50 and k=200. Directors flagged k=100 as too many. **Superseded by thesisV2: k=70** (see note above).

Top-5 features (RF-Gini, §4.3, reproducible en `p2_top20_importancia.csv`): Red_mean, SI_std, GRVI_p25, RENDVI_std, VARI_p25.

**Low-contribution indices (bottom 40% — 17 indices, kept in pipeline):**
EVI, NIR_Red, GRVI, ExGR, CI_soil, GNDVI, NDWI, RENDVI, NIR_Blue, Red_Blue, RENI, RedEdge_Red, Red_Green, CIVE, MSAVI, NIR_Green, NIR_RedEdge. Removing them reduces features to 225 but costs -0.005 AUC; kept for robustness.

Figure: `FINAL_DATASET/figuras/seleccion_features_xgb.png`

## Segmented Groundtruth Experiment

Motivated by advisor observation that top features were all visible-spectrum (VARI, NDRB, HI) with no NIR or Red Edge. Hypothesis: coarse labeling (entire mine-zone image = mina=1) masked the spectral stress signal in those bands.

**Method:** 40 training images annotated in Roboflow with bounding boxes (class MAP, fixed size ~212×199 px — all mines same physical size at 1m height). COCO JSON exported to `FINAL_DATASET/referencia_roboflow/minas-groundtruth.v1i.coco/`. Patches re-labeled: center inside bbox → mina=1, otherwise → mina=0.

**Script:** `tests/clasificadores/extraccion_segmentada.py` → `FINAL_DATASET/features_combinado_segmentado.csv`

**Results (LOSO, S2–S7, 378 features):**
- AUC = **0.913** (vs 0.882 Enfoque A) — +0.031 improvement
- Class balance: 487 positive vs 14,414 negative (30:1 ratio)
- **NIR and Red Edge in top-10** — advisor's hypothesis confirmed
- Top-5 (S2–S7): Red_mean, SI_std, RENDVI_std, NDVI_std, GRVI_p25

**Reference images** for Roboflow annotation: `FINAL_DATASET/referencia_roboflow/UBI/` (plunger visible, band-aligned) and `referencia_roboflow/entrenamiento/` (calibrated RGB for annotation).

## Architecture

Two independent layers communicate through files on disk:

**Capture layer** (root directory): Scripts that talk to the MicaSense camera HTTP API (`http://192.168.10.254`) to capture and download 5-band multispectral TIF images. `micasense_utils.py` holds shared HTTP/download logic used by both `micasense_capture_cli.py` and `micasense_preview_capture.py`. `micasense_mock_server.py` simulates the camera API for testing without hardware.

**Processing layer** (`process/`): Three-stage pipeline orchestrated by `pipeline_completo.py`:
1. `preprocesamiento.py` — Radiometric calibration using CRP panel (shared mask detected on Red band, applied to all 5 bands), band alignment (SIFT+RANSAC with ECC fallback), border crop, reflectance conversion. Outputs calibrated multi-band GeoTIFF.
2. `extraccion_features.py` — Slides 128×128 patches (50% overlap) over each calibrated image, computes 40+ spectral indices (vegetation, soil, humidity, ratios), extracts 9 statistics per index per patch. Each patch becomes one row in the CSV dataset.
3. `clasificador_rf.py` — Trains Random Forest (100 trees, balanced classes), evaluates with 5-fold stratified CV. For honest evaluation use `tests/evaluar_loso.py` instead. Once 6+ sessions are available, compare against XGBoost and SVM (see Classifier Strategy below).

Labels come automatically from folder names: any folder containing 'MINA' → `tiene_mina=1`; anything else → `tiene_mina=0`.

## Key Design Decisions

- **Patch-based extraction** generates ~216 training samples per image (one row per 128×128 patch with 50% overlap on 1185×840px images), making RF training viable with few captures.
- **No feature scaling**: Random Forest is scale-invariant, so no StandardScaler is used.
- **Band alignment**: Red band (band 3) is reference. Panel detection uses Red band mask shared across all 5 bands — critical because Red Edge (band 4) vegetation is too bright for independent Otsu detection.
- **Honest validation**: use Leave-One-Session-Out (LOSO) via `tests/evaluar_loso.py`, not random split. Random split causes data leakage because overlapping patches from the same image end up in both train and test.
- `opencv-contrib-python` (not `opencv-python`) is required because SIFT is in the contrib package.
- Recommended validation split when enough sessions exist: train on days 1–3, test on day 4.

## Session Protocol

**Official protocol starts at SESION_2.** SESION_1 is permanently excluded from all analyses. Rationale: captured ~09:40 (low sun angle, possible dew), only 1 control zone, AUC≈0.47 as test fold (worse than random). `combinar_datasets.py` enforces this via `SESIONES_EXCLUIDAS = [1]`.

## Model Strategy

**Enfoque A (coarse labeling):** entire mine-zone image labeled mina=1. Used as methodological baseline only — motivates the need for precise labeling. Dataset: `features_combinado.csv`.

**Enfoque B (segmented groundtruth) — OFFICIAL FINAL MODEL:** patches re-labeled using Roboflow bounding box annotations. Only patches whose center falls inside the bbox → mina=1. Dataset: `features_combinado_segmentado.csv`. All future analyses use this approach.

**Final results (thesisV2 — bounding-box labeling, S2–S7, 14,901 patches, 487 positive):**
- **FINAL MODEL — EasyEnsemble x10 XGBoost, top-70 features (XGB-selected), random 1:1 undersampling, LOSO:**
  - AUC = **0.947**, Sens(0.5) = **0.823**, Spec(0.5) = 0.884, ~4.2 FP/mina
  - TP=401, FN=86, FP=1670, TN=12744
  - Per fold AUC: S2=0.936, S3=0.946, S4=0.942, S5=0.950, S6=0.958, S7=0.948
- (Historical, original run) RF top-50 with object quota: LOSO AUC RF = 0.913; EasyEnsemble top-50 = 0.931 / Sens 0.832
- Top-5 features (RF-Gini, §4.3): Red_mean, SI_std, GRVI_p25, RENDVI_std, VARI_p25

## Dataset Status

**CLOSED. No more sessions will be captured.** Official dataset: S2–S7 (6 sessions). All analyses complete. See `docs/resultados_finales.md` for full results.

## Classifier Strategy (COMPLETED)

Initial comparison run on Enfoque B, top-100 features (RF-selected), LOSO S2–S7:

| Classifier | Mean AUC | Std | Notes |
|---|---|---|---|
| Random Forest | 0.929 | 0.019 | Baseline |
| XGBoost (base) | 0.945 | 0.016 | scale_pos_weight=30 |
| SVM (RBF) | 0.937 | 0.019 | StandardScaler per fold |

**Final classifier: EasyEnsemble x10 XGBoost** — addresses 30:1 class imbalance without Youden threshold adjustment.

| Strategy | AUC | Sens(0.5) | FP/mina |
|---|---|---|---|
| XGBoost base (scale_pos_weight=30) | 0.945 | 0.534 | — |
| XGBoost undersampling 1:1 | 0.930 | 0.817 | 4.7 |
| **EasyEnsemble x10 XGBoost** | **0.931** | **0.832** | **4.4** |

EasyEnsemble trains 10 XGBoost classifiers, each with all 487 positives + a different random subset of 487 negatives (1:1 balance). Probabilities are averaged. Achieves Sens=0.832 at threshold 0.5 without Youden adjustment — more practical for deployment.

Scripts: `tests/clasificadores/comparacion_clasificadores.py`, `tests/clasificadores/experimento_smote_ensemble_umbral.py`, `tests/clasificadores/validacion_splits.py`

## Known Issues and Calibration Notes

- **Red Edge (band 4) calibration**: Vegetation is almost as bright as the calibration panel in the 717nm band, making independent panel detection unreliable. The shared-mask approach (detect panel in Red band, apply to all bands) partially solves this. NDRE values remain slightly biased but consistent across zones.
- **Band misalignment at 1m**: Parallax between the 5 physical lenses is significant at 1m capture height. SIFT corrects the global shift but 3D vegetation causes residual color fringing. This is inherent to close-range capture and acceptable for patch-level statistical features.
- **Class imbalance**: 8 mine zones vs 1–4 control zones per session. RF uses `class_weight='balanced'`. From SESION_3 onward, capture 4 control zones per session (2:1 ratio) — this was a key factor in improving AUC from 0.52 to 0.755.
- **SESION_1 excluded (permanent)**: Captured ~09:40 (low sun angle, possible dew), only 1 control zone, AUC≈0.47 as test fold. Officially excluded from the protocol. `combinar_datasets.py` skips it via `SESIONES_EXCLUIDAS = [1]`. The official dataset is SESION_2 onward.

## Border Crop Configuration

Asymmetric crop applied after band alignment to remove zero-padding artifacts:
```python
'RECORTE': {'top': 20, 'bottom': 60, 'left': 35, 'right': 20}
```
Bottom and left are larger because warpPerspective zero-padding concentrates there.

## Data Structure Convention

```
FINAL_DATASET/
├── SESION_N/
│   ├── PANEL_INICIO/        # 5 TIF files (calibration panel)
│   ├── MINA_1CM_1/          # 5 TIF files per zone
│   ├── MINA_1CM_2/
│   ├── MINA_3CM_1/, MINA_3CM_2/
│   ├── MINA_5CM_1/, MINA_5CM_2/
│   ├── MINA_7CM_1/, MINA_7CM_2/
│   ├── ZONA_CONTROL/
│   ├── ZONA_CONTROL_2/      # recommended: use 4 control zones from SESION_3 onward
│   ├── ZONA_CONTROL_3/
│   ├── ZONA_CONTROL_4/
│   ├── PANEL_FINAL/
│   └── PROCESADAS/          # generated by pipeline
│       ├── *_calibrado.tif
│       └── features_dataset.csv
├── SESION_N_UBI/            # reference only, not used for training
├── features_combinado.csv   # sessions S2–S5+ combined (SESION_1 excluded), includes 'sesion' column
├── modelo_rf_loso.joblib
└── resultados_loso.png
```

Each zone folder contains exactly 5 TIF files named `cap_0001_band1.tif` … `cap_0001_band5.tif` (Blue 475nm, Green 560nm, Red 668nm, Red Edge 717nm, NIR 842nm). Panel albedo values: [0.490, 0.491, 0.491, 0.488, 0.490].

## Capture Sessions and FINAL_DATASET Organization

Raw captures are saved to `captures/` with auto-generated timestamps. After each field day, captures are copied (never moved) into `FINAL_DATASET/` with clean names.

### Name mapping (captures/ → FINAL_DATASET/)
| captures/ folder            | FINAL_DATASET/ destination      |
|-----------------------------|----------------------------------|
| PANEL_INICIO_*              | SESION_N/PANEL_INICIO/           |
| PANEL_FINAL_*               | SESION_N/PANEL_FINAL/            |
| MINA_7.1_* or MINA_7CM_1_* | SESION_N/MINA_7CM_1/             |
| CONTROL_* or ZONA_CONTROL_* | SESION_N/ZONA_CONTROL/           |
| PANEL_INICIO_UBI_*          | SESION_N_UBI/PANEL_INICIO/       |
| PANEL_FINAL_UBI_*           | SESION_N_UBI/PANEL_FINAL/        |
| UBI_7.1_*                   | SESION_N_UBI/UBI_7CM_1/          |

**Special case:** if UBI photos were taken interleaved with normal captures (no separate UBI panel), copy the same PANEL_INICIO and PANEL_FINAL from SESION_N into SESION_N_UBI.

### Workflow when user asks to organize captures
1. Read `captures/` and identify all sessions present (ignore `captures/REVISADAS/`).
2. Confirm the full mapping with the user before copying anything.
3. Use `cp -r` to copy into `FINAL_DATASET/SESION_N/` — never delete originals.
4. After copying, move the raw capture folders into `captures/REVISADAS/SESION_N/` to keep `captures/` clean. Each session gets its own subfolder under `REVISADAS/`.

## Style

- User-facing messages in **Spanish**
- Capture indexes use 4 digits (`cap_0001`)
- JSON output: UTF-8 with `indent=2`
- HTTP retries: 3 attempts with linear backoff
