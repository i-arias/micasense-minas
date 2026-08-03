# thesisV2 — Bitácora y plan de ejecución

> ## ⭐ ESTADO OFICIAL / CANÓNICO (leer esto primero)
>
> **Modelo final:** EasyEnsemble x10 XGBoost · top-70 (XGBoost-gain) · submuestreo random 1:1 (sin cuota) · LOSO.
> **AUC = 0.947 ± 0.007 · Sens(0.5) = 0.823 · Spec(0.5) = 0.884 · FP/mina = 4.16.**
> Confusión: TP 401 · FN 86 · FP 1670 · TN 12744. Dataset: S2–S7, 14 901 ventanas, 487 pos / 14 414 neg (30:1).
> AUC por fold: S2 .936 · S3 .946 · S4 .942 · S5 .950 · S6 .958 · S7 .948.
> LOZO (deja un emplazamiento fuera, top-70): AUC **0.933**. Profundidad: Sens 87.5 % (1 cm) → 70.7 % (7 cm).
> Clasificadores (378): RF .915 · SVM .929 · XGB .941 · EE .945. FP: 93 % sobre fondo (de 1363 en zonas de mina).
>
> Números oficiales = sección **"RESULTADOS DEFINITIVOS (sin cuota)"** (al final) + CSV de `resultados/`.
> Todo lo marcado `[DEPRECATED]` es de la versión **con cuota** (k=80), descartada tras P8; se conserva
> solo como registro histórico del proceso.

> Reestructuración del Capítulo 4 "desde cero". **Todo Enfoque B** (el Enfoque A se elimina).
> El código original NO se toca: todo lo nuevo vive aquí. El documento (.tex) se reescribe
> al final, cuando todas las pruebas estén listas.

## Decisiones fijas
- **Clasificador final:** EasyEnsemble x10 XGBoost.
- **Balanceo:** submuestreo **1:1 desde el entrenamiento**, aplicado **dentro de cada fold LOSO**
  (la sesión de prueba se evalúa con su distribución real). Descrito en metodología.
- **Cuota de objetos: DESCARTADA.** Se probó (hard-negative mining, obj_frac=1/3) pero la
  comparación limpia (P8) muestra que **empeora** todas las métricas. El balanceo final es
  **submuestreo random 1:1** (que ya incluye los objetos en su proporción natural ~5.8%).
  La cuota queda solo como experimento con resultado negativo en P8.
- **k final = 70** (Pareto-óptimo en el plateau del 99.5%; ver P4).
- **Métricas reportadas en TODAS las pruebas:** AUC (LOSO), Sensibilidad(0.5), Especificidad(0.5),
  FP/mina. Posible desde el inicio gracias al 1:1.
- **Cálculo de índices:** se evalúa imagen-completa vs ventana (Prueba 1) y se adopta el de
  imagen completa (más eficiente, resultado idéntico).

## Ruta de pruebas (orden)
| # | Prueba | Clasificador | Features | Salida |
|---|---|---|---|---|
| **P1** | Índices imagen completa vs ventana por ventana | RF (1:1) | 378 | AUC/Sens/Spec/FP de ambos + violín + tiempos + equivalencia |
| **P2** | Importancia de características (global vs ventana) | RF (1:1) | 378 | ¿mismas características? |
| **P3** | Comparación de clasificadores | RF vs SVM vs XGBoost vs EasyEnsemble (1:1) | 378 | métricas → define EasyEnsemble |
| **P4** | Selección del mejor k | EasyEnsemble (1:1) | barrido k | k óptimo |
| **P5** | Modelo final por fold | EasyEnsemble (1:1), top-k | LOSO | tabla por fold |
| **P6** | Rendimiento por sesión / condiciones | — | — | tabla |
| **P7** | Rendimiento por profundidad | EasyEnsemble | — | tabla/figura |
| **P8** | Falsos positivos vs objetos (con cuota) | EasyEnsemble | — | tabla/figura |
| **P9** | Discusión / resumen | — | — | — |

## Archivos
- `scripts/comun.py` — carga de datos, submuestreo 1:1 con cuota de objetos, RF-LOSO con métricas.
- `scripts/extraccion_v2.py` — extracción unificada (modo `global` y modo `ventana`), con `clase`.
- `scripts/p1_indices.py` — Prueba 1.
- `resultados/` — CSVs de features y métricas.
- `figuras/` — imágenes nuevas.

---

# Resultados (se van llenando)

> ⚠️ **[DEPRECATED — versión con cuota de objetos]** Todo lo que sigue, hasta la sección
> **"RESULTADOS DEFINITIVOS"**, usó `obj_frac≈1/3` (con cuota) y/o k=80. **Superado** por los
> RESULTADOS DEFINITIVOS (sin cuota, k=70) al final. Se conserva como historia del proceso.

## Prueba 4 — Selección de k ✅ → **k = 80**

Barrido uniforme (paso 10, k=10..378) con EasyEnsemble, selección XGBoost-gain dentro de cada fold.

| Criterio | k |
|---|---|
| Tolerancia 0.005 / 99.5% AUC | **80** |
| Tolerancia 0.010 AUC | 50 |
| Importancia acumulada 90% / 95% | 235 / 274 |

**k elegido = 80** (codo real; AUC 0.940, Sens 0.815, Spec 0.875, FP/mina 4.54 — el menor; std 0.012).
La importancia acumulada (235–274) se **descarta** como criterio: la ganancia está repartida, no hay
features dominantes, así que pediría guardar casi todo para el mismo AUC. Criterio válido = plateau de rendimiento.

**Método de selección (b):** XGBoost-gain ≈ RF-Gini (k=50: 0.935 vs 0.934; k=100: 0.939 vs 0.939) →
usar la importancia propia de XGBoost está justificado (consistente, sin penalización).

Figuras: `p4_barrido_k.png`, `p4b_importancia_acumulada.png`, `p4_metodo_seleccion.png`.
Tablas: `p4_barrido_k.csv`, `p4b_criterios.csv`, `p4_metodo_seleccion.csv`.

## Prueba 5 — Modelo final ✅ (EasyEnsemble x10, top-80, LOSO)

| Fold | AUC | Sens | Spec | TP | FN | FP | TN |
|---|---|---|---|---|---|---|---|
| S2 | 0.914 | 0.787 | 0.885 | 59 | 16 | 215 | 1654 |
| S3 | 0.946 | 0.964 | 0.671 | 81 | 3 | 825 | 1683 |
| S4 | 0.938 | 0.770 | 0.909 | 67 | 20 | 228 | 2277 |
| S5 | 0.946 | 0.815 | 0.904 | 66 | 15 | 240 | 2270 |
| S6 | 0.953 | 0.872 | 0.926 | 68 | 10 | 185 | 2327 |
| S7 | 0.943 | 0.683 | 0.956 | 56 | 26 | 110 | 2400 |
| **Prom.** | **0.940** | **0.815** | **0.875** | 397 | 90 | 1803 | 12611 |

**Modelo final: AUC=0.940, Sens=0.815, Spec=0.875, FP/mina=4.54.**

Comparación con el final del documento viejo (top-50, sin cuota): AUC 0.937→**0.940**, Spec 0.858→**0.875**,
FP/mina ~5.1→**4.54**; Sens 0.823→0.815 (−0.008, costo de parsimonia + cuota más conservadora).
Patrón de folds extremos igual que antes: S3 (Sens alta 0.964 / Spec baja 0.671) vs S7 (Spec alta 0.956 / Sens baja 0.683).

Figura (matriz balanceada 1:1): `figuras/p5_matriz_confusion.png` · Tabla: `resultados/p5_loso_por_fold.csv`.

## Prueba 1 — Índices imagen completa vs ventana ✅

**Dataset:** 14.901 parches, 487 positivos (Enfoque B, S2–S7). Clasificador: RF con
submuestreo 1:1 + cuota de objetos (≈1/3), LOSO, semilla fija (misma para ambos modos).

### 1. Equivalencia numérica
`max |global − ventana| = 0.0` sobre las 378 features → **idénticos bit a bit**.
(Esperado: los índices son operaciones píxel a píxel).

### 2. Métricas LOSO (idénticas en ambos modos)
| Modo | AUC | Sens(0.5) | Spec(0.5) | FP/mina |
|---|---|---|---|---|
| ventana | 0.908 ± 0.030 | 0.795 | 0.862 | 5.13 |
| **global** | **0.908 ± 0.030** | **0.795** | **0.862** | **5.13** |

### 3. Tiempo de extracción
| Modo | Tiempo | |
|---|---|---|
| ventana (recalcula índices por parche) | 1856.1 s | |
| **global (índices 1× sobre imagen)** | **1692.8 s** | **speedup 1.10×** |

**Hallazgo clave:** el ahorro es modesto (~9 %), NO ~4× como se esperaría por el solapamiento.
Razón: el cuello de botella son las **9 estadísticas por ventana** (media, std, percentiles,
skewness, kurtosis sobre 45 características × 216 ventanas), que son por-ventana en ambos modos.
El cálculo del índice es una fracción pequeña del costo total. La redundancia de recalcular
índices existe, pero pesa poco frente al cálculo de estadísticas.

### 4. Violín (firma por clase)
Idéntico en ambos modos. MAP se separa en NDVI/NDSI (NIR) y es la más alta en Rojo.
Conteos: MAP=487, Botella=374, Lata=372, Piedra=178, Control=4535.
Figura: `figuras/p1_violin_firma.png`.

### Conclusión P1
Los dos modos dan **exactamente el mismo resultado** (features, métricas y violín). El modo
**global** es ~9 % más rápido en extracción → **se adopta el cálculo sobre imagen completa**:
mismo rendimiento, algo más eficiente. El tiempo de **entrenamiento no cambia** (mismas features).

**Línea base RF (Enfoque B, 378 features, 1:1+objetos):** AUC=0.908, Sens=0.795, Spec=0.862, FP/mina=5.13.

## Prueba 2 — Importancia de características (global vs ventana) ✅

Importancia Gini de RF promediada sobre los 6 folds LOSO (1:1 + cuota de objetos, misma semilla).

### Equivalencia
`max |imp_global − imp_ventana| = 0.0`; el **top-20 es idéntico y en el mismo orden**.
→ Confirma de nuevo que los dos modos son intercambiables.

### Top-20 (dataset global) — 11 de 20 involucran NIR o Red Edge
| # | Índice | Estad. | Imp. | NIR/RE |
|---|---|---|---|---|
| 1 | Red | mean | 0.0337 | |
| 2 | GRVI | p25 | 0.0279 | |
| 3 | SI | std | 0.0257 | |
| 4 | RENDVI | std | 0.0241 | * |
| 5 | NIR_Red | p25 | 0.0198 | * |
| 6 | NDVI | std | 0.0183 | * |
| 7 | Red_Green | p75 | 0.0173 | |
| 8 | VARI | p25 | 0.0167 | |
| 9 | NDSI | std | 0.0163 | * |
| 10 | NDVI | p25 | 0.0160 | * |
| 11 | NDSVI | std | 0.0155 | * |
| 12 | Red | p75 | 0.0154 | |
| 13 | RedEdge_Red | p25 | 0.0143 | * |
| 14 | NDSI | p75 | 0.0138 | * |
| 15 | Red | std | 0.0128 | |
| 16 | BSI | p75 | 0.0118 | * |
| 17 | RENDVI | p25 | 0.0114 | * |
| 18 | NDVI | skewness | 0.0114 | * |
| 19 | CI_soil | skewness | 0.0113 | |
| 20 | ExR | p75 | 0.0106 | |

**Observaciones:**
- **11/20 con NIR o Red Edge** → las bandas infrarrojas aportan de forma clara.
- Domina el estadístico **std** (RENDVI_std, NDVI_std, NDSI_std, SI_std, NDSVI_std, Red_std):
  la mina genera un **gradiente espectral localizado** que la desviación capta mejor que la media.
- **Red_mean** es la #1 (coherente con P1: MAP es la más alta en el Rojo por el suelo desnudo).

Figura: `figuras/p2_top20_importancia.png` · Tabla: `resultados/p2_top20_importancia.csv`

### Conclusión P2
Importancias **idénticas** global vs ventana (equivalencia confirmada por 2ª vez). El conjunto de
características discriminativas combina visible (Red, GRVI, VARI) e infrarrojo (NDVI, NDSI, RENDVI,
NIR_Red), con el `std` como estadístico dominante.

## Prueba 3 — Comparación de clasificadores ✅

Dataset global, 378 features, submuestreo 1:1 + cuota de objetos, LOSO, umbral 0.5.

| Clasificador | AUC | Sens(0.5) | Spec(0.5) | FP/mina |
|---|---|---|---|---|
| Random Forest | 0.908 ± 0.030 | 0.795 | 0.862 | 5.13 |
| SVM (RBF) | 0.919 ± 0.023 | 0.764 | 0.868 | 5.13 |
| XGBoost | 0.934 ± 0.021 | 0.821 | 0.872 | 4.61 |
| **EasyEnsemble x10** | **0.942 ± 0.017** | **0.823** | **0.883** | **4.20** |

**EasyEnsemble es el mejor en las 4 métricas y el más estable (std 0.017).** → clasificador final.

**Efecto de la cuota de objetos** (vs EasyEnsemble documentado sin cuota): AUC 0.937→**0.942**,
FP/mina ~5.1→**4.20**, misma Sens (0.823). Los negativos difíciles mejoran el rechazo de falsas
alarmas sin perder detección. (Se cuantifica formalmente en P8.)

Figura: `figuras/p3_clasificadores.png` · Tabla: `resultados/p3_clasificadores.csv`

### Conclusión P3
Clasificador final = **EasyEnsemble x10 XGBoost** (1:1 + cuota objetos). AUC=0.942, Sens=0.823,
Spec=0.883, FP/mina=4.20 con las 378 features. Siguiente: P4 (selección de k).

## Prueba 3b — Eficiencia computacional (complejidad)

| Clasificador | AUC | Sens | Modelos | Entren. total | Infer. µs/ventana | vs XGBoost |
|---|---|---|---|---|---|---|
| Random Forest | 0.908 | 0.795 | 6 | 1.9 s | 22.9 | 8.8× |
| SVM (RBF) | 0.919 | 0.764 | 6 | 1.9 s | 108.3 | 41.7× |
| **XGBoost** | 0.934 | 0.821 | 6 | 16.1 s | **2.6** | **1.0×** |
| **EasyEnsemble** | **0.942** | **0.823** | 60 | 160.7 s | 29.7 | 11.4× |

**Lecturas:**
- XGBoost = inferencia más rápida (árboles depth 4 + motor C++).
- EasyEnsemble = ~11× inferencia / ~10× entrenamiento que XGBoost (son 10 modelos), por +0.008 AUC.
- SVM = peor trato (métricas medias + inferencia más cara, 42×).

**En absoluto (lo decisivo):** inferencia por imagen (~216 ventanas) — EasyEnsemble ≈ 6.4 ms,
XGBoost ≈ 0.56 ms; pero la **extracción ≈ 24.5 s/imagen**. La inferencia es ~0.03 % del costo por
imagen → el sobrecosto de EasyEnsemble es **irrelevante** para análisis post-captura.

**Conclusión:** EasyEnsemble se mantiene como final (mejores métricas, sobrecosto despreciable).
XGBoost queda documentado como **alternativa ligera** (11× más rápido, −0.008 AUC) para despliegue
en hardware limitado / tiempo real. Tabla: `resultados/p3b_eficiencia.csv`.

## Prueba 3c — Matrices de confusión por clasificador (criterio extra, sugerido por la profe)

Conteos agregados LOSO (umbral 0.5), misma semilla que P3. Fila Mina = recall de minas; fila No mina = rechazo de falsas alarmas.

| Clasificador | TP (mina✓) | FN | FP | TN (no-mina✓) | Sens | Spec |
|---|---|---|---|---|---|---|
| Random Forest | 387 (79.5%) | 100 | 1987 | 12427 (86.2%) | 0.795 | 0.862 |
| SVM (RBF) | 372 (76.4%) | 115 | 1907 | 12507 (86.8%) | 0.764 | 0.868 |
| XGBoost | 400 (82.1%) | 87 | 1845 | 12569 (87.2%) | 0.821 | 0.872 |
| **EasyEnsemble x10** | **401 (82.3%)** | **86** | **1684** | **12730 (88.3%)** | **0.823** | **0.883** |

**EasyEnsemble domina las dos filas simultáneamente** (más minas detectadas Y menos falsas alarmas:
FP 1684, el menor). Confirma la elección. Figura: `figuras/p3c_matrices.png`.

> **Decisión de estilo de figuras (matrices de confusión):**
> - **P3c (comparación de clasificadores):** conteos REALES + % por fila (recall). Para comparar.
> - **P5 (modelo final):** visualización BALANCEADA 1:1 (negativos reescalados a 487, % reales),
>   estilo de la figura `matriz_confusion_final.png` del documento viejo — más limpia para un solo modelo.

---

# RESULTADOS DEFINITIVOS (sin cuota) — supersede a los parciales con cuota de arriba

> Tras P8, el pipeline final usa **submuestreo random 1:1 (sin cuota)**. Estos son los números oficiales.

## P3 — Clasificadores (sin cuota, 378 features)
| Clasificador | AUC | Sens | Spec | FP/mina |
|---|---|---|---|---|
| Random Forest | 0.915 | 0.805 | 0.856 | 5.29 |
| SVM (RBF) | 0.929 | 0.758 | 0.880 | 4.70 |
| XGBoost | 0.941 | 0.817 | 0.885 | 4.15 |
| **EasyEnsemble x10** | **0.945** | **0.832** | 0.883 | 4.16 |
EasyEnsemble = clasificador final. P3c: EE TP=405 FN=82 FP=1684 TN=12730. P3b: EE ~10x inferencia vs XGBoost (despreciable vs extracción).

## P4 — Selección de k (sin cuota) → **k = 70**
Criterio 99.5%/tolerancia 0.005 → k=60; se elige **k=70** por ser Pareto-óptimo (domina a k=60 en AUC, Sens y FP/mina) priorizando sensibilidad. Importancia acumulada (235) descartada. Método: XGBoost-gain > RF-Gini (justifica importancia propia).

## P5 — Modelo final: EasyEnsemble x10, top-70, sin cuota
| Fold | AUC | Sens | Spec | TP | FN | FP | TN |
|---|---|---|---|---|---|---|---|
| S2 | 0.936 | 0.827 | 0.898 | 62 | 13 | 191 | 1678 |
| S3 | 0.946 | 0.964 | 0.669 | 81 | 3 | 831 | 1677 |
| S4 | 0.942 | 0.770 | 0.921 | 67 | 20 | 198 | 2307 |
| S5 | 0.950 | 0.815 | 0.916 | 66 | 15 | 212 | 2298 |
| S6 | 0.958 | 0.872 | 0.938 | 68 | 10 | 156 | 2356 |
| S7 | 0.948 | 0.695 | 0.967 | 57 | 25 | 82 | 2428 |
| **Prom.** | **0.947** | **0.823** | **0.884** | 401 | 86 | 1670 | 12744 |

**MODELO FINAL: AUC=0.947, Sens=0.823, Spec=0.884, FP/mina=4.16.** Figura: `p5_matriz_confusion.png`.

## P6 — Condiciones de captura
AUC por sesión 0.936–0.958 (rango 0.022). Sin patrón con hora/HR/radiación → señal estable (n=6, inspección cualitativa). Tabla: `p6_condiciones.csv`.

## P7 — Profundidad
| Prof | AUC | Sens | Spec |
|---|---|---|---|
| 1 cm | 0.942 | 0.875 | 0.858 |
| 3 cm | 0.914 | 0.880 | 0.813 |
| 5 cm | 0.928 | 0.835 | 0.855 |
| 7 cm | 0.915 | 0.707 | 0.923 |
Sensibilidad baja con la profundidad (0.875→0.707): más profunda, más difícil. Figura: `p7_profundidad.png`.

## P8 — Falsos positivos vs objetos (+ experimento de la cuota)
- FP del modelo final en zonas de mina: **1363**. **93.2% sobre fondo (suelo perturbado)**, solo **6.8% sobre objetos** (Botella 57, Lata 17, Piedra 19). → los objetos NO son la causa de las falsas alarmas.
- **Experimento cuota:** final (sin cuota) 1363 FP vs con cuota 1547 FP; sobre objetos 93 vs 97. → la cuota **no reduce** FP sobre objetos y **aumenta** los totales. Resultado negativo. Figura: `p8_fp_objetos.png`.

## P9 — Progresión metodológica (resumen)
| Configuración | AUC | Sens | FP/mina |
|---|---|---|---|
| RF, 378 feat (línea base) | 0.915 | 0.805 | 5.29 |
| XGBoost, 378 feat | 0.941 | 0.817 | 4.15 |
| EasyEnsemble, 378 feat | 0.945 | 0.832 | 4.16 |
| **EasyEnsemble, top-70 (FINAL)** | **0.947** | **0.823** | **4.16** |

Frente a la tesis vieja (EE top-50 con cuota: 0.937 / 0.823 / ~5.1) el nuevo modelo mejora AUC (0.947) y FP/mina (4.16), con Sens igual. Cálculo de índices a nivel de imagen (P1, idéntico y más eficiente). Selección por XGBoost-gain (consistente). Cuota descartada con evidencia (P8).
