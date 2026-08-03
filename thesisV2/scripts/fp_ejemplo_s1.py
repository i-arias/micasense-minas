"""
Figura espacial de falsos positivos sobre una imagen — ejemplo S1 MINA_7CM_1.

El modelo final (EasyEnsemble x10, top-70, sin cuota) se entrena con S2-S7 y se
aplica a S1 MINA_7CM_1 (imagen NO vista en entrenamiento). Se dibujan el bbox de
la MAP y de los objetos, y los FP (ventanas sin mina predichas como mina),
coloreados según si caen cerca (<=30 px) de un objeto o sobre el fondo.

Ejecutar desde la raiz del proyecto:
    python thesisV2/scripts/fp_ejemplo_s1.py
"""
import sys, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')
import rasterio

sys.path.insert(0, 'thesisV2/scripts')
import comun
import extraccion_v2 as E
from xgboost import XGBClassifier

FIG = 'thesisV2/figuras'
K = 70
N_ENS = 10
SEED = 42
SES, ZONA = 1, 'MINA_7CM_1'
TIF = f'FINAL_DATASET/SESION_{SES}/PROCESADAS/{ZONA}_calibrado.tif'
DIST_OBJ = 30  # px para considerar un FP "cerca de objeto"

# ---------------------------------------------------------------- extraer S1
print(f"Extrayendo {ZONA} (S{SES}) ...")
img = E.cargar_imagen_calibrada(TIF)
bb = E.BBOX[(SES, ZONA)]
regs = E.extraer_imagen(img, ZONA, bb, 'global')
ds1 = pd.DataFrame(regs)
print(f"  {len(ds1)} ventanas")

# ---------------------------------------------------------------- entrenar en S2-S7
print("Entrenando modelo final (EasyEnsemble x10, top-70) en S2-S7 ...")
dg = comun.cargar('thesisV2/resultados/features_global.csv')
cols = comun.feature_cols(dg)
Xtr, ytr, clase_tr = dg[cols].values, dg['tiene_mina'].values, dg['clase'].values
Xs1 = ds1[cols].values
rng = np.random.default_rng(SEED)

# top-70 por XGBoost-gain sobre un submuestreo 1:1
sub0, _, _ = comun.undersample_quota(ytr, clase_tr, rng, 0.0)
rk = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1, subsample=0.8,
                   colsample_bytree=0.8, eval_metric='logloss', importance_type='gain',
                   verbosity=0, random_state=SEED, n_jobs=-1)
rk.fit(Xtr[sub0], ytr[sub0])
top = np.argsort(rk.feature_importances_)[::-1][:K]

proba = np.zeros(len(ds1))
for j in range(N_ENS):
    sub, _, _ = comun.undersample_quota(ytr, clase_tr, rng, 0.0)
    clf = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1, subsample=0.8,
                        colsample_bytree=0.8, eval_metric='logloss', verbosity=0,
                        random_state=j, n_jobs=-1)
    clf.fit(Xtr[sub][:, top], ytr[sub])
    proba += clf.predict_proba(Xs1[:, top])[:, 1]
proba /= N_ENS
ds1['proba'] = proba

# ---------------------------------------------------------------- RGB + bboxes
with rasterio.open(TIF) as src:
    blue, green, red = (src.read(i).astype(float) for i in (1, 2, 3))
    h, w = red.shape

def nb(b):
    p2, p98 = np.percentile(b, 2), np.percentile(b, 98)
    return np.clip((b - p2) / (p98 - p2 + 1e-8), 0, 1)
rgb = np.stack([nb(red), nb(green), nb(blue)], axis=-1)

# bboxes en coords TIF
OBJ_COL = {'MAP': '#2ECC71', 'Botella': '#00B0FF', 'Lata': '#FF9100', 'Piedra': '#E040FB'}
boxes = {}
for nombre, (bbox, pw, ph) in bb.items():
    sx, sy = w / pw, h / ph
    x0, y0, bw, bh = bbox
    boxes[nombre] = (x0*sx, y0*sy, bw*sx, bh*sy)

def cerca_objeto(cx, cy):
    for nombre in ['Botella', 'Lata', 'Piedra']:
        if nombre not in boxes:
            continue
        x0, y0, bw, bh = boxes[nombre]
        dx = max(x0 - cx, 0, cx - (x0 + bw))
        dy = max(y0 - cy, 0, cy - (y0 + bh))
        if (dx*dx + dy*dy) ** 0.5 <= DIST_OBJ:
            return True
    return False

# FP = ventanas sin mina (clase != MAP) predichas positivas
fp = ds1[(ds1['tiene_mina'] == 0) & (ds1['proba'] >= 0.5)].copy()
fp['cx'] = fp['parche_x'] + E.TAMANO / 2
fp['cy'] = fp['parche_y'] + E.TAMANO / 2
fp['cerca'] = fp.apply(lambda r: cerca_objeto(r['cx'], r['cy']), axis=1)
n_cerca = int(fp['cerca'].sum())
n_fondo = len(fp) - n_cerca
print(f"  FP totales: {len(fp)}  (cerca de objeto: {n_cerca}, fondo: {n_fondo})")

# ---------------------------------------------------------------- figura
fig, ax = plt.subplots(figsize=(9, 6.5))
ax.imshow(rgb)
estilo = {'MAP': '-', 'Botella': '--', 'Lata': '--', 'Piedra': '--'}
for nombre, (x0, y0, bw, bh) in boxes.items():
    ax.add_patch(mpatches.Rectangle((x0, y0), bw, bh, lw=2.2, edgecolor=OBJ_COL[nombre],
                 facecolor='none', linestyle=estilo[nombre], zorder=4))
    ax.text(x0 + bw/2, y0 - 6, nombre, ha='center', va='bottom', fontsize=8,
            fontweight='bold', color=OBJ_COL[nombre], zorder=5,
            bbox=dict(boxstyle='round,pad=0.12', fc='black', ec='none', alpha=0.45))

for _, r in fp.iterrows():
    if r['cerca']:
        ax.plot(r['cx'], r['cy'], 'o', color='#E74C3C', ms=7, markeredgecolor='white',
                markeredgewidth=0.8, zorder=6)
    else:
        ax.plot(r['cx'], r['cy'], 'o', color='#F1948A', ms=6, markeredgecolor='white',
                markeredgewidth=0.6, alpha=0.9, zorder=5)
ax.axis('off')
ax.set_title(f'Falsos positivos — Sesión 1, mina a 7 cm (imagen no vista)\n'
             f'{len(fp)} FP: {n_cerca} cerca de objeto, {n_fondo} sobre fondo',
             fontsize=11, fontweight='bold')

leg = [
    mpatches.Patch(facecolor='none', edgecolor='#2ECC71', label='bbox MAP'),
    mpatches.Patch(facecolor='none', edgecolor='#7F8C8D', linestyle='--', label='bbox objetos'),
    plt.Line2D([0],[0], marker='o', color='w', markerfacecolor='#E74C3C', markersize=9,
               label=f'FP cerca de objeto ($\\leq$30 px): {n_cerca}'),
    plt.Line2D([0],[0], marker='o', color='w', markerfacecolor='#F1948A', markersize=8,
               label=f'FP sobre fondo: {n_fondo}'),
]
ax.legend(handles=leg, loc='lower center', bbox_to_anchor=(0.5, -0.14), ncol=2, fontsize=9)

plt.tight_layout()
os.makedirs(FIG, exist_ok=True)
ruta = f'{FIG}/fp_ejemplo_s1.png'
plt.savefig(ruta, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f"figura -> {ruta}")
