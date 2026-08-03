"""
Figura espacial de falsos positivos sobre una imagen de S2-S7 (modelo final, LOSO).
Usa la probabilidad LOSO ya calculada (p5_predicciones.csv): cada sesión fue
predicha por el modelo entrenado sin ella, así que es una predicción honesta.

Dibuja el bbox de la MAP y de los objetos, y los FP (ventanas sin mina predichas
positivas) coloreados según si caen cerca (<=30 px) de un objeto o sobre el fondo.

Ejecutar desde la raiz del proyecto:
    python thesisV2/scripts/fp_ejemplo.py
"""
import sys, os, json, re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')
import rasterio

SES, ZONA = 2, 'MINA_7CM_1'          # imagen de ejemplo (editable)
PARCHE = 128
DIST_OBJ = 30
FIG = 'thesisV2/figuras'
TIF = f'FINAL_DATASET/SESION_{SES}/PROCESADAS/{ZONA}_calibrado.tif'
COCO = 'FINAL_DATASET/referencia_roboflow/minas-groundtruth.v3i.coco/train/_annotations.coco.json'

# ---------------------------------------------------------------- predicciones LOSO
df = pd.read_csv('thesisV2/resultados/p5_predicciones.csv')
dz = df[(df['sesion'] == SES) & (df['zona'] == ZONA)].copy()
dz['cx'] = dz['parche_x'] + PARCHE / 2
dz['cy'] = dz['parche_y'] + PARCHE / 2

# ---------------------------------------------------------------- bboxes (COCO V3)
coco = json.load(open(COCO)); idm = {i['id']: i for i in coco['images']}
CAT = {3: 'MAP', 1: 'Botella', 2: 'Lata', 4: 'Piedra'}
bb = {}
for a in coco['annotations']:
    f = idm[a['image_id']]['extra']['name']
    m = re.search(r'SESION_(\d+)_MINA_(\d+)-(\d+)', f)
    if m and (int(m.group(1)), f"MINA_{m.group(2)}CM_{m.group(3)}") == (SES, ZONA):
        bb[CAT[a['category_id']]] = (a['bbox'], idm[a['image_id']]['width'], idm[a['image_id']]['height'])

# ---------------------------------------------------------------- RGB
with rasterio.open(TIF) as src:
    blue, green, red = (src.read(i).astype(float) for i in (1, 2, 3))
    h, w = red.shape

def nb(b):
    p2, p98 = np.percentile(b, 2), np.percentile(b, 98)
    return np.clip((b - p2) / (p98 - p2 + 1e-8), 0, 1)
rgb = np.stack([nb(red), nb(green), nb(blue)], axis=-1)

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

fp = dz[(dz['tiene_mina'] == 0) & (dz['proba'] >= 0.5)].copy()
fp['cerca'] = fp.apply(lambda r: cerca_objeto(r['cx'], r['cy']), axis=1)
n_cerca = int(fp['cerca'].sum()); n_fondo = len(fp) - n_cerca
tp = dz[(dz['tiene_mina'] == 1) & (dz['proba'] >= 0.5)]
print(f"S{SES} {ZONA}: FP={len(fp)} (cerca={n_cerca}, fondo={n_fondo}), TP={len(tp)}")

# ---------------------------------------------------------------- figura
fig, ax = plt.subplots(figsize=(9, 6.5))
ax.imshow(rgb)
estilo = {'MAP': '-', 'Botella': '--', 'Lata': '--', 'Piedra': '--'}
for nombre, (x0, y0, bw, bh) in boxes.items():
    ax.add_patch(mpatches.Rectangle((x0, y0), bw, bh, lw=2.2, edgecolor=OBJ_COL[nombre],
                 facecolor='none', linestyle=estilo[nombre], zorder=4))
    ax.text(x0 + bw/2, y0 - 6, nombre, ha='center', va='bottom', fontsize=8,
            fontweight='bold', color=OBJ_COL[nombre], zorder=6,
            bbox=dict(boxstyle='round,pad=0.12', fc='black', ec='none', alpha=0.45))
for _, r in fp.iterrows():
    c = '#E74C3C' if r['cerca'] else '#F1948A'
    ms = 7.5 if r['cerca'] else 6
    ax.plot(r['cx'], r['cy'], 'o', color=c, ms=ms, markeredgecolor='white',
            markeredgewidth=0.8, zorder=5)
ax.axis('off')
ax.set_title(f'Falsos positivos — Sesión {SES}, mina a 7 cm\n'
             f'{len(fp)} FP: {n_cerca} cerca de objeto, {n_fondo} sobre el fondo',
             fontsize=11, fontweight='bold')
leg = [
    mpatches.Patch(facecolor='none', edgecolor='#2ECC71', label='bbox MAP'),
    mpatches.Patch(facecolor='none', edgecolor='#7F8C8D', linestyle='--', label='bbox objetos'),
    plt.Line2D([0],[0], marker='o', color='w', markerfacecolor='#E74C3C', markersize=9,
               label='FP cerca de objeto'),
    plt.Line2D([0],[0], marker='o', color='w', markerfacecolor='#F1948A', markersize=8,
               label='FP sobre el fondo'),
]
ax.legend(handles=leg, loc='lower center', bbox_to_anchor=(0.5, -0.13), ncol=2, fontsize=9)
plt.tight_layout()
os.makedirs(FIG, exist_ok=True)
ruta = f'{FIG}/fp_ejemplo.png'
plt.savefig(ruta, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f"figura -> {ruta}")
