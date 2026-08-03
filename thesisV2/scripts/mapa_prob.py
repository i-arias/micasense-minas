"""
Mapa de probabilidades de detección de MAP sobre una imagen (S2-S7).
Usa la probabilidad LOSO del modelo final (p5_predicciones.csv): la sesión fue
predicha por el modelo entrenado sin ella.

Ejecutar desde la raiz del proyecto:
    python thesisV2/scripts/mapa_prob.py
"""
import os, json, re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import warnings
warnings.filterwarnings('ignore')
import rasterio

SES, ZONA = 2, 'MINA_7CM_1'
PARCHE, PASO = 128, 64
FIG = 'thesisV2/figuras'
TIF = f'FINAL_DATASET/SESION_{SES}/PROCESADAS/{ZONA}_calibrado.tif'
COCO = 'FINAL_DATASET/referencia_roboflow/minas-groundtruth.v3i.coco/train/_annotations.coco.json'

# ---- predicciones LOSO de la zona
df = pd.read_csv('thesisV2/resultados/p5_predicciones.csv')
dz = df[(df['sesion'] == SES) & (df['zona'] == ZONA)].copy()

# ---- RGB
with rasterio.open(TIF) as src:
    blue, green, red = (src.read(i).astype(float) for i in (1, 2, 3))
    h, w = red.shape

def nb(b):
    p2, p98 = np.percentile(b, 2), np.percentile(b, 98)
    return np.clip((b - p2) / (p98 - p2 + 1e-8), 0, 1)
rgb = np.stack([nb(red), nb(green), nb(blue)], axis=-1)

# ---- mapa de probabilidad (promedio donde se solapan las ventanas)
pm = np.zeros((h, w)); cnt = np.zeros((h, w))
for _, r in dz.iterrows():
    x, y = int(r['parche_x']), int(r['parche_y'])
    pm[y:y+PARCHE, x:x+PARCHE] += r['proba']
    cnt[y:y+PARCHE, x:x+PARCHE] += 1
with np.errstate(invalid='ignore'):
    pm = np.where(cnt > 0, pm / cnt, np.nan)

# ---- bbox MAP (COCO V3) -> coords TIF
coco = json.load(open(COCO)); idm = {i['id']: i for i in coco['images']}
mapbox = None
for a in coco['annotations']:
    if a['category_id'] != 3:
        continue
    f = idm[a['image_id']]['extra']['name']
    m = re.search(r'SESION_(\d+)_MINA_(\d+)-(\d+)', f)
    if m and (int(m.group(1)), f"MINA_{m.group(2)}CM_{m.group(3)}") == (SES, ZONA):
        x0, y0, bw, bh = a['bbox']
        sx, sy = w / idm[a['image_id']]['width'], h / idm[a['image_id']]['height']
        mapbox = (x0*sx, y0*sy, bw*sx, bh*sy)

cmap = LinearSegmentedColormap.from_list('p', [(0, '#2471A3'), (0.5, '#F4D03F'), (1, '#C0392B')])
fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))
fig.suptitle(f'Mapa de probabilidades — Sesión {SES}, {ZONA.replace("_", " ")} '
             f'(mina a 7 cm)', fontsize=12, fontweight='bold')

for ax in axes:
    ax.imshow(rgb); ax.axis('off')
    if mapbox:
        ax.add_patch(mpatches.Rectangle(mapbox[:2], mapbox[2], mapbox[3], lw=2.5,
                     edgecolor='#27AE60', facecolor='none', linestyle='--', zorder=4))
axes[0].set_title('Imagen RGB calibrada', fontsize=11, fontweight='bold')
im = axes[1].imshow(pm, cmap=cmap, alpha=0.6, vmin=0, vmax=1, interpolation='bilinear')
try:
    axes[1].contour((pm >= 0.5).astype(float), levels=[0.5], colors=['white'], linewidths=1.3)
except Exception:
    pass
axes[1].set_title('Probabilidad de MAP (umbral 0.5)', fontsize=11, fontweight='bold')
cb = plt.colorbar(im, ax=axes[1], fraction=0.035, pad=0.02)
cb.set_label('P(MAP)'); cb.set_ticks([0, 0.25, 0.5, 0.75, 1])
axes[1].legend(handles=[mpatches.Patch(facecolor='none', edgecolor='#27AE60',
               linestyle='--', label='bbox MAP')], loc='upper right', fontsize=8)

tp = int(((dz['tiene_mina'] == 1) & (dz['proba'] >= 0.5)).sum())
fn = int(((dz['tiene_mina'] == 1) & (dz['proba'] < 0.5)).sum())
fp = int(((dz['tiene_mina'] == 0) & (dz['proba'] >= 0.5)).sum())
plt.tight_layout(rect=[0, 0, 1, 0.96])
os.makedirs(FIG, exist_ok=True)
ruta = f'{FIG}/mapa_prob_S{SES}_7cm.png'
plt.savefig(ruta, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f"S{SES} {ZONA}: TP={tp} FN={fn} FP={fp}")
print(f"figura -> {ruta}")
