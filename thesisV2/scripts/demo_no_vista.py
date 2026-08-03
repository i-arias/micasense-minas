"""
Demostración del modelo final sobre IMÁGENES NO VISTAS (validación LOSO).
Dos zonas de la Sesión 2 (predichas por el modelo entrenado sin la S2):
  - MINA_7CM_1  -> el modelo enciende sobre la MAP.
  - ZONA_CONTROL -> el modelo se mantiene apagado (sin mina).
Probabilidades del modelo final en p5_predicciones.csv.

Ejecutar desde la raiz del proyecto:
    python thesisV2/scripts/demo_no_vista.py
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

SES = 2
PARCHE = 128
FIG = 'thesisV2/figuras'
COCO = 'FINAL_DATASET/referencia_roboflow/minas-groundtruth.v3i.coco/train/_annotations.coco.json'
ZONAS = [
    ('MINA_7CM_1',   'Zona de mina (MAP a 7 cm)'),
    ('ZONA_CONTROL', 'Zona de control (sin mina)'),
]

df = pd.read_csv('thesisV2/resultados/p5_predicciones.csv')
coco = json.load(open(COCO)); idm = {i['id']: i for i in coco['images']}

def cargar_rgb(tif):
    with rasterio.open(tif) as src:
        blue, green, red = (src.read(i).astype(float) for i in (1, 2, 3))
    def nb(b):
        p2, p98 = np.percentile(b, 2), np.percentile(b, 98)
        return np.clip((b - p2) / (p98 - p2 + 1e-8), 0, 1)
    return np.stack([nb(red), nb(green), nb(blue)], axis=-1)

def mapa_prob(dz, h, w):
    pm = np.zeros((h, w)); cnt = np.zeros((h, w))
    for _, r in dz.iterrows():
        x, y = int(r['parche_x']), int(r['parche_y'])
        pm[y:y+PARCHE, x:x+PARCHE] += r['proba']
        cnt[y:y+PARCHE, x:x+PARCHE] += 1
    with np.errstate(invalid='ignore'):
        return np.where(cnt > 0, pm / cnt, np.nan)

def bbox_map(zona, h, w):
    for a in coco['annotations']:
        if a['category_id'] != 3:
            continue
        f = idm[a['image_id']]['extra']['name']
        m = re.search(r'SESION_(\d+)_MINA_(\d+)-(\d+)', f)
        if m and (int(m.group(1)), f"MINA_{m.group(2)}CM_{m.group(3)}") == (SES, zona):
            x0, y0, bw, bh = a['bbox']
            sx, sy = w / idm[a['image_id']]['width'], h / idm[a['image_id']]['height']
            return (x0*sx, y0*sy, bw*sx, bh*sy)
    return None

cmap = LinearSegmentedColormap.from_list('p', [(0, '#2471A3'), (0.5, '#F4D03F'), (1, '#C0392B')])
fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.6))
fig.suptitle('Demostración sobre imágenes no vistas en entrenamiento '
             '(Sesión 2, predicción LOSO)', fontsize=13, fontweight='bold')

im_ref = None
for fila, (zona, titulo) in enumerate(ZONAS):
    tif = f'FINAL_DATASET/SESION_{SES}/PROCESADAS/{zona}_calibrado.tif'
    rgb = cargar_rgb(tif)
    h, w = rgb.shape[:2]
    dz = df[(df['sesion'] == SES) & (df['zona'] == zona)].copy()
    pm = mapa_prob(dz, h, w)
    mb = bbox_map(zona, h, w)

    axL, axR = axes[fila]
    for ax in (axL, axR):
        ax.imshow(rgb); ax.axis('off')
        if mb:
            ax.add_patch(mpatches.Rectangle(mb[:2], mb[2], mb[3], lw=2.5,
                         edgecolor='#27AE60', facecolor='none', linestyle='--', zorder=4))
    axL.set_title(f'{titulo} — RGB calibrada', fontsize=11, fontweight='bold')
    im_ref = axR.imshow(pm, cmap=cmap, alpha=0.6, vmin=0, vmax=1, interpolation='bilinear')
    try:
        axR.contour((pm >= 0.5).astype(float), levels=[0.5], colors=['white'], linewidths=1.3)
    except Exception:
        pass

    tp = int(((dz['tiene_mina'] == 1) & (dz['proba'] >= 0.5)).sum())
    fn = int(((dz['tiene_mina'] == 1) & (dz['proba'] < 0.5)).sum())
    fp = int(((dz['tiene_mina'] == 0) & (dz['proba'] >= 0.5)).sum())
    if (dz['tiene_mina'] == 1).any():
        txt = f'Detecta {tp}/{tp+fn} ventanas de mina · {fp} falsas alarmas'
    else:
        txt = f'Sin mina · {fp} de {len(dz)} ventanas activas'
    axR.set_title(f'P(MAP) ≥ 0.5 — {txt}', fontsize=10.5, fontweight='bold')
    print(f'S{SES} {zona}: TP={tp} FN={fn} FP={fp} (n={len(dz)})')

cb = fig.colorbar(im_ref, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02)
cb.set_label('P(MAP)'); cb.set_ticks([0, 0.25, 0.5, 0.75, 1])

os.makedirs(FIG, exist_ok=True)
ruta = f'{FIG}/demo_no_vista.png'
plt.savefig(ruta, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f'figura -> {ruta}')
