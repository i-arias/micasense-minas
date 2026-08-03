"""
Extraccion unificada de features (thesisV2) — dos modos:

  modo='ventana'  -> indices calculados sobre los pixeles de CADA ventana (como el original).
  modo='global'   -> indices calculados UNA vez sobre la imagen completa; cada ventana
                     solo recorta el mapa de indice y saca estadisticas.

Ambos producen EXACTAMENTE las mismas columnas y etiquetas (Enfoque B). La unica diferencia
es donde se calcula el indice; como son operaciones pixel-a-pixel, el resultado debe ser
identico. Esta es la base de la Prueba 1 (equivalencia + ahorro de tiempo).

Ademas asigna 'clase' (MAP/Botella/Lata/Piedra/Control/Fondo) para violines y cuota de objetos.

Salida: thesisV2/resultados/features_{modo}.csv

Ejecutar desde la raiz del proyecto:
    python thesisV2/scripts/extraccion_v2.py
"""
import sys, os, json, re, time
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.join('process'))
from extraccion_features import (
    calcular_indices_vegetativos,
    calcular_estadisticas,
    cargar_imagen_calibrada,
    CONFIG as EXT_CONFIG,
)

COCO_JSON = 'FINAL_DATASET/referencia_roboflow/minas-groundtruth.v3i.coco/train/_annotations.coco.json'
SESIONES = [2, 3, 4, 5, 6, 7]
ZONAS_MINA = ['1CM_1', '1CM_2', '3CM_1', '3CM_2', '5CM_1', '5CM_2', '7CM_1', '7CM_2']
CARPETAS_CONTROL = ['ZONA_CONTROL', 'ZONA_CONTROL_2', 'ZONA_CONTROL_3', 'ZONA_CONTROL_4']

TAMANO = EXT_CONFIG['TAMANO_PARCHE']
SOLAP = EXT_CONFIG['SOLAPAMIENTO']
MINVAL = EXT_CONFIG['MIN_PIXELES_VALIDOS']
BANDAS = EXT_CONFIG['BANDAS_NOMBRES']
PASO = int(TAMANO * (1 - SOLAP))

CAT = {3: 'MAP', 1: 'Botella', 2: 'Lata', 4: 'Piedra'}

# ---------------------------------------------------------------- bbox lookup
with open(COCO_JSON) as f:
    coco = json.load(f)
img_por_id = {im['id']: im for im in coco['images']}
# (sesion, zona) -> {clase: (bbox, png_w, png_h)}
BBOX = {}
for a in coco['annotations']:
    im = img_por_id[a['image_id']]
    m = re.search(r'SESION_(\d+)_MINA_(\d+)-(\d+)', im['file_name'])
    if not m:
        continue
    key = (int(m.group(1)), f"MINA_{m.group(2)}CM_{m.group(3)}")
    BBOX.setdefault(key, {})[CAT[a['category_id']]] = (a['bbox'], im['width'], im['height'])


def clase_de(cx, cy, zona, bboxes, tif_w, tif_h):
    """Devuelve (clase, tiene_mina) segun el centro del parche."""
    if 'CONTROL' in zona.upper():
        return 'Control', 0
    # mine zone: probar MAP primero, luego objetos
    for nombre in ['MAP', 'Botella', 'Lata', 'Piedra']:
        if nombre not in bboxes:
            continue
        (x0, y0, bw, bh), pw, ph = bboxes[nombre]
        sx, sy = tif_w / pw, tif_h / ph
        if x0 * sx <= cx <= (x0 + bw) * sx and y0 * sy <= cy <= (y0 + bh) * sy:
            return nombre, (1 if nombre == 'MAP' else 0)
    return 'Fondo', 0


def stats_dict(valores):
    return calcular_estadisticas(valores)


def extraer_imagen(imagen, zona, bboxes, modo):
    h, w, _ = imagen.shape
    umbral = TAMANO * TAMANO * MINVAL
    regs = []

    if modo == 'global':
        bandas2d = [imagen[:, :, i] for i in range(5)]
        idx_maps = calcular_indices_vegetativos(*bandas2d)  # dict de mapas 2D

    for y in range(0, h - TAMANO + 1, PASO):
        for x in range(0, w - TAMANO + 1, PASO):
            parche = imagen[y:y+TAMANO, x:x+TAMANO, :]
            mask = np.all(parche > 0.01, axis=2)
            if np.sum(mask) < umbral:
                continue

            cx, cy = x + TAMANO / 2, y + TAMANO / 2
            clase, mina = clase_de(cx, cy, zona, bboxes, w, h)

            reg = {'tiene_mina': mina, 'parche_x': x, 'parche_y': y,
                   'zona': zona, 'clase': clase}

            # estadisticas de bandas (iguales en ambos modos)
            for i, nb in enumerate(BANDAS):
                for st, v in stats_dict(parche[:, :, i][mask]).items():
                    reg[f'{nb}_{st}'] = v

            # estadisticas de indices
            if modo == 'ventana':
                pix = parche[mask]
                idx = calcular_indices_vegetativos(pix[:, 0], pix[:, 1], pix[:, 2], pix[:, 3], pix[:, 4])
                for nombre, val in idx.items():
                    for st, v in stats_dict(val).items():
                        reg[f'{nombre}_{st}'] = v
            else:  # global
                for nombre, fullmap in idx_maps.items():
                    win = fullmap[y:y+TAMANO, x:x+TAMANO][mask]
                    for st, v in stats_dict(win).items():
                        reg[f'{nombre}_{st}'] = v

            regs.append(reg)
    return regs


def extraer(modo):
    t0 = time.perf_counter()
    todos = []
    for n in SESIONES:
        proc = f'FINAL_DATASET/SESION_{n}/PROCESADAS'
        for mz in ZONAS_MINA:
            zona = f'MINA_{mz}'
            tif = os.path.join(proc, f'{zona}_calibrado.tif')
            if not os.path.exists(tif) or (n, zona) not in BBOX:
                continue
            imagen = cargar_imagen_calibrada(tif)
            regs = extraer_imagen(imagen, zona, BBOX[(n, zona)], modo)
            for r in regs:
                r['sesion'] = n
            todos.extend(regs)
        for ctrl in CARPETAS_CONTROL:
            tif = os.path.join(proc, f'{ctrl}_calibrado.tif')
            if not os.path.exists(tif):
                continue
            imagen = cargar_imagen_calibrada(tif)
            regs = extraer_imagen(imagen, ctrl, {}, modo)
            for r in regs:
                r['sesion'] = n
            todos.extend(regs)
        print(f"  S{n} listo ({modo})")
    elapsed = time.perf_counter() - t0
    df = pd.DataFrame(todos)
    return df, elapsed


if __name__ == '__main__':
    os.makedirs('thesisV2/resultados', exist_ok=True)
    tiempos = {}
    for modo in ['ventana', 'global']:
        print(f"\n=== Extraccion modo '{modo}' ===")
        df, el = extraer(modo)
        out = f'thesisV2/resultados/features_{modo}.csv'
        df.to_csv(out, index=False)
        tiempos[modo] = el
        pos = int((df['tiene_mina'] == 1).sum())
        print(f"  {len(df)} parches, {pos} positivos -> {out}  ({el:.1f}s)")
    print("\n=== Tiempos ===")
    for m, t in tiempos.items():
        print(f"  {m:8s}: {t:.1f}s")
    speedup = tiempos['ventana'] / tiempos['global'] if tiempos['global'] else float('nan')
    print(f"  speedup (ventana/global): {speedup:.2f}x")
    pd.DataFrame([tiempos]).to_csv('thesisV2/resultados/tiempos_extraccion.csv', index=False)
