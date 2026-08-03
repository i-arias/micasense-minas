"""
PRUEBA 2 — Importancia de caracteristicas: imagen completa (global) vs ventana.

¿Son las mismas caracteristicas importantes en ambos modos?
Importancia Gini de RF promediada sobre los 6 folds LOSO (mismo submuestreo 1:1 + cuota
de objetos, misma semilla). Muestra el TOP-20.

Como las features son identicas (Prueba 1), las importancias deben coincidir.

Ejecutar desde la raiz del proyecto:
    python thesisV2/scripts/p2_importancia.py
"""
import sys, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, 'thesisV2/scripts')
import comun

RES = 'thesisV2/resultados'
FIG = 'thesisV2/figuras'
TOP = 20

STATS = {'mean', 'std', 'min', 'max', 'median', 'p25', 'p75', 'skewness', 'kurtosis'}
# indices que involucran NIR (842) o Red Edge (717)
NIR_RE = {
    'NDVI', 'NDRE', 'GNDVI', 'SAVI', 'CI_RedEdge', 'MSAVI', 'EVI', 'OSAVI', 'NDWI',
    'NDSI', 'BSI', 'NMDI', 'NDSVI', 'RENDVI', 'RENI',
    'NIR_Red', 'NIR_Green', 'NIR_Blue', 'NIR_RedEdge', 'RedEdge_Red', 'RedEdge_Green', 'RedEdge',
}


def split_feat(col):
    base, st = col.rsplit('_', 1)
    return (base, st) if st in STATS else (col, '')


def importancia_media(df, seed=42, obj_frac=0.0):
    cols = comun.feature_cols(df)
    X = df[cols].values
    y = df['tiene_mina'].values
    ses = df['sesion'].values
    clase = df['clase'].values
    rng = np.random.default_rng(seed)
    acc = np.zeros(len(cols))
    for s in comun.SESIONES:
        tr = ses != s
        idx_tr = np.where(tr)[0]
        sub, _, _ = comun.undersample_quota(y[tr], clase[tr], rng, obj_frac)
        abs_idx = idx_tr[sub]
        clf = comun.rf_builder(seed=seed)()
        clf.fit(X[abs_idx], y[abs_idx])
        acc += clf.feature_importances_
    return pd.Series(acc / len(comun.SESIONES), index=cols)


print("=" * 70)
print("PRUEBA 2 — importancia de caracteristicas (global vs ventana)")
print("=" * 70)

dv = comun.cargar(f'{RES}/features_ventana.csv')
dg = comun.cargar(f'{RES}/features_global.csv')

imp_v = importancia_media(dv)
imp_g = importancia_media(dg)

# ---- equivalencia
maxdiff = np.max(np.abs(imp_v.values - imp_g.values))
# coincidencia del orden del top-20
top_v = list(imp_v.sort_values(ascending=False).head(TOP).index)
top_g = list(imp_g.sort_values(ascending=False).head(TOP).index)
print(f"\n[1] max |imp_global - imp_ventana| = {maxdiff:.3e}")
print(f"    top-{TOP} en el mismo orden: {top_v == top_g}")
print(f"    mismas {TOP} features (sin importar orden): {set(top_v) == set(top_g)}")

# ---- tabla top-20 (sobre global)
imp = imp_g.sort_values(ascending=False).head(TOP)
filas = []
n_nirre = 0
for rank, (col, val) in enumerate(imp.items(), 1):
    base, st = split_feat(col)
    es = base in NIR_RE
    n_nirre += es
    filas.append(dict(rank=rank, indice=base, estadistico=st,
                      importancia=round(val, 5), NIR_RedEdge='*' if es else ''))
tabla = pd.DataFrame(filas)
tabla.to_csv(f'{RES}/p2_top20_importancia.csv', index=False)
print(f"\n[2] TOP-{TOP} caracteristicas (dataset global):")
print(tabla.to_string(index=False))
print(f"\n    De las top-{TOP}, {n_nirre} involucran NIR o Red Edge.")

# ---- figura barh
os.makedirs(FIG, exist_ok=True)
fig, ax = plt.subplots(figsize=(8, 7))
labels = [f"{r['indice']}_{r['estadistico']}" for _, r in tabla.iloc[::-1].iterrows()]
vals = tabla['importancia'].values[::-1]
colors = ['#C0392B' if r['NIR_RedEdge'] == '*' else '#5D6D7E'
          for _, r in tabla.iloc[::-1].iterrows()]
ax.barh(range(TOP), vals, color=colors)
ax.set_yticks(range(TOP)); ax.set_yticklabels(labels, fontsize=8)
ax.set_xlabel('Importancia Gini media (LOSO)')
ax.set_title(f'Prueba 2 — Top-{TOP} características (RF, Enfoque B)\n'
             'rojo = involucra NIR o Red Edge', fontsize=10, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{FIG}/p2_top20_importancia.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f"\n    figura -> {FIG}/p2_top20_importancia.png")
print("\nPrueba 2 completada.")
