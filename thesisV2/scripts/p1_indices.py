"""
PRUEBA 1 — Calculo de indices: imagen completa (global) vs ventana por ventana.

Demuestra:
  1. Equivalencia numerica de las features (deberian ser identicas).
  2. Mismas metricas LOSO (AUC, Sens, Spec, FP/mina) con RF + submuestreo 1:1 (cuota objetos).
  3. Mismo reparto del violin (firma espectral por clase).
  4. Ahorro de tiempo de extraccion (global < ventana).

Ejecutar desde la raiz del proyecto:
    python thesisV2/scripts/p1_indices.py
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

sys.path.insert(0, 'thesisV2/scripts')
import comun

RES = 'thesisV2/resultados'
FIG = 'thesisV2/figuras'
os.makedirs(FIG, exist_ok=True)

dv = comun.cargar(f'{RES}/features_ventana.csv')
dg = comun.cargar(f'{RES}/features_global.csv')

print("=" * 70)
print("PRUEBA 1 — indices imagen completa (global) vs ventana por ventana")
print("=" * 70)

# ---------------------------------------------------------------- 1. equivalencia
feat = comun.feature_cols(dv)
assert len(dv) == len(dg), "distinto numero de parches!"
maxdiff = np.nanmax(np.abs(dv[feat].values - dg[feat].values))
print(f"\n[1] Equivalencia numerica de features")
print(f"    parches: {len(dv)}  | features: {len(feat)}")
print(f"    max |global - ventana| = {maxdiff:.3e}  -> {'IDENTICOS' if maxdiff < 1e-9 else 'DIFIEREN'}")

# ---------------------------------------------------------------- 2. metricas LOSO
print(f"\n[2] Metricas LOSO (RF, submuestreo 1:1 con cuota de objetos)")
filas_metricas = []
for nombre, df in [('ventana', dv), ('global', dg)]:
    res, agg = comun.loso(df, comun.rf_builder(seed=42), seed=42, obj_frac=0.0)
    print(f"    {nombre:8s}: AUC={agg['AUC']:.4f}±{agg['AUC_std']:.3f}  "
          f"Sens={agg['Sens']:.3f}  Spec={agg['Spec']:.3f}  FP/mina={agg['FP_mina']:.2f}")
    filas_metricas.append(dict(modo=nombre, AUC=round(agg['AUC'], 4),
                               AUC_std=round(agg['AUC_std'], 3),
                               Sens=round(agg['Sens'], 3), Spec=round(agg['Spec'], 3),
                               FP_mina=round(agg['FP_mina'], 2)))
pd.DataFrame(filas_metricas).to_csv(f'{RES}/p1_metricas.csv', index=False)

# ---------------------------------------------------------------- 3. tiempos
print(f"\n[3] Tiempos de extraccion")
try:
    t = pd.read_csv(f'{RES}/tiempos_extraccion.csv').iloc[0]
    sp = t['ventana'] / t['global']
    print(f"    ventana={t['ventana']:.1f}s  global={t['global']:.1f}s  speedup={sp:.2f}x")
except Exception as e:
    print("    (sin tiempos)", e)

# ---------------------------------------------------------------- 4. violin
print(f"\n[4] Violin (firma por clase) sobre el dataset global")
CLASES = ['MAP', 'Botella', 'Lata', 'Piedra', 'Control']
COL = {'MAP': '#C0392B', 'Botella': '#2980B9', 'Lata': '#E67E22',
       'Piedra': '#7D3C98', 'Control': '#27AE60'}
BANDAS = ['Blue', 'Green', 'Red', 'RedEdge', 'NIR']
LBL_B = ['Azul\n(475)', 'Verde\n(560)', 'Rojo\n(668)', 'Red Edge\n(717)', 'NIR\n(842)']
IDX = ['NDVI', 'NDRE', 'RENDVI', 'CI_RedEdge', 'BSI', 'NDSI']
LBL_I = ['NDVI', 'NDRE', 'RENDVI', 'CI\nRedEdge', 'BSI', 'NDSI']

df = dg
nstr = {c: int((df['clase'] == c).sum()) for c in CLASES}
fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor='white')
w = 0.15
off = np.linspace(-2, 2, 5) * w

def panel(ax, cols, xt):
    x = np.arange(len(cols))
    for i, c in enumerate(CLASES):
        sub = df[df['clase'] == c]
        data = [sub[f'{col}_mean'].dropna().values for col in cols]
        data = [d if len(d) > 1 else np.array([0., 0.]) for d in data]
        parts = ax.violinplot(data, positions=x + off[i], widths=w*0.95, showmedians=True, showextrema=False)
        for pc in parts['bodies']:
            pc.set_facecolor(COL[c]); pc.set_edgecolor(COL[c]); pc.set_alpha(0.65)
        parts['cmedians'].set_color('black'); parts['cmedians'].set_linewidth(1.1)
    ax.set_xticks(x); ax.set_xticklabels(xt, fontsize=9)
    ax.grid(axis='y', alpha=0.3)

panel(axes[0], BANDAS, LBL_B)
axes[0].set_title('Reflectancia por banda', fontweight='bold')
axes[0].set_ylabel('Reflectancia del parche')
panel(axes[1], IDX, LBL_I)
axes[1].set_title('Índices espectrales clave', fontweight='bold')
axes[1].set_ylabel('Valor del índice')
axes[1].axhline(0, color='#888', lw=0.8, ls='--')
leg = [mpatches.Patch(facecolor=COL[c], alpha=0.65, label=f'{c} (n={nstr[c]})') for c in CLASES]
axes[0].legend(handles=leg, fontsize=8, loc='upper left')
fig.suptitle('Firma espectral por clase (dataset global, S2–S7)',
             fontweight='bold')
plt.tight_layout()
plt.savefig(f'{FIG}/p1_violin_firma.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f"    figura -> {FIG}/p1_violin_firma.png")

print("\nPrueba 1 completada.")
