"""
Violín de las TOP características (por importancia XGBoost-gain) por clase.

Responde la pregunta de la profe: ¿el violín con las características más importantes
se ve distinto al de los índices fijos (medias)? Aquí cada subpanel es una de las top-9
características reales que usa el modelo (incluyen std/p25, no solo medias), con su propia
escala, mostrando las 5 clases.

Ejecutar desde la raiz del proyecto:
    python thesisV2/scripts/violin_top_features.py
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
from xgboost import XGBClassifier

RES = 'thesisV2/resultados'
FIG = 'thesisV2/figuras'
N_TOP = 9

dg = comun.cargar(f'{RES}/features_global.csv')
cols = comun.feature_cols(dg)
X = dg[cols].values
y = dg['tiene_mina'].values
ses = dg['sesion'].values
clase = dg['clase'].values

# importancia XGBoost-gain media sobre folds (random 1:1, como el modelo final)
rng = np.random.default_rng(42)
imp = np.zeros(len(cols))
for s in comun.SESIONES:
    tr = ses != s
    idx_tr = np.where(tr)[0]
    sub, _, _ = comun.undersample_quota(y[tr], clase[tr], rng, 0.0)
    ai = idx_tr[sub]
    rk = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                       subsample=0.8, colsample_bytree=0.8, eval_metric='logloss',
                       importance_type='gain', verbosity=0, random_state=42, n_jobs=-1)
    rk.fit(X[ai], y[ai])
    imp += rk.feature_importances_
imp = pd.Series(imp / len(comun.SESIONES), index=cols).sort_values(ascending=False)
top = list(imp.head(N_TOP).index)
print("Top características (XGBoost-gain):")
for i, f in enumerate(top, 1):
    print(f"  {i}. {f}  ({imp[f]:.4f})")

CLASES = ['MAP', 'Botella', 'Lata', 'Piedra', 'Control']
COL = {'MAP': '#C0392B', 'Botella': '#2980B9', 'Lata': '#E67E22',
       'Piedra': '#7D3C98', 'Control': '#27AE60'}

fig, axes = plt.subplots(3, 3, figsize=(14, 11))
for ax, feat in zip(axes.ravel(), top):
    data = [dg[dg['clase'] == c][feat].dropna().values for c in CLASES]
    parts = ax.violinplot(data, positions=range(len(CLASES)), showmedians=True, showextrema=False)
    for pc, c in zip(parts['bodies'], CLASES):
        pc.set_facecolor(COL[c]); pc.set_edgecolor(COL[c]); pc.set_alpha(0.65)
    parts['cmedians'].set_color('black')
    ax.set_xticks(range(len(CLASES)))
    ax.set_xticklabels(CLASES, fontsize=8, rotation=20)
    ax.set_title(feat, fontsize=10, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
leg = [mpatches.Patch(facecolor=COL[c], alpha=0.65, label=c) for c in CLASES]
fig.legend(handles=leg, loc='lower center', ncol=5, fontsize=10, bbox_to_anchor=(0.5, -0.01))
fig.suptitle('Violín de las top-9 características (XGBoost-gain) por clase — S2–S7',
             fontsize=13, fontweight='bold')
plt.tight_layout(rect=[0, 0.03, 1, 1])
os.makedirs(FIG, exist_ok=True)
plt.savefig(f'{FIG}/violin_top_features.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f"\nfigura -> {FIG}/violin_top_features.png")
