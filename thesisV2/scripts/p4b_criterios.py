"""
PRUEBA 4b — Criterios de selección de k (convergencia).

Aplica 3 criterios explícitos sobre el barrido (p4_barrido_k.csv) y la importancia:
  1. Tolerancia: menor k cuya AUC esté dentro de 0.005 / 0.01 del AUC máximo.
  2. Importancia acumulada: menor k que acumula 90% / 95% de la ganancia XGBoost (media folds).
  3. 99.5% del AUC máximo.

Genera la curva de importancia acumulada.

Ejecutar desde la raiz del proyecto:
    python thesisV2/scripts/p4b_criterios.py
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
from xgboost import XGBClassifier

RES = 'thesisV2/resultados'
FIG = 'thesisV2/figuras'

sweep = pd.read_csv(f'{RES}/p4_barrido_k.csv')
auc_max = sweep['AUC'].max()

def menor_k_auc(umbral):
    cand = sweep.loc[sweep['AUC'] >= umbral, 'k']
    return int(cand.min()) if len(cand) else None

# ---- importancia acumulada (gain media sobre folds) ----
dg = comun.cargar(f'{RES}/features_global.csv')
cols = comun.feature_cols(dg)
X = dg[cols].values; y = dg['tiene_mina'].values
ses = dg['sesion'].values; clase = dg['clase'].values
rng = np.random.default_rng(42)
imp = np.zeros(len(cols))
for s in comun.SESIONES:
    tr = ses != s; idx_tr = np.where(tr)[0]
    sub, _, _ = comun.undersample_quota(y[tr], clase[tr], rng, 0.0)
    ai = idx_tr[sub]
    rk = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                       subsample=0.8, colsample_bytree=0.8, eval_metric='logloss',
                       importance_type='gain', verbosity=0, random_state=42, n_jobs=-1)
    rk.fit(X[ai], y[ai])
    imp += rk.feature_importances_
imp /= len(comun.SESIONES)
imp_sorted = np.sort(imp)[::-1]
cum = np.cumsum(imp_sorted) / imp_sorted.sum()

def menor_k_cum(frac):
    return int(np.searchsorted(cum, frac) + 1)

# ---- tabla de criterios ----
criterios = [
    ('Tolerancia 0.005 (AUC)', menor_k_auc(auc_max - 0.005)),
    ('Tolerancia 0.010 (AUC)', menor_k_auc(auc_max - 0.010)),
    ('99.5% del AUC máximo',   menor_k_auc(0.995 * auc_max)),
    ('Importancia acum. 90%',  menor_k_cum(0.90)),
    ('Importancia acum. 95%',  menor_k_cum(0.95)),
]
tab = pd.DataFrame(criterios, columns=['criterio', 'k_sugerido'])
# anexar metricas de ese k
def met(k):
    r = sweep[sweep['k'] == k]
    if len(r) == 0:  # k no esta en la malla; tomar el k de la malla >= k
        r = sweep[sweep['k'] >= k].head(1)
    r = r.iloc[0]
    return f"AUC={r['AUC']:.4f} Sens={r['Sens']:.3f} Spec={r['Spec']:.3f} FP/mina={r['FP_mina']:.2f}"
tab['metricas (k de la malla)'] = tab['k_sugerido'].apply(met)
tab.to_csv(f'{RES}/p4b_criterios.csv', index=False)

print(f"AUC max = {auc_max:.4f}")
print(tab.to_string(index=False))

# ---- figura importancia acumulada ----
os.makedirs(FIG, exist_ok=True)
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(range(1, len(cum)+1), cum*100, color='#2C3E50')
for frac, c in [(0.90, '#E67E22'), (0.95, '#C0392B')]:
    kf = menor_k_cum(frac)
    ax.axhline(frac*100, color=c, ls='--', lw=1)
    ax.axvline(kf, color=c, ls=':', lw=1)
    ax.text(kf+3, frac*100-4, f'{int(frac*100)}% -> k={kf}', color=c, fontsize=9)
ax.set_xlabel('k (número de características, ordenadas por ganancia)')
ax.set_ylabel('Importancia acumulada (%)')
ax.set_title('Prueba 4b — Importancia acumulada (XGBoost-gain, media LOSO)', fontweight='bold')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f'{FIG}/p4b_importancia_acumulada.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f"\nfigura -> {FIG}/p4b_importancia_acumulada.png")
