"""
P7 — Rendimiento por profundidad de enterramiento (1, 3, 5, 7 cm).

Reagrupa las ventanas de prueba del modelo final por la profundidad de su zona de mina.
AUC/Sens/Spec por profundidad (y por fold para la figura). Solo zonas de mina (tienen
positivos y negativos); las de control no tienen profundidad.

Ejecutar desde la raiz del proyecto:
    python thesisV2/scripts/p7_profundidad.py
"""
import sys, os, re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

RES = 'thesisV2/resultados'
FIG = 'thesisV2/figuras'

df = pd.read_csv(f'{RES}/p5_predicciones.csv')

def prof(zona):
    m = re.search(r'MINA_(\d+)CM', str(zona))
    return int(m.group(1)) if m else None

df['prof'] = df['zona'].apply(prof)
dm = df[df['prof'].notna()].copy()   # solo zonas de mina

PROFS = [1, 3, 5, 7]
SES = [2, 3, 4, 5, 6, 7]

# tabla por profundidad (global) + por fold
filas = []
porfold = {p: [] for p in PROFS}
for p in PROFS:
    sub = dm[dm['prof'] == p]
    auc = roc_auc_score(sub['tiene_mina'], sub['proba'])
    pred = (sub['proba'] >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(sub['tiene_mina'], pred, labels=[0, 1]).ravel()
    sens = tp/(tp+fn) if (tp+fn) else np.nan
    spec = tn/(tn+fp) if (tn+fp) else np.nan
    # auc por fold
    aucs_fold = []
    for s in SES:
        ss = sub[sub['sesion'] == s]
        if ss['tiene_mina'].nunique() == 2:
            aucs_fold.append(roc_auc_score(ss['tiene_mina'], ss['proba']))
        else:
            aucs_fold.append(np.nan)
    porfold[p] = aucs_fold
    filas.append(dict(prof=f'{p} cm', AUC=round(auc, 3),
                      AUC_std=round(np.nanstd(aucs_fold), 3),
                      Sens=round(sens, 3), Spec=round(spec, 3),
                      n_pos=int(tp+fn)))
tab = pd.DataFrame(filas)
tab.to_csv(f'{RES}/p7_profundidad.csv', index=False)
print("P7 — Rendimiento por profundidad (modelo final)")
print(tab.to_string(index=False))

# figura: AUC por profundidad (media + folds)
os.makedirs(FIG, exist_ok=True)
fig, ax = plt.subplots(figsize=(8, 5))
xs = PROFS
medias = [tab.loc[i, 'AUC'] for i in range(len(PROFS))]
stds = [tab.loc[i, 'AUC_std'] for i in range(len(PROFS))]
colors = plt.cm.tab10(np.linspace(0, 1, len(SES)))
for i, s in enumerate(SES):
    ys = [porfold[p][i] for p in PROFS]
    ax.plot(xs, ys, '--o', ms=4, color=colors[i], alpha=0.6, label=f'S{s}')
ax.plot(xs, medias, '-o', color='black', lw=2.5, ms=7, label='Media LOSO')
ax.fill_between(xs, np.array(medias)-np.array(stds), np.array(medias)+np.array(stds),
                color='gray', alpha=0.2)
ax.axhline(0.5, color='red', ls=':', lw=1, label='Azar')
ax.set_xticks(PROFS); ax.set_xlabel('Profundidad de enterramiento (cm)')
ax.set_ylabel('AUC (ROC)'); ax.set_ylim(0.5, 1.0)
ax.set_title('Rendimiento del modelo final por profundidad', fontweight='bold')
ax.legend(fontsize=8, ncol=2); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f'{FIG}/p7_profundidad.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f"\nfigura -> {FIG}/p7_profundidad.png")
