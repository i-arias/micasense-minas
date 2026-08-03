"""
PRUEBA 5 — Modelo final: EasyEnsemble x10 XGBoost, top-80 (XGBoost-gain), LOSO.

Selección de características dentro de cada fold (sin fuga). Mismo procedimiento y
semilla que el barrido P4 → reproduce la fila k=80.

Salidas:
  - resultados/p5_loso_por_fold.csv  (métricas por fold)
  - figuras/p5_matriz_confusion.png  (visualización BALANCEADA 1:1, estilo modelo final)

Ejecutar desde la raiz del proyecto:
    python thesisV2/scripts/p5_modelo_final.py
"""
import sys, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, 'thesisV2/scripts')
import comun
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix

RES = 'thesisV2/resultados'
FIG = 'thesisV2/figuras'
K = 70
N_ENS = 10
OBJ_FRAC = 0.0   # sin cuota (random 1:1) — la cuota empeora (P8)
SEED = 42

dg = comun.cargar(f'{RES}/features_global.csv')
cols = comun.feature_cols(dg)
X = dg[cols].values
y = dg['tiene_mina'].values
ses = dg['sesion'].values
clase = dg['clase'].values


def xgb_sub(seed):
    return XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                         subsample=0.8, colsample_bytree=0.8, eval_metric='logloss',
                         verbosity=0, random_state=seed, n_jobs=-1)


# ----- precompute identico a P4 (para reproducir k=80) -----
rng = np.random.default_rng(SEED)
folds = {}
for s in comun.SESIONES:
    tr = ses != s
    te = ses == s
    idx_tr = np.where(tr)[0]
    sub0, _, _ = comun.undersample_quota(y[tr], clase[tr], rng, OBJ_FRAC)
    ai0 = idx_tr[sub0]
    rk = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                       subsample=0.8, colsample_bytree=0.8, eval_metric='logloss',
                       importance_type='gain', verbosity=0, random_state=SEED, n_jobs=-1)
    rk.fit(X[ai0], y[ai0])
    rank_xgb = np.argsort(rk.feature_importances_)[::-1]
    # (RF ranking en P4 no consume rng; se omite aqui sin afectar las muestras)
    unders = [idx_tr[comun.undersample_quota(y[tr], clase[tr], rng, OBJ_FRAC)[0]]
              for _ in range(N_ENS)]
    folds[s] = dict(te=te, yte=y[te], rank_xgb=rank_xgb, unders=unders)

# ----- evaluar k=80 por fold -----
print(f"Modelo final: EasyEnsemble x{N_ENS}, top-{K}, LOSO")
filas = []
TP = FN = FP = TN = 0
for s in comun.SESIONES:
    f = folds[s]
    topk = f['rank_xgb'][:K]
    Xte = X[f['te']][:, topk]
    yte = f['yte']
    proba = np.zeros(len(yte))
    for j, ai in enumerate(f['unders']):
        proba += xgb_sub(j).fit(X[ai][:, topk], y[ai]).predict_proba(Xte)[:, 1]
    proba /= N_ENS
    auc = roc_auc_score(yte, proba)
    tn, fp, fn, tp = confusion_matrix(yte, (proba >= 0.5).astype(int), labels=[0, 1]).ravel()
    sens = tp/(tp+fn); spec = tn/(tn+fp)
    filas.append(dict(fold=f'S{s}', AUC=round(auc, 3), Sens=round(sens, 3),
                      Spec=round(spec, 3), TP=tp, FN=fn, FP=fp, TN=tn))
    TP += tp; FN += fn; FP += fp; TN += tn

res = pd.DataFrame(filas)
auc_prom = res['AUC'].mean()
sens_glob = TP/(TP+FN); spec_glob = TN/(TN+FP); fp_mina = FP/TP
res_tot = pd.DataFrame([dict(fold='Prom.', AUC=round(auc_prom, 3),
                             Sens=round(sens_glob, 3), Spec=round(spec_glob, 3),
                             TP=TP, FN=FN, FP=FP, TN=TN)])
out = pd.concat([res, res_tot], ignore_index=True)
out.to_csv(f'{RES}/p5_loso_por_fold.csv', index=False)
print(out.to_string(index=False))
print(f"\nAUC={auc_prom:.3f}  Sens={sens_glob:.3f}  Spec={spec_glob:.3f}  FP/mina={fp_mina:.2f}")

# ----- figura: matriz de confusion estilo P3c (conteos REALES, normalizada por fila) -----
labels = ['No mina', 'Mina']
cm = np.array([[TN, FP], [FN, TP]], dtype=float)        # filas: real [No mina, Mina]
cm_row = cm / cm.sum(axis=1, keepdims=True)             # normalizado por fila (recall)
fig, ax = plt.subplots(figsize=(6.6, 5.8))
im = ax.imshow(cm_row, cmap='Blues', vmin=0, vmax=1)
ax.set_xticks([0, 1]); ax.set_xticklabels(labels, fontsize=12)
ax.set_yticks([0, 1]); ax.set_yticklabels(labels, fontsize=12)
ax.set_xlabel('Predicción del modelo', fontsize=12, fontweight='bold')
ax.set_ylabel('Clase real', fontsize=12, fontweight='bold')
for i in range(2):
    for j in range(2):
        txt = f"{int(cm[i, j]):,}\n({cm_row[i, j]*100:.1f}%)"
        ax.text(j, i, txt, ha='center', va='center', fontsize=15,
                color='white' if cm_row[i, j] > 0.5 else 'black')
ax.set_title('Matriz de confusión — Modelo final\n'
             f'EasyEnsemble x10 XGBoost, top-{K}, LOSO S2–S7, umbral 0.5\n'
             f'AUC={auc_prom:.3f}  Sens={sens_glob:.3f}  Spec={spec_glob:.3f}  FP/mina={fp_mina:.2f}',
             fontsize=10, fontweight='bold')
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Recall por fila')
plt.tight_layout()
plt.savefig(f'{FIG}/p5_matriz_confusion.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f"figura -> {FIG}/p5_matriz_confusion.png")
print("Prueba 5 completada.")
