"""
PRUEBA 4 — Selección del número de características (k).

(a) Barrido de k con PASO UNIFORME (10, 20, ..., 370, 378), selección por la
    importancia de ganancia (gain) de XGBoost calculada DENTRO de cada fold.
    Clasificador: EasyEnsemble x10. Criterio del k óptimo: el menor k que alcanza
    el 99.5 % del AUC máximo.

(b) Comparación del MÉTODO de selección: XGBoost-gain vs RF-Gini, en k=50 y k=100,
    para justificar usar la importancia propia del modelo final.

Eficiencia: el ranking (XGB y RF) y los 10 submuestreos se calculan UNA vez por fold
y se reutilizan para todos los k; solo cambia el subconjunto de columnas.

Ejecutar desde la raiz del proyecto:
    python thesisV2/scripts/p4_seleccion.py
"""
import sys, os, time
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix

RES = 'thesisV2/resultados'
FIG = 'thesisV2/figuras'
N_ENS = 10
OBJ_FRAC = 0.0   # sin cuota (random 1:1) — la cuota empeora (ver P8)
K_VALUES = list(range(10, 371, 10)) + [378]
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


# ---------------------------------------------------------------- precompute por fold
print("Precomputando rankings y submuestreos por fold ...")
rng = np.random.default_rng(SEED)
folds = {}
for s in comun.SESIONES:
    tr = ses != s
    te = ses == s
    idx_tr = np.where(tr)[0]

    # ranking XGBoost-gain (sobre un submuestreo del fold, todas las features)
    sub0, _, _ = comun.undersample_quota(y[tr], clase[tr], rng, OBJ_FRAC)
    ai0 = idx_tr[sub0]
    rk = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                       subsample=0.8, colsample_bytree=0.8, eval_metric='logloss',
                       importance_type='gain', verbosity=0, random_state=SEED, n_jobs=-1)
    rk.fit(X[ai0], y[ai0])
    rank_xgb = np.argsort(rk.feature_importances_)[::-1]

    # ranking RF-Gini (mismo submuestreo)
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=SEED, n_jobs=-1)
    rf.fit(X[ai0], y[ai0])
    rank_rf = np.argsort(rf.feature_importances_)[::-1]

    # 10 submuestreos fijos (se reutilizan para todos los k)
    unders = [idx_tr[comun.undersample_quota(y[tr], clase[tr], rng, OBJ_FRAC)[0]]
              for _ in range(N_ENS)]

    folds[s] = dict(te=te, yte=y[te], rank_xgb=rank_xgb, rank_rf=rank_rf, unders=unders)
print("  listo.")


def eval_k(k, rankkey):
    aucs = []
    TP = FN = FP = TN = 0
    for s in comun.SESIONES:
        f = folds[s]
        topk = f[rankkey][:k]
        Xte = X[f['te']][:, topk]
        yte = f['yte']
        proba = np.zeros(len(yte))
        for j, ai in enumerate(f['unders']):
            clf = xgb_sub(j)
            clf.fit(X[ai][:, topk], y[ai])
            proba += clf.predict_proba(Xte)[:, 1]
        proba /= len(f['unders'])
        aucs.append(roc_auc_score(yte, proba))
        tn, fp, fn, tp = confusion_matrix(yte, (proba >= 0.5).astype(int), labels=[0, 1]).ravel()
        TP += tp; FN += fn; FP += fp; TN += tn
    return dict(k=k, AUC=np.mean(aucs), AUC_std=np.std(aucs),
                Sens=TP/(TP+FN), Spec=TN/(TN+FP), FP_mina=FP/TP if TP else np.nan)


# ---------------------------------------------------------------- (a) barrido
print(f"\n(a) Barrido de k (XGBoost-gain), {len(K_VALUES)} puntos ...")
t0 = time.perf_counter()
filas = []
for k in K_VALUES:
    r = eval_k(k, 'rank_xgb')
    filas.append(r)
    print(f"  k={k:3d}  AUC={r['AUC']:.4f}  Sens={r['Sens']:.3f}  "
          f"Spec={r['Spec']:.3f}  FP/mina={r['FP_mina']:.2f}")
sweep = pd.DataFrame(filas)
sweep.to_csv(f'{RES}/p4_barrido_k.csv', index=False)

# criterio: menor k con AUC >= 99.5% del AUC maximo
auc_max = sweep['AUC'].max()
umbral = 0.995 * auc_max
kopt = int(sweep.loc[sweep['AUC'] >= umbral, 'k'].min())
print(f"\n  AUC max = {auc_max:.4f} (en k={int(sweep.loc[sweep.AUC.idxmax(),'k'])})")
print(f"  criterio 99.5% -> umbral {umbral:.4f} -> k OPTIMO = {kopt}")
print(f"  barrido en {time.perf_counter()-t0:.0f}s")

# ---------------------------------------------------------------- (b) metodo seleccion
print(f"\n(b) Metodo de seleccion XGBoost-gain vs RF-Gini ...")
filas_b = []
for k in [50, 100]:
    rx = sweep[sweep['k'] == k].iloc[0]
    rr = eval_k(k, 'rank_rf')
    filas_b.append(dict(k=k, metodo='XGBoost-gain', AUC=round(rx['AUC'], 4),
                        Sens=round(rx['Sens'], 3), Spec=round(rx['Spec'], 3), FP_mina=round(rx['FP_mina'], 2)))
    filas_b.append(dict(k=k, metodo='RF-Gini', AUC=round(rr['AUC'], 4),
                        Sens=round(rr['Sens'], 3), Spec=round(rr['Spec'], 3), FP_mina=round(rr['FP_mina'], 2)))
comp = pd.DataFrame(filas_b)
comp.to_csv(f'{RES}/p4_metodo_seleccion.csv', index=False)
print(comp.to_string(index=False))

# ---------------------------------------------------------------- figuras
os.makedirs(FIG, exist_ok=True)
fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
ax[0].plot(sweep['k'], sweep['AUC'], '-o', ms=3, color='#2980B9')
ax[0].axhline(umbral, color='#C0392B', ls='--', lw=1, label=f'99.5% del AUC máx')
ax[0].axvline(kopt, color='#27AE60', ls='--', lw=1.5, label=f'k óptimo = {kopt}')
ax[0].set_xlabel('k (número de características)'); ax[0].set_ylabel('AUC (LOSO)')
ax[0].set_title('AUC vs k — paso uniforme 10', fontweight='bold'); ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)

ax[1].plot(sweep['k'], sweep['Sens'], '-o', ms=3, color='#E67E22', label='Sensibilidad')
ax[1].plot(sweep['k'], sweep['Spec'], '-o', ms=3, color='#7D3C98', label='Especificidad')
ax[1].axvline(kopt, color='#27AE60', ls='--', lw=1.5)
ax[1].set_xlabel('k (número de características)'); ax[1].set_ylabel('Métrica (umbral 0.5)')
ax[1].set_title('Sens / Spec vs k', fontweight='bold'); ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)

fig.suptitle('Selección de k (EasyEnsemble, XGBoost-gain)', fontweight='bold')
plt.tight_layout()
plt.savefig(f'{FIG}/p4_barrido_k.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

# figura comparacion metodo
fig, ax = plt.subplots(figsize=(7, 4.2))
ks = [50, 100]
w = 0.35
xg = comp[comp.metodo == 'XGBoost-gain'].set_index('k')
rf = comp[comp.metodo == 'RF-Gini'].set_index('k')
xpos = np.arange(len(ks))
ax.bar(xpos - w/2, [xg.loc[k, 'AUC'] for k in ks], w, label='XGBoost-gain', color='#2980B9')
ax.bar(xpos + w/2, [rf.loc[k, 'AUC'] for k in ks], w, label='RF-Gini', color='#95A5A6')
ax.set_xticks(xpos); ax.set_xticklabels([f'k={k}' for k in ks])
ax.set_ylim(0.90, 0.95); ax.set_ylabel('AUC (LOSO)')
ax.set_title('Método de selección: XGBoost-gain vs RF-Gini', fontweight='bold')
ax.legend()
for i, k in enumerate(ks):
    ax.text(i - w/2, xg.loc[k, 'AUC'] + 0.001, f"{xg.loc[k,'AUC']:.3f}", ha='center', fontsize=8)
    ax.text(i + w/2, rf.loc[k, 'AUC'] + 0.001, f"{rf.loc[k,'AUC']:.3f}", ha='center', fontsize=8)
plt.tight_layout()
plt.savefig(f'{FIG}/p4_metodo_seleccion.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

print(f"\nk OPTIMO = {kopt}")
print("Prueba 4 completada.")
