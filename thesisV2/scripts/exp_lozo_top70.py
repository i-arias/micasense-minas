"""
LOZO con el MODELO FINAL (top-70, EasyEnsemble x10) — coherente con §4.6.
Replica EXACTAMENTE el procedimiento de p5_modelo_final.py (selección top-70 por
ganancia de XGBoost dentro de cada fold + EasyEnsemble x10), cambiando solo el
agrupamiento del pliegue: por SESIÓN (LOSO, reproduce 0.947) vs por ZONA física
(LOZO, emplazamiento no visto). NO modifica el documento.
"""
import sys
sys.path.insert(0, 'thesisV2/scripts')
import comun
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix

RES = 'thesisV2/resultados'
K, N_ENS, SEED, OBJ_FRAC = 70, 10, 42, 0.0
MINAS = [f'MINA_{cm}CM_{i}' for cm in (1, 3, 5, 7) for i in (1, 2)]

dg = comun.cargar(f'{RES}/features_global.csv')
cols = comun.feature_cols(dg)
X = dg[cols].values
y = dg['tiene_mina'].values
clase = dg['clase'].values


def xgb_sub(seed):
    return XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                         subsample=0.8, colsample_bytree=0.8, eval_metric='logloss',
                         verbosity=0, random_state=seed, n_jobs=-1)


def rank_gain(ai):
    rk = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                       subsample=0.8, colsample_bytree=0.8, eval_metric='logloss',
                       importance_type='gain', verbosity=0, random_state=SEED, n_jobs=-1)
    rk.fit(X[ai], y[ai])
    return np.argsort(rk.feature_importances_)[::-1]


def evaluar(group_vec, groups, etiqueta):
    rng = np.random.default_rng(SEED)
    filas = []
    TP = FN = FP = TN = 0
    for g in groups:
        te = group_vec == g
        tr = ~te
        idx_tr = np.where(tr)[0]
        sub0, _, _ = comun.undersample_quota(y[tr], clase[tr], rng, OBJ_FRAC)
        topk = rank_gain(idx_tr[sub0])[:K]
        unders = [idx_tr[comun.undersample_quota(y[tr], clase[tr], rng, OBJ_FRAC)[0]]
                  for _ in range(N_ENS)]
        Xte = X[te][:, topk]; yte = y[te]
        proba = np.zeros(len(yte))
        for j, ai in enumerate(unders):
            proba += xgb_sub(j).fit(X[ai][:, topk], y[ai]).predict_proba(Xte)[:, 1]
        proba /= N_ENS
        auc = roc_auc_score(yte, proba)
        tn, fp, fn, tp = confusion_matrix(yte, (proba >= 0.5).astype(int), labels=[0, 1]).ravel()
        filas.append(dict(fold=str(g), AUC=auc, Sens=tp/(tp+fn), Spec=tn/(tn+fp),
                          TP=tp, FN=fn, FP=fp, TN=tn))
        TP += tp; FN += fn; FP += fp; TN += tn
    res = pd.DataFrame(filas)
    print(f"\n=== {etiqueta} (top-{K}) ===")
    print(res.to_string(index=False, float_format=lambda v: f'{v:.3f}'))
    print(f"  -> AUC={res.AUC.mean():.3f} ± {res.AUC.std():.3f} | "
          f"Sens={TP/(TP+FN):.3f} Spec={TN/(TN+FP):.3f} FP/mina={FP/TP:.2f} | "
          f"rango AUC {res.AUC.min():.3f}–{res.AUC.max():.3f}")
    return res.AUC.mean(), res.AUC.std()


a_loso, s_loso = evaluar(dg['sesion'].values, comun.SESIONES, "LOSO (sesión fuera)")
a_lozo, s_lozo = evaluar(dg['zona'].values, MINAS, "LOZO (zona/emplazamiento fuera)")
print("\n=== RESUMEN (top-70, modelo final) ===")
print(f"LOSO : AUC = {a_loso:.3f} ± {s_loso:.3f}")
print(f"LOZO : AUC = {a_lozo:.3f} ± {s_lozo:.3f}")
print(f"Delta AUC (LOZO-LOSO) = {a_lozo-a_loso:+.3f}")
