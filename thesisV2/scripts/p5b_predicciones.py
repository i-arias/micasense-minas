"""
P5b — Predicciones por ventana del modelo final (EasyEnsemble x10, top-80, LOSO).

Guarda la probabilidad LOSO de cada ventana (cada una es test en su fold), para que
P6 (condiciones), P7 (profundidad) y P8 (FP vs objetos) solo lean este archivo.

Genera dos columnas: con cuota de objetos (obj_frac=1/3, el modelo final) y SIN cuota
(obj_frac=0), para cuantificar el efecto de la cuota en P8.

Salida: resultados/p5_predicciones.csv

Ejecutar desde la raiz del proyecto:
    python thesisV2/scripts/p5b_predicciones.py
"""
import sys, os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, 'thesisV2/scripts')
import comun
from xgboost import XGBClassifier

RES = 'thesisV2/resultados'
K = 70
N_ENS = 10
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


def run_final(obj_frac):
    proba_all = np.full(len(dg), np.nan)
    rng = np.random.default_rng(SEED)
    for s in comun.SESIONES:
        tr = ses != s
        te = ses == s
        idx_tr = np.where(tr)[0]
        sub0, _, _ = comun.undersample_quota(y[tr], clase[tr], rng, obj_frac)
        ai0 = idx_tr[sub0]
        rk = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                           subsample=0.8, colsample_bytree=0.8, eval_metric='logloss',
                           importance_type='gain', verbosity=0, random_state=SEED, n_jobs=-1)
        rk.fit(X[ai0], y[ai0])
        topk = np.argsort(rk.feature_importances_)[::-1][:K]
        Xte = X[te][:, topk]
        proba = np.zeros(int(te.sum()))
        for j in range(N_ENS):
            sub, _, _ = comun.undersample_quota(y[tr], clase[tr], rng, obj_frac)
            ai = idx_tr[sub]
            proba += xgb_sub(j).fit(X[ai][:, topk], y[ai]).predict_proba(Xte)[:, 1]
        proba_all[te] = proba / N_ENS
        print(f"  fold S{s} listo (obj_frac={obj_frac})")
    return proba_all


print("Modelo FINAL sin cuota (obj_frac=0) ...")
p_final = run_final(0.0)
print("Alternativa CON cuota (obj_frac=1/3) — solo para experimento P8 ...")
p_cuota = run_final(1/3)

out = dg[['sesion', 'zona', 'clase', 'tiene_mina', 'parche_x', 'parche_y']].copy()
out['proba'] = p_final            # modelo final (sin cuota)
out['proba_concuota'] = p_cuota   # alternativa con cuota (P8)
out.to_csv(f'{RES}/p5_predicciones.csv', index=False)
print(f"\nGuardado: {RES}/p5_predicciones.csv  ({len(out)} ventanas)")
