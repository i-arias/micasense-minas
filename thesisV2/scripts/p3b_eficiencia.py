"""
PRUEBA 3b — Eficiencia computacional de los clasificadores.

Mide, sobre LOSO (mismas condiciones que P3):
  - tiempo total de ENTRENAMIENTO (los 6 folds),
  - tiempo de INFERENCIA por ventana (coste de desplegar),
  - numero de modelos entrenados (complejidad),
para sopesar metricas vs coste.

Ejecutar desde la raiz del proyecto:
    python thesisV2/scripts/p3b_eficiencia.py
"""
import sys, os, time
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, 'thesisV2/scripts')
import comun

RES = 'thesisV2/resultados'


def medir(df, build, ensemble=False, n_ens=10, seed=42, obj_frac=1/3):
    cols = comun.feature_cols(df)
    X = df[cols].values
    y = df['tiene_mina'].values
    ses = df['sesion'].values
    clase = df['clase'].values
    rng = np.random.default_rng(seed)
    t_train = t_inf = 0.0
    n_win = n_models = 0
    for s in comun.SESIONES:
        tr = ses != s
        te = ses == s
        idx_tr = np.where(tr)[0]
        Xte = X[te]
        reps = n_ens if ensemble else 1
        for _ in range(reps):
            sub, _, _ = comun.undersample_quota(y[tr], clase[tr], rng, obj_frac)
            ai = idx_tr[sub]
            clf = build()
            t0 = time.perf_counter(); clf.fit(X[ai], y[ai]); t_train += time.perf_counter() - t0
            t0 = time.perf_counter(); _ = clf.predict_proba(Xte)[:, 1]; t_inf += time.perf_counter() - t0
            n_models += 1
        n_win += int(te.sum())   # ventanas unicas (una vez por fold)
    return dict(train_s=t_train, infer_us_win=t_inf / n_win * 1e6, n_models=n_models)


print("=" * 70)
print("PRUEBA 3b — eficiencia computacional")
print("=" * 70)

dg = comun.cargar(f'{RES}/features_global.csv')

config = [
    ('Random Forest',     comun.rf_builder(),  False),
    ('SVM (RBF)',         comun.svm_builder(), False),
    ('XGBoost',           comun.xgb_builder(), False),
    ('EasyEnsemble x10',  comun.xgb_builder(), True),
]

# metricas de P3 (para la tabla coste-beneficio)
metr = pd.read_csv(f'{RES}/p3_clasificadores.csv').set_index('clasificador')

filas = []
for nombre, build, ens in config:
    print(f"  midiendo {nombre} ...")
    m = medir(dg, build, ensemble=ens)
    filas.append(dict(
        clasificador=nombre,
        AUC=metr.loc[nombre, 'AUC'], Sens=metr.loc[nombre, 'Sens'],
        modelos=m['n_models'],
        train_total_s=round(m['train_s'], 2),
        infer_us_por_ventana=round(m['infer_us_win'], 1),
    ))

tabla = pd.DataFrame(filas)
# coste relativo de inferencia vs XGBoost (1 modelo)
base = tabla.loc[tabla.clasificador == 'XGBoost', 'infer_us_por_ventana'].values[0]
tabla['infer_rel_vs_XGB'] = (tabla['infer_us_por_ventana'] / base).round(1)
tabla.to_csv(f'{RES}/p3b_eficiencia.csv', index=False)

print("\n" + tabla.to_string(index=False))
print("\nPrueba 3b completada.")
