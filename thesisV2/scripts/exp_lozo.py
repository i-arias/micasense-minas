"""
Experimento de control para C1 (fuga por emplazamiento físico).
Compara, con EL MISMO modelo (EasyEnsemble x10 XGBoost, 378 features, 1:1):
  - LOSO  : dejar una SESIÓN fuera  (baseline del documento)
  - LOZO  : dejar una ZONA de mina fuera (emplazamiento físico nunca visto)
Solo cambia el agrupamiento del fold. NO modifica el documento.
"""
import sys, time
sys.path.insert(0, 'thesisV2/scripts')
import comun
import numpy as np
import pandas as pd

df = comun.cargar('thesisV2/resultados/features_global.csv')
cols = comun.feature_cols(df)
X = df[cols].values
y = df['tiene_mina'].values
clase = df['clase'].values

MINAS = [f'MINA_{cm}CM_{i}' for cm in (1, 3, 5, 7) for i in (1, 2)]


def easy_group(group_col, groups, n_ens=10, seed=42):
    g = df[group_col].values
    rng = np.random.default_rng(seed)
    filas = []
    TP = FN = FP = TN = 0
    for grp in groups:
        te = g == grp
        tr = ~te
        idx_tr = np.where(tr)[0]
        Xte, yte = X[te], y[te]
        proba = np.zeros(int(te.sum()))
        for _ in range(n_ens):
            sub, _, _ = comun.undersample_quota(y[tr], clase[tr], rng, 0.0)
            ai = idx_tr[sub]
            clf = comun.xgb_builder(seed)()
            clf.fit(X[ai], y[ai])
            proba += clf.predict_proba(Xte)[:, 1]
        proba /= n_ens
        m = comun.metricas_fold(yte, proba)
        m['fold'] = grp
        filas.append(m)
        TP += m['TP']; FN += m['FN']; FP += m['FP']; TN += m['TN']
    res = pd.DataFrame(filas)[['fold', 'AUC', 'Sens', 'Spec', 'TP', 'FN', 'FP', 'TN']]
    agg = dict(AUC=res.AUC.mean(), AUC_std=res.AUC.std(),
               Sens=TP/(TP+FN), Spec=TN/(TN+FP), FP_mina=FP/TP if TP else np.nan)
    return res, agg


t0 = time.time()
print("=== BASELINE: LOSO (dejar una SESIÓN fuera) ===")
res_s, agg_s = comun.loso_easy(df, comun.xgb_builder(), n_ens=10, seed=42, obj_frac=0.0)
print(res_s.to_string(index=False, float_format=lambda v: f'{v:.3f}'))
print(f"  -> AUC={agg_s['AUC']:.3f} ± {agg_s['AUC_std']:.3f} | Sens={agg_s['Sens']:.3f} "
      f"Spec={agg_s['Spec']:.3f} FP/mina={agg_s['FP_mina']:.2f}")

print("\n=== EXPERIMENTO: LOZO (dejar una ZONA DE MINA fuera) ===")
res_z, agg_z = easy_group('zona', MINAS, n_ens=10, seed=42)
print(res_z.to_string(index=False, float_format=lambda v: f'{v:.3f}'))
print(f"  -> AUC={agg_z['AUC']:.3f} ± {agg_z['AUC_std']:.3f} | Sens={agg_z['Sens']:.3f} "
      f"Spec={agg_z['Spec']:.3f} FP/mina={agg_z['FP_mina']:.2f}")

print("\n=== RESUMEN ===")
print(f"LOSO (sesión fuera) : AUC={agg_s['AUC']:.3f}  Sens={agg_s['Sens']:.3f}  Spec={agg_s['Spec']:.3f}  FP/mina={agg_s['FP_mina']:.2f}")
print(f"LOZO (zona fuera)   : AUC={agg_z['AUC']:.3f}  Sens={agg_z['Sens']:.3f}  Spec={agg_z['Spec']:.3f}  FP/mina={agg_z['FP_mina']:.2f}")
print(f"Delta AUC (LOZO-LOSO) = {agg_z['AUC']-agg_s['AUC']:+.3f}")
print(f"\n(tiempo: {time.time()-t0:.0f}s)")
