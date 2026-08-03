"""
PRUEBA 3c — Matrices de confusión de cada clasificador (criterio de selección extra).

RF, SVM, XGBoost, EasyEnsemble — 378 features, 1:1 + cuota objetos, LOSO, umbral 0.5.
Conteos agregados sobre los 6 folds (distribución real). Misma semilla que P3 → coinciden.

Ejecutar desde la raiz del proyecto:
    python thesisV2/scripts/p3c_matrices.py
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

RES = 'thesisV2/resultados'
FIG = 'thesisV2/figuras'

dg = comun.cargar(f'{RES}/features_global.csv')

print("Calculando matrices de confusion (mismas condiciones que P3) ...")
resultados = []
print("  RF ...");          _, a_rf  = comun.loso(dg, comun.rf_builder(),  seed=42); resultados.append(('Random Forest', a_rf))
print("  SVM ...");         _, a_sv  = comun.loso(dg, comun.svm_builder(), seed=42); resultados.append(('SVM (RBF)', a_sv))
print("  XGBoost ...");     _, a_xg  = comun.loso(dg, comun.xgb_builder(), seed=42); resultados.append(('XGBoost', a_xg))
print("  EasyEnsemble ..."); _, a_ee = comun.loso_easy(dg, comun.xgb_builder(), n_ens=10, seed=42); resultados.append(('EasyEnsemble x10', a_ee))

# guardar conteos
filas = []
for nom, a in resultados:
    filas.append(dict(clasificador=nom, TP=a['TP'], FN=a['FN'], FP=a['FP'], TN=a['TN'],
                      Sens=round(a['Sens'], 3), Spec=round(a['Spec'], 3),
                      AUC=round(a['AUC'], 4), FP_mina=round(a['FP_mina'], 2)))
pd.DataFrame(filas).to_csv(f'{RES}/p3c_matrices.csv', index=False)

# ---------------------------------------------------------------- figura 2x2
os.makedirs(FIG, exist_ok=True)
fig, axes = plt.subplots(2, 2, figsize=(11, 9))
labels = ['No mina', 'Mina']

for ax, (nom, a) in zip(axes.ravel(), resultados):
    TP, FN, FP, TN = a['TP'], a['FN'], a['FP'], a['TN']
    cm = np.array([[TN, FP], [FN, TP]], dtype=float)        # filas: real [No mina, Mina]
    cm_row = cm / cm.sum(axis=1, keepdims=True)             # normalizado por fila (recall)
    im = ax.imshow(cm_row, cmap='Blues', vmin=0, vmax=1)
    ax.set_xticks([0, 1]); ax.set_xticklabels(labels)
    ax.set_yticks([0, 1]); ax.set_yticklabels(labels)
    ax.set_xlabel('Predicción'); ax.set_ylabel('Clase real')
    ax.set_title(f"{nom}\nAUC={a['AUC']:.3f}  Sens={a['Sens']:.3f}  "
                 f"Spec={a['Spec']:.3f}  FP/mina={a['FP_mina']:.2f}", fontsize=9, fontweight='bold')
    for i in range(2):
        for j in range(2):
            txt = f"{int(cm[i, j]):,}\n({cm_row[i, j]*100:.1f}%)"
            ax.text(j, i, txt, ha='center', va='center', fontsize=11,
                    color='white' if cm_row[i, j] > 0.5 else 'black')

fig.suptitle('Matrices de confusión por clasificador (LOSO, umbral 0.5)\n'
             '% = por fila (recall de cada clase real)', fontweight='bold')
plt.tight_layout()
plt.savefig(f'{FIG}/p3c_matrices.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f"\nfigura -> {FIG}/p3c_matrices.png")
print(pd.DataFrame(filas).to_string(index=False))
print("Prueba 3c completada.")
