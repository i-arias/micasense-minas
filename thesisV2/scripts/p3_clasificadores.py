"""
PRUEBA 3 — Comparacion de clasificadores: RF vs SVM vs XGBoost vs EasyEnsemble.

Todos con las 378 features (dataset global), submuestreo 1:1 + cuota de objetos,
LOSO, umbral 0.5. Define el clasificador final.

Ejecutar desde la raiz del proyecto:
    python thesisV2/scripts/p3_clasificadores.py
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

print("=" * 70)
print("PRUEBA 3 — comparacion de clasificadores (378 features, 1:1 + objetos)")
print("=" * 70)

dg = comun.cargar(f'{RES}/features_global.csv')

filas = []

print("\n  RF ...")
_, a = comun.loso(dg, comun.rf_builder(), seed=42)
filas.append(('Random Forest', a))

print("  SVM (RBF) ...")
_, a = comun.loso(dg, comun.svm_builder(), seed=42)
filas.append(('SVM (RBF)', a))

print("  XGBoost ...")
_, a = comun.loso(dg, comun.xgb_builder(), seed=42)
filas.append(('XGBoost', a))

print("  EasyEnsemble x10 XGBoost ...")
_, a = comun.loso_easy(dg, comun.xgb_builder(), n_ens=10, seed=42)
filas.append(('EasyEnsemble x10', a))

tabla = pd.DataFrame([
    dict(clasificador=nom,
         AUC=round(ag['AUC'], 4), AUC_std=round(ag['AUC_std'], 3),
         Sens=round(ag['Sens'], 3), Spec=round(ag['Spec'], 3),
         FP_mina=round(ag['FP_mina'], 2))
    for nom, ag in filas
])
tabla.to_csv(f'{RES}/p3_clasificadores.csv', index=False)
print("\n" + tabla.to_string(index=False))

# ---- figura: AUC y Sensibilidad por clasificador
os.makedirs(FIG, exist_ok=True)
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
nombres = tabla['clasificador'].values
x = np.arange(len(nombres))
cols = ['#5D6D7E', '#5D6D7E', '#2980B9', '#C0392B']

axes[0].bar(x, tabla['AUC'], yerr=tabla['AUC_std'], color=cols, capsize=4)
axes[0].set_ylim(0.85, 0.96); axes[0].set_xticks(x)
axes[0].set_xticklabels(nombres, rotation=20, ha='right', fontsize=8)
axes[0].set_ylabel('AUC (LOSO)'); axes[0].set_title('AUC', fontweight='bold')
for i, v in enumerate(tabla['AUC']):
    axes[0].text(i, v + tabla['AUC_std'].iloc[i] + 0.002, f'{v:.3f}', ha='center', fontsize=8)

axes[1].bar(x, tabla['Sens'], color=cols)
axes[1].set_ylim(0.6, 0.9); axes[1].set_xticks(x)
axes[1].set_xticklabels(nombres, rotation=20, ha='right', fontsize=8)
axes[1].set_ylabel('Sensibilidad (umbral 0.5)'); axes[1].set_title('Sensibilidad', fontweight='bold')
for i, v in enumerate(tabla['Sens']):
    axes[1].text(i, v + 0.005, f'{v:.3f}', ha='center', fontsize=8)

fig.suptitle('Prueba 3 — Clasificadores (Enfoque B, 378 features, 1:1 + cuota objetos)',
             fontweight='bold')
plt.tight_layout()
plt.savefig(f'{FIG}/p3_clasificadores.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f"\nfigura -> {FIG}/p3_clasificadores.png")
print("Prueba 3 completada.")
