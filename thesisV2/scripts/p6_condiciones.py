"""
P6 — Rendimiento por sesión y condiciones de captura.

AUC por sesión del modelo final vs hora / humedad / radiación. Inspección cualitativa
(n=6: no se calcula correlación, se observa si hay degradación sistemática).

Ejecutar desde la raiz del proyecto:
    python thesisV2/scripts/p6_condiciones.py
"""
import sys, os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

RES = 'thesisV2/resultados'

# condiciones de captura (Tabla del documento)
COND = {
    2: ('13:10', 72.8, 778.7),
    3: ('12:34', 60.7, 346.7),
    4: ('17:21', 61.8, 64.5),
    5: ('17:17', 63.6, 73.3),
    6: ('07:30', 84.5, 161.5),
    7: ('16:40', 65.5, 143.9),
}

df = pd.read_csv(f'{RES}/p5_predicciones.csv')
filas = []
for s in sorted(COND, key=lambda k: COND[k][0]):   # ordenar por hora
    sub = df[df['sesion'] == s]
    auc = roc_auc_score(sub['tiene_mina'], sub['proba'])
    hora, hr, rad = COND[s]
    filas.append(dict(sesion=f'S{s}', hora=hora, HR=hr, Rad=rad, AUC=round(auc, 3)))
tab = pd.DataFrame(filas)
tab.to_csv(f'{RES}/p6_condiciones.csv', index=False)
print("P6 — AUC por sesión vs condiciones de captura (modelo final, ordenado por hora)")
print(tab.to_string(index=False))
print(f"\nAUC min={tab.AUC.min()}  max={tab.AUC.max()}  rango={tab.AUC.max()-tab.AUC.min():.3f}")
print("Inspección cualitativa (n=6): no se calcula correlación; se observa estabilidad.")
