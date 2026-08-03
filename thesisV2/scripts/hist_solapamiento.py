"""
Histograma de solapamiento de P(MAP): distribución de la probabilidad que el
modelo final asigna a las ventanas de MINA vs NO-MINA (LOSO, p5_predicciones).
La zona donde ambas distribuciones se solapan = casos que el modelo no puede
separar limpiamente. Cada clase se normaliza a su propio total (densidad por
clase) por el desbalance 30:1.

Ejecutar desde la raiz del proyecto:
    python thesisV2/scripts/hist_solapamiento.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FIG = 'thesisV2/figuras'
df = pd.read_csv('thesisV2/resultados/p5_predicciones.csv')
p_mina = df.loc[df['tiene_mina'] == 1, 'proba'].values
p_nomina = df.loc[df['tiene_mina'] == 0, 'proba'].values
UMBRAL = 0.5

# métricas para anotar
fn = int((p_mina < UMBRAL).sum())
tp = int((p_mina >= UMBRAL).sum())
fp = int((p_nomina >= UMBRAL).sum())
tn = int((p_nomina < UMBRAL).sum())
print(f"minas: {len(p_mina)} (TP={tp}, FN={fn}) | no-minas: {len(p_nomina)} (TN={tn}, FP={fp})")

bins = np.linspace(0, 1, 31)
ROJO, AZUL = '#C0392B', '#2471A3'

fig, ax = plt.subplots(figsize=(8.2, 4.8), facecolor='white')

# densidad por clase (cada clase integra a 1)
for vals, col, lab in [(p_nomina, AZUL, 'Ventanas sin mina (n=%d)' % len(p_nomina)),
                       (p_mina,   ROJO, 'Ventanas de mina (n=%d)' % len(p_mina))]:
    w = np.ones_like(vals) / len(vals)
    ax.hist(vals, bins=bins, weights=w, color=col, alpha=0.55,
            edgecolor=col, linewidth=0.8, label=lab)

ax.axvline(UMBRAL, color='black', linestyle='--', linewidth=1.4)
ax.text(UMBRAL + 0.012, ax.get_ylim()[1]*0.96, 'umbral 0.5',
        fontsize=9, va='top', rotation=0)

# anotaciones de los casos de error
ymax = ax.get_ylim()[1]
ax.annotate(f'Minas no detectadas\n(FN = {fn} de {len(p_mina)})',
            xy=(0.22, 0.06), xytext=(0.30, ymax*0.55),
            fontsize=9, color=ROJO, ha='center', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=ROJO, lw=1.2))
ax.annotate(f'Falsas alarmas\n(FP = {fp})',
            xy=(0.70, 0.012), xytext=(0.78, ymax*0.45),
            fontsize=9, color=AZUL, ha='center', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=AZUL, lw=1.2))

# sombrear la zona de solapamiento (donde ambas clases tienen masa)
ax.axvspan(0.2, 0.8, color='#F4D03F', alpha=0.12, zorder=0)
ax.text(0.5, ymax*0.66, 'zona de solapamiento\n(casos no separables)',
        fontsize=8.5, ha='center', color='#7D6608', style='italic')

ax.set_xlabel('Probabilidad asignada por el modelo  P(MAP)')
ax.set_ylabel('Fracción de ventanas de la clase')
ax.set_xlim(0, 1)
ax.set_title('Solapamiento de las distribuciones de probabilidad por clase '
             '(modelo final, LOSO)', fontsize=11, fontweight='bold')
ax.legend(loc='upper center', fontsize=9, framealpha=0.95)
ax.grid(axis='y', alpha=0.25)

plt.tight_layout()
os.makedirs(FIG, exist_ok=True)
ruta = f'{FIG}/hist_solapamiento.png'
plt.savefig(ruta, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f"figura -> {ruta}")
