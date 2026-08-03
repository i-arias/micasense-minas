"""
P8 — Falsos positivos vs objetos de interferencia, y EFECTO DE LA CUOTA.

(a) De los FP del modelo final dentro de zonas de mina, ¿cuántos caen sobre objetos
    (botella/lata/piedra) y cuántos sobre fondo del suelo perturbado?
(b) Efecto de la cuota de objetos: compara FP-sobre-objetos CON cuota vs SIN cuota,
    para demostrar que meter negativos difíciles reduce esas falsas alarmas.

Ejecutar desde la raiz del proyecto:
    python thesisV2/scripts/p8_fp_objetos.py
"""
import sys, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

RES = 'thesisV2/resultados'
FIG = 'thesisV2/figuras'
OBJ = ['Botella', 'Lata', 'Piedra']

df = pd.read_csv(f'{RES}/p5_predicciones.csv')
# FP en zonas de mina = ventanas con tiene_mina=0 en zona de mina y prediccion positiva
mina_zone = df['zona'].str.contains('MINA')
neg = df[mina_zone & (df['tiene_mina'] == 0)].copy()


def resumen_fp(proba_col):
    pred = neg[proba_col] >= 0.5
    fp = neg[pred]
    total = len(fp)
    por_clase = fp['clase'].value_counts().to_dict()
    n_obj = sum(por_clase.get(o, 0) for o in OBJ)
    n_fondo = por_clase.get('Fondo', 0)
    return dict(total=total, objetos=n_obj, fondo=n_fondo,
                Botella=por_clase.get('Botella', 0), Lata=por_clase.get('Lata', 0),
                Piedra=por_clase.get('Piedra', 0),
                pct_objetos=100*n_obj/total if total else 0,
                pct_fondo=100*n_fondo/total if total else 0)


final = resumen_fp('proba')            # modelo final (sin cuota)
cuota = resumen_fp('proba_concuota')   # alternativa con cuota (experimento)

print("P8 — Falsos positivos en zonas de mina (MODELO FINAL, sin cuota)")
print(f"  Total FP: {final['total']}")
print(f"  Sobre objetos: {final['objetos']} ({final['pct_objetos']:.1f}%)  "
      f"[Botella={final['Botella']}, Lata={final['Lata']}, Piedra={final['Piedra']}]")
print(f"  Sobre fondo (suelo perturbado): {final['fondo']} ({final['pct_fondo']:.1f}%)")

print("\nExperimento: efecto de la cuota de objetos (FINAL sin cuota vs CON cuota):")
print(f"  FP totales:          {final['total']}  vs  {cuota['total']}")
print(f"  FP sobre objetos:    {final['objetos']}  vs  {cuota['objetos']}")
print(f"    - Botella:         {final['Botella']}  vs  {cuota['Botella']}")
print(f"    - Lata:            {final['Lata']}  vs  {cuota['Lata']}")
print(f"    - Piedra:          {final['Piedra']}  vs  {cuota['Piedra']}")
print("  -> la cuota NO reduce los FP sobre objetos y aumenta los FP totales.")

pd.DataFrame([dict(escenario='final_sin_cuota', **final), dict(escenario='con_cuota', **cuota)]
             ).to_csv(f'{RES}/p8_fp_objetos.csv', index=False)
con = final   # alias para la figura (composicion del modelo final)
sin = cuota   # alias para la comparacion

# ---- dos figuras separadas: composicion (torta) y efecto de la cuota (barras)
os.makedirs(FIG, exist_ok=True)

# Figura A: composicion de los FP (torta con leyenda)
fig, ax = plt.subplots(figsize=(6.2, 5.2))
vals = [con['fondo'], con['Botella'], con['Lata'], con['Piedra']]
names = ['Fondo (suelo)', 'Botella', 'Lata', 'Piedra']
tot = sum(vals)
wedges, _ = ax.pie(vals, colors=['#5D6D7E', '#2980B9', '#E67E22', '#7D3C98'],
                   startangle=90, wedgeprops={'edgecolor': 'white', 'linewidth': 1})
ax.legend(wedges, [f"{n}: {v} ({100*v/tot:.1f}%)" for n, v in zip(names, vals)],
          loc='upper center', bbox_to_anchor=(0.5, -0.02), ncol=2, fontsize=9, frameon=False)
ax.set_title(f"Composición de los falsos positivos (total = {con['total']})",
             fontweight='bold', fontsize=11)
plt.tight_layout()
plt.savefig(f'{FIG}/p8_composicion.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

# Figura B: efecto de la cuota de objetos (barras con vs sin cuota)
fig, ax = plt.subplots(figsize=(7, 4.6))
cats = ['Botella', 'Lata', 'Piedra', 'Total objetos']
v_con = [con['Botella'], con['Lata'], con['Piedra'], con['objetos']]
v_sin = [sin['Botella'], sin['Lata'], sin['Piedra'], sin['objetos']]
x = np.arange(len(cats)); w = 0.38
ax.bar(x - w/2, v_con, w, label='Final (sin cuota)', color='#27AE60')
ax.bar(x + w/2, v_sin, w, label='Con cuota', color='#95A5A6')
ax.set_xticks(x); ax.set_xticklabels(cats, fontsize=9)
ax.set_ylabel('FP sobre objetos'); ax.legend(fontsize=9)
ax.set_title('Efecto de la cuota de objetos en los FP', fontweight='bold', fontsize=11)
for i, (a, b) in enumerate(zip(v_con, v_sin)):
    ax.text(i - w/2, a + 1, str(a), ha='center', fontsize=8)
    ax.text(i + w/2, b + 1, str(b), ha='center', fontsize=8)
plt.tight_layout()
plt.savefig(f'{FIG}/p8_cuota.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f"figuras -> {FIG}/p8_composicion.png y {FIG}/p8_cuota.png")
