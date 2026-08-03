"""
Utilidades compartidas para thesisV2.

- Carga de datos (Enfoque B).
- Submuestreo 1:1 con cuota de objetos de interferencia, aplicado por fold.
- RF-LOSO con metricas completas (AUC, Sens, Spec, FP/mina) a umbral 0.5.

Las funciones de balanceo se reutilizan en todas las pruebas (P1..P8).
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix

META = ['tiene_mina', 'parche_x', 'parche_y', 'sesion', 'zona', 'clase']
SESIONES = [2, 3, 4, 5, 6, 7]
OBJ_CLASES = ['Botella', 'Lata', 'Piedra']


def feature_cols(df):
    return [c for c in df.columns if c not in META]


def undersample_quota(y, clase, rng, obj_frac=0.0):
    """
    Submuestreo 1:1: todos los positivos + igual numero de negativos.
    De esos negativos, una fraccion 'obj_frac' son ventanas de objetos
    (negativos dificiles), garantizando su presencia. El resto, aleatorios.

    y, clase: arrays del CONJUNTO sobre el que se sortea (p.ej. el train del fold).
    Devuelve los indices (posiciones dentro de y) seleccionados.
    """
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    n = len(pos_idx)  # 1:1

    es_obj = np.isin(clase, OBJ_CLASES)
    neg_obj = neg_idx[es_obj[neg_idx]]
    neg_otro = neg_idx[~es_obj[neg_idx]]

    n_obj = min(len(neg_obj), int(round(n * obj_frac)))
    sel_obj = rng.choice(neg_obj, size=n_obj, replace=False) if n_obj > 0 else np.array([], dtype=int)

    n_otro = min(n - n_obj, len(neg_otro))
    sel_otro = rng.choice(neg_otro, size=n_otro, replace=False)

    sel_neg = np.concatenate([sel_obj, sel_otro])
    return np.concatenate([pos_idx, sel_neg]), len(sel_obj), len(sel_otro)


def cargar(csv):
    df = pd.read_csv(csv)
    if 'clase' not in df.columns:
        df['clase'] = np.where(df['tiene_mina'] == 1, 'MAP', 'Fondo')
    return df


def metricas_fold(yt, proba, umbral=0.5):
    auc = roc_auc_score(yt, proba)
    pred = (proba >= umbral).astype(int)
    tn, fp, fn, tp = confusion_matrix(yt, pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else np.nan
    spec = tn / (tn + fp) if (tn + fp) else np.nan
    return dict(AUC=auc, Sens=sens, Spec=spec, TP=tp, FN=fn, FP=fp, TN=tn)


def loso(df, build_clf, seed=42, obj_frac=0.0):
    """
    Validacion LOSO generica. 'build_clf' es una funcion () -> clasificador sklearn-like.
    Submuestreo 1:1 con cuota de objetos en el train de cada fold.
    Devuelve (tabla_por_fold, agregado).
    """
    cols = feature_cols(df)
    X = df[cols].values
    y = df['tiene_mina'].values
    ses = df['sesion'].values
    clase = df['clase'].values
    rng = np.random.default_rng(seed)

    filas = []
    TP = FN = FP = TN = 0
    for s in SESIONES:
        tr = ses != s
        te = ses == s
        idx_tr = np.where(tr)[0]

        sub, _, _ = undersample_quota(y[tr], clase[tr], rng, obj_frac)
        abs_idx = idx_tr[sub]

        clf = build_clf()
        clf.fit(X[abs_idx], y[abs_idx])
        proba = clf.predict_proba(X[te])[:, 1]

        m = metricas_fold(y[te], proba)
        m['fold'] = f'S{s}'
        filas.append(m)
        TP += m['TP']; FN += m['FN']; FP += m['FP']; TN += m['TN']

    res = pd.DataFrame(filas)[['fold', 'AUC', 'Sens', 'Spec', 'TP', 'FN', 'FP', 'TN']]
    agg = dict(
        AUC=res.AUC.mean(), AUC_std=res.AUC.std(),
        Sens=TP / (TP + FN), Spec=TN / (TN + FP),
        FP_mina=FP / TP if TP else np.nan,
        TP=TP, FN=FN, FP=FP, TN=TN,
    )
    return res, agg


def loso_easy(df, build_clf, n_ens=10, seed=42, obj_frac=0.0):
    """
    LOSO con EasyEnsemble: en cada fold entrena 'n_ens' clasificadores, cada uno con
    un submuestreo 1:1 distinto (con cuota de objetos), y promedia las probabilidades.
    """
    cols = feature_cols(df)
    X = df[cols].values
    y = df['tiene_mina'].values
    ses = df['sesion'].values
    clase = df['clase'].values
    rng = np.random.default_rng(seed)

    filas = []
    TP = FN = FP = TN = 0
    for s in SESIONES:
        tr = ses != s
        te = ses == s
        idx_tr = np.where(tr)[0]
        Xte = X[te]
        proba = np.zeros(int(te.sum()))
        for _ in range(n_ens):
            sub, _, _ = undersample_quota(y[tr], clase[tr], rng, obj_frac)
            abs_idx = idx_tr[sub]
            clf = build_clf()
            clf.fit(X[abs_idx], y[abs_idx])
            proba += clf.predict_proba(Xte)[:, 1]
        proba /= n_ens
        m = metricas_fold(y[te], proba)
        m['fold'] = f'S{s}'
        filas.append(m)
        TP += m['TP']; FN += m['FN']; FP += m['FP']; TN += m['TN']

    res = pd.DataFrame(filas)[['fold', 'AUC', 'Sens', 'Spec', 'TP', 'FN', 'FP', 'TN']]
    agg = dict(
        AUC=res.AUC.mean(), AUC_std=res.AUC.std(),
        Sens=TP / (TP + FN), Spec=TN / (TN + FP),
        FP_mina=FP / TP if TP else np.nan,
        TP=TP, FN=FN, FP=FP, TN=TN,
    )
    return res, agg


def rf_builder(seed=42, n_estimators=100, max_depth=10):
    return lambda: RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth,
        random_state=seed, n_jobs=-1)


def xgb_builder(seed=42):
    from xgboost import XGBClassifier
    return lambda: XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, eval_metric='logloss',
        verbosity=0, random_state=seed, n_jobs=-1)


def svm_builder(seed=42):
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    return lambda: make_pipeline(
        StandardScaler(),
        SVC(C=1.0, gamma='scale', kernel='rbf', probability=True, random_state=seed))
