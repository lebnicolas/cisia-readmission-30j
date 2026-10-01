"""Modelisation, calibration, choix du seuil et equite.

Le protocole est fixe une fois pour toutes ici : decoupage groupe par patient,
predictions hors echantillon, metriques adaptees au desequilibre. Aucun modele
ne voit un patient present dans son echantillon d'apprentissage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, brier_score_loss, confusion_matrix,
    precision_recall_curve, roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config as cfg


# --------------------------------------------------------------------------
# Protocole
# --------------------------------------------------------------------------

def make_cv() -> StratifiedGroupKFold:
    """Decoupage groupe par patient et stratifie sur la cible.

    Groupe : aucun patient n'est reparti entre apprentissage et test (J1-07).
    Stratifie : chaque pli conserve le taux de readmission, indispensable avec
    seulement 141 evenements positifs.
    """
    return StratifiedGroupKFold(n_splits=cfg.N_SPLITS, shuffle=True,
                                random_state=cfg.SEED)


def _column_types(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    cat = [c for c in X.columns if str(X[c].dtype) == "category"]
    num = [c for c in X.columns if c not in cat]
    return num, cat


def make_linear_pipeline(X: pd.DataFrame) -> Pipeline:
    """Regression logistique : imputation, encodage, standardisation.

    Modele de reference interpretable. Ses coefficients sont directement
    lisibles, ce qui repond a l'exigence d'explicabilite (section RGPD).
    """
    num, cat = _column_types(X)
    pre = ColumnTransformer([
        ("num", Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
        ]), num),
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                     min_frequency=10, sparse_output=False)),
        ]), cat),
    ])
    return Pipeline([
        ("pre", pre),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced",
                                   random_state=cfg.SEED)),
    ])


def make_models(X: pd.DataFrame) -> dict[str, object]:
    """Six familles comparees, du plus simple au plus souple (cf. J2-25)."""
    from lightgbm import LGBMClassifier
    from sklearn.neural_network import MLPClassifier

    return {
        "Référence (taux de base)": DummyClassifier(strategy="prior"),
        "Régression logistique": make_linear_pipeline(X),
        "Forêt aléatoire": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(
                n_estimators=400, min_samples_leaf=5, class_weight="balanced",
                random_state=cfg.SEED, n_jobs=-1)),
        ]),
        "Gradient boosting": HistGradientBoostingClassifier(
            categorical_features="from_dtype", random_state=cfg.SEED,
            max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
            l2_regularization=1.0, early_stopping=True, validation_fraction=0.15),
        # Le standard industriel de la meme famille — temoin d'equivalence.
        "LightGBM": LGBMClassifier(
            n_estimators=300, learning_rate=0.05, num_leaves=15,
            reg_lambda=1.0, random_state=cfg.SEED, verbose=-1),
        # Le reseau de neurones : mesure pour que « pourquoi pas du deep »
        # soit une experience, pas un argument d'autorite.
        "Réseau de neurones (MLP)": Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
            ("clf", MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-2,
                                  max_iter=800, random_state=cfg.SEED)),
        ]),
    }


# Modeles sans support natif des colonnes categorielles : codes entiers.
_SANS_CATEGORIES = {"Forêt aléatoire", "LightGBM", "Réseau de neurones (MLP)"}


def _prepare_for(name: str, X: pd.DataFrame) -> pd.DataFrame:
    if name not in _SANS_CATEGORIES:
        return X
    out = X.copy()
    for c in out.columns:
        if str(out[c].dtype) == "category":
            out[c] = out[c].cat.codes.replace(-1, np.nan)
    return out


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------

def out_of_fold_predictions(models: dict, X, y, groups) -> pd.DataFrame:
    """Probabilites predites hors echantillon, un modele par colonne."""
    cv = make_cv()
    preds = {}
    for name, model in models.items():
        Xm = _prepare_for(name, X)
        preds[name] = cross_val_predict(
            model, Xm, y, cv=cv, groups=groups, method="predict_proba"
        )[:, 1]
    return pd.DataFrame(preds, index=X.index)


def score_table(preds: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Metriques comparees.

    PR-AUC est la metrique principale : avec 18 % de positifs, elle reflete la
    capacite a hierarchiser le risque bien mieux que l'exactitude ou le ROC.
    Le score de Brier mesure la qualite des probabilites elles-memes, ce qui
    compte des lors qu'on veut choisir un seuil.
    """
    rows = []
    for name in preds.columns:
        p = preds[name]
        rows.append({
            "modèle": name,
            "PR-AUC": average_precision_score(y, p),
            "ROC-AUC": roc_auc_score(y, p),
            "Brier": brier_score_loss(y, p),
        })
    out = pd.DataFrame(rows).set_index("modèle")
    out["gain vs référence"] = out["PR-AUC"] / out["PR-AUC"].min()
    return out.sort_values("PR-AUC", ascending=False)


def calibration_table(p: pd.Series, y: pd.Series, n_bins: int = 5) -> pd.DataFrame:
    """Confronte probabilite predite et frequence observee, par tranche."""
    frac, mean_pred = calibration_curve(y, p, n_bins=n_bins, strategy="quantile")
    counts = pd.qcut(p, n_bins, duplicates="drop").value_counts().sort_index()
    return pd.DataFrame({
        "risque prédit moyen": mean_pred,
        "réadmissions observées": frac,
        "séjours": counts.to_numpy()[:len(frac)],
    })


# --------------------------------------------------------------------------
# Choix du seuil
# --------------------------------------------------------------------------

def threshold_table(p: pd.Series, y: pd.Series,
                    thresholds: np.ndarray | None = None) -> pd.DataFrame:
    """Consequences operationnelles de chaque seuil.

    Les colonnes sont exprimees en actes, pas en taux : c'est ainsi que la
    decision se pose pour un service — combien de patients signales, combien de
    readmissions rattrapees, combien de visites inutiles.
    """
    if thresholds is None:
        thresholds = np.arange(0.05, 0.65, 0.05)
    rows = []
    for t in thresholds:
        pred = (p >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        rows.append({
            "seuil": round(float(t), 2),
            "patients signalés": int(tp + fp),
            "réadmissions rattrapées": int(tp),
            "réadmissions manquées": int(fn),
            "alertes inutiles": int(fp),
            "rappel": tp / (tp + fn) if (tp + fn) else np.nan,
            "précision": tp / (tp + fp) if (tp + fp) else np.nan,
        })
    return pd.DataFrame(rows).set_index("seuil")


def optimal_threshold(p: pd.Series, y: pd.Series, cost_ratio: float) -> dict:
    """Seuil minimisant le cout total, pour un rapport de cout donne.

    `cost_ratio` est le cout d'un faux negatif rapporte a celui d'un faux
    positif : une readmission non anticipee coute `cost_ratio` fois plus cher
    qu'une visite a domicile inutile. C'est une donnee metier, pas un
    hyperparametre a optimiser.
    """
    grid = np.arange(0.02, 0.90, 0.01)
    best, best_cost = None, np.inf
    for t in grid:
        pred = (p >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        cost = cost_ratio * fn + fp
        if cost < best_cost:
            best, best_cost = t, cost
    pred = (p >= best).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "seuil": round(float(best), 2),
        "rapport de coût": cost_ratio,
        "patients signalés": int(tp + fp),
        "réadmissions rattrapées": int(tp),
        "réadmissions manquées": int(fn),
        "alertes inutiles": int(fp),
        "rappel": tp / (tp + fn),
        "précision": tp / (tp + fp) if (tp + fp) else np.nan,
    }


def cost_sensitivity(p: pd.Series, y: pd.Series,
                     ratios: list[float]) -> pd.DataFrame:
    """Sensibilite du seuil retenu au rapport de cout.

    Rend explicite le fait que le seuil est un arbitrage metier : il se deplace
    avec la valeur qu'on accorde a une readmission evitee.
    """
    return pd.DataFrame([optimal_threshold(p, y, r) for r in ratios]) \
             .set_index("rapport de coût")


# --------------------------------------------------------------------------
# Interpretabilite
# --------------------------------------------------------------------------

def permutation_scores(model, X, y, groups, n_repeats: int = 10) -> pd.DataFrame:
    """Importance par permutation, mesuree hors echantillon sur les cinq plis.

    Preferee a l'importance interne des arbres, qui surestime les variables a
    forte cardinalite. Chaque pli entraine le modele sur sa partie
    d'apprentissage et permute sur sa partie de test ; l'importance rapportee
    est la moyenne des plis, et l'ecart-type mesure leur dispersion. Mesuree
    sur un seul pli, elle reposait sur une trentaine d'evenements positifs
    (cf. J3-02).
    """
    from sklearn.base import clone

    cv = make_cv()
    par_pli = []
    for train_idx, test_idx in cv.split(X, y, groups=groups):
        m = clone(model).fit(X.iloc[train_idx], y.iloc[train_idx])
        r = permutation_importance(
            m, X.iloc[test_idx], y.iloc[test_idx],
            scoring="average_precision", n_repeats=n_repeats,
            random_state=cfg.SEED, n_jobs=1,
        )
        par_pli.append(r.importances_mean)
    par_pli = np.asarray(par_pli)
    return (pd.DataFrame({"variable": X.columns,
                          "importance": par_pli.mean(axis=0),
                          "écart-type": par_pli.std(axis=0)})
            .sort_values("importance", ascending=False)
            .set_index("variable"))


def logistic_coefficients(pipeline: Pipeline, top: int = 20) -> pd.DataFrame:
    """Coefficients de la regression logistique, en rapport de cotes."""
    names = pipeline.named_steps["pre"].get_feature_names_out()
    coefs = pipeline.named_steps["clf"].coef_[0]
    out = pd.DataFrame({"variable": names, "coefficient": coefs})
    out["rapport de cotes"] = np.exp(out.coefficient)
    out["effet"] = np.where(out.coefficient > 0, "augmente le risque",
                            "diminue le risque")
    out["|coefficient|"] = out.coefficient.abs()
    return (out.sort_values("|coefficient|", ascending=False)
              .head(top).drop(columns="|coefficient|").set_index("variable"))


# --------------------------------------------------------------------------
# Equite
# --------------------------------------------------------------------------

def fairness_table(p: pd.Series, y: pd.Series, groups: pd.Series,
                   threshold: float) -> pd.DataFrame:
    """Performance par sous-population, au seuil retenu.

    Un modele peut afficher un bon score global tout en signalant mal une
    sous-population. On compare le taux de signalement, le rappel et la
    precision groupe par groupe.
    """
    pred = (p >= threshold).astype(int)
    rows = []
    for name, idx in groups.groupby(groups, observed=True).groups.items():
        yy, pp, dd = y.loc[idx], p.loc[idx], pred.loc[idx]
        if yy.nunique() < 2 or len(yy) < 20:
            continue
        tn, fp, fn, tp = confusion_matrix(yy, dd, labels=[0, 1]).ravel()
        rows.append({
            "sous-population": name,
            "séjours": len(yy),
            "taux réel": yy.mean(),
            "taux signalé": dd.mean(),
            "rappel": tp / (tp + fn) if (tp + fn) else np.nan,
            "précision": tp / (tp + fp) if (tp + fp) else np.nan,
            "PR-AUC": average_precision_score(yy, pp),
        })
    return pd.DataFrame(rows).set_index("sous-population")
