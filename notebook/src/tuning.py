"""Recherche d'hyperparametres du modele retenu, en validation croisee imbriquee.

Le reglage du gradient boosting retenu (cf. `modeling.make_models`) a ete pose
a la main pour un petit jeu : arbres brides a 15 feuilles, regularisation L2 a
1.0, pas d'apprentissage a 0,05, arret anticipe. Un reglage de bon sens n'est
pas une optimisation — ce module verifie par la mesure si une recherche
automatique fait mieux, et rapporte l'ecart meme s'il est nul.

Le protocole est **imbrique**, et ce n'est pas un ornement : la recherche vit
entierement dans les plis internes, l'evaluation se fait sur des plis externes
qu'aucun essai n'a vus. Une recherche evaluee sur les plis qui l'ont guidee
rapporterait le score de sa propre selection — optimiste, et d'autant plus
trompeur que le jeu est petit.

    ..\\.venv\\Scripts\\python.exe recherche_hyperparametres.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict

from . import config as cfg

# Reglage manuel du dossier — le temoin a battre.
MANUEL = dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
              l2_regularization=1.0, early_stopping=True,
              validation_fraction=0.15)


def modele(params: dict) -> HistGradientBoostingClassifier:
    """Estimateur du dossier, hyperparametres au choix."""
    return HistGradientBoostingClassifier(
        categorical_features="from_dtype", random_state=cfg.SEED, **params)


def espace(trial) -> dict:
    """Espace explore par Optuna.

    Les bornes encadrent largement le reglage manuel sans autoriser
    l'absurde : jusqu'a 63 feuilles (soit quatre fois le reglage retenu) et
    une regularisation pouvant descendre a 1e-3, de quoi laisser la recherche
    conclure au surapprentissage si elle en trouve le chemin.
    """
    return dict(
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.30, log=True),
        max_leaf_nodes=trial.suggest_int("max_leaf_nodes", 4, 63),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 5, 50),
        l2_regularization=trial.suggest_float("l2_regularization", 1e-3, 10.0,
                                              log=True),
        max_features=trial.suggest_float("max_features", 0.4, 1.0),
        max_iter=trial.suggest_int("max_iter", 100, 600, step=50),
        early_stopping=True,
        validation_fraction=0.15,
    )


def _pr_auc_interne(params: dict, X, y, groups, n_plis: int) -> float:
    """PR-AUC hors echantillon sur les plis internes, predictions mises en
    commun.

    Les predictions des plis sont concatenees avant de calculer la metrique :
    avec une trentaine d'evenements positifs par pli, une moyenne de PR-AUC
    par pli serait dominee par le bruit d'echantillonnage.
    """
    cv = StratifiedGroupKFold(n_splits=n_plis, shuffle=True,
                              random_state=cfg.SEED)
    p = cross_val_predict(modele(params), X, y, cv=cv, groups=groups,
                          method="predict_proba")[:, 1]
    return float(average_precision_score(y, p))


def recherche_imbriquee(X: pd.DataFrame, y: pd.Series, groups: pd.Series,
                        n_essais: int = 50, n_plis_internes: int = 4,
                        verbose: bool = True) -> dict:
    """Validation croisee imbriquee : Optuna dedans, evaluation dehors.

    Retourne, pli externe par pli externe, le score du reglage manuel et celui
    du reglage trouve par la recherche, ainsi que les hyperparametres retenus.
    Les deux modeles sont evalues sur exactement les memes plis externes : la
    comparaison ne depend pas du decoupage.
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    cv_externe = StratifiedGroupKFold(n_splits=cfg.N_SPLITS, shuffle=True,
                                      random_state=cfg.SEED)

    plis, p_manuel, p_optuna = [], np.zeros(len(y)), np.zeros(len(y))

    for k, (i_tr, i_te) in enumerate(cv_externe.split(X, y, groups), start=1):
        Xtr, ytr, gtr = X.iloc[i_tr], y.iloc[i_tr], groups.iloc[i_tr]
        Xte, yte = X.iloc[i_te], y.iloc[i_te]

        # --- la recherche ne voit que le pli d'apprentissage
        etude = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=cfg.SEED + k))
        etude.optimize(
            lambda t: _pr_auc_interne(espace(t), Xtr, ytr, gtr, n_plis_internes),
            n_trials=n_essais, show_progress_bar=False)

        params = dict(etude.best_params, early_stopping=True,
                      validation_fraction=0.15)

        # --- evaluation sur le pli externe, jamais vu par la recherche
        m_opt = modele(params).fit(Xtr, ytr)
        m_man = modele(MANUEL).fit(Xtr, ytr)
        po = m_opt.predict_proba(Xte)[:, 1]
        pm = m_man.predict_proba(Xte)[:, 1]
        p_optuna[i_te], p_manuel[i_te] = po, pm

        ligne = {
            "pli": k,
            "séjours test": int(len(i_te)),
            "positifs test": int(yte.sum()),
            "manuel": float(average_precision_score(yte, pm)),
            "Optuna": float(average_precision_score(yte, po)),
            "interne (recherche)": float(etude.best_value),
            "params": params,
        }
        plis.append(ligne)
        if verbose:
            print(f"  pli {k} — manuel {ligne['manuel']:.3f} | "
                  f"Optuna {ligne['Optuna']:.3f} | "
                  f"interne {ligne['interne (recherche)']:.3f}")

    # Metriques mises en commun : directement comparables au tableau du dossier.
    groupe = {
        "manuel": float(average_precision_score(y, p_manuel)),
        "Optuna": float(average_precision_score(y, p_optuna)),
    }
    par_pli = {
        "manuel": float(np.mean([p["manuel"] for p in plis])),
        "Optuna": float(np.mean([p["Optuna"] for p in plis])),
        "écart-type manuel": float(np.std([p["manuel"] for p in plis])),
        "écart-type Optuna": float(np.std([p["Optuna"] for p in plis])),
    }
    return {
        "n_essais": n_essais,
        "n_plis_internes": n_plis_internes,
        "plis": plis,
        "mise_en_commun": groupe,
        "moyenne_par_pli": par_pli,
        "écart": groupe["Optuna"] - groupe["manuel"],
    }


def table_plis(resultat: dict) -> pd.DataFrame:
    """Le resultat, pli par pli, sous forme de tableau lisible."""
    lignes = [{k: v for k, v in p.items() if k != "params"}
              for p in resultat["plis"]]
    df = pd.DataFrame(lignes).set_index("pli")
    df["écart"] = df["Optuna"] - df["manuel"]
    return df


def table_params(resultat: dict) -> pd.DataFrame:
    """Les hyperparametres retenus dans chaque pli externe.

    Leur dispersion est le resultat le plus parlant : si chaque pli designe un
    reglage different, c'est que la recherche suit le bruit du pli et non une
    structure du probleme.
    """
    lignes = []
    for p in resultat["plis"]:
        ligne = {"pli": p["pli"]}
        ligne.update({k: v for k, v in p["params"].items()
                      if k not in ("early_stopping", "validation_fraction")})
        lignes.append(ligne)
    return pd.DataFrame(lignes).set_index("pli")
