r"""Entraine le modele final de l'etage A et fige les artefacts de service.

Le modele servi par l'API est entraine sur la totalite de la population
d'etude — les performances annoncees restent celles de la validation croisee
(cf. notebook, sections 7 a 9), jamais celles du modele reajuste.

Produit dans api/artefacts/ :
  modele.joblib      HistGradientBoosting entraine, memes hyperparametres
  donnees.parquet    la table de variables des 786 sejours (pour la demo)
  meta.json          version, seuil, importance globale, contexte patient
  contributions.*    contributions SHAP par sejour (via expliquer.py, ~2 min)

    ..\.venv\Scripts\python.exe entrainer.py
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import joblib
import pandas as pd

API_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(API_DIR.parent / "notebook"))

from src import assemble, clean, config as cfg, loading, modeling as mdl  # noqa: E402

ARTEFACTS = API_DIR / "artefacts"
ARTEFACTS.mkdir(exist_ok=True)

VERSION = f"etageA-{date.today().isoformat()}-seed{cfg.SEED}"


def main() -> None:
    tables = loading.load_raw()
    population = clean.build_population(tables)
    X, y, groups = assemble.build_dataset(tables, stages=("A",),
                                          population=population, verbose=False)

    # Memes hyperparametres que le protocole d'evaluation.
    modele = mdl.make_models(X)["Gradient boosting"]
    modele.fit(X, y)

    # Importance globale par permutation, pour expliquer les scores servis.
    imp = mdl.permutation_scores(mdl.make_models(X)["Gradient boosting"],
                                 X, y, groups, n_repeats=5)
    top = imp[imp.importance > 0].head(12)

    # ----- Arbre de substitution : un arbre unique et lisible qui imite le
    # modele complet, avec sa fidelite mesuree. Il ne remplace pas le modele
    # (300 arbres) — il l'explique. Numeriques seulement : les variables
    # dominantes le sont toutes.
    from sklearn.tree import DecisionTreeRegressor

    proba = modele.predict_proba(X)[:, 1]
    colonnes_num = [c for c in X.columns if str(X[c].dtype) != "category"]
    Xn = X[colonnes_num].astype(float)

    candidats = {}
    for prof in (3, 4):
        t = DecisionTreeRegressor(max_depth=prof, min_samples_leaf=25,
                                  random_state=cfg.SEED).fit(Xn, proba)
        candidats[prof] = (t, t.score(Xn, proba),
                           float(((t.predict(Xn) >= cfg.DECISION_THRESHOLD)
                                  == (proba >= cfg.DECISION_THRESHOLD)).mean()))
    # La profondeur 4 n'est retenue que si elle paie vraiment sa complexite.
    prof = 4 if candidats[4][1] - candidats[3][1] > 0.04 else 3
    arbre, r2, accord = candidats[prof]

    tr = arbre.tree_
    noeuds = []
    for i in range(tr.node_count):
        f = int(tr.feature[i])
        noeuds.append({
            "feature": None if f < 0 else colonnes_num[f],
            "seuil": None if f < 0 else round(float(tr.threshold[i]), 2),
            "gauche": int(tr.children_left[i]),
            "droite": int(tr.children_right[i]),
            "n": int(tr.n_node_samples[i]),
            "risque": round(float(tr.value[i][0][0]), 3),
        })
    joblib.dump(arbre, ARTEFACTS / "arbre.joblib")
    (ARTEFACTS / "arbre.json").write_text(json.dumps({
        "colonnes": colonnes_num, "profondeur": prof, "noeuds": noeuds,
        "fidelite": {"r2": round(float(r2), 3),
                     "accord_seuil": round(accord, 3)},
        "note": "Arbre de substitution entraine a imiter le modele complet "
                "(300 arbres) — il explique, il ne predit pas en production.",
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"  arbre de substitution : profondeur {prof}, "
          f"R2 {r2:.3f}, accord au seuil {accord:.1%}")

    joblib.dump(modele, ARTEFACTS / "modele.joblib")
    Xs = X.copy()
    for c in Xs.select_dtypes("category").columns:
        Xs[c] = Xs[c].astype(str)          # parquet-compatible ; recategorise au service
    Xs.to_parquet(ARTEFACTS / "donnees.parquet")

    # Contexte lisible par sejour, pour l'ecran de demo.
    contexte = pd.DataFrame({
        "age": X.demo_age,
        "sexe": X.demo_sexe.astype(str),
        "service": X.sej_service.astype(str),
        "diagnostic": X.dx_diagnostic_principal.astype(str),
        "duree": X.sej_duree,
        "nb_pathologies": X.demo_nb_pathologies.astype(float),
        "reel": y,
    })
    contexte.to_parquet(ARTEFACTS / "contexte.parquet")

    meta = {
        "version": VERSION,
        "seuil": cfg.DECISION_THRESHOLD,
        "rapport_cout": cfg.COST_RATIO_FN_FP,
        "sejours": len(X),
        "readmissions": int(y.sum()),
        "importance_globale": {k: round(float(v), 4)
                               for k, v in top.importance.items()},
        "colonnes_categorielle": [c for c in X.columns
                                  if str(X[c].dtype) == "category"],
        "performance_cv": {
            "note": "PR-AUC de 0,39 a 0,44 selon la partition (cf. J3-11) ; "
                    "les chiffres servis proviennent de la validation croisee, "
                    "pas du modele reajuste.",
            "pr_auc_seed42": 0.435, "roc_auc_seed42": 0.778,
        },
        "etage_b": {
            "statut": "specifie, non entrainable",
            "patients_couverts": 6, "patients_requis": cfg.IOT_MIN_PATIENTS,
            "fenetre_jours": cfg.IOT_WINDOW_DAYS,
            "decisions_visees": ["adaptation de la medication",
                                 "visite a domicile",
                                 "anticipation d'une rehospitalisation"],
        },
    }
    (ARTEFACTS / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    # Contributions SHAP par sejour — l'explication individuelle du score
    # (cf. expliquer.py). Calculees ici pour que l'etape 2 du README suffise.
    import expliquer
    expliquer.main()

    print(f"artefacts ecrits dans {ARTEFACTS}")
    print(f"  version {VERSION} — {len(X)} sejours, {X.shape[1]} variables")


if __name__ == "__main__":
    main()
