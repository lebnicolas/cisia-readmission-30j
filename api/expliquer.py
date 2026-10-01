r"""Contributions SHAP par sejour — l'explication individuelle du score servi.

L'importance globale (meta.json) dit ce qui pese dans le modele, pour tout le
monde. Elle ne dit pas pourquoi CE patient est a 69 % : c'est le role des
contributions calculees ici, une par variable et par sejour, qui se somment
exactement au score (base de population + contributions = probabilite).

Methode : valeurs de Shapley estimees par permutations (shap.explainers.Permutation)
sur la sortie predict_proba du modele complet, avec un fond de 100 sejours tires
au sort. Cette voie est retenue plutot que TreeExplainer parce que ce dernier ne
gere pas les coupures categorielles natives de HistGradientBoosting : il repond
sans erreur mais faux (somme des contributions loin du score, age absent chez un
patient de 78 ans — verifie le 08/09/2026).

Produit dans api/artefacts/ :
  contributions.parquet   786 sejours x 132 variables, en points de probabilite
  contributions.json      base de population, methode, parametres, controle

Appele par entrainer.py ; executable seul (~2 min sur 16 coeurs) :

    ..\.venv\Scripts\python.exe expliquer.py [--permutations 8] [--fond 100]
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

API_DIR = Path(__file__).resolve().parent
ARTEFACTS = API_DIR / "artefacts"
GRAINE = 42


def _charger():
    modele = joblib.load(ARTEFACTS / "modele.joblib")
    meta = json.loads((ARTEFACTS / "meta.json").read_text(encoding="utf-8"))
    X = pd.read_parquet(ARTEFACTS / "donnees.parquet")
    for c in meta["colonnes_categorielle"]:
        X[c] = X[c].astype("category")
    return modele, meta, X


def _encoder(X: pd.DataFrame, cats: list[str]) -> np.ndarray:
    """Le masqueur de shap compare des flottants : les categorielles passent en
    codes entiers (-1 = manquant), redecodes avant chaque appel au modele."""
    out = X.copy()
    for c in cats:
        out[c] = out[c].cat.codes.astype(float)
    return out.to_numpy(dtype=float)


def _fonction_modele(modele, colonnes, categories: dict):
    def f(arr: np.ndarray) -> np.ndarray:
        df = pd.DataFrame(arr, columns=colonnes)
        for c, cat in categories.items():
            codes = df[c].round().astype(int)
            df[c] = pd.Categorical.from_codes(codes.where(codes >= 0, -1),
                                              categories=cat)
        return modele.predict_proba(df)[:, 1]
    return f


def _expliquer_lot(ids: list[str], permutations: int, fond_n: int) -> np.ndarray:
    """Un lot de sejours, dans un processus de travail (chaque processus
    recharge les artefacts : rien de lourd ne transite entre processus)."""
    import shap
    from threadpoolctl import threadpool_limits
    warnings.filterwarnings("ignore")
    # Un fil par processus : sans cela, chaque predict_proba ouvre autant de
    # fils OpenMP que de coeurs, et seize processus se paralysent mutuellement.
    threadpool_limits(1)
    modele, meta, X = _charger()
    cats = meta["colonnes_categorielle"]
    categories = {c: X[c].cat.categories for c in cats}
    rng = np.random.default_rng(GRAINE)
    fond = X.iloc[rng.choice(len(X), fond_n, replace=False)]
    masker = shap.maskers.Independent(_encoder(fond, cats), max_samples=fond_n)
    ex = shap.explainers.Permutation(_fonction_modele(modele, list(X.columns), categories),
                                     masker, seed=GRAINE)
    evals = 2 * X.shape[1] * permutations + 1
    res = ex(_encoder(X.loc[ids], cats), max_evals=evals, silent=True)
    return np.column_stack([np.full(len(ids), np.asarray(res.base_values).ravel()[0]),
                            res.values])


def main(permutations: int = 8, fond_n: int = 100, n_jobs: int = -1) -> None:
    modele, meta, X = _charger()
    ids = list(X.index)
    taille = max(8, len(ids) // 64)
    lots = [ids[i:i + taille] for i in range(0, len(ids), taille)]
    print(f"  contributions SHAP : {len(ids)} sejours x {X.shape[1]} variables, "
          f"{permutations} permutations, fond {fond_n}, {len(lots)} lots")
    from joblib import Parallel, delayed
    blocs = Parallel(n_jobs=n_jobs)(
        delayed(_expliquer_lot)(lot, permutations, fond_n) for lot in lots)
    mat = np.vstack(blocs)
    base = float(mat[0, 0])
    contrib = pd.DataFrame(mat[:, 1:], index=X.index, columns=X.columns)

    # Controle : base + somme des contributions = probabilite servie, a 1e-9 pres.
    proba = modele.predict_proba(X)[:, 1]
    ecart = float(np.abs(base + contrib.sum(axis=1).to_numpy() - proba).max())
    if ecart > 1e-6:
        sys.exit(f"additivite rompue : ecart max {ecart:.2e}")

    contrib.to_parquet(ARTEFACTS / "contributions.parquet")
    (ARTEFACTS / "contributions.json").write_text(json.dumps({
        "base": round(base, 4),
        "methode": "valeurs de Shapley par permutations (shap.explainers.Permutation) "
                   "sur predict_proba du modele complet ; contributions en probabilite, "
                   "additives : base + somme = score",
        "permutations": permutations, "fond": fond_n, "graine": GRAINE,
        "shap_version": __import__("shap").__version__,
        "date": date.today().isoformat(),
        "controle": {"ecart_additivite_max": ecart,
                     "version_modele": meta["version"]},
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  base de population {base:.3f} ; additivite verifiee (ecart max {ecart:.1e})")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--permutations", type=int, default=8)
    p.add_argument("--fond", type=int, default=100)
    p.add_argument("--jobs", type=int, default=-1)
    a = p.parse_args()
    main(a.permutations, a.fond, a.jobs)
