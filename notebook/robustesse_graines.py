"""Contrôle de robustesse d'une extension sur plusieurs graines de découpage.

Rejoue la comparaison de la section 12 (référence contre référence + Charlson)
avec quatre graines de découpage en plis ; l'aléa du modèle reste fixé à la
graine du dossier, pour que seule la partition varie. Sert à séparer un gain
réel de la variance de partition (journal J2-06, J3-02, J3-11). Jusqu'ici fait
à la main, il est désormais versionné pour pouvoir être rejoué.

    ..\\.venv\\Scripts\\python.exe robustesse_graines.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.metrics import average_precision_score  # noqa: E402
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict  # noqa: E402

from src import assemble, clean, config as cfg, extensions as ext, loading  # noqa: E402

GRAINES = [42, 7, 123, 2026]


def pr_auc(tables, population, extras, graine: int) -> float:
    X, y, g = assemble.build_dataset(tables, stages=("A",), population=population,
                                     extra_blocks=extras, verbose=False)
    m = HistGradientBoostingClassifier(
        categorical_features="from_dtype", random_state=cfg.SEED, max_iter=300,
        learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0,
        early_stopping=True, validation_fraction=0.15)
    cv = StratifiedGroupKFold(n_splits=cfg.N_SPLITS, shuffle=True, random_state=graine)
    p = cross_val_predict(m, X, y, cv=cv, groups=g, method="predict_proba")[:, 1]
    return average_precision_score(y, p)


def main() -> None:
    tables = loading.load_raw()
    population = clean.build_population(tables)
    charlson = [b for b in ext.EXTENSIONS if b.name == "charlson"]
    print("| Graine | Référence | +Charlson | Écart |")
    print("|---:|---:|---:|---:|")
    ecarts = []
    for graine in GRAINES:
        ref = pr_auc(tables, population, [], graine)
        avec = pr_auc(tables, population, charlson, graine)
        ecarts.append(avec - ref)
        print(f"| {graine} | {ref:.3f} | {avec:.3f} | {avec - ref:+.3f} |".replace(".", ","))
    print(f"\nécart moyen {sum(ecarts) / len(ecarts):+.3f}, positif sur "
          f"{sum(e > 0 for e in ecarts)} graines sur {len(ecarts)}")


if __name__ == "__main__":
    main()
