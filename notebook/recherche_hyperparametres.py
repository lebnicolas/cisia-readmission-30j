"""Lance la recherche d'hyperparametres imbriquee et fige son resultat.

Le calcul est trop long pour vivre dans une cellule du notebook (quelques
minutes). Il est execute une fois ici, son resultat est ecrit dans
outputs/hyperparametres.json, et le notebook se contente de le relire.

    ..\\.venv\\Scripts\\python.exe recherche_hyperparametres.py [n_essais]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import assemble, clean, config as cfg, loading, tuning  # noqa: E402


def main() -> None:
    n_essais = int(sys.argv[1]) if len(sys.argv) > 1 else 50

    print("chargement des donnees...")
    tables = loading.load_raw()
    population = clean.build_population(tables)
    X, y, groups = assemble.build_dataset(tables, stages=("A",),
                                          population=population, verbose=False)
    print(f"  {len(X)} sejours, {X.shape[1]} variables, "
          f"{int(y.sum())} readmissions")

    print(f"\nrecherche imbriquee — {n_essais} essais par pli externe "
          f"({cfg.N_SPLITS} plis) :")
    res = tuning.recherche_imbriquee(X, y, groups, n_essais=n_essais)

    print("\n--- mise en commun des plis externes ---")
    print(f"  reglage manuel : PR-AUC {res['mise_en_commun']['manuel']:.4f}")
    print(f"  reglage Optuna : PR-AUC {res['mise_en_commun']['Optuna']:.4f}")
    print(f"  ecart          : {res['écart']:+.4f}")

    print("\n--- moyenne par pli ---")
    m = res["moyenne_par_pli"]
    print(f"  manuel : {m['manuel']:.4f} (ecart-type {m['écart-type manuel']:.4f})")
    print(f"  Optuna : {m['Optuna']:.4f} (ecart-type {m['écart-type Optuna']:.4f})")

    print("\n--- hyperparametres retenus par pli ---")
    print(tuning.table_params(res).to_string())

    sortie = cfg.OUTPUT_DIR / "hyperparametres.json"
    sortie.write_text(json.dumps(res, indent=2, ensure_ascii=False),
                      encoding="utf-8")
    print(f"\nresultat ecrit : {sortie}")


if __name__ == "__main__":
    main()
