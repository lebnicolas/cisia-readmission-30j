"""Assemblage des blocs en une table d'apprentissage.

L'assemblage filtre les blocs sur l'etage demande. C'est ce filtre, et non une
vigilance manuelle, qui garantit qu'aucune donnee posterieure a la sortie
n'alimente le modele de sortie.
"""

from __future__ import annotations

import pandas as pd

from . import blocks as blk, clean, config as cfg


def build_dataset(tables: dict[str, pd.DataFrame],
                  stages: tuple[str, ...] = ("A",),
                  population: pd.DataFrame | None = None,
                  extra_blocks: list = (),
                  verbose: bool = True) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Construit la table d'apprentissage pour les etages demandes.

    Renvoie le triplet (X, y, groupes). Les groupes servent au decoupage
    groupe par patient, qui evite qu'un meme patient soit reparti entre
    apprentissage et test (J1-07).

    `extra_blocks` accepte des blocs supplementaires (cf. extensions.py),
    soumis au meme filtre d'etage que les blocs du registre.
    """
    if population is None:
        population = clean.build_population(tables)

    selected = [b for b in list(blk.BLOCKS) + list(extra_blocks)
                if b.stage in stages]
    parts = []
    for b in selected:
        part = b.build(tables, population)
        parts.append(part)
        if verbose:
            print(f"  bloc {b.name:<18} etage {b.stage}  "
                  f"{part.shape[1]:>3} variables  "
                  f"{part.notna().any(axis=1).sum():>4} sejours renseignes")

    X = pd.concat(parts, axis=1)
    y = population[cfg.TARGET].astype(int)
    groups = population[cfg.GROUP]

    if verbose:
        print(f"\n  total : {X.shape[0]} sejours x {X.shape[1]} variables")
        print(f"  cible : {y.sum()} readmissions ({y.mean():.1%})")
        print(f"  groupes : {groups.nunique()} patients")
    return X, y, groups


def block_contribution(X: pd.DataFrame) -> pd.DataFrame:
    """Volume et taux de remplissage par bloc, pour la section d'assemblage."""
    rows = []
    for b in blk.BLOCKS:
        cols = [c for c in X.columns if c.startswith(_prefix(b.name))]
        if not cols:
            continue
        sub = X[cols]
        rows.append({
            "bloc": b.name,
            "etage": b.stage,
            "variables": len(cols),
            "remplissage_moyen": float(sub.notna().mean().mean()),
            "sejours_couverts": int(sub.notna().any(axis=1).sum()),
        })
    return pd.DataFrame(rows).set_index("bloc")


_PREFIXES = {
    "demographie": "demo_", "sejour": "sej_", "diagnostics": "dx_",
    "actes": "act_", "biologie": "bio_", "constantes": "cst_",
    "medications": "rx_", "texte": "cr_", "historique": "hist_",
    "territoire": "geo_", "telesurveillance": "iot_",
}


def _prefix(name: str) -> str:
    return _PREFIXES[name]
