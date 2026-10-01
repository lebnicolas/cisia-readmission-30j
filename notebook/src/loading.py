"""Chargement des tables brutes.

Aucune transformation ici : le contrat de ce module est de rendre les fichiers
tels qu'ils sont, pour que l'audit qualite porte sur la donnee reelle et non
sur une version deja corrigee.
"""

from __future__ import annotations

import pandas as pd

from . import config as cfg

# Colonnes a interpreter comme des dates, par table.
_DATE_COLUMNS = {
    "patients": ["DateNaissance"],
    "historique": ["DateEvenement"],
    "sejours": ["DateAdmission", "DateSortie"],
    "actes": ["DateActe"],
    "biologies": ["DatePrelevement"],
    "signes_vitaux": ["Horodatage"],
    "medications": ["DateDebut", "DateFin"],
    "comptes_rendus": ["DateCR"],
    "objets_connectes": ["Horodatage"],
}


def load_raw() -> dict[str, pd.DataFrame]:
    """Charge les 11 tables sans aucune correction.

    Les colonnes de date sont converties en datetime, ce qui n'est pas une
    correction mais un typage : les chaines sont conservees telles quelles en
    cas d'echec de parsing.
    """
    tables: dict[str, pd.DataFrame] = {}
    for name in cfg.TABLES:
        df = pd.read_csv(cfg.DATA_DIR / f"{name}.csv")
        for col in _DATE_COLUMNS.get(name, []):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")
        tables[name] = df
    return tables


def table_overview(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Recapitulatif de volumetrie, pour la section de presentation du jeu."""
    rows = []
    for name, df in tables.items():
        rows.append({
            "table": name,
            "lignes": len(df),
            "colonnes": df.shape[1],
            "cle_primaire": df.columns[0],
            "unicite_cle": df[df.columns[0]].is_unique,
        })
    return pd.DataFrame(rows).set_index("table")
