"""Nettoyage et harmonisation.

Chaque correction renvoie a la mesure qui la justifie dans `audit.py`. Les
fonctions renvoient une copie : les tables brutes restent disponibles pour
comparaison avant/apres dans le notebook.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as cfg


def is_text_column(s: pd.Series) -> bool:
    """Colonne de texte, que pandas la type `object` (2.x) ou `str` (3.x).

    Le test `dtype == object` seul est faux sous pandas 3, dont le type par
    defaut des chaines est `str` : la normalisation des sentinelles devenait
    silencieusement inoperante (J3-02).
    """
    return s.dtype == object or pd.api.types.is_string_dtype(s)


def normalize_sentinels(df: pd.DataFrame) -> pd.DataFrame:
    """Convertit les sentinelles textuelles en vraies valeurs manquantes.

    Sans cette etape, "Non renseigné" est traite comme une modalite a part
    entiere et les manquants de patients.csv restent invisibles (J1-04).
    """
    out = df.copy()
    for col in out.columns:
        if is_text_column(out[col]):
            out[col] = out[col].replace(cfg.TEXT_SENTINELS, np.nan)
    return out


def clean_sejours(sejours: pd.DataFrame) -> pd.DataFrame:
    """Repare la duree de sejour et signale les valeurs d'origine corrompues.

    La colonne DureeSejour est ecartee au profit d'un recalcul sur les dates,
    valide sur la totalite des sejours et confirme par les comptes rendus
    (J1-11).
    """
    out = normalize_sentinels(sejours)
    out["duree_sejour"] = (
        (out.DateSortie - out.DateAdmission).dt.total_seconds() / 86400
    ).round()
    out["duree_source_corrompue"] = (
        (out.DureeSejour < 0) | (out.DureeSejour != out.duree_sejour)
    )
    return out.drop(columns=["DureeSejour"])


def build_population(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Constitue la population d'etude : un sejour par ligne.

    Exclut les sejours termines par un deces : la question « ce patient
    sera-t-il readmis » ne se pose pas pour lui. Le jeu de donnees marque
    pourtant 17 de ces 92 sejours comme readmis (J1-07 revise).
    """
    sej = clean_sejours(tables["sejours"])
    pop = sej[~sej.ModeSortie.isin(cfg.EXCLUDED_DISCHARGE_MODES)].copy()
    return pop.set_index(cfg.UNIT).sort_index()


def clean_biologies(biologies: pd.DataFrame,
                    valid_stays: set | None = None) -> pd.DataFrame:
    """Harmonise les unites et ecarte les valeurs non exploitables.

    Trois corrections, chacune mesuree a l'audit :
      - suppression des lignes orphelines (SejourID inconnu),
      - mise a NaN des sentinelles 999 et des valeurs negatives,
      - conversion vers l'unite canonique du panel.

    Les bornes de reference du fichier source ne sont pas reprises : elles ne
    suivent pas l'unite de la ligne. On utilise celles de `config`.
    """
    out = biologies.copy()

    if valid_stays is not None:
        out["orpheline"] = ~out.SejourID.isin(valid_stays)
        out = out[~out.orpheline].drop(columns=["orpheline"])

    # Sentinelles et negatives : ce ne sont pas des resultats, ce sont des
    # absences de resultat deguisees.
    out.loc[out.Valeur.isin(cfg.NUMERIC_SENTINELS), "Valeur"] = np.nan
    out.loc[out.Valeur < 0, "Valeur"] = np.nan

    # Conversion vers l'unite canonique.
    factors = out.set_index(["Panel", "Unite"]).index.map(
        lambda k: cfg.UNIT_CONVERSIONS.get(k, 1.0)
    )
    out["valeur_harmonisee"] = out.Valeur * np.asarray(factors, dtype=float)
    out["unite_harmonisee"] = out.Panel.map(cfg.CANONICAL_UNITS)

    # Plausibilite physiologique, appliquee apres conversion.
    lo = out.Panel.map(lambda p: cfg.BIOLOGY_PLAUSIBLE[p][0])
    hi = out.Panel.map(lambda p: cfg.BIOLOGY_PLAUSIBLE[p][1])
    implausible = (out.valeur_harmonisee < lo) | (out.valeur_harmonisee > hi)
    out.loc[implausible, "valeur_harmonisee"] = np.nan

    # Position par rapport aux bornes de reference corrigees.
    ref_lo = out.Panel.map(lambda p: cfg.REFERENCE_RANGES[p][0])
    ref_hi = out.Panel.map(lambda p: cfg.REFERENCE_RANGES[p][1])
    # Type flottant et non booleen : l'indicateur doit pouvoir rester indefini
    # quand la valeur elle-meme a ete ecartee.
    out["hors_reference"] = np.where(
        out.valeur_harmonisee.isna(), np.nan,
        ((out.valeur_harmonisee < ref_lo) | (out.valeur_harmonisee > ref_hi)).astype(float),
    )

    return out


def clean_vitals(signes_vitaux: pd.DataFrame) -> pd.DataFrame:
    """Ecarte les constantes physiologiquement impossibles.

    Une frequence cardiaque a zero ou une saturation a 140 % ne sont pas des
    valeurs extremes mais des defaillances de capteur : on les traite comme des
    manquants plutot que de les tronquer, pour ne pas fabriquer de valeur.
    """
    out = signes_vitaux.copy()
    for col, (lo, hi) in cfg.VITALS_PLAUSIBLE.items():
        bad = (out[col] < lo) | (out[col] > hi)
        out.loc[bad, col] = np.nan
    return out


def add_relative_time(df: pd.DataFrame, stay_col: str, time_col: str,
                      out_col: str = "t_relatif_h") -> pd.DataFrame:
    """Recale les horodatages sur le premier releve de chaque sejour.

    Les horodatages absolus sont inexploitables (J1-06), mais l'ecart interne
    au sejour est coherent : recale sur son propre debut, chaque sejour
    retrouve une chronologie utilisable.
    """
    out = df.copy()
    t0 = out.groupby(stay_col)[time_col].transform("min")
    out[out_col] = (out[time_col] - t0).dt.total_seconds() / 3600
    span = out.groupby(stay_col)[out_col].transform("max")
    # Position relative dans le sejour, entre 0 (admission) et 1 (sortie).
    out["position_sejour"] = np.where(span > 0, out[out_col] / span, np.nan)
    return out


def clean_iot(objets_connectes: pd.DataFrame) -> pd.DataFrame:
    """Ne conserve que les mesures de qualite exploitable."""
    out = objets_connectes.copy()
    out["exploitable"] = out.QualiteSignal.isin(cfg.IOT_VALID_QUALITY)
    return out
