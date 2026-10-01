"""Extensions de blocs — variables candidates issues de la litterature.

Quatre blocs supplementaires, construits comme les autres (une fonction, un
prefixe, l'etage A) et branches au pipeline sans toucher a l'existant. Leur
apport est mesure au meme protocole avant toute integration au modele final
(cf. journal J2-06).

  - charlson        indice de comorbidite de Charlson, depuis les codes CIM-10
  - lace            score LACE (duree, admission urgente, comorbidites, urgences)
  - atc             classes therapeutiques nominatives
  - bio_tendances   tendances biologiques intra-sejour, en temps relatif
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import clean, config as cfg
from .blocks import Block, _reindex

# --------------------------------------------------------------------------
# Charlson — poids des composantes presentes dans le jeu de donnees
# --------------------------------------------------------------------------

# Ponderations de l'indice de Charlson (cartographie CIM-10 de Quan 2005),
# restreintes aux six composantes que le jeu contient reellement. Les dix
# autres codes du jeu (HTA, arthrose, sepsis...) ne sont pas des composantes
# de l'indice.
CHARLSON_WEIGHTS = {
    "I50": 1,   # insuffisance cardiaque congestive
    "J44": 1,   # BPCO (maladie pulmonaire chronique)
    "E11": 1,   # diabete sans complication
    "F03": 1,   # demence
    "N18": 2,   # insuffisance renale moderee a severe
    "C50": 2,   # tumeur maligne
}


def _charlson_par_sejour(tables, population) -> pd.Series:
    dx = tables["diagnostics"]
    dx = dx[dx.SejourID.isin(population.index)]
    # Chaque composante compte une fois par sejour, quel que soit le nombre de
    # lignes qui la codent (principal + associe ne doublent pas le poids).
    comp = dx[dx.CodeCIM10.isin(CHARLSON_WEIGHTS)].drop_duplicates(
        ["SejourID", "CodeCIM10"])
    poids = comp.CodeCIM10.map(CHARLSON_WEIGHTS)
    return poids.groupby(comp.SejourID).sum().reindex(population.index).fillna(0)


def build_charlson(tables, population):
    out = pd.DataFrame(index=population.index)
    out["indice"] = _charlson_par_sejour(tables, population).astype("Int16")
    out["indice_3plus"] = (out.indice >= 3).astype("Int8")
    return _reindex(out, population, "chl")


# --------------------------------------------------------------------------
# LACE
# --------------------------------------------------------------------------

def _points_duree(d: float) -> int:
    if d < 1: return 0
    if d < 2: return 1
    if d < 3: return 2
    if d < 4: return 3
    if d < 7: return 4
    if d < 14: return 5
    return 7


def build_lace(tables, population):
    """Score LACE : Length of stay, Acuity, Comorbidity, Emergency visits.

    L'indice canonique de la litterature readmission (van Walraven 2010).
    La composante E compte les passages aux urgences dans les six mois
    precedant l'admission — l'historique etant la seule table aux dates
    coherentes, ce calcul est legitime (cf. J1-06).
    """
    out = pd.DataFrame(index=population.index)

    out["l_duree"] = population.duree_sejour.map(_points_duree).astype("Int8")
    out["a_urgent"] = (
        (population.TypeSejour == "Urgent") | (population.Service == "Urgences")
    ).astype("Int8") * 3

    charlson = _charlson_par_sejour(tables, population)
    out["c_comorbidites"] = charlson.clip(upper=4).where(charlson < 4, 5).astype("Int8")

    h = tables["historique"]
    urg = h[h.TypeEvenement == "Urgence"]
    m = population[["PatientID", "DateAdmission"]].reset_index().merge(
        urg, on="PatientID", how="left")
    fenetre = m[(m.DateEvenement < m.DateAdmission)
                & (m.DateEvenement >= m.DateAdmission - pd.Timedelta(days=182))]
    out["e_urgences_6mois"] = (
        fenetre.groupby(cfg.UNIT).size().reindex(population.index)
        .fillna(0).clip(upper=4).astype("Int8"))

    out["score"] = (out.l_duree + out.a_urgent + out.c_comorbidites
                    + out.e_urgences_6mois).astype("Int8")
    # Seuil usuel de haut risque dans la litterature.
    out["haut_risque"] = (out.score >= 10).astype("Int8")
    return _reindex(out, population, "lace")


# --------------------------------------------------------------------------
# Classes ATC nominatives
# --------------------------------------------------------------------------

def build_atc(tables, population):
    """Indicatrice par classe therapeutique.

    Le bloc medications compte les medicaments sans regarder lesquels ; or
    diuretiques, antithrombotiques ou psychotropes sont des marqueurs de
    fragilite documentes. Les douze classes du jeu sont toutes frequentes
    (283 a 354 prescriptions) : on les encode toutes.
    """
    rx = tables["medications"]
    rx = rx[rx.SejourID.isin(population.index)]
    ind = (rx.drop_duplicates(["SejourID", "CodeATC"])
             .assign(un=1)
             .pivot_table(index="SejourID", columns="CodeATC", values="un",
                          fill_value=0))
    out = ind.reindex(population.index).fillna(0).astype("Int8")
    out.columns = [c.lower() for c in out.columns]
    return _reindex(out, population, "atc")


# --------------------------------------------------------------------------
# Tendances biologiques intra-sejour
# --------------------------------------------------------------------------

def _slope(y, x) -> float:
    mask = y.notna() & x.notna()
    if mask.sum() < 2:
        return np.nan
    xv, yv = x[mask].to_numpy(float), y[mask].to_numpy(float)
    if np.ptp(xv) == 0:
        return np.nan
    return float(np.polyfit(xv, yv, 1)[0])


def build_bio_tendances(tables, population):
    """Pente et variation des panels biologiques, en temps relatif.

    Le recalage applique aux constantes vitales (J1-06) ne l'avait jamais ete
    a la biologie. Meme technique : chaque sejour est recale sur son premier
    prelevement, et la pente se calcule en unites par jour. Couverture
    partielle attendue — 1608 couples sejour-panel n'ont qu'un prelevement.
    """
    bio = clean.clean_biologies(tables["biologies"],
                               valid_stays=set(population.index))
    bio = bio[bio.SejourID.isin(population.index)]
    bio = clean.add_relative_time(bio, "SejourID", "DatePrelevement",
                                  out_col="t_relatif_h")
    bio["t_relatif_j"] = bio.t_relatif_h / 24

    out = pd.DataFrame(index=population.index)
    for panel in cfg.CANONICAL_UNITS:
        sub = bio[bio.Panel == panel]
        g = sub.groupby("SejourID")
        key = panel.lower()
        out[f"{key}_pente_j"] = g.apply(
            lambda d: _slope(d.valeur_harmonisee, d.t_relatif_j),
            include_groups=False)
        # Variation brute dernier - premier prelevement du sejour.
        tri = sub.sort_values("t_relatif_h")
        gg = tri.groupby("SejourID").valeur_harmonisee
        out[f"{key}_delta"] = gg.last() - gg.first()
    return _reindex(out, population, "biot")


# --------------------------------------------------------------------------
# Registre des extensions
# --------------------------------------------------------------------------

EXTENSIONS: list[Block] = [
    Block("charlson", "A", ["diagnostics"],
          "Indice de comorbidite de Charlson (6 composantes presentes)",
          build_charlson),
    Block("lace", "A", ["sejours", "diagnostics", "historique"],
          "Score LACE : duree, admission urgente, comorbidites, urgences 6 mois",
          build_lace),
    Block("atc", "A", ["medications"],
          "Indicatrices des 12 classes therapeutiques",
          build_atc),
    Block("bio_tendances", "A", ["biologies"],
          "Pentes et variations biologiques intra-sejour, temps relatif",
          build_bio_tendances),
]
