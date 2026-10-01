"""Blocs de variables — un bloc par source.

Chaque bloc est une fonction independante qui prend les tables et la population
d'etude, et renvoie un tableau indexe par SejourID. Les blocs ne se connaissent
pas entre eux : ajouter une source revient a ecrire une fonction et a
l'enregistrer dans `BLOCKS`, sans toucher au reste du pipeline.

Chaque bloc declare son etage :
  A — la source est disponible au moment de la sortie ;
  B — la source est posterieure a la sortie, donc reservee a la reevaluation.

Cette declaration est ce qui empeche une fuite temporelle : l'assemblage filtre
sur l'etage demande, il n'y a pas de vigilance manuelle a exercer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from . import clean, config as cfg


@dataclass(frozen=True)
class Block:
    name: str
    stage: str                 # "A" ou "B"
    sources: list[str]
    description: str
    build: Callable[[dict, pd.DataFrame], pd.DataFrame] = field(repr=False)


def _reindex(df: pd.DataFrame, population: pd.DataFrame,
             prefix: str) -> pd.DataFrame:
    """Aligne un bloc sur la population et prefixe ses colonnes."""
    out = df.reindex(population.index)
    out.columns = [f"{prefix}_{c}" for c in out.columns]
    return out


# --------------------------------------------------------------------------
# Bloc 1 — Demographie (patients)
# --------------------------------------------------------------------------

def build_demographie(tables, population):
    pat = clean.normalize_sentinels(tables["patients"])
    m = population[["PatientID", "DateAdmission"]].merge(
        pat, on="PatientID", how="left"
    ).set_index(population.index)

    out = pd.DataFrame(index=population.index)
    out["age"] = ((m.DateAdmission - m.DateNaissance).dt.days / 365.25).round(1)
    out["sexe"] = m.Sexe.astype("category")
    # RegimeAssurance n'est pas repris. L'enonce l'autorise (collecte a
    # l'hopital), mais l'AME, accordee sous condition de ressources, en fait un
    # marqueur individuel de precarite ; mesure avant retrait : derniere des 133
    # variables a la permutation, aucun effet sur la PR-AUC. Minimisation (J3-11).
    out["situation_familiale"] = m.SituationFamiliale.astype("category")
    out["vit_seul"] = (m.SituationFamiliale == "Vit seul").astype("Int8")
    out["situation_inconnue"] = m.SituationFamiliale.isna().astype("Int8")
    out["medecin_traitant_declare"] = m.MedecinTraitant.notna().astype("Int8")
    out["aidant_declare"] = m.PersonneAPrevenir.notna().astype("Int8")

    # Pathologies chroniques : liste separee par "|", "Aucune" signifiant
    # l'absence de pathologie declaree.
    path = m.PathologiesChroniques.fillna("")
    listed = path.str.split("|").apply(
        lambda xs: [x for x in xs if x and x != "Aucune"]
    )
    out["nb_pathologies"] = listed.apply(len).astype("Int16")
    vocab = sorted({p for xs in listed for p in xs})
    for p in vocab:
        out[f"patho_{p}"] = listed.apply(lambda xs, p=p: int(p in xs)).astype("Int8")

    return _reindex(out, population, "demo")


# --------------------------------------------------------------------------
# Bloc 2 — Sejour (sejours)
# --------------------------------------------------------------------------

def build_sejour(tables, population):
    out = pd.DataFrame(index=population.index)
    out["duree"] = population.duree_sejour
    out["duree_corrompue_source"] = population.duree_source_corrompue.astype("Int8")
    out["service"] = population.Service.astype("category")
    out["type_sejour"] = population.TypeSejour.astype("category")
    out["urgent"] = (population.TypeSejour == "Urgent").astype("Int8")
    out["mode_sortie"] = population.ModeSortie.astype("category")
    out["retour_domicile"] = (population.ModeSortie == "Domicile").astype("Int8")
    # Le GHM encode l'activite (chiffre) et la severite (dernier caractere).
    out["ghm_categorie"] = population.GHM.str.slice(0, 3).astype("category")
    out["ghm_severite"] = population.GHM.str.slice(-1).astype("category")
    out["mois_admission"] = population.DateAdmission.dt.month.astype("Int8")
    out["admission_hiver"] = population.DateAdmission.dt.month.isin(
        [6, 7, 8]  # hiver austral : le jeu de donnees est reunionnais
    ).astype("Int8")
    out["sortie_weekend"] = (population.DateSortie.dt.dayofweek >= 5).astype("Int8")
    return _reindex(out, population, "sej")


# --------------------------------------------------------------------------
# Bloc 3 — Diagnostics (PMSI)
# --------------------------------------------------------------------------

def build_diagnostics(tables, population):
    dx = tables["diagnostics"]
    dx = dx[dx.SejourID.isin(population.index)]

    out = pd.DataFrame(index=population.index)
    out["nb_total"] = dx.groupby("SejourID").size()
    for role in ["Principal", "Associe", "Complication"]:
        out[f"nb_{role.lower()}"] = (
            dx[dx.Role == role].groupby("SejourID").size()
        )
    out = out.fillna(0).astype("Int16")

    principal = (dx[dx.Role == "Principal"]
                 .drop_duplicates("SejourID")
                 .set_index("SejourID"))
    out["diagnostic_principal"] = principal.LibelleDiagnostic.astype("category")
    # Le chapitre CIM-10 (premiere lettre) regroupe les grandes familles.
    out["chapitre_cim10"] = principal.CodeCIM10.str.slice(0, 1).astype("category")
    out["nb_chapitres_distincts"] = (
        dx.assign(ch=dx.CodeCIM10.str.slice(0, 1))
          .groupby("SejourID").ch.nunique()
    ).reindex(population.index).fillna(0).astype("Int16")

    return _reindex(out, population, "dx")


# --------------------------------------------------------------------------
# Bloc 4 — Actes (PMSI)
# --------------------------------------------------------------------------

def build_actes(tables, population):
    ac = tables["actes"]
    ac = ac[ac.SejourID.isin(population.index)]

    out = pd.DataFrame(index=population.index)
    g = ac.groupby("SejourID")
    out["nb_actes"] = g.size()
    out["nb_actes_distincts"] = g.CodeCCAM.nunique()
    out["nb_executants"] = g.CodeExecutant.nunique()
    out = out.reindex(population.index).fillna(0).astype("Int16")
    # 229 sejours n'ont aucun acte : l'absence est informative (sejour medical
    # sans geste), pas un manquant a imputer.
    out["aucun_acte"] = (out.nb_actes == 0).astype("Int8")
    return _reindex(out, population, "act")


# --------------------------------------------------------------------------
# Bloc 5 — Biologie (DPI)
# --------------------------------------------------------------------------

def build_biologie(tables, population):
    bio = clean.clean_biologies(tables["biologies"],
                               valid_stays=set(population.index))
    bio = bio[bio.SejourID.isin(population.index)]

    out = pd.DataFrame(index=population.index)
    out["nb_prelevements"] = bio.groupby("SejourID").size()
    out["nb_panels"] = bio.groupby("SejourID").Panel.nunique()

    for panel in cfg.CANONICAL_UNITS:
        sub = bio[bio.Panel == panel]
        g = sub.groupby("SejourID").valeur_harmonisee
        key = panel.lower()
        out[f"{key}_moy"] = g.mean()
        out[f"{key}_min"] = g.min()
        out[f"{key}_max"] = g.max()
        out[f"{key}_n"] = g.count()
        # Part des prelevements hors bornes de reference corrigees.
        out[f"{key}_hors_ref"] = sub.groupby("SejourID").hors_reference.mean()

    out["nb_prelevements"] = out.nb_prelevements.fillna(0)
    out["nb_panels"] = out.nb_panels.fillna(0)
    return _reindex(out, population, "bio")


# --------------------------------------------------------------------------
# Bloc 6 — Constantes vitales (DPI)
# --------------------------------------------------------------------------

def _slope(y: pd.Series, x: pd.Series) -> float:
    """Pente d'une regression lineaire simple, robuste aux series courtes."""
    mask = y.notna() & x.notna()
    if mask.sum() < 3:
        return np.nan
    xv, yv = x[mask].to_numpy(float), y[mask].to_numpy(float)
    if np.ptp(xv) == 0:
        return np.nan
    return float(np.polyfit(xv, yv, 1)[0])


def build_constantes(tables, population):
    sv = clean.clean_vitals(tables["signes_vitaux"])
    sv = sv[sv.SejourID.isin(population.index)]
    # Les horodatages absolus sont inexploitables ; on recale chaque sejour sur
    # son propre premier releve (J1-06).
    sv = clean.add_relative_time(sv, "SejourID", "Horodatage")

    out = pd.DataFrame(index=population.index)
    g = sv.groupby("SejourID")
    out["nb_releves"] = g.size()
    out["etendue_releves_h"] = g.t_relatif_h.max()

    # Dernier quart du sejour : etat du patient a l'approche de la sortie.
    fin = sv[sv.position_sejour >= 0.75]

    for col in cfg.VITALS:
        key = col.lower()
        out[f"{key}_moy"] = g[col].mean()
        out[f"{key}_min"] = g[col].min()
        out[f"{key}_max"] = g[col].max()
        out[f"{key}_ecart_type"] = g[col].std()
        out[f"{key}_fin_sejour"] = fin.groupby("SejourID")[col].mean()
        out[f"{key}_pente"] = g.apply(
            lambda d, c=col: _slope(d[c], d.t_relatif_h), include_groups=False
        )
        # Taux de defaillance capteur, informatif en soi.
        out[f"{key}_taux_manquant"] = g[col].apply(lambda s: s.isna().mean())

    out["nb_releves"] = out.nb_releves.fillna(0)
    return _reindex(out, population, "cst")


# --------------------------------------------------------------------------
# Bloc 7 — Medicaments (DPI)
# --------------------------------------------------------------------------

def build_medications(tables, population):
    rx = tables["medications"]
    rx = rx[rx.SejourID.isin(population.index)]

    out = pd.DataFrame(index=population.index)
    g = rx.groupby("SejourID")
    out["nb_prescriptions"] = g.size()
    out["nb_classes_atc"] = g.CodeATC.nunique()
    # DateFin absente signifie traitement en cours a la sortie : manquant
    # informatif, a compter plutot qu'a imputer (J1-04).
    out["nb_traitements_en_cours"] = g.DateFin.apply(lambda s: s.isna().sum())
    out["part_intraveineuse"] = g.Voie.apply(lambda s: (s == "IV").mean())
    out = out.reindex(population.index)
    out[["nb_prescriptions", "nb_classes_atc", "nb_traitements_en_cours"]] = (
        out[["nb_prescriptions", "nb_classes_atc", "nb_traitements_en_cours"]]
        .fillna(0)
    )
    # Seuil usuel de polymedication en geriatrie.
    out["polymedication"] = (out.nb_classes_atc >= 5).astype("Int8")
    return _reindex(out, population, "rx")


# --------------------------------------------------------------------------
# Bloc 8 — Comptes rendus (texte)
# --------------------------------------------------------------------------

_ANTECEDENT_PATTERNS = {
    "insuffisance_renale": "insuffisance rénale",
    "bpco_tabac": "BPCO",
    "cardiopathie": "cardiopathie",
    "hta_diabete": "HTA",
    "aucun": "aucun antécédent",
}


def build_texte(tables, population):
    cr = tables["comptes_rendus"]
    cr = cr[(cr.TypeCR == cfg.CR_TYPE) & cr.SejourID.isin(population.index)]
    cr = cr.drop_duplicates("SejourID").set_index("SejourID")
    t = cr.TexteCR

    out = pd.DataFrame(index=cr.index)
    # Gabarit du document : conditionne les champs disponibles (J1-10).
    out["gabarit"] = np.select(
        [t.str.startswith("Patient admis"),
         t.str.startswith("Séjour de"),
         t.str.startswith("Hospitalisation")],
        ["A", "B", "C"], default="autre",
    )
    # Seules les variables ayant montre un lien a la cible sont construites.
    out["nb_comorbidites"] = (
        t.str.extract(r"Comorbidités\s*:\s*(\d+)", expand=False).astype("Float64")
    )
    out["evolution_favorable"] = t.str.contains("Évolution favorable").astype("Int8")
    for name, pattern in _ANTECEDENT_PATTERNS.items():
        flag = t.str.contains(pattern, regex=False)
        # L'information n'existe que sur le gabarit A.
        out[f"atcd_{name}"] = np.where(out.gabarit == "A", flag.astype(float), np.nan)

    out["gabarit"] = out.gabarit.astype("category")
    return _reindex(out, population, "cr")


# --------------------------------------------------------------------------
# Bloc 9 — Historique de recours aux soins
# --------------------------------------------------------------------------

def build_historique(tables, population):
    h = tables["historique"]
    # Contrairement aux tables filles du sejour, les dates de l'historique sont
    # coherentes : 100 % des evenements precedent la premiere admission.
    m = population[["PatientID", "DateAdmission"]].reset_index().merge(
        h, on="PatientID", how="left"
    )
    anterieur = m[m.DateEvenement < m.DateAdmission]

    out = pd.DataFrame(index=population.index)
    g = anterieur.groupby(cfg.UNIT)
    out["nb_evenements"] = g.size()
    for kind in ["Hospitalisation", "Urgence", "Consultation"]:
        out[f"nb_{kind.lower()}"] = (
            anterieur[anterieur.TypeEvenement == kind].groupby(cfg.UNIT).size()
        )
    out["duree_hospit_anterieure"] = g.DureeJours.sum()
    out["nb_etablissements"] = g.CodeEtablissement.nunique()
    out["jours_depuis_dernier"] = (
        g.apply(lambda d: (d.DateAdmission - d.DateEvenement).dt.days.min(),
                include_groups=False)
    )
    out = out.reindex(population.index)
    counts = [c for c in out.columns if c.startswith("nb_") or c.startswith("duree_")]
    out[counts] = out[counts].fillna(0)
    return _reindex(out, population, "hist")


# --------------------------------------------------------------------------
# Bloc 10 — Territoire (source externe INSEE)
# --------------------------------------------------------------------------

def build_territoire(tables, population):
    pat = tables["patients"][["PatientID", "Commune", "CodePostal"]]
    ins = tables["territoire_insee"]
    m = (population[["PatientID"]]
         .merge(pat, on="PatientID", how="left")
         .merge(ins, on=["Commune", "CodePostal"], how="left"))
    m.index = population.index

    out = pd.DataFrame(index=population.index)
    # Seule source socio-economique autorisee : agregee a la commune, jamais
    # individuelle (contrainte explicite de l'enonce).
    out["indice_defavorisation"] = m.IndiceDefavorisation
    out["densite_medicale"] = m.DensiteMedicale
    out["population_commune"] = m.PopulationCommune
    # Un desert medical se definit par une densite basse ; seuil au premier
    # quintile observe, calcule sur les communes et non sur les sejours pour
    # ne pas ponderer par le volume d'hospitalisation.
    seuil = ins.DensiteMedicale.quantile(0.20)
    out["desert_medical"] = (m.DensiteMedicale <= seuil).astype("Int8")
    return _reindex(out, population, "geo")


# --------------------------------------------------------------------------
# Bloc 11 — Telesurveillance (etage B, source externe)
# --------------------------------------------------------------------------

def build_telesurveillance(tables, population):
    """Mesures issues des objets connectes dans la fenetre post-sortie.

    Ce bloc appartient a l'etage B : ses variables ne sont disponibles qu'apres
    la sortie et ne doivent jamais alimenter le modele de sortie. Il est
    implemente et executable ; sa couverture reelle sur ce jeu de donnees est
    mesuree dans l'audit et se revele insuffisante pour un entrainement.
    """
    oc = clean.clean_iot(tables["objets_connectes"])
    oc = oc[oc.exploitable]

    m = (population[["PatientID", "DateSortie"]].reset_index()
         .merge(oc, on="PatientID", how="left"))
    m["delai_j"] = (m.Horodatage - m.DateSortie).dt.days
    fenetre = m[(m.delai_j >= 0) & (m.delai_j <= cfg.IOT_WINDOW_DAYS)]

    out = pd.DataFrame(index=population.index)
    g = fenetre.groupby(cfg.UNIT)
    out["nb_mesures"] = g.size()
    out["nb_jours_couverts"] = g.delai_j.nunique()
    for kind in ["FrequenceCardiaque", "ActivitePas", "Poids", "SpO2"]:
        sub = fenetre[fenetre.TypeMesure == kind].groupby(cfg.UNIT).Valeur
        out[f"{kind.lower()}_moy"] = sub.mean()
        out[f"{kind.lower()}_min"] = sub.min()

    out = out.reindex(population.index)
    out["nb_mesures"] = out.nb_mesures.fillna(0)
    out["nb_jours_couverts"] = out.nb_jours_couverts.fillna(0)
    out["sous_telesurveillance"] = (out.nb_mesures > 0).astype("Int8")
    return _reindex(out, population, "iot")


# --------------------------------------------------------------------------
# Registre
# --------------------------------------------------------------------------

BLOCKS: list[Block] = [
    Block("demographie", "A", ["patients"],
          "Age, sexe, couverture, isolement, pathologies chroniques",
          build_demographie),
    Block("sejour", "A", ["sejours"],
          "Duree reparee, service, type, mode de sortie, saisonnalite",
          build_sejour),
    Block("diagnostics", "A", ["diagnostics"],
          "Codage CIM-10 : volume, roles, diagnostic principal, chapitres",
          build_diagnostics),
    Block("actes", "A", ["actes"],
          "Codage CCAM : volume, diversite, absence informative",
          build_actes),
    Block("biologie", "A", ["biologies"],
          "Panels harmonises en unite canonique, ecarts aux bornes corrigees",
          build_biologie),
    Block("constantes", "A", ["signes_vitaux"],
          "Constantes recalees sur le temps relatif : niveau, dispersion, "
          "tendance, etat en fin de sejour",
          build_constantes),
    Block("medications", "A", ["medications"],
          "Volume de prescription, classes ATC, polymedication, voie",
          build_medications),
    Block("texte", "A", ["comptes_rendus"],
          "Variables extraites du CRH par regles cliniques, par gabarit",
          build_texte),
    Block("historique", "A", ["historique"],
          "Recours aux soins anterieur a l'admission",
          build_historique),
    Block("territoire", "A", ["patients", "territoire_insee"],
          "Indicateurs INSEE agreges a la commune (seul proxy autorise)",
          build_territoire),
    Block("telesurveillance", "B", ["objets_connectes"],
          "Mesures post-sortie dans la fenetre de reevaluation",
          build_telesurveillance),
]

BLOCKS_BY_NAME = {b.name: b for b in BLOCKS}


def registry() -> pd.DataFrame:
    """Vue tabulaire du registre, pour presentation dans le notebook."""
    return pd.DataFrame([
        {"bloc": b.name, "etage": b.stage,
         "sources": ", ".join(b.sources), "role": b.description}
        for b in BLOCKS
    ]).set_index("bloc")
