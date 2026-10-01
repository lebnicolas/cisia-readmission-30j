"""Audit qualite des sources.

Chaque fonction renvoie un tableau lisible destine a etre affiche dans le
notebook. L'audit ne corrige rien : il mesure et documente. Les corrections
sont dans `clean.py`, et chacune renvoie a une mesure faite ici.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import clean, config as cfg


# --------------------------------------------------------------------------
# Manquants, sentinelles comprises
# --------------------------------------------------------------------------

def missing_report(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Manquants par colonne, en distinguant nuls et sentinelles textuelles.

    La distinction est essentielle : dans patients.csv l'absence est codee par
    la chaine "Non renseigné", qu'un isna() ne voit pas (cf. J1-04).
    """
    rows = []
    for name, df in tables.items():
        for col in df.columns:
            n_null = int(df[col].isna().sum())
            n_sent = 0
            if clean.is_text_column(df[col]):
                n_sent = int(df[col].isin(cfg.TEXT_SENTINELS).sum())
            if n_null or n_sent:
                rows.append({
                    "table": name,
                    "colonne": col,
                    "nuls": n_null,
                    "sentinelles_texte": n_sent,
                    "total": n_null + n_sent,
                    "taux": (n_null + n_sent) / len(df),
                })
    out = pd.DataFrame(rows)
    return out.sort_values("total", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------
# Integrite referentielle
# --------------------------------------------------------------------------

_FOREIGN_KEYS = [
    ("historique", "PatientID", "patients", "PatientID"),
    ("sejours", "PatientID", "patients", "PatientID"),
    ("diagnostics", "SejourID", "sejours", "SejourID"),
    ("actes", "SejourID", "sejours", "SejourID"),
    ("biologies", "SejourID", "sejours", "SejourID"),
    ("signes_vitaux", "SejourID", "sejours", "SejourID"),
    ("medications", "SejourID", "sejours", "SejourID"),
    ("comptes_rendus", "SejourID", "sejours", "SejourID"),
    ("objets_connectes", "PatientID", "patients", "PatientID"),
]


def referential_integrity(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Cles etrangeres orphelines et couverture de la table parente."""
    rows = []
    for child, fk, parent, pk in _FOREIGN_KEYS:
        c, p = tables[child], tables[parent]
        known = set(p[pk])
        orphans = int((~c[fk].isin(known)).sum())
        rows.append({
            "table": child,
            "cle": fk,
            "reference": f"{parent}.{pk}",
            "orphelins": orphans,
            "parents_couverts": c[fk].nunique(),
            "parents_total": p[pk].nunique(),
            "couverture": c[fk].nunique() / p[pk].nunique(),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Coherence temporelle
# --------------------------------------------------------------------------

_CHILD_DATES = [
    ("biologies", "DatePrelevement"),
    ("signes_vitaux", "Horodatage"),
    ("actes", "DateActe"),
    ("comptes_rendus", "DateCR"),
]


def temporal_coherence(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Part des lignes filles tombant dans la fenetre de leur propre sejour.

    Mesure centrale de la seance 1 : elle etablit que les horodatages des
    tables filles ont ete tires independamment des dates de sejour (J1-06).
    """
    sej = tables["sejours"]
    window = sej[["SejourID", "DateAdmission", "DateSortie"]]
    rows = []
    for name, datecol in _CHILD_DATES:
        m = tables[name].merge(window, on="SejourID", how="inner")
        inside = (
            (m[datecol] >= m.DateAdmission.dt.normalize())
            & (m[datecol] <= m.DateSortie + pd.Timedelta(days=1))
        )
        shift = (m[datecol] - m.DateAdmission).dt.days
        rows.append({
            "table": name,
            "lignes": len(m),
            "dans_la_fenetre": float(inside.mean()),
            "decalage_median_j": float(shift.median()),
        })
    return pd.DataFrame(rows)


def stay_internal_coherence(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Confronte la colonne DureeSejour a l'ecart entre ses propres dates.

    Etablit que la colonne derivee est corrompue alors que les dates brutes
    sont saines (J1-11).
    """
    sej = tables["sejours"].copy()
    delta = (sej.DateSortie - sej.DateAdmission).dt.total_seconds() / 86400
    sej["duree_recalculee"] = delta.round()
    return pd.DataFrame([
        {"source": "colonne DureeSejour",
         "min": sej.DureeSejour.min(), "max": sej.DureeSejour.max(),
         "negatives": int((sej.DureeSejour < 0).sum()),
         "sup_28j": int((sej.DureeSejour > 28).sum())},
        {"source": "recalcul DateSortie - DateAdmission",
         "min": sej.duree_recalculee.min(), "max": sej.duree_recalculee.max(),
         "negatives": int((sej.duree_recalculee < 0).sum()),
         "sup_28j": int((sej.duree_recalculee > 28).sum())},
    ]).set_index("source")


def duration_triangulation(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Confronte les trois sources de duree : colonne, dates, compte rendu.

    Le compte rendu joue le role de troisieme source independante et departage
    la colonne du recalcul (J1-11).
    """
    sej = tables["sejours"].copy()
    sej["duree_recalculee"] = (
        (sej.DateSortie - sej.DateAdmission).dt.total_seconds() / 86400
    ).round()
    cr = tables["comptes_rendus"]
    cr = cr[cr.TypeCR == cfg.CR_TYPE].copy()
    cr["duree_cr"] = cr.TexteCR.str.extract(
        r"(?:Séjour de|Hospitalisation de)\s+(\d+)\s*j"
    ).astype("Float64")

    m = sej.merge(cr[["SejourID", "duree_cr"]], on="SejourID", how="inner")
    m = m.dropna(subset=["duree_cr"])
    suspect = m[(m.DureeSejour < 0) | (m.DureeSejour > 28)]

    return pd.DataFrame([
        {"comparaison": "compte rendu vs colonne DureeSejour",
         "n": len(m), "concordance": float((m.duree_cr == m.DureeSejour).mean())},
        {"comparaison": "compte rendu vs recalcul sur dates",
         "n": len(m), "concordance": float((m.duree_cr == m.duree_recalculee).mean())},
        {"comparaison": "compte rendu vs recalcul, sur sejours aberrants",
         "n": len(suspect),
         "concordance": float((suspect.duree_cr == suspect.duree_recalculee).mean())},
    ]).set_index("comparaison")


# --------------------------------------------------------------------------
# Heterogeneite des unites
# --------------------------------------------------------------------------

def unit_heterogeneity(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Unites par panel, avec les bornes de reference annoncees.

    Met en evidence que les bornes du fichier source ne suivent pas l'unite de
    la ligne : elles sont identiques pour les deux unites d'un meme panel.
    """
    b = tables["biologies"]
    g = b.groupby(["Panel", "Unite"]).agg(
        n=("Valeur", "size"),
        mediane=("Valeur", "median"),
        ref_bas=("ValeurReferenceBas", "median"),
        ref_haut=("ValeurReferenceHaut", "median"),
    ).round(2)
    g["unite_canonique"] = [cfg.CANONICAL_UNITS[p] for p, _ in g.index]
    g["conversion"] = [
        cfg.UNIT_CONVERSIONS.get((p, u), 1.0) for p, u in g.index
    ]
    return g


# --------------------------------------------------------------------------
# Valeurs implausibles
# --------------------------------------------------------------------------

def implausible_biology(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Sentinelles et valeurs negatives dans les resultats biologiques."""
    b = tables["biologies"]
    rows = []
    for panel, sub in b.groupby("Panel"):
        rows.append({
            "panel": panel,
            "n": len(sub),
            "manquants": int(sub.Valeur.isna().sum()),
            "sentinelle_999": int(sub.Valeur.isin(cfg.NUMERIC_SENTINELS).sum()),
            "negatives": int((sub.Valeur < 0).sum()),
        })
    out = pd.DataFrame(rows).set_index("panel")
    out["total_ecarte"] = out.manquants + out.sentinelle_999 + out.negatives
    out["taux_ecarte"] = out.total_ecarte / out.n
    return out


def implausible_vitals(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Constantes vitales hors bornes physiologiques."""
    sv = tables["signes_vitaux"]
    rows = []
    for col, (lo, hi) in cfg.VITALS_PLAUSIBLE.items():
        s = sv[col]
        rows.append({
            "constante": col,
            "borne_basse": lo,
            "borne_haute": hi,
            "min_observe": s.min(),
            "max_observe": s.max(),
            "manquants": int(s.isna().sum()),
            "hors_bornes": int(((s < lo) | (s > hi)).sum()),
        })
    return pd.DataFrame(rows).set_index("constante")


# --------------------------------------------------------------------------
# Couverture de la telesurveillance
# --------------------------------------------------------------------------

def iot_coverage(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Patients couverts par la telesurveillance apres leur sortie.

    Fonde la conclusion sur l'etage B : le connecteur fonctionne, la donnee
    manque (J1-06).
    """
    oc = tables["objets_connectes"]
    sej = tables["sejours"]
    last = sej.sort_values("DateSortie").groupby("PatientID").DateSortie.last()
    m = oc.merge(last.rename("sortie"), on="PatientID", how="left")
    m["delai_j"] = (m.Horodatage - m.sortie).dt.days

    total_patients = sej.PatientID.nunique()
    rows = []
    for lo, hi in [(0, 7), (0, 30), (0, 90)]:
        sub = m[(m.delai_j >= lo) & (m.delai_j <= hi)]
        rows.append({
            "fenetre": f"J+{lo} a J+{hi}",
            "mesures": len(sub),
            "patients": sub.PatientID.nunique(),
            "patients_total": total_patients,
            "couverture": sub.PatientID.nunique() / total_patients,
            "seuil_entrainabilite": cfg.IOT_MIN_PATIENTS,
        })
    return pd.DataFrame(rows).set_index("fenetre")


def iot_signal_quality(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Repartition de la qualite de signal des objets connectes."""
    oc = tables["objets_connectes"]
    g = oc.QualiteSignal.value_counts().rename("mesures").to_frame()
    g["part"] = g.mesures / len(oc)
    g["exploitable"] = [q in cfg.IOT_VALID_QUALITY for q in g.index]
    return g


# --------------------------------------------------------------------------
# Cible
# --------------------------------------------------------------------------

def target_report(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Distribution de la cible, avant et apres exclusion des deces."""
    sej = tables["sejours"]
    kept = sej[~sej.ModeSortie.isin(cfg.EXCLUDED_DISCHARGE_MODES)]
    rows = []
    for label, df in [("population brute", sej), ("apres exclusion des deces", kept)]:
        rows.append({
            "population": label,
            "sejours": len(df),
            "patients": df.PatientID.nunique(),
            "readmissions": int(df[cfg.TARGET].sum()),
            "taux": float(df[cfg.TARGET].mean()),
        })
    return pd.DataFrame(rows).set_index("population")


def target_by_discharge_mode(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Taux de readmission par mode de sortie.

    Rend visible une incoherence clinique : le taux des sejours termines par
    un deces est indiscernable de celui des autres modalites, alors qu'il
    devrait etre nul.
    """
    sej = tables["sejours"]
    g = sej.groupby("ModeSortie")[cfg.TARGET].agg(["size", "sum", "mean"])
    g.columns = ["sejours", "readmissions", "taux"]
    return g.sort_values("taux", ascending=False)


def stays_per_patient(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Nombre de sejours par patient — justifie le decoupage groupe (J1-07)."""
    counts = tables["sejours"].groupby("PatientID").size()
    g = counts.value_counts().sort_index().rename("patients").to_frame()
    g.index.name = "sejours_par_patient"
    g["sejours_concernes"] = g.index * g.patients
    return g
