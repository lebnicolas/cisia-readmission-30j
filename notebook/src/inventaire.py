"""Inventaire des colonnes sources : ce que devient chacune, et pourquoi.

Chaque colonne des onze tables livrées y est déclarée une fois, avec son
devenir et sa raison. C'est la réponse, colonne par colonne, à deux questions
de la section 11.1 : qu'est-il advenu des données personnelles, et pourquoi
telle colonne n'a-t-elle pas servi ?

L'inventaire est déclaré à la main, parce que la raison ne se déduit pas du
code. Le test `tests/test_inventaire.py` le confronte aux tables réelles : une
colonne source absente d'ici, ou déclarée ici sans exister, fait échouer la
chaîne d'intégration. Il ne peut donc pas devenir faux en silence (cf. J3-12).
"""

from __future__ import annotations

import pandas as pd

# Les devenirs possibles, du plus au moins utilisé.
VARIABLE = "variable du modèle"
TRANSFORMEE = "variable, transformée"
CLE = "clé de jointure"
TECHNIQUE = "identifiant technique"
NETTOYAGE = "nettoyage ou audit"
CIBLE = "cible"
ETAGE_B = "étage B, non entraîné"
ECARTEE = "écartée"

DEVENIRS = [VARIABLE, TRANSFORMEE, CLE, TECHNIQUE, NETTOYAGE, CIBLE, ETAGE_B, ECARTEE]

_LIGNE = "numéro de ligne, sans information"
_DATE_FILLE = "horodatage désynchronisé du séjour, tiré indépendamment (J1-06) : audit seulement"

INVENTAIRE: dict[str, dict[str, tuple[str, str]]] = {
    "patients": {
        "PatientID": (CLE, "relie le patient à ses séjours, son historique, son territoire"),
        "NomPrenom": (ECARTEE, "identifiant direct : minimisation (§1.4)"),
        "DateNaissance": (TRANSFORMEE, "âge à l'admission ; la date elle-même n'entre pas"),
        "Sexe": (VARIABLE, ""),
        "CodePostal": (CLE, "jointure vers territoire_insee, avec la commune"),
        "Commune": (CLE, "jointure vers territoire_insee, avec le code postal"),
        "RegimeAssurance": (ECARTEE, "permis par l'énoncé, retiré par minimisation : l'AME en fait "
                                     "un marqueur individuel de précarité, sans effet mesuré (J3-11)"),
        "MedecinTraitant": (TRANSFORMEE, "seule la présence est encodée, jamais le nom"),
        "PersonneAPrevenir": (TRANSFORMEE, "seule la présence est encodée, jamais le contact"),
        "SituationFamiliale": (VARIABLE, "plus deux indicateurs : vit seul, situation inconnue"),
        "PathologiesChroniques": (TRANSFORMEE, "nombre de pathologies et un indicateur par pathologie"),
    },
    "historique": {
        "EvenementID": (TECHNIQUE, _LIGNE),
        "PatientID": (CLE, "rattache l'événement au patient"),
        "TypeEvenement": (TRANSFORMEE, "nombre d'hospitalisations, d'urgences, de consultations"),
        "DateEvenement": (TRANSFORMEE, "seuls les événements antérieurs à l'admission ; délai depuis le dernier"),
        "DureeJours": (TRANSFORMEE, "durée cumulée d'hospitalisation antérieure"),
        "CodeEtablissement": (TRANSFORMEE, "nombre d'établissements fréquentés ; le code n'entre pas"),
    },
    "sejours": {
        "SejourID": (CLE, "grain d'observation : une ligne du jeu de données par séjour"),
        "PatientID": (CLE, "découpage de la validation croisée par patient (J1-07)"),
        "DateAdmission": (TRANSFORMEE, "mois et saison d'admission ; date de référence de l'âge et de l'historique"),
        "DateSortie": (TRANSFORMEE, "sortie le week-end ; durée de séjour recalculée"),
        "DureeSejour": (NETTOYAGE, "fausse sur 54 séjours : remplacée par la durée recalculée sur les dates (§3.7, J1-11)"),
        "Service": (VARIABLE, ""),
        "TypeSejour": (VARIABLE, "plus un indicateur d'admission en urgence"),
        "GHM": (TRANSFORMEE, "catégorie et niveau de sévérité"),
        "ModeSortie": (VARIABLE, "et filtre de population : les séjours terminés par un décès sont exclus (J1-07)"),
        "Readmission30j": (CIBLE, "ce que le modèle prédit ; jamais une variable d'entrée"),
    },
    "diagnostics": {
        "DiagnosticID": (TECHNIQUE, _LIGNE),
        "SejourID": (CLE, "rattache le diagnostic au séjour"),
        "CodeCIM10": (TRANSFORMEE, "chapitre du diagnostic principal, nombre de chapitres distincts"),
        "LibelleDiagnostic": (VARIABLE, "diagnostic principal"),
        "Role": (TRANSFORMEE, "nombre de diagnostics principaux, associés, complications"),
    },
    "actes": {
        "ActeID": (TECHNIQUE, _LIGNE),
        "SejourID": (CLE, "rattache l'acte au séjour"),
        "CodeCCAM": (TRANSFORMEE, "nombre d'actes distincts"),
        "LibelleActe": (ECARTEE, "redondant : un libellé par code CCAM, sans exception (12 pour 12)"),
        "DateActe": (NETTOYAGE, _DATE_FILLE),
        "CodeExecutant": (TRANSFORMEE, "nombre d'exécutants ; le code du praticien n'entre pas"),
    },
    "biologies": {
        "BiologieID": (TECHNIQUE, _LIGNE),
        "SejourID": (CLE, "rattache le prélèvement au séjour ; 73 orphelins sur SEJ-999999 écartés (§4)"),
        "DatePrelevement": (NETTOYAGE, _DATE_FILLE),
        "Panel": (TRANSFORMEE, "un groupe de variables par panel"),
        "Valeur": (TRANSFORMEE, "harmonisée, puis moyenne, minimum, maximum, part hors bornes par panel"),
        "Unite": (NETTOYAGE, "sert à harmoniser les deux unités de la créatinine et de l'hémoglobine (§3.8)"),
        "ValeurReferenceBas": (ECARTEE, "bornes non adaptées à l'unité de la ligne : redéfinies dans config (J1-12)"),
        "ValeurReferenceHaut": (ECARTEE, "bornes non adaptées à l'unité de la ligne : redéfinies dans config (J1-12)"),
    },
    "signes_vitaux": {
        "ConstanteID": (TECHNIQUE, _LIGNE),
        "SejourID": (CLE, "rattache le relevé au séjour"),
        "Horodatage": (TRANSFORMEE, "temps relatif au premier relevé du séjour, l'absolu étant désynchronisé (J1-06)"),
        "FrequenceCardiaque": (TRANSFORMEE, "état en fin de séjour, valeurs impossibles écartées"),
        "TensionSystolique": (TRANSFORMEE, "état en fin de séjour, valeurs impossibles écartées"),
        "TensionDiastolique": (TRANSFORMEE, "état en fin de séjour, valeurs impossibles écartées"),
        "Temperature": (TRANSFORMEE, "état en fin de séjour, valeurs impossibles écartées"),
        "FrequenceRespiratoire": (TRANSFORMEE, "état en fin de séjour, valeurs impossibles écartées"),
        "SpO2": (TRANSFORMEE, "état en fin de séjour, valeurs impossibles écartées"),
    },
    "medications": {
        "PrescriptionID": (TECHNIQUE, _LIGNE),
        "PatientID": (TECHNIQUE, "doublon du patient du séjour, identique sur toutes les lignes : "
                                 "le rattachement passe par le séjour"),
        "SejourID": (CLE, "rattache la prescription au séjour"),
        "CodeATC": (TRANSFORMEE, "nombre de classes ATC"),
        "LibelleMedicament": (ECARTEE, "redondant : un libellé par code ATC, sans exception (12 pour 12)"),
        "Posologie": (ECARTEE, "indépendante du médicament et contradictoire avec la voie (« 1 IV x3/j » "
                               "en voie orale 366 fois) : générée indépendamment, comme le champ du §5.4"),
        "DateDebut": (ECARTEE, "désynchronisée du séjour comme les autres dates filles (0,5 % dans la "
                               "fenêtre, décalage médian −224 jours)"),
        "DateFin": (TRANSFORMEE, "son absence signale un traitement en cours à la sortie"),
        "Voie": (TRANSFORMEE, "part de prescriptions intraveineuses ; ne dépend pas du médicament, sans doute "
                              "générée indépendamment ; poids +0,003 de PR-AUC, sous la variance de "
                              "partition : conservée, sans effet"),
    },
    "comptes_rendus": {
        "CompteRenduID": (TECHNIQUE, _LIGNE),
        "SejourID": (CLE, "rattache le document au séjour"),
        "TypeCR": (NETTOYAGE, "filtre : seul le compte-rendu d'hospitalisation, un par séjour (J1-10)"),
        "DateCR": (NETTOYAGE, _DATE_FILLE),
        "TexteCR": (TRANSFORMEE, "gabarit, comorbidités, évolution, antécédents ; le texte lui-même n'entre jamais"),
    },
    "objets_connectes": {
        "MesureID": (TECHNIQUE, _LIGNE),
        "PatientID": (CLE, "rattache la mesure au patient ; 125 lignes sur PAT-999999, sans effet (J3-10)"),
        "Horodatage": (ETAGE_B, "mesures postérieures à la sortie : 6 patients couverts à J+7, trop peu (J1-06)"),
        "TypeMesure": (ETAGE_B, "idem"),
        "Valeur": (ETAGE_B, "idem"),
        "QualiteSignal": (NETTOYAGE, "filtre des mesures exploitables"),
    },
    "territoire_insee": {
        "Commune": (CLE, "jointure depuis le patient"),
        "CodePostal": (CLE, "jointure depuis le patient"),
        "IndiceDefavorisation": (VARIABLE, "agrégé à la commune, jamais individuel"),
        "DensiteMedicale": (VARIABLE, "plus un indicateur de désert médical"),
        "PopulationCommune": (VARIABLE, ""),
    },
}


def ecarts(tables: dict[str, pd.DataFrame]) -> tuple[list[str], list[str]]:
    """Colonnes présentes dans les tables mais absentes de l'inventaire, et l'inverse."""
    reelles = {f"{t}.{c}" for t, df in tables.items() for c in df.columns}
    declarees = {f"{t}.{c}" for t, cols in INVENTAIRE.items() for c in cols}
    return sorted(reelles - declarees), sorted(declarees - reelles)


def tableau(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """L'inventaire complet, dans l'ordre des tables et des colonnes livrées."""
    manquantes, en_trop = ecarts(tables)
    if manquantes or en_trop:
        raise ValueError(f"inventaire désaligné : non déclarées {manquantes}, inexistantes {en_trop}")
    return pd.DataFrame(
        [(t, c, *INVENTAIRE[t][c]) for t, df in tables.items() for c in df.columns],
        columns=["table", "colonne", "devenir", "raison"],
    )


def synthese(inv: pd.DataFrame) -> pd.DataFrame:
    """Nombre de colonnes par table et par devenir."""
    return (pd.crosstab(inv.table, inv.devenir, margins=True, margins_name="total")
              .reindex(columns=[d for d in DEVENIRS if d in inv.devenir.values] + ["total"])
              .reindex([*dict.fromkeys(inv.table), "total"]))
