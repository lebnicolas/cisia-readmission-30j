"""Constantes du pipeline.

Toutes les valeurs de ce module sont issues de l'audit qualite mene en seance 1
(cf. JOURNAL.md, entrees J1-04 a J1-11). Aucune n'est un reglage arbitraire :
chaque seuil est justifie par une mesure sur les donnees ou par une borne
physiologique documentee.
"""

from pathlib import Path

# --------------------------------------------------------------------------
# Chemins
# --------------------------------------------------------------------------

# Le dossier de donnees porte un accent compose dans son nom ; on le resout par
# motif plutot que par chaine litterale pour eviter les problemes d'encodage.
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = next(p for p in PROJECT_DIR.parent.iterdir()
                if p.is_dir() and p.name.startswith("CISIA"))
OUTPUT_DIR = PROJECT_DIR / "outputs"

TABLES = [
    "patients", "historique", "sejours", "diagnostics", "actes",
    "biologies", "signes_vitaux", "medications", "comptes_rendus",
    "objets_connectes", "territoire_insee",
]

# --------------------------------------------------------------------------
# Cible et protocole
# --------------------------------------------------------------------------

TARGET = "Readmission30j"
GROUP = "PatientID"       # cle de regroupement du decoupage (cf. J1-07)
UNIT = "SejourID"         # grain d'observation : une ligne = un sejour
SEED = 42
N_SPLITS = 5

# Modes de sortie exclus de la population d'etude. Un patient decede ne peut
# pas etre readmis : ces sejours n'appartiennent pas a la population sur
# laquelle la question se pose. Le jeu de donnees ne respecte pas cette
# contrainte — 17 des 92 deces sont marques readmis — ce qui en fait une
# incoherence a corriger, non une fuite de cible (cf. J1-07 revise).
EXCLUDED_DISCHARGE_MODES = ["Deces"]

# Rapport entre le cout d'un faux negatif et celui d'un faux positif : une
# readmission non anticipee est tenue pour 5 fois plus couteuse qu'une alerte
# inutile. Ce n'est pas un hyperparametre a optimiser mais un arbitrage metier
# (cf. J1-15). L'argument purement economique donnerait un rapport bien
# superieur ; c'est la capacite de suivi du service qui borne la valeur.
COST_RATIO_FN_FP = 5

# Seuil de decision qui en decoule, fige apres analyse de sensibilite.
DECISION_THRESHOLD = 0.11

# --------------------------------------------------------------------------
# Sentinelles textuelles
# --------------------------------------------------------------------------

# Dans patients.csv, l'absence est codee par une chaine et non par une valeur
# nulle : un isna() naif renvoie zero manquant (cf. J1-04).
TEXT_SENTINELS = ["Non renseigné", "Non renseigne", "NR", "Inconnu", ""]

# --------------------------------------------------------------------------
# Biologie
# --------------------------------------------------------------------------

# Valeur sentinelle numerique reperee sur 86 lignes, tous panels confondus.
NUMERIC_SENTINELS = [999, 9999, -999]

# Unite canonique retenue par panel, et facteur de conversion depuis l'unite
# alternative. Facteurs verifies sur les medianes observees :
#   Creatinine  mediane 10,25 mg/L x 8,84 = 90,6 ~ 93,9 umol/L observes
#   Hemoglobine mediane 11,84 g/dL x 10   = 118,4 ~ 121,5 g/L observes
CANONICAL_UNITS = {
    "CRP": "mg/L",
    "Creatinine": "µmol/L",
    "GlobulesBlancs": "G/L",
    "Hemoglobine": "g/L",
    "Natremie": "mmol/L",
}

UNIT_CONVERSIONS = {
    ("Creatinine", "mg/L"): 8.84,   # mg/L -> umol/L
    ("Hemoglobine", "g/dL"): 10.0,  # g/dL -> g/L
}

# Bornes de reference exprimees dans l'unite canonique.
#
# ATTENTION : les colonnes ValeurReferenceBas / ValeurReferenceHaut du fichier
# source ne sont PAS adaptees a l'unite de la ligne. Elles annoncent 60-110
# pour la creatinine et 12-16 pour l'hemoglobine quelle que soit l'unite,
# c'est-a-dire les plages umol/L et g/dL. Un indicateur "hors bornes" construit
# sur ces colonnes marquerait la totalite des creatinines en mg/L comme
# effondrees. On redefinit donc les bornes ici (cf. J1-12).
REFERENCE_RANGES = {
    "CRP": (0.0, 5.0),
    "Creatinine": (60.0, 110.0),
    "GlobulesBlancs": (4.0, 10.0),
    "Hemoglobine": (120.0, 160.0),
    "Natremie": (135.0, 145.0),
}

# Bornes de plausibilite physiologique, dans l'unite canonique. Au-dela, la
# valeur est consideree comme une erreur de mesure ou de saisie, pas comme un
# resultat extreme. Volontairement larges : on ecarte l'impossible, pas le rare.
BIOLOGY_PLAUSIBLE = {
    "CRP": (0.0, 500.0),
    "Creatinine": (20.0, 1500.0),
    "GlobulesBlancs": (0.1, 100.0),
    "Hemoglobine": (30.0, 220.0),
    "Natremie": (100.0, 180.0),
}

# --------------------------------------------------------------------------
# Constantes vitales
# --------------------------------------------------------------------------

# Bornes physiologiques. Mesure a l'audit : 1145 frequences cardiaques hors
# plage dont 382 exactement a zero (capteur debranche), et 340 saturations
# superieures a 100 % (impossible). Les quatre autres constantes sont propres.
VITALS_PLAUSIBLE = {
    "FrequenceCardiaque": (20.0, 220.0),
    "TensionSystolique": (50.0, 260.0),
    "TensionDiastolique": (30.0, 150.0),
    "Temperature": (32.0, 43.0),
    "FrequenceRespiratoire": (5.0, 60.0),
    "SpO2": (50.0, 100.0),
}

VITALS = list(VITALS_PLAUSIBLE)

# --------------------------------------------------------------------------
# Objets connectes
# --------------------------------------------------------------------------

# Seule la qualite "Bon" est exploitable ; 18,6 % des mesures sont degradees.
IOT_VALID_QUALITY = ["Bon"]

# Fenetre de reevaluation de l'etage B (cf. J1-05 / J1-06).
IOT_WINDOW_DAYS = 7

# Volumetrie minimale en dessous de laquelle l'etage B n'est pas entrainable.
# Retenu : 30 evenements positifs attendus, regle usuelle pour un modele de
# classification a faible dimension. Avec un taux de readmission de 18 %, cela
# suppose environ 170 patients couverts. Le jeu de donnees en fournit 6.
IOT_MIN_PATIENTS = 170

# --------------------------------------------------------------------------
# Comptes rendus
# --------------------------------------------------------------------------

# Un seul type de document est retenu : le CRH existe pour les 878 sejours,
# a raison d'un par sejour (cf. J1-10).
CR_TYPE = "CRH"
