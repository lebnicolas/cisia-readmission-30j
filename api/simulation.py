"""Simulation de telesurveillance — le patient rentre chez lui.

Genere au fil du temps les mesures des objets connectes d'un patient sorti,
et applique la couche d'alerte de la phase 2 : des regles cliniques
explicites, en attendant la volumetrie necessaire a l'entrainement du modele
(6 patients couverts sur les 170 requis — cf. notebook, section 5.5).

La trajectoire est PROPRE A CHAQUE PATIENT :
  - la graine derive de l'identifiant du sejour — deux patients ne jouent
    jamais la meme sequence ;
  - les valeurs de base viennent du profil reel — age, sexe, pathologies
    (un BPCO part avec une SpO2 plus basse, un insuffisant cardiaque avec
    un poids plus instable) ;
  - le TYPE de decompensation depend des pathologies (respiratoire,
    cardiaque, mixte), et son debut comme sa vitesse sont modules par le
    score de risque de la phase 1 : un patient a haut risque decompense
    plus tot et plus vite.

Chaque alerte porte sa regle et sa preconisation — adaptation de la
medication, teleconsultation, ou anticipation d'une rehospitalisation.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field

# Un pas de simulation = une remontee de capteurs (toutes les ~6 h en
# conditions reelles, toutes les quelques secondes en demonstration).
PAS_PAR_JOUR = 4

# Regles cliniques de la couche d'alerte. Chaque regle est explicite,
# contestable, et porte sa preconisation — jamais une alerte nue.
REGLES = [
    {
        "id": "prise-de-poids",
        "condition": lambda m, base: m["poids"] - base["poids"] >= 2.0,
        "gravite": "serieux",
        "libelle": "Prise de poids ≥ 2 kg depuis la sortie",
        "clinique": "signe classique de rétention hydrique — décompensation "
                    "d'une insuffisance cardiaque",
        "preconisation": "adaptation du traitement diurétique à évaluer sous "
                         "24 h (médecin traitant ou cardiologue)",
    },
    {
        "id": "desaturation",
        "condition": lambda m, base: m["spo2"] < 92.0,
        "gravite": "serieux",
        "libelle": "SpO2 sous 92 %",
        "clinique": "hypoxémie — dégradation respiratoire",
        "preconisation": "téléconsultation dans la journée ; oxymétrie de "
                         "contrôle rapprochée",
    },
    {
        "id": "tachycardie",
        "condition": lambda m, base: m["fc"] >= 105.0,
        "gravite": "modere",
        "libelle": "Fréquence cardiaque de repos ≥ 105 bpm",
        "clinique": "tachycardie de repos persistante",
        "preconisation": "contrôle clinique ; vérifier l'observance du "
                         "traitement bêtabloquant",
    },
    {
        "id": "sedentarite",
        "condition": lambda m, base: m["pas"] < base["pas"] * 0.35,
        "gravite": "modere",
        "libelle": "Effondrement de l'activité quotidienne",
        "clinique": "sédentarisation brutale — asthénie ou aggravation",
        "preconisation": "appel infirmier de suivi pour évaluation",
    },
]

ESCALADE = ("Anticiper une réhospitalisation — contact du service d'origine "
            "pour une admission programmée, plutôt qu'un retour par les "
            "urgences.")

# Entre la decision d'escalade et l'admission effective, le patient reste a
# domicile SOUS SURVEILLANCE — c'est la fenetre ou elle compte le plus. Un
# jour simule, puis l'admission cloture l'episode.
ATTENTE_ADMISSION_PAS = PAS_PAR_JOUR

# Vigilance personnalisee par l'IA : chez un patient dont le score appris de
# sortie atteint ce niveau (calibre — trois patients sur dix comme lui
# reviennent), UN SEUL signe serieux suffit a anticiper la rehospitalisation.
# C'est ainsi que le modele de phase 1 entre dans la decision de phase 2 —
# sans pretendre apprendre de donnees post-sortie qui n'existent pas encore.
RISQUE_VIGILANCE_RENFORCEE = 0.30


def seuil_escalade(risque: float) -> int:
    """Nombre de signes serieux requis pour escalader, selon le score IA."""
    return 1 if risque >= RISQUE_VIGILANCE_RENFORCEE else 2

# Derives par pas selon le type de decompensation : (fc, spo2, poids, pas).
# Le type est choisi d'apres les pathologies du patient.
TYPES_DECOMPENSATION = {
    "cardiaque": {
        "libelle": "décompensation cardiaque — prise de poids dominante, "
                   "tachycardie progressive",
        "derive": (+1.8, -0.16, +0.21, -80.0),
    },
    "respiratoire": {
        "libelle": "exacerbation respiratoire — désaturation dominante",
        "derive": (+1.3, -0.55, +0.05, -110.0),
    },
    "mixte": {
        "libelle": "dégradation mixte — désaturation et prise de poids",
        "derive": (+1.5, -0.34, +0.14, -90.0),
    },
}


def _graine(sejour: str) -> int:
    """Graine deterministe propre au sejour — rejouable, jamais identique
    d'un patient a l'autre."""
    return int(hashlib.sha256(sejour.encode()).hexdigest()[:8], 16)


def _type_decompensation(profil: dict, rng: random.Random) -> str:
    bpco = bool(profil.get("bpco"))
    icc = bool(profil.get("insuffisance_cardiaque"))
    if bpco and not icc:
        return "respiratoire"
    if icc and not bpco:
        return "cardiaque"
    if bpco and icc:
        return rng.choice(["respiratoire", "cardiaque", "mixte"])
    return "mixte"


def _base_physiologique(profil: dict, rng: random.Random) -> dict:
    """Valeurs de repos derivees du profil reel du patient."""
    age = float(profil.get("age", 75))
    bpco = bool(profil.get("bpco"))
    icc = bool(profil.get("insuffisance_cardiaque"))
    femme = profil.get("sexe") == "F"
    nb_patho = float(profil.get("nb_pathologies", 1) or 0)

    return {
        "fc": round(66 + 0.10 * age + (4 if icc else 0) + rng.gauss(0, 3), 1),
        "spo2": round(min(98.5, 97.8 - 0.02 * age - (2.6 if bpco else 0)
                          + rng.gauss(0, 0.4)), 1),
        "poids": round((62 if femme else 76) + rng.gauss(0, 6), 1),
        "pas": max(700, round(6200 - 42 * age - 280 * nb_patho
                              + rng.gauss(0, 400))),
    }


@dataclass
class Simulation:
    sejour: str
    scenario: str                    # "stable" ou "decompensation"
    profil: dict = field(default_factory=dict)
    t: int = 0
    historique: list = field(default_factory=list)
    escalade_t: int | None = None    # pas de la decision d'escalade
    admis: bool = False              # l'admission cloture l'episode
    deja: set = field(default_factory=set)
    dernier: dict | None = None

    def __post_init__(self):
        self.rng = random.Random(_graine(self.sejour))
        self.base = _base_physiologique(self.profil, self.rng)
        risque = float(self.profil.get("risque", 0.2))

        if self.scenario == "decompensation":
            self.type = _type_decompensation(self.profil, self.rng)
            # Plus le risque de phase 1 est haut, plus la degradation
            # commence tot et va vite.
            self.debut_derive = (2.0 + self.rng.uniform(0.0, 4.5)
                                 * (1.15 - min(risque, 0.9)))
            self.intensite = 0.75 + 0.75 * min(risque, 0.9) \
                + self.rng.uniform(-0.1, 0.15)
            self.derive = TYPES_DECOMPENSATION[self.type]["derive"]
            self.description = (
                f"{TYPES_DECOMPENSATION[self.type]['libelle']} — début vers "
                f"J+{self.debut_derive:.0f}, vitesse modulée par le risque "
                f"de phase 1 ({risque:.0%})")
        else:
            self.type = "aucune"
            self.debut_derive = float("inf")
            self.intensite = 0.0
            self.derive = (0.0, 0.0, 0.0, +30.0)
            self.description = ("convalescence normale — bruit de mesure et "
                                "reprise progressive de l'activité")

    def tick(self) -> dict:
        """Avance d'un pas : genere une remontee, applique les regles.

        L'escalade n'interrompt pas la surveillance : elle ouvre une fenetre
        d'attente d'admission d'un jour, pendant laquelle les capteurs
        continuent d'emettre. L'admission, elle, cloture l'episode.
        """
        if self.admis and self.dernier:
            d = dict(self.dernier); d["nouvelles"] = []
            return d
        self.t += 1
        jour = self.t / PAS_PAR_JOUR
        actif = max(0.0, (jour - self.debut_derive) * PAS_PAR_JOUR) \
            * self.intensite
        d_fc, d_spo2, d_poids, d_pas = self.derive
        # En convalescence, l'activite reprend meme sans derive pathologique.
        reprise = min(self.t, 20) * 30 if self.scenario == "stable" else 0

        m = {
            "fc": round(self.base["fc"] + d_fc * actif
                        + self.rng.gauss(0, 2.2), 1),
            "spo2": round(min(99.0, self.base["spo2"] + d_spo2 * actif
                              + self.rng.gauss(0, 0.5)), 1),
            "poids": round(self.base["poids"] + d_poids * actif
                           + self.rng.gauss(0, 0.15), 2),
            "pas": max(0, round(self.base["pas"] + d_pas * actif + reprise
                                + self.rng.gauss(0, 260))),
            "jour": round(jour, 2),
        }
        self.historique.append(m)

        declenchees = [
            {k: r[k] for k in
             ("id", "gravite", "libelle", "clinique", "preconisation")}
            for r in REGLES if r["condition"](m, self.base)
        ]
        n_serieux = sum(a["gravite"] == "serieux" for a in declenchees)
        requis = seuil_escalade(float(self.profil.get("risque", 0.2)))
        declenche = n_serieux >= requis
        nouvelles = [a for a in declenchees if a["id"] not in self.deja]
        self.deja |= {a["id"] for a in declenchees}

        if declenche and self.escalade_t is None:
            self.escalade_t = self.t
            motif = (f"{n_serieux} signe(s) sérieux — seuil personnalisé à "
                     f"{requis} par le score de sortie"
                     + (" (vigilance renforcée : patient à haut risque)"
                        if requis == 1 else ""))
            nouvelles.append({"id": "escalade", "gravite": "critique",
                              "libelle": "ESCALADE — réhospitalisation à anticiper",
                              "clinique": motif,
                              "preconisation": ESCALADE})

        en_escalade = self.escalade_t is not None
        grave = [a["gravite"] for a in declenchees]
        statut = ("escalade" if en_escalade else
                  "serieux" if "serieux" in grave else
                  "modere" if grave else "calme")

        # Fin de la fenetre d'attente : le patient est admis, l'episode se clot.
        if en_escalade and self.t >= self.escalade_t + ATTENTE_ADMISSION_PAS:
            self.admis = True
            statut = "admis"
            nouvelles.append({
                "id": "admission", "gravite": "admission",
                "libelle": "Admission réalisée — fin de la télésurveillance",
                "clinique": "le patient est pris en charge par le service",
                "preconisation": "relais hospitalier — l'épisode de "
                                 "télésurveillance est clos",
            })

        self.dernier = {
            "sejour": self.sejour,
            "scenario": self.scenario,
            "type": self.type,
            "description": self.description,
            "jour": m["jour"],
            "mesure": m,
            "statut": statut,
            "historique": self.historique[-24:],
            "alertes": declenchees,
            "nouvelles": nouvelles,
            "escalade": ESCALADE if en_escalade and not self.admis else None,
            "base": self.base,
        }
        return self.dernier


# Registre des simulations en cours, par sejour.
EN_COURS: dict[str, Simulation] = {}

SCENARIOS = ("stable", "decompensation")


def demarrer(sejour: str, scenario: str, profil: dict | None = None) -> Simulation:
    if scenario not in SCENARIOS:
        raise KeyError(scenario)
    sim = Simulation(sejour, scenario, profil or {})
    EN_COURS[sejour] = sim
    return sim


def obtenir(sejour: str) -> Simulation | None:
    return EN_COURS.get(sejour)


# --------------------------------------------------------------------------
# Cohorte — le mur de supervision, navigable par curseur de jour
# --------------------------------------------------------------------------

# La trajectoire complete de la cohorte est PRECALCULEE au demarrage
# (deterministe : les graines derivent des identifiants). Le curseur de
# position du jour navigue ensuite librement dedans, en avant comme en
# arriere, sans attendre aucun flux.

JOURS_MAX = 30
TRAJECTOIRES: dict[str, list] = {}     # sejour -> un etat par pas (1..120)
EVENEMENTS: list[dict] = []            # alertes + marqueurs de jour, tries par pas
INFOS: dict[str, str] = {}             # sejour -> "82 ans · Insuffisance cardiaque"
BASES: dict[str, dict] = {}            # sejour -> valeurs de repos simulees


def precalculer_cohorte(patients: list[tuple[str, dict]]) -> int:
    """Deroule 30 jours de telesurveillance pour toute la cohorte.

    Le scenario de chaque patient est tire au sort — cache a l ecran — avec
    une probabilite de decompensation liee a son risque de phase 1. Le tirage
    est ancre sur l identifiant du sejour : la demonstration est rejouable.
    """
    TRAJECTOIRES.clear()
    EVENEMENTS.clear()
    BASES.clear()
    for sejour, profil in patients:
        rng = random.Random(_graine(sejour) ^ 0xC0F0)
        risque = float(profil.get("risque", 0.2))
        p_decomp = min(0.90, 0.12 + 1.1 * risque)
        scenario = "decompensation" if rng.random() < p_decomp else "stable"
        sim = Simulation(sejour, scenario, profil)
        EN_COURS[sejour] = sim          # la fiche lit le profil (seuil IA) ici

        etats = []
        for pas in range(1, JOURS_MAX * PAS_PAR_JOUR + 1):
            d = sim.tick()
            etats.append({"jour": d["jour"], "statut": d["statut"],
                          "mesure": d["mesure"],
                          "alertes": [a["libelle"] for a in d["alertes"]]})
            for a in d["nouvelles"]:
                EVENEMENTS.append({"pas": pas, "jour": d["jour"],
                                   "sejour": sejour, **a})
        TRAJECTOIRES[sejour] = etats
        BASES[sejour] = sim.base

    # Marqueurs de position du jour, inseres dans le fil.
    for j in range(1, JOURS_MAX + 1):
        EVENEMENTS.append({"pas": j * PAS_PAR_JOUR, "jour": j,
                           "gravite": "jour", "sejour": "",
                           "libelle": f"Jour {j} à domicile",
                           "preconisation": ""})
    EVENEMENTS.sort(key=lambda e: (e["pas"], 0 if e["gravite"] == "jour" else 1))
    return len(TRAJECTOIRES)


def etat_a(pas: int) -> tuple[list[dict], list[dict], float]:
    """Etat de la cohorte a une position donnee du curseur (0..120 pas)."""
    pas = max(0, min(pas, JOURS_MAX * PAS_PAR_JOUR))
    etats = []
    for sejour, traj in TRAJECTOIRES.items():
        if pas == 0:
            etats.append({"sejour": sejour, "jour": 0, "statut": "calme",
                          "mesure": None, "alertes": []})
        else:
            etats.append({"sejour": sejour, **traj[pas - 1]})
    journal = [{**e, "qui": INFOS.get(e["sejour"], "")}
               for e in EVENEMENTS if e["pas"] <= pas]
    return etats, journal, pas / PAS_PAR_JOUR


def _releves_jusqu_a(traj: list, pas: int) -> list:
    """La fenetre d'attente d'admission fait partie de la main courante
    (le patient reste surveille) ; l'admission, elle, clot les remontees."""
    releves = []
    for e in traj[:pas]:
        if e["statut"] == "admis":
            break
        releves.append(e["mesure"] | {"jour": e["jour"], "statut": e["statut"],
                                      "alertes": e["alertes"]})
    return releves


def fiche_patient(sejour: str, pas: int) -> dict | None:
    """La fiche de telesurveillance d un patient, a la position du curseur.

    Rend l etat courant, l historique complet des remontees jusqu a la
    position (le tableau des metriques), et les evenements d alerte du
    patient — chacun avec sa preconisation.
    """
    traj = TRAJECTOIRES.get(sejour)
    if traj is None:
        return None
    pas = max(0, min(pas, len(traj)))
    courant = traj[pas - 1] if pas else {"jour": 0, "statut": "calme",
                                         "mesure": None, "alertes": []}
    sim = EN_COURS.get(sejour)
    risque = float(sim.profil.get("risque", 0.2)) if sim else 0.2
    return {
        "base": BASES.get(sejour, {}),
        "seuil_escalade": seuil_escalade(risque),
        "vigilance_renforcee": seuil_escalade(risque) == 1,
        "jour": pas / PAS_PAR_JOUR,
        "statut": courant["statut"],
        "alertes_actives": courant["alertes"],
        "releves": _releves_jusqu_a(traj, pas),
        "evenements": [e for e in EVENEMENTS
                       if e["sejour"] == sejour and e["pas"] <= pas],
    }
