r"""API de scoring — demonstration de l'integration SIH (section 12 du notebook).

Deux phases, conformes a l'architecture :
  - etage A (operationnel)  : score de risque a la sortie, explique et trace ;
  - etage B (specifie)      : reevaluation par telesurveillance — l'endpoint
    existe et repond honnetement qu'il attend la donnee.

Demarrage :
    ..\.venv\Scripts\python.exe -m uvicorn app:app --port 8077

Demo : http://127.0.0.1:8077
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

import simulation

API_DIR = Path(__file__).resolve().parent
ART = API_DIR / "artefacts"

app = FastAPI(title="Readmission 30 jours — demonstration",
              description="Etage A operationnel, etage B specifie.")

# Artefacts charges une fois au demarrage.
MODELE = joblib.load(ART / "modele.joblib")
META = json.loads((ART / "meta.json").read_text(encoding="utf-8"))
X = pd.read_parquet(ART / "donnees.parquet")
for c in META["colonnes_categorielle"]:
    X[c] = X[c].astype("category")
CONTEXTE = pd.read_parquet(ART / "contexte.parquet")
SEUIL = META["seuil"]
# Risque de phase 1 pour tous les sejours — calcule une fois au demarrage.
RISQUES = pd.Series(MODELE.predict_proba(X)[:, 1], index=X.index)
# Arbre de substitution — l'explication visuelle du score (cf. entrainer.py).
ARBRE = joblib.load(ART / "arbre.joblib")
ARBRE_META = json.loads((ART / "arbre.json").read_text(encoding="utf-8"))
# Contributions SHAP par sejour — l'explication individuelle (cf. expliquer.py).
# Optionnelles : sans elles, le score est explique par l'importance globale.
if (ART / "contributions.parquet").exists():
    CONTRIB = pd.read_parquet(ART / "contributions.parquet")
    CONTRIB_META = json.loads((ART / "contributions.json").read_text(encoding="utf-8"))
else:
    CONTRIB, CONTRIB_META = None, None


def _profil(sejour_id: str) -> dict:
    """Profil reel du patient, transmis a la simulation."""
    ligne, ctx = X.loc[[sejour_id]], CONTEXTE.loc[sejour_id]
    return {
        "age": float(ctx.age),
        "sexe": ctx.sexe,
        "nb_pathologies": float(ctx.nb_pathologies),
        "bpco": bool(int(ligne["demo_patho_BPCO"].iloc[0] or 0)),
        "insuffisance_cardiaque":
            bool(int(ligne["demo_patho_InsuffisanceCardiaque"].iloc[0] or 0)),
        "risque": float(RISQUES[sejour_id]),
    }


@app.get("/")
def accueil():
    return FileResponse(API_DIR / "index.html")


@app.get("/arbre")
def page_arbre():
    """Page dediee a l'arbre de decision — plein ecran, zoom et deplacement."""
    return FileResponse(API_DIR / "arbre.html")


@app.get("/api/info")
def info():
    """Version, seuil et performances — la tracabilite exigee en section 11."""
    return {**{k: META[k] for k in
               ["version", "seuil", "rapport_cout", "sejours", "readmissions",
                "performance_cv"]},
            "explication_individuelle": CONTRIB is not None}


@app.get("/api/sejours")
def sejours(n: int = 40):
    """Echantillon de sejours pour la demonstration, contexte compris."""
    ech = CONTEXTE.sample(n=min(n, len(CONTEXTE)), random_state=0)
    ech = ech.sort_values("age", ascending=False)
    return [{"sejour": i, **{k: (None if pd.isna(v) else v)
                             for k, v in r.items()}}
            for i, r in ech.drop(columns="reel").iterrows()]


@app.get("/api/score/{sejour_id}")
def score(sejour_id: str):
    """Etage A : score calibre + facteurs + version — jamais un score nu."""
    if sejour_id not in X.index:
        raise HTTPException(404, f"sejour inconnu : {sejour_id}")
    ligne = X.loc[[sejour_id]]
    proba = float(MODELE.predict_proba(ligne)[0, 1])

    def valeur(var: str):
        v = ligne[var].iloc[0]
        if pd.isna(v):
            return None
        return str(v) if not isinstance(v, (int, float)) else round(float(v), 2)

    # Facteurs globaux : les variables qui pesent dans le modele, pour tous.
    facteurs = [{"variable": var, "valeur": valeur(var), "poids_global": poids}
                for var, poids in META["importance_globale"].items()]

    # Explication individuelle : la contribution de chaque variable a CE score,
    # en points de probabilite, additive (base + somme = score). C'est elle qui
    # rend l'alerte contestable sur un fait precis.
    explication = None
    if CONTRIB is not None:
        c = CONTRIB.loc[sejour_id]
        ordre = c.abs().sort_values(ascending=False)
        tete = ordre.index[:8]
        explication = {
            "base": CONTRIB_META["base"],
            "methode": CONTRIB_META["methode"],
            "contributions": [{"variable": var, "valeur": valeur(var),
                               "points": round(float(c[var]) * 100, 1)}
                              for var in tete],
            "reste_points": round(float(c.drop(tete).sum()) * 100, 1),
            "nb_reste": int(len(c) - len(tete)),
        }

    ctx = CONTEXTE.loc[sejour_id]
    alerte = proba >= SEUIL
    return {
        "sejour": sejour_id,
        "risque": round(proba, 3),
        "seuil": SEUIL,
        "alerte": bool(alerte),
        "preconisation": ("plan de sortie renforcé : évaluation par le cadre "
                          "de santé avant la sortie" if alerte
                          else "sortie standard — pas de signal particulier"),
        "contexte": {"age": float(ctx.age), "sexe": ctx.sexe,
                     "service": ctx.service, "diagnostic": ctx.diagnostic,
                     "duree_sejour": float(ctx.duree)},
        "facteurs": facteurs[:6],
        "explication": explication,
        "version_modele": META["version"],
        "avertissements": (["patient de moins de 65 ans : le score reflète "
                            "surtout l'âge — l'évaluation clinique reste la "
                            "seule référence sur cette population"]
                           if ctx.age < 65 else []),
    }


@app.post("/api/simulation/{sejour_id}/demarrer")
def simulation_demarrer(sejour_id: str, scenario: str = "decompensation"):
    """Demarre la simulation du retour a domicile d'un patient.

    Genere le flux d'objets connectes de la phase 2 — le « jeu de donnees
    auto-genere » de la demonstration.
    """
    if sejour_id not in X.index:
        raise HTTPException(404, f"sejour inconnu : {sejour_id}")

    # Profil reel du patient : la trajectoire simulee en depend.
    profil = _profil(sejour_id)
    try:
        sim = simulation.demarrer(sejour_id, scenario, profil)
    except KeyError:
        raise HTTPException(422, f"scenario inconnu : {scenario} "
                                 f"(choix : {list(simulation.SCENARIOS)})")
    return {"sejour": sejour_id, "scenario": sim.scenario,
            "type": sim.type, "description": sim.description,
            "base": sim.base, "profil": profil,
            "regles_actives": [r["libelle"] for r in simulation.REGLES]}


@app.get("/api/simulation/{sejour_id}/tick")
def simulation_tick(sejour_id: str):
    """Une remontee de capteurs : mesure generee, regles evaluees, alerte levee.

    C'est la phase 2 en fonctionnement — regles cliniques explicites en
    attendant la volumetrie du modele appris.
    """
    sim = simulation.obtenir(sejour_id)
    if sim is None:
        raise HTTPException(409, "aucune simulation en cours pour ce sejour — "
                                 "appeler d'abord POST /demarrer")
    return sim.tick()


@app.get("/api/telesurveillance/{sejour_id}")
def telesurveillance(sejour_id: str):
    """Etage B : l'endpoint existe, et repond ce que la donnee permet.

    En production, il reevaluerait le risque a J+7 depuis les objets
    connectes et preconiserait une adaptation de la medication, une visite a
    domicile ou l'anticipation d'une rehospitalisation. Sur ce jeu de
    donnees, la couverture est insuffisante pour entrainer le modele — le
    systeme le dit au lieu de l'inventer.
    """
    if sejour_id not in X.index:
        raise HTTPException(404, f"sejour inconnu : {sejour_id}")
    b = META["etage_b"]
    return {
        "sejour": sejour_id,
        "statut": b["statut"],
        "decisions_visees": b["decisions_visees"],
        "couverture": {"patients_avec_mesures_j0_j7": b["patients_couverts"],
                       "patients_requis_pour_entrainer": b["patients_requis"],
                       "fenetre_jours": b["fenetre_jours"]},
        "message": ("Réévaluation à J+7 non disponible : 6 patients disposent "
                    "de télésurveillance post-sortie sur les 620 du jeu de "
                    "données, pour un minimum de 170. Le connecteur est "
                    "implémenté et s'activera quand la volumétrie sera "
                    "atteinte."),
    }


# --------------------------------------------------------------------------
# Le mur de supervision — navigable au curseur de position du jour
# --------------------------------------------------------------------------

@app.post("/api/cohorte/demarrer")
def cohorte_demarrer(n: int = 12):
    """Precalcule 30 jours de telesurveillance pour une cohorte melangee.

    Qui decompensera n est pas devoile : chaque patient tire son scenario
    avec une probabilite liee a son risque de phase 1.
    """
    n = max(6, min(n, 24))
    # Tirage aleatoire STRATIFIE : un tiers parmi les hauts risques, un tiers
    # au milieu, un tiers en bas — chaque appel renouvelle la cohorte. La
    # sequence de tirages est deterministe (compteur) : le premier tirage est
    # identique a chaque lancement, pratique pour repeter la demonstration.
    import itertools, random as _random
    global _TIRAGES
    try:
        _TIRAGES
    except NameError:
        _TIRAGES = itertools.count()
    rng = _random.Random(1000 + next(_TIRAGES))
    tri = RISQUES.sort_values(ascending=False)
    t = len(tri) // 3
    k = n // 3
    ids = (rng.sample(list(tri.index[:t]), k)
           + rng.sample(list(tri.index[t:2 * t]), k)
           + rng.sample(list(tri.index[2 * t:]), n - 2 * k))
    simulation.precalculer_cohorte([(i, _profil(i)) for i in ids])
    simulation.INFOS.update({
        i: f"{CONTEXTE.loc[i].age:.0f} ans · {CONTEXTE.loc[i].diagnostic}"
        for i in ids})
    patients = []
    for i in ids:
        ctx = CONTEXTE.loc[i]
        patients.append({
            "sejour": i, "age": float(ctx.age), "sexe": ctx.sexe,
            "service": ctx.service, "diagnostic": ctx.diagnostic,
            "risque": round(float(RISQUES[i]), 3),
            "alerte_phase1": bool(RISQUES[i] >= SEUIL),
        })
    return {"patients": patients, "seuil": SEUIL,
            "pas_max": simulation.JOURS_MAX * simulation.PAS_PAR_JOUR,
            "pas_par_jour": simulation.PAS_PAR_JOUR}


@app.get("/api/cohorte/etat")
def cohorte_etat(pas: int = 0):
    """L etat de la cohorte a la position demandee du curseur.

    Navigable librement, en avant comme en arriere : la trajectoire est
    precalculee, le journal renvoye couvre tout ce qui precede la position.
    """
    if not simulation.TRAJECTOIRES:
        raise HTTPException(409, "aucune cohorte precalculee — POST /demarrer")
    etats, journal, jour = simulation.etat_a(pas)
    return {"etats": etats, "journal": journal, "jour": jour, "pas": pas}


@app.get("/api/cohorte/patient/{sejour_id}")
def cohorte_patient(sejour_id: str, pas: int = 0):
    """La fiche patient a la position du curseur — identite, etat, tableau
    des releves, evenements d alerte."""
    if sejour_id not in X.index:
        raise HTTPException(404, f"sejour inconnu : {sejour_id}")
    fiche = simulation.fiche_patient(sejour_id, pas)
    ctx = CONTEXTE.loc[sejour_id]
    identite = {"age": float(ctx.age), "sexe": ctx.sexe,
                "service": ctx.service, "diagnostic": ctx.diagnostic,
                "duree_sejour": float(ctx.duree),
                "risque": round(float(RISQUES[sejour_id]), 3),
                "alerte_phase1": bool(RISQUES[sejour_id] >= SEUIL)}
    if fiche is None:
        return {"sejour": sejour_id, "dans_cohorte": False,
                "identite": identite}
    return {"sejour": sejour_id, "dans_cohorte": True,
            "identite": identite, **fiche}


@app.get("/api/arbre/{sejour_id}")
def arbre_patient(sejour_id: str):
    """L arbre de decision de substitution, avec le chemin de CE patient.

    L arbre unique n existe pas dans le modele reel (300 arbres agreges) :
    celui-ci est entraine a l imiter, et sa fidelite est renvoyee avec lui —
    jamais l arbre sans son degre d approximation.
    """
    if sejour_id not in X.index:
        raise HTTPException(404, f"sejour inconnu : {sejour_id}")
    ligne = X.loc[[sejour_id], ARBRE_META["colonnes"]].astype(float)
    chemin = [int(i) for i in ARBRE.decision_path(ligne).indices]

    # Les valeurs du patient sur les variables rencontrees en chemin.
    valeurs = {}
    for i in chemin:
        f = ARBRE_META["noeuds"][i]["feature"]
        if f is not None:
            v = ligne[f].iloc[0]
            valeurs[f] = None if pd.isna(v) else round(float(v), 2)

    return {"sejour": sejour_id,
            "risque_modele": round(float(RISQUES[sejour_id]), 3),
            "risque_arbre": round(float(ARBRE.predict(ligne)[0]), 3),
            "seuil": SEUIL,
            "chemin": chemin, "valeurs": valeurs,
            "noeuds": ARBRE_META["noeuds"],
            "profondeur": ARBRE_META["profondeur"],
            "fidelite": ARBRE_META["fidelite"],
            "note": ARBRE_META["note"]}
