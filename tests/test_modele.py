"""Evaluation automatisee du modele.

C'est le test qui transforme une performance annoncee en performance
verifiee. Il rejoue le protocole du dossier — validation croisee groupee par
patient, predictions hors echantillon — et refuse que les chiffres sortent
des bornes documentees, dans un sens comme dans l'autre.

Les bornes sont volontairement larges : la section 13.2 bis etablit que la
performance varie de 0,39 a 0,44 selon la partition. Un test plus serre
echouerait sur du bruit, un test plus lache ne detecterait rien.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import average_precision_score, brier_score_loss

from src import config as cfg, modeling as mdl

PR_AUC_MIN, PR_AUC_MAX = 0.35, 0.50
RAPPEL_MIN = 0.70


@pytest.fixture(scope="module")
def predictions(jeu):
    """Probabilites hors echantillon du modele retenu, protocole du dossier."""
    X, y, groupes = jeu
    modele = {"Gradient boosting": mdl.make_models(X)["Gradient boosting"]}
    preds = mdl.out_of_fold_predictions(modele, X, y, groupes)
    return preds["Gradient boosting"], y


def test_la_performance_reste_dans_la_fourchette_documentee(predictions):
    """Le dossier annonce 0,39 a 0,44 selon la partition."""
    p, y = predictions
    pr_auc = average_precision_score(y, p)
    assert PR_AUC_MIN <= pr_auc <= PR_AUC_MAX, (
        f"PR-AUC = {pr_auc:.3f}, hors des bornes [{PR_AUC_MIN}, {PR_AUC_MAX}]. "
        "Une valeur basse signale une regression ; une valeur haute, une fuite."
    )


def test_le_modele_fait_mieux_que_le_taux_de_base(predictions):
    """Section 7 — deux fois mieux que le hasard, sur toutes les partitions testees."""
    p, y = predictions
    pr_auc = average_precision_score(y, p)
    taux_de_base = y.mean()
    assert pr_auc > 1.8 * taux_de_base, (
        f"PR-AUC {pr_auc:.3f} contre un taux de base de {taux_de_base:.3f} : "
        "le gain sur le hasard n'est plus au rendez-vous"
    )


def test_les_probabilites_restent_calibrees(predictions):
    """Section 7.1 — un score de 0,30 doit valoir trois readmissions sur dix.

    Sans quoi le seuil de decision ne veut plus rien dire cliniquement.
    """
    p, y = predictions
    assert abs(p.mean() - y.mean()) < 0.05, (
        f"risque moyen predit {p.mean():.3f} contre {y.mean():.3f} observe : "
        "la calibration a derive"
    )
    reference = np.full(len(y), y.mean())
    assert brier_score_loss(y, p) <= brier_score_loss(y, reference) + 0.005, (
        "le score de Brier est moins bon que celui de la reference constante"
    )


def test_le_seuil_retenu_tient_sa_promesse_de_depistage(predictions):
    """Section 8 — au seuil 0,11, le dossier annonce huit readmissions sur dix."""
    p, y = predictions
    signales = p >= cfg.DECISION_THRESHOLD
    rappel = (signales & (y == 1)).sum() / y.sum()
    assert rappel >= RAPPEL_MIN, (
        f"rappel {rappel:.3f} au seuil {cfg.DECISION_THRESHOLD} : "
        f"le reglage de depistage ne tient plus (minimum attendu {RAPPEL_MIN})"
    )


def test_le_seuil_ne_signale_pas_toute_la_population(predictions):
    """Un rappel parfait obtenu en signalant tout le monde n'est pas un resultat.

    Le pendant du test precedent : la sensibilite ne doit pas etre achetee au
    prix d'une charge que le service ne peut pas absorber (section 8.2).
    """
    p, _ = predictions
    part_signalee = (p >= cfg.DECISION_THRESHOLD).mean()
    assert part_signalee < 0.60, (
        f"{part_signalee:.1%} des sorties signalees : aucun service ne peut suivre"
    )
