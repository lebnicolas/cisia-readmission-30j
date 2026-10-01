"""Les proprietes structurelles du jeu d'apprentissage.

L'etancheite temporelle entre les deux etages est la plus importante : elle
est annoncee comme une propriete du code (section 1.5), pas comme une
vigilance a exercer. Une propriete qu'aucun test ne verifie redevient une
vigilance.
"""

from __future__ import annotations

import pandas as pd

from src import assemble, blocks


def test_aucune_variable_de_l_etage_b_dans_le_jeu_de_sortie(jeu):
    """Section 1.1 — les objets connectes mesurent APRES la sortie.

    Les faire entrer dans le modele de sortie serait une fuite temporelle :
    le modele lirait le futur du patient qu'il est cense predire.
    """
    X, _, _ = jeu
    prefixes_b = {
        b.name for b in blocks.BLOCKS if b.stage == "B"
    }
    intruses = [
        c for c in X.columns
        if any(c.startswith(p[:3] + "_") for p in prefixes_b) or c.startswith("iot_")
    ]
    assert not intruses, f"variables d'etage B dans le jeu d'etage A : {intruses}"


def test_chaque_bloc_declare_son_etage(tables, population):
    """Le filtre d'assemblage repose sur cette declaration : elle doit exister."""
    sans_etage = [b.name for b in blocks.BLOCKS if b.stage not in ("A", "B")]
    assert not sans_etage, f"blocs sans etage valide : {sans_etage}"


def test_le_grain_est_le_sejour(jeu, population):
    """Une ligne par sejour, ni plus ni moins — la decision se prend a la sortie."""
    X, y, groupes = jeu
    assert len(X) == len(population)
    assert X.index.is_unique, "index duplique : le grain n'est plus le sejour"
    assert (X.index == population.index).all()
    assert len(y) == len(X) and len(groupes) == len(X)


def test_aucune_variable_anormalement_correlee_a_la_cible(jeu):
    """Section 6 — un coefficient proche de 1 signalerait une fuite de cible.

    Le seuil de 0,60 est large : la correlation la plus forte du jeu vaut
    moins de 0,40. Il ne se declenchera que sur une vraie anomalie.
    """
    X, y, _ = jeu
    num = X.select_dtypes("number").astype(float)
    corr = num.corrwith(y.astype(float)).abs().dropna()
    suspectes = corr[corr > 0.60]
    assert suspectes.empty, (
        f"correlation anormale a la cible — fuite probable : "
        f"{suspectes.round(3).to_dict()}"
    )


def test_le_jeu_a_la_forme_documentee(jeu):
    """Le dossier annonce 786 sejours et 132 variables."""
    X, y, groupes = jeu
    assert X.shape == (786, 132), f"forme inattendue : {X.shape}"
    assert int(y.sum()) == 141, f"{int(y.sum())} readmissions au lieu de 141"
    assert groupes.nunique() == 575


def test_ajouter_une_source_ne_demande_pas_de_refonte(tables, population):
    """Section 12.3 — l'extensibilite est verifiee, pas affirmee.

    Un bloc supplementaire declare en etage A doit entrer dans le jeu sans
    qu'aucun autre code ne soit touche ; declare en etage B, il doit en etre
    exclu par le seul filtre d'etage.
    """
    sonde_a = blocks.Block(
        "sonde", "A", ["sejours"],
        "bloc de test",
        lambda tables, population: pd.DataFrame(
            {"sonde_valeur": 1}, index=population.index),
    )
    sonde_b = blocks.Block(
        "sonde", "B", ["sejours"],
        "bloc de test",
        lambda tables, population: pd.DataFrame(
            {"sonde_valeur": 1}, index=population.index),
    )
    avec_a, _, _ = assemble.build_dataset(
        tables, stages=("A",), population=population,
        extra_blocks=[sonde_a], verbose=False)
    avec_b, _, _ = assemble.build_dataset(
        tables, stages=("A",), population=population,
        extra_blocks=[sonde_b], verbose=False)
    assert any("sonde" in c for c in avec_a.columns), (
        "un bloc d'etage A declare n'entre pas dans le jeu"
    )
    assert not any("sonde" in c for c in avec_b.columns), (
        "un bloc d'etage B franchit le filtre : l'etancheite n'est pas garantie"
    )
