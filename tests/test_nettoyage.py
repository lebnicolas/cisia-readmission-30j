"""Les corrections de la section 4 s'appliquent-elles reellement ?

Chaque test de ce fichier correspond a une correction documentee du dossier.
Le premier existe a cause de J3-02 : la normalisation des sentinelles avait
cesse de s'appliquer sous pandas 3, silencieusement, pendant deux semaines.
Une correction annoncee se verifie sur sa sortie, pas sur son code.
"""

from __future__ import annotations

import pandas as pd

from src import clean

SENTINELLES = ("Non renseigné", "Non renseigne", "NR", "Inconnu")


def test_sentinelles_textuelles_converties_en_manquants(tables):
    """J3-02 — le test qui aurait detecte la correction muette."""
    patients = clean.normalize_sentinels(tables["patients"].copy())
    restantes = {
        col: int(patients[col].isin(SENTINELLES).sum())
        for col in patients.columns
        if patients[col].dtype == object or str(patients[col].dtype) == "str"
    }
    en_trop = {c: n for c, n in restantes.items() if n}
    assert not en_trop, f"sentinelles encore presentes apres nettoyage : {en_trop}"


def test_la_normalisation_trouve_bien_des_manquants(tables):
    """Symetrique du precedent : une correction qui ne corrige rien est suspecte.

    Sans ce controle, un `normalize_sentinels` devenu inoperant passerait le
    test ci-dessus les yeux fermes — c'est exactement ce qui s'est produit.
    """
    avant = tables["patients"].isna().sum().sum()
    apres = clean.normalize_sentinels(tables["patients"].copy()).isna().sum().sum()
    assert apres > avant, (
        "la normalisation n'a converti aucune sentinelle : elle ne s'applique plus"
    )


def test_aucune_duree_de_sejour_negative_apres_reparation(tables):
    """Section 3.7 — la duree est recalculee sur les dates, jamais lue telle quelle."""
    sejours = clean.clean_sejours(tables["sejours"].copy())
    duree = (sejours.DateSortie - sejours.DateAdmission).dt.total_seconds() / 86400
    assert (duree >= 0).all(), f"{int((duree < 0).sum())} sejours de duree negative"


def test_population_sans_sejour_termine_par_un_deces(population):
    """Section 1.2 — la question ne se pose pas pour un patient qui vient de mourir."""
    assert "Deces" not in set(population.ModeSortie.astype(str)), (
        "des sejours termines par un deces subsistent dans la population d'etude"
    )


def test_population_conforme_au_volume_documente(population, config):
    """Le dossier annonce 786 sejours et 575 patients : un ecart signale une derive."""
    assert len(population) == 786
    assert population[config.GROUP].nunique() == 575


def test_integrite_referentielle_de_la_biologie(tables):
    """Section 3.4 — 73 lignes orphelines, ecartees au nettoyage."""
    valides = set(tables["sejours"].SejourID)
    biologies = clean.clean_biologies(tables["biologies"].copy(), valid_stays=valides)
    inconnus = set(biologies.SejourID) - valides
    assert not inconnus, f"{len(inconnus)} SejourID de biologie absents de sejours.csv"
    assert len(biologies) < len(tables["biologies"]), (
        "aucune ligne orpheline ecartee : le filtre ne s'applique plus"
    )


def test_une_seule_unite_par_panel_apres_harmonisation(tables):
    """Section 3.8 — deux panels arrivent en deux unites, un seul doit ressortir."""
    biologies = clean.clean_biologies(tables["biologies"].copy())
    multiples = {
        panel: sorted(set(grp.unite_harmonisee.dropna().astype(str)))
        for panel, grp in biologies.groupby("Panel")
        if grp.unite_harmonisee.dropna().nunique() > 1
    }
    assert not multiples, f"panels encore en plusieurs unites : {multiples}"


def test_les_deux_unites_d_un_panel_se_rejoignent_apres_conversion(tables):
    """Section 3.8 — le controle qui prouve que la conversion a vraiment eu lieu.

    Deux panels arrivent en deux unites. Apres harmonisation, les valeurs issues
    de l'une et de l'autre doivent decrire la meme grandeur : leurs medianes ne
    peuvent pas differer d'un facteur. Une conversion oubliee laisserait un
    ecart de 8,84 pour la creatinine, de 10 pour l'hemoglobine — c'est-a-dire
    exactement ce que ce test refuse.
    """
    biologies = clean.clean_biologies(tables["biologies"].copy())
    ecarts = {}
    for panel, grp in biologies.groupby("Panel"):
        medianes = grp.groupby("Unite", observed=True).valeur_harmonisee.median().dropna()
        if len(medianes) > 1:
            ecarts[panel] = float(medianes.max() / medianes.min())
    divergents = {p: round(r, 2) for p, r in ecarts.items() if r > 1.5}
    assert not divergents, (
        f"unites non reconciliees apres conversion (rapport des medianes) : {divergents}"
    )
    assert ecarts, "aucun panel multi-unites trouve : le jeu ou le nettoyage a change"
