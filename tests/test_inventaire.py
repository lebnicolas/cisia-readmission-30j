"""L'inventaire des colonnes reste aligné sur les tables livrées.

L'inventaire de la section 11.1 est déclaré à la main dans `src/inventaire.py`,
parce que la raison d'un devenir ne se déduit pas du code. Ce test l'empêche de
se détacher des données : une colonne source ajoutée, renommée ou retirée le
fait échouer (cf. J3-12).
"""

from __future__ import annotations

from src import inventaire


def test_chaque_colonne_source_est_declaree(tables):
    manquantes, _ = inventaire.ecarts(tables)
    assert not manquantes, f"colonnes absentes de l'inventaire : {manquantes}"


def test_aucune_colonne_declaree_n_est_imaginaire(tables):
    _, en_trop = inventaire.ecarts(tables)
    assert not en_trop, f"colonnes déclarées mais absentes des tables : {en_trop}"


def test_chaque_devenir_est_connu_et_chaque_exclusion_justifiee():
    for table, colonnes in inventaire.INVENTAIRE.items():
        for colonne, (devenir, raison) in colonnes.items():
            assert devenir in inventaire.DEVENIRS, f"{table}.{colonne} : devenir inconnu « {devenir} »"
            if devenir == inventaire.ECARTEE:
                assert raison, f"{table}.{colonne} est écartée sans raison"


def test_aucun_identifiant_direct_n_entre_dans_le_modele():
    """Minimisation (§11.1) : le nom et le contact ne sont jamais des variables."""
    patients = inventaire.INVENTAIRE["patients"]
    assert patients["NomPrenom"][0] == inventaire.ECARTEE
    for colonne in ("MedecinTraitant", "PersonneAPrevenir"):
        assert "présence" in patients[colonne][1]
