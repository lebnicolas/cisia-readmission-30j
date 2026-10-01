"""Fixtures partagees par la suite de tests.

Le pipeline est lourd a construire : les fixtures sont de portee `session`
pour que le jeu de donnees ne soit assemble qu'une fois, quel que soit le
nombre de tests qui s'en servent.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

NOTEBOOK = Path(__file__).resolve().parent.parent / "notebook"
sys.path.insert(0, str(NOTEBOOK))

from src import assemble, clean, config as cfg, loading  # noqa: E402


@pytest.fixture(scope="session")
def tables():
    """Les onze tables brutes, telles que les CSV les livrent."""
    return loading.load_raw()


@pytest.fixture(scope="session")
def population(tables):
    """La population d'etude : sejours retenus apres exclusion des deces."""
    return clean.build_population(tables)


@pytest.fixture(scope="session")
def jeu(tables, population):
    """Le triplet (X, y, groupes) de l'etage A, tel que le notebook le construit."""
    return assemble.build_dataset(tables, stages=("A",),
                                  population=population, verbose=False)


@pytest.fixture(scope="session")
def config():
    return cfg
