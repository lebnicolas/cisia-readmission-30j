"""Reporte JOURNAL.md dans la section 14 du notebook.

Le journal est tenu dans un fichier markdown, plus commode a editer au fil de
l'eau, et doit figurer dans le notebook puisque c'est lui le livrable. Cet
utilitaire evite que les deux divergent.

Il ne touche qu'aux cellules markdown situees apres le titre de la section 14 :
aucune sortie de calcul n'est perdue, et le notebook n'a pas besoin d'etre
reexecute.

    python sync_journal.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import nbformat as nbf

NOTEBOOK = Path(__file__).parent / "soutenance_cisia.ipynb"
JOURNAL = Path(__file__).parent / "JOURNAL.md"
ANCRE = "# 14. Journal de bord"


def journal_cells() -> list:
    """Decoupe le journal en cellules, une par entree.

    Le corps commence a la premiere seance ; le decoupage se fait sur les
    entrees quel que soit leur numero de seance (J1-, J2-, ...), et les titres
    de seance restent attaches a l'entree qui les suit.
    """
    texte = JOURNAL.read_text(encoding="utf-8")
    depart = re.search(r"^## \d{2}/\d{2}/\d{4}", texte, flags=re.M)
    if not depart:
        raise SystemExit("Aucun titre de seance trouve dans JOURNAL.md")
    corps = texte[depart.start():]
    return [nbf.v4.new_markdown_cell(p)
            for p in re.split(r"\n(?=### J\d+-)", corps) if p.strip()]


def main() -> None:
    nb = nbf.read(str(NOTEBOOK), as_version=4)
    ancres = [i for i, c in enumerate(nb.cells)
              if c.cell_type == "markdown" and ANCRE in c.source]
    if not ancres:
        raise SystemExit(f"Ancre « {ANCRE} » introuvable dans le notebook")
    debut = ancres[0] + 1

    # Tout ce qui suit l'ancre doit etre du markdown : on refuse d'ecraser du
    # code, au cas ou la structure du notebook aurait change.
    reste = nb.cells[debut:]
    if any(c.cell_type != "markdown" for c in reste):
        raise SystemExit("Des cellules de code suivent l'ancre — synchronisation refusee")

    nouvelles = journal_cells()
    nb.cells = nb.cells[:debut] + nouvelles
    nbf.write(nb, str(NOTEBOOK))
    print(f"Journal synchronise : {len(reste)} cellules remplacees par {len(nouvelles)}")


if __name__ == "__main__":
    main()
