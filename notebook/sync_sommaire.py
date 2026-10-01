"""Écrit le sommaire du notebook à partir de ses titres.

Le sommaire est une cellule markdown placée juste après l'introduction. Il liste
les sections de la production (titres « ## ») et leurs sous-sections (« ### »),
puis les séances du journal de bord. Il est régénéré plutôt qu'écrit à la main :
un titre renommé ou ajouté ne laisse pas de lien mort (même logique que
sync_journal.py).

Les liens suivent la convention de l'export HTML de nbconvert : l'ancre d'un titre
est son texte, espaces remplacées par des tirets, caractères non ASCII encodés en
pourcent, apostrophe laissée telle quelle. Un lien markdown encoderait aussi
l'apostrophe et manquerait sa cible : les entrées sont donc des liens HTML.
JupyterLab décode l'ancre avant de chercher le titre, il les suit aussi.

Il ne touche qu'à la cellule du sommaire : aucune sortie de calcul n'est perdue,
et le notebook n'a pas besoin d'être réexécuté.

    python sync_sommaire.py
"""

from __future__ import annotations

import html
import re
import sys
from pathlib import Path
from urllib.parse import quote

import nbformat as nbf

NOTEBOOK = Path(__file__).parent / "soutenance_cisia.ipynb"
MARQUE = "<!-- sommaire : généré par sync_sommaire.py, ne pas éditer à la main -->"
JOURNAL = "14. Journal de bord"


def texte_visible(titre: str) -> str:
    """Le texte d'un titre tel que le lecteur le voit : sans gras, italique ni code."""
    return re.sub(r"\*\*|__|`|(?<!\w)[*_](?=\S)|(?<=\S)[*_](?!\w)", "", titre).strip()


def lien(titre: str) -> str:
    visible = texte_visible(titre)
    cible = quote(visible.replace(" ", "-"), safe="/:',;()!*$&+=@?~")
    return f'<a href="#{html.escape(cible)}">{html.escape(visible, quote=False)}</a>'


def sommaire(cellules) -> str:
    lignes = [MARQUE, "", "**Sommaire**", ""]
    dans_journal = False
    for c in cellules:
        if c.cell_type != "markdown" or c.source.startswith(MARQUE):
            continue
        for niveau, titre in re.findall(r"^(#{1,3}) (.+?)\s*$", c.source, re.M):
            if len(niveau) == 1 and texte_visible(titre) == JOURNAL:
                dans_journal = True
                lignes.append(f"- {lien(titre)}")
            elif len(niveau) == 2:
                lignes.append(f"{'  ' if dans_journal else ''}- {lien(titre)}")
            elif len(niveau) == 3 and not dans_journal:
                lignes.append(f"  - {lien(titre)}")
    return "\n".join(lignes)


def main() -> None:
    nb = nbf.read(NOTEBOOK, as_version=4)
    existante = [i for i, c in enumerate(nb.cells) if c.cell_type == "markdown" and c.source.startswith(MARQUE)]
    texte = sommaire(nb.cells)
    if existante:
        nb.cells[existante[0]].source = texte
    else:
        nb.cells.insert(1, nbf.v4.new_markdown_cell(texte))
    nbf.write(nb, NOTEBOOK)
    print(f"Sommaire {'mis à jour' if existante else 'ajouté'} : {texte.count('<a href=')} entrées")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
