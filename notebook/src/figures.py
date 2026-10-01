"""Figures du notebook.

Chaque figure porte **un** constat et se lit sans son paragraphe. Les couleurs
suivent une palette validee pour les daltonismes deutan et tritan : bleu et
orange en series, rouge reserve aux zones defectueuses, gris pour les reperes.

Regles tenues ici :
  - une seule echelle de valeur par graphique, jamais deux axes ;
  - grille et axes discrets, marques fines ;
  - etiquettes directes selectives, jamais un nombre sur chaque point ;
  - legende seulement a partir de deux series.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import clean, config as cfg

# --------------------------------------------------------------------------
# Palette et style
# --------------------------------------------------------------------------

SERIE_1 = "#2a78d6"     # bleu — serie principale
SERIE_2 = "#eb6834"     # orange — seconde serie
DEFAUT = "#d03b3b"      # rouge statut — zones defectueuses, jamais une serie
SURFACE = "#fcfcfb"
ENCRE = "#0b0b0b"
ENCRE_2 = "#52514e"
MUET = "#898781"
GRILLE = "#e1e0d9"
AXE = "#c3c2b7"


def appliquer_style() -> None:
    """Style global, applique une fois en debut de notebook."""
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "semibold",
        "axes.titlecolor": ENCRE,
        "axes.titlepad": 12,
        "axes.labelcolor": ENCRE_2,
        "axes.edgecolor": AXE,
        "axes.linewidth": 0.8,
        "axes.grid": False,
        "xtick.color": MUET,
        "ytick.color": MUET,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "figure.dpi": 110,
    })


def _habiller(ax, titre: str, sous_titre: str = "", grille: str = "y") -> None:
    """Retire le superflu et pose une grille discrete derriere les marques."""
    for cote in ("top", "right"):
        ax.spines[cote].set_visible(False)
    if grille:
        ax.grid(axis=grille, color=GRILLE, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
    if sous_titre:
        ax.set_title(f"{titre}\n", loc="left")
        ax.text(0, 1.02, sous_titre, transform=ax.transAxes,
                fontsize=9.5, color=ENCRE_2, va="bottom")
    else:
        ax.set_title(titre, loc="left")


def _etiqueter(ax, barres, valeurs, format_="{:.0%}", decalage=0.004,
               couleur=None) -> None:
    """Etiquettes directes en bout de barre verticale."""
    for b, v in zip(barres, valeurs):
        ax.text(b.get_x() + b.get_width() / 2, v + decalage, format_.format(v),
                ha="center", va="bottom", fontsize=9,
                color=couleur or ENCRE_2)


# --------------------------------------------------------------------------
# 1 — L'incoherence clinique des sejours termines par un deces
# --------------------------------------------------------------------------

def fig_incoherence_deces(tables):
    """Les deces portent le meme taux de readmission que les autres sorties.

    Cliniquement impossible, et invisible sans controle de coherence : le taux
    des deces (18,5 %) est indiscernable de celui des sorties a domicile.
    """
    sej = tables["sejours"].copy()
    g = sej.groupby("ModeSortie")[cfg.TARGET].agg(["mean", "size", "sum"])
    g = g.sort_values("mean", ascending=False)
    base = sej[cfg.TARGET].mean()

    # Sejours debutant apres le deces du meme patient.
    morts = sej[sej.ModeSortie == "Deces"][["PatientID", "DateSortie"]]
    apres = sej.merge(morts, on="PatientID", suffixes=("", "_deces"))
    apres = apres[apres.DateAdmission > apres.DateSortie_deces]

    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    couleurs = [DEFAUT if m == "Deces" else SERIE_1 for m in g.index]
    barres = ax.bar(range(len(g)), g["mean"], width=0.62, color=couleurs, zorder=3)
    _etiqueter(ax, barres, g["mean"])

    ax.axhline(base, color=MUET, linewidth=1.2, linestyle=(0, (4, 3)), zorder=2)
    ax.text(-0.42, base + 0.006, f"taux global {base:.1%}", ha="left",
            fontsize=9, color=MUET)

    # Effectifs dans l'etiquette d'axe, pour eviter toute collision.
    ax.set_xticks(range(len(g)))
    ax.set_xticklabels([f"{m}\nn={n}" for m, n in zip(g.index, g["size"])])

    i_deces = list(g.index).index("Deces")
    n_deces, r_deces = int(g.loc["Deces", "size"]), int(g.loc["Deces", "sum"])
    ax.annotate(f"{r_deces} des {n_deces} séjours terminés par un décès\n"
                f"sont marqués comme réadmis.\n"
                f"{len(apres)} séjours débutent après le décès du patient.",
                xy=(i_deces + 0.32, g.loc["Deces", "mean"]),
                xytext=(0.46, 0.74), textcoords="axes fraction",
                fontsize=9.5, color=DEFAUT, ha="left", va="center",
                arrowprops=dict(arrowstyle="-", color=DEFAUT, linewidth=1.1,
                                connectionstyle="arc3,rad=-0.25"))

    ax.set_ylim(0, g["mean"].max() * 1.55)
    ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))
    ax.set_ylabel("taux de réadmission")
    _habiller(ax, "Les décès portent le même taux que les autres sorties",
              "Ce qui est cliniquement impossible — et indétectable sans "
              "contrôle de cohérence")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# 2 — La desynchronisation des horodatages
# --------------------------------------------------------------------------

def fig_desynchronisation(tables):
    sej = tables["sejours"][["SejourID", "DateAdmission", "DateSortie"]]
    sources = [("biologies", "DatePrelevement"), ("signes_vitaux", "Horodatage"),
               ("actes", "DateActe"), ("comptes_rendus", "DateCR")]

    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True)
    for ax, (nom, col) in zip(axes.ravel(), sources):
        m = tables[nom].merge(sej, on="SejourID", how="inner")
        ecart = (m[col] - m.DateAdmission).dt.days
        dedans = ((m[col] >= m.DateAdmission.dt.normalize())
                  & (m[col] <= m.DateSortie + pd.Timedelta(days=1))).mean()

        # Fenetre d'affichage plutot qu'ecretage : entasser les valeurs
        # extremes dans la barre de bord creerait un mode qui n'existe pas.
        ax.hist(ecart[ecart.between(-750, 750)], bins=60, color=SERIE_1, zorder=3)
        ax.set_xlim(-780, 780)
        ax.axvspan(-1, 30, color=DEFAUT, alpha=0.14, zorder=2)
        ax.axvline(0, color=DEFAUT, linewidth=1.2, zorder=4)
        ax.set_title(f"{nom} — {dedans:.1%} dans la fenêtre", loc="left",
                     fontsize=10.5)
        for cote in ("top", "right"):
            ax.spines[cote].set_visible(False)
        ax.grid(axis="y", color=GRILLE, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        ax.set_yticks([])

    axes[0, 0].annotate("fenêtre du séjour", xy=(15, 0), xytext=(300, 0.72),
                        textcoords=("data", "axes fraction"), fontsize=9,
                        color=DEFAUT, ha="left",
                        arrowprops=dict(arrowstyle="-", color=DEFAUT,
                                        linewidth=1.1,
                                        connectionstyle="arc3,rad=-0.25"))
    for ax in axes[1]:
        ax.set_xlabel("jours par rapport à l'admission")

    fig.suptitle("Les horodatages des tables filles sont désynchronisés",
                 x=0.008, ha="left", fontsize=12.5, fontweight="semibold",
                 color=ENCRE)
    fig.text(0.008, 0.928,
             "Écart entre chaque ligne fille et l'admission de son propre séjour — "
             "moins de 2 % tombent dans la fenêtre",
             fontsize=9.5, color=ENCRE_2, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    return fig


# --------------------------------------------------------------------------
# 3 — La couverture de telesurveillance
# --------------------------------------------------------------------------

def fig_couverture_iot(tables):
    oc, sej = tables["objets_connectes"], tables["sejours"]
    derniere = sej.sort_values("DateSortie").groupby("PatientID").DateSortie.last()
    m = oc.merge(derniere.rename("sortie"), on="PatientID", how="left")
    delai = (m.Horodatage - m.sortie).dt.days

    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.hist(delai.clip(-400, 400), bins=70, color=SERIE_1, zorder=3)
    ax.axvspan(0, cfg.IOT_WINDOW_DAYS, color=DEFAUT, alpha=0.16, zorder=2)
    ax.axvline(0, color=DEFAUT, linewidth=1.4, zorder=4)

    n_pat = m[(delai >= 0) & (delai <= cfg.IOT_WINDOW_DAYS)].PatientID.nunique()
    ax.annotate(f"fenêtre de réévaluation J+0 à J+7\n"
                f"{n_pat} patients sur {sej.PatientID.nunique()}",
                xy=(4, 0), xytext=(0.60, 0.72), textcoords="axes fraction",
                fontsize=9.5, color=DEFAUT, ha="left",
                arrowprops=dict(arrowstyle="-", color=DEFAUT, linewidth=1.1,
                                connectionstyle="arc3,rad=0.25"))
    ax.text(-390, ax.get_ylim()[1] * 0.9, "avant la sortie", fontsize=9,
            color=MUET, ha="left")

    ax.set_xlabel("jours par rapport à la sortie")
    ax.set_ylabel("mesures")
    _habiller(ax, "La télésurveillance ne couvre pas l'après-sortie",
              "6 297 mesures d'objets connectés, réparties sur une année "
              "autour du séjour")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# 4 — L'harmonisation des unites
# --------------------------------------------------------------------------

def fig_unites_biologie(tables):
    brut = tables["biologies"]
    net = clean.clean_biologies(brut, valid_stays=set(tables["sejours"].SejourID))

    fig, (g, d) = plt.subplots(1, 2, figsize=(10, 4.2))

    borne_vue = 260   # fenetre d'affichage, sans ecretage
    a = brut[(brut.Panel == "Creatinine") & (brut.Unite == "mg/L")].Valeur
    b = brut[(brut.Panel == "Creatinine") & (brut.Unite == "µmol/L")].Valeur
    g.hist(a[a.between(0, borne_vue)], bins=45, color=SERIE_1, alpha=0.92,
           label=f"mg/L  (n={len(a)})", zorder=3)
    g.hist(b[b.between(0, borne_vue)], bins=45, color=SERIE_2, alpha=0.82,
           label=f"µmol/L  (n={len(b)})", zorder=3)
    g.legend(loc="upper right")
    g.set_title("Avant — deux populations disjointes", loc="left", fontsize=10.5)

    c = net[net.Panel == "Creatinine"].valeur_harmonisee
    d.hist(c[c.between(0, borne_vue)], bins=45, color=SERIE_1, zorder=3)
    lo, hi = cfg.REFERENCE_RANGES["Creatinine"]
    d.axvspan(lo, hi, color=MUET, alpha=0.16, zorder=2)
    d.set_title("Après — une seule échelle, en µmol/L", loc="left", fontsize=10.5)

    for ax in (g, d):
        for cote in ("top", "right"):
            ax.spines[cote].set_visible(False)
        ax.grid(axis="y", color=GRILLE, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        ax.set_xlabel("créatinine")
        ax.set_yticks([])
        ax.set_xlim(0, borne_vue)

    # Pose apres coup : l'echelle verticale n'est connue qu'une fois tracee.
    d.annotate("bornes de référence\n60 – 110 µmol/L", xy=(hi, d.get_ylim()[1] * 0.42),
               xytext=(0.56, 0.80), textcoords="axes fraction", fontsize=9,
               color=ENCRE_2, ha="left", va="center",
               arrowprops=dict(arrowstyle="-", color=MUET, linewidth=1,
                               connectionstyle="arc3,rad=0.2"))

    fig.suptitle("Harmoniser les unités avant toute comparaison",
                 x=0.008, ha="left", fontsize=12.5, fontweight="semibold",
                 color=ENCRE)
    fig.text(0.008, 0.905,
             "Le facteur de conversion est vérifié sur les médianes observées : "
             "10,25 mg/L × 8,84 = 90,6 contre 93,9 µmol/L",
             fontsize=9.5, color=ENCRE_2, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    return fig


# --------------------------------------------------------------------------
# 5 — Les constantes physiologiquement impossibles
# --------------------------------------------------------------------------

def fig_constantes_impossibles(tables):
    sv = tables["signes_vitaux"]
    cas = [("FrequenceCardiaque", "fréquence cardiaque (bpm)",
            "382 relevés à zéro — capteur débranché"),
           ("SpO2", "saturation en oxygène (%)",
            "340 relevés au-dessus de 100 % — impossible")]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, (col, libelle, note) in zip(axes, cas):
        lo, hi = cfg.VITALS_PLAUSIBLE[col]
        s = sv[col].dropna()
        ax.hist(s, bins=60, color=SERIE_1, zorder=3)
        for borne in (lo, hi):
            ax.axvline(borne, color=DEFAUT, linewidth=1.2,
                       linestyle=(0, (4, 3)), zorder=4)
        # Les deux zones impossibles sont ombrees, pas seulement celle qui
        # porte le constat : la borne opposee doit se lire aussi.
        ax.axvspan(min(s.min(), lo) - 2, lo, color=DEFAUT, alpha=0.16, zorder=2)
        ax.axvspan(hi, max(s.max(), hi) + 2, color=DEFAUT, alpha=0.16, zorder=2)
        ax.set_title(note, loc="left", fontsize=10.5, color=DEFAUT)
        ax.set_xlabel(libelle)
        for cote in ("top", "right"):
            ax.spines[cote].set_visible(False)
        ax.grid(axis="y", color=GRILLE, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        ax.set_yticks([])

    fig.suptitle("Écarter l'impossible, ne pas le tronquer",
                 x=0.008, ha="left", fontsize=12.5, fontweight="semibold",
                 color=ENCRE)
    fig.text(0.008, 0.905,
             "Les traits marquent les bornes physiologiques. Ramener une valeur "
             "à sa borne fabriquerait une mesure qui n'a pas eu lieu.",
             fontsize=9.5, color=ENCRE_2, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    return fig


# --------------------------------------------------------------------------
# 6 — Le gradient clinique
# --------------------------------------------------------------------------

def fig_gradient_diagnostic(tables, effectif_min: int = 30):
    dx = tables["diagnostics"]
    sej = tables["sejours"]
    p = dx[dx.Role == "Principal"].merge(
        sej[["SejourID", cfg.TARGET]], on="SejourID")
    g = p.groupby("LibelleDiagnostic")[cfg.TARGET].agg(["mean", "size"])
    g = g[g["size"] >= effectif_min].sort_values("mean")
    base = sej[cfg.TARGET].mean()

    fig, ax = plt.subplots(figsize=(9, 5.4))
    barres = ax.barh(g.index, g["mean"], height=0.66, color=SERIE_1, zorder=3)
    ax.axvline(base, color=MUET, linewidth=1.2, linestyle=(0, (4, 3)), zorder=4)
    ax.text(base + 0.004, len(g) - 0.35, f"taux global {base:.1%}", fontsize=9,
            color=MUET, va="center")

    for b, (v, n) in zip(barres, g[["mean", "size"]].to_numpy()):
        ax.text(v + 0.006, b.get_y() + b.get_height() / 2,
                f"{v:.0%}   n={int(n)}", va="center", fontsize=9, color=ENCRE_2)

    ax.set_xlim(0, g["mean"].max() * 1.30)
    ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))
    ax.set_xlabel("taux de réadmission")
    _habiller(ax, "Le gradient est cliniquement cohérent",
              "Taux de réadmission par diagnostic principal — séjours ≥ 30",
              grille="x")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# 7 — La completude des blocs
# --------------------------------------------------------------------------

def fig_completude_blocs(X, contribution):
    g = contribution.sort_values("remplissage_moyen")
    fig, ax = plt.subplots(figsize=(9, 4.6))
    barres = ax.barh(g.index, g.remplissage_moyen, height=0.66,
                     color=SERIE_1, zorder=3)
    for b, (r, n) in zip(barres, g[["remplissage_moyen", "variables"]].to_numpy()):
        ax.text(r + 0.012, b.get_y() + b.get_height() / 2,
                f"{r:.0%}   {int(n)} variables", va="center", fontsize=9,
                color=ENCRE_2)

    ax.set_xlim(0, 1.30)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))
    ax.set_xlabel("taux de remplissage moyen des variables du bloc")
    _habiller(ax, "Deux blocs sont partiellement renseignés, par construction",
              "Biologie : tous les séjours n'ont pas tous les panels. "
              "Texte : les gabarits de compte-rendu portent des champs disjoints.",
              grille="x")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# 8 — L'age, variable dominante
# --------------------------------------------------------------------------

def fig_age_readmission(X, y):
    tranches = pd.cut(X.demo_age, [0, 55, 65, 75, 85, 120],
                      labels=["< 55", "55-64", "65-74", "75-84", "85 +"])
    g = pd.DataFrame({"tranche": tranches, "cible": y}).groupby(
        "tranche", observed=True).cible.agg(["mean", "size"])
    base = y.mean()

    fig, ax = plt.subplots(figsize=(8, 4.2))
    barres = ax.bar(range(len(g)), g["mean"], width=0.62,
                    color=SERIE_1, zorder=3)
    _etiqueter(ax, barres, g["mean"])
    ax.axhline(base, color=MUET, linewidth=1.2, linestyle=(0, (4, 3)), zorder=2)
    ax.text(-0.42, base + 0.010, f"taux global {base:.1%}",
            ha="left", fontsize=9, color=MUET)

    ax.set_xticks(range(len(g)))
    ax.set_xticklabels([f"{t}\nn={n}"
                        for t, n in zip(g.index.astype(str), g["size"])])

    ax.set_ylim(0, g["mean"].max() * 1.24)
    ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))
    ax.set_ylabel("taux de réadmission")
    _habiller(ax, "Le risque croît fortement avec l'âge",
              "C'est ce gradient que le modèle apprend — et presque rien d'autre")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# 9 — L'apport marginal des sources
# --------------------------------------------------------------------------

def fig_apport_marginal(apport, taux_de_base: float):
    g = apport.sort_values("PR-AUC")
    couleurs = [SERIE_2 if "Âge seul" in i else SERIE_1 for i in g.index]

    fig, ax = plt.subplots(figsize=(9, 4.4))
    barres = ax.barh(g.index, g["PR-AUC"], height=0.62, color=couleurs, zorder=3)
    ax.axvline(taux_de_base, color=MUET, linewidth=1.2,
               linestyle=(0, (4, 3)), zorder=4)
    ax.text(taux_de_base + 0.006, len(g) - 0.35,
            f"taux de base {taux_de_base:.3f}", fontsize=9, color=MUET,
            va="center")

    for b, (v, n) in zip(barres, g[["PR-AUC", "variables"]].to_numpy()):
        ax.text(v + 0.008, b.get_y() + b.get_height() / 2,
                f"{v:.3f}   {int(n)} var.", va="center", fontsize=9,
                color=ENCRE_2)

    ax.set_xlim(0, g["PR-AUC"].max() * 1.28)
    ax.set_xlabel("PR-AUC")
    # Titre calcule sur les valeurs : un chiffre ecrit a la main finit par
    # contredire la figure qu'il commente (J3-06, J3-11).
    age = g.loc[[i for i in g.index if "Âge seul" in i][0]]
    complet = g.loc[[i for i in g.index if "complet" in i][0]]
    gain = complet["PR-AUC"] / age["PR-AUC"] - 1
    fr = lambda x: f"{x:.3f}".replace(".", ",")
    _habiller(ax, f"Dix sources supplémentaires apportent {gain:.0%} en relatif".replace("%", " %"),
              f"L'âge seul atteint {fr(age['PR-AUC'])} ; le pipeline complet à "
              f"{int(complet['variables'])} variables, {fr(complet['PR-AUC'])}",
              grille="x")
    fig.tight_layout()
    return fig
