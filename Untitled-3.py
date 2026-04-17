"""
Analyse descriptive — Health Disparities in Gynecology
Dataset TidyTuesday 2025-02-25
"""

import warnings

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import pandas as pd

warnings.filterwarnings("ignore")

# ── 1. Chargement ──────────────────────────────────────────────────────────────

URL = "https://raw.githubusercontent.com/rfordatascience/tidytuesday/main/data/2025/2025-02-25/article_dat.csv"
df = pd.read_csv(URL, low_memory=False)

print("=" * 60)
print("APERÇU GÉNÉRAL")
print("=" * 60)
print(f"Dimensions         : {df.shape[0]:,} lignes × {df.shape[1]} colonnes")
print(f"Journaux uniques   : {df['journal'].nunique()}")
print(f"Années couvertes   : {df['year'].min()} – {df['year'].max()}")
print(f"États US uniques   : {df['study_location'].nunique()}")
print(
    f"DOI renseignés     : {df['doi'].notna().sum():,} ({df['doi'].notna().mean():.1%})"
)


# ── 2. Valeurs manquantes ──────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("VALEURS MANQUANTES (colonnes > 10 %)")
print("=" * 60)
missing = df.isnull().mean().sort_values(ascending=False)
print(missing[missing > 0.10].apply(lambda x: f"{x:.1%}").to_string())


# ── 3. Stats descriptives — variables numériques ───────────────────────────────

print("\n" + "=" * 60)
print("STATISTIQUES DESCRIPTIVES — variables numériques clés")
print("=" * 60)
num_cols = ["year", "study_year_start", "study_year_end"]
print(df[num_cols].describe().round(1).to_string())

# Durée des études
df["study_duration"] = df["study_year_end"] - df["study_year_start"]
print(f"\nDurée moyenne des études : {df['study_duration'].mean():.1f} ans")
print(f"Médiane                  : {df['study_duration'].median():.1f} ans")
print(f"Max                      : {df['study_duration'].max():.0f} ans")


# ── 4. Publications par année ──────────────────────────────────────────────────

pubs_year = df.groupby("year").size().reset_index(name="n_articles")

print("\n" + "=" * 60)
print("PUBLICATIONS PAR ANNÉE")
print("=" * 60)
print(pubs_year.to_string(index=False))


# ── 5. Types d'études ──────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("TYPES D'ÉTUDE (top 10)")
print("=" * 60)
study_types = df["study_type"].value_counts().head(10)
print(study_types.to_string())


# ── 6. Journaux les plus représentés ──────────────────────────────────────────

print("\n" + "=" * 60)
print("TOP 10 JOURNAUX")
print("=" * 60)
top_journals = df["journal"].value_counts().head(10)
print(top_journals.to_string())


# ── 7. Domaines de recherche (colonnes binaires) ───────────────────────────────

domain_cols = {
    "cancer_ovarian": "Cancer ovaire",
    "cancer_uterine": "Cancer utérin",
    "cancer_cervical": "Cancer cervical",
    "cancer_vulvar": "Cancer vulvaire",
    "endo": "Endométriose",
    "fibroids": "Fibromes",
    "fert": "Fertilité",
    "matmorbmort": "Mortalité mat.",
    "other_preg": "Autres grossesse",
    "other_gyn_surg": "Chirurgie gyn.",
    "other_gyn_onc": "Autres cancers",
    "access_to_care": "Accès aux soins",
    "treatment_received": "Traitement reçu",
    "health_outcome": "Résultat santé",
    "covid": "COVID",
}

print("\n" + "=" * 60)
print("FRÉQUENCE DES DOMAINES DE RECHERCHE")
print("=" * 60)
domain_counts = {}
for col, label in domain_cols.items():
    if col in df.columns:
        domain_counts[label] = df[col].sum()

domain_series = pd.Series(domain_counts).sort_values(ascending=False)
for label, count in domain_series.items():
    pct = count / len(df) * 100
    print(f"  {label:<22} {count:5,}  ({pct:.1f}%)")


# ── 8. Groupes raciaux ────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("GROUPES RACIAUX MENTIONNÉS (race1 à race8)")
print("=" * 60)
race_cols = [c for c in df.columns if c.startswith("race") and not c.endswith("_ss")]
all_races = pd.concat([df[c] for c in race_cols]).dropna()
race_counts = all_races.value_counts().head(12)
print(race_counts.to_string())


# ── 9. Sources de données ──────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("SOURCES DE DONNÉES (top 10)")
print("=" * 60)
print(df["data_source"].value_counts().head(10).to_string())


# ── 10. Évolution thématique dans le temps ─────────────────────────────────────

print("\n" + "=" * 60)
print("ÉVOLUTION — % d'articles sur le cancer de l'ovaire par année")
print("=" * 60)
if "cancer_ovarian" in df.columns:
    ov_trend = df.groupby("year")["cancer_ovarian"].mean().mul(100).round(1)
    print(ov_trend.to_string())


# ── 11. VISUALISATIONS ────────────────────────────────────────────────────────

fig = plt.figure(figsize=(18, 14))
fig.suptitle(
    "Disparités de santé gynécologique — Analyse descriptive",
    fontsize=15,
    fontweight="bold",
    y=0.98,
)
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

colors_blue = ["#3266ad", "#5a88cc", "#88a8db", "#b0c6e8"]
colors_multi = ["#3266ad", "#1d9e75", "#d85a30", "#73726c", "#9966cc", "#ba7517"]

# — 1. Publications par année
ax1 = fig.add_subplot(gs[0, :2])
ax1.bar(pubs_year["year"], pubs_year["n_articles"], color="#3266ad", width=0.7)
ax1.set_title("Publications par année", fontsize=11)
ax1.set_xlabel("Année")
ax1.set_ylabel("Nombre d'articles")
ax1.tick_params(axis="x", rotation=45)
for spine in ["top", "right"]:
    ax1.spines[spine].set_visible(False)

# — 2. Types d'étude (donut)
ax2 = fig.add_subplot(gs[0, 2])
top_types = df["study_type"].value_counts().head(5)
wedges, texts, autotexts = ax2.pie(
    top_types.values,
    labels=top_types.index,
    autopct="%1.0f%%",
    colors=colors_multi,
    startangle=140,
    pctdistance=0.8,
    wedgeprops=dict(width=0.55),
)
for t in texts:
    t.set_fontsize(8)
for at in autotexts:
    at.set_fontsize(8)
ax2.set_title("Types d'étude", fontsize=11)

# — 3. Top domaines
ax3 = fig.add_subplot(gs[1, :2])
domains_plot = domain_series.head(10)
bars = ax3.barh(domains_plot.index[::-1], domains_plot.values[::-1], color="#3266ad")
ax3.set_title("Fréquence des domaines de recherche (top 10)", fontsize=11)
ax3.set_xlabel("Nombre d'articles")
for spine in ["top", "right"]:
    ax3.spines[spine].set_visible(False)
for bar, val in zip(bars, domains_plot.values[::-1], strict=True):
    ax3.text(
        bar.get_width() + 20,
        bar.get_y() + bar.get_height() / 2,
        f"{val:,}",
        va="center",
        fontsize=8,
    )

# — 4. Top journaux
ax4 = fig.add_subplot(gs[1, 2])
top_j = df["journal"].value_counts().head(6)
ax4.barh(top_j.index[::-1], top_j.values[::-1], color="#1d9e75")
ax4.set_title("Top 6 journaux", fontsize=11)
ax4.set_xlabel("Nombre d'articles")
for spine in ["top", "right"]:
    ax4.spines[spine].set_visible(False)
for i, v in enumerate(top_j.values[::-1]):
    ax4.text(v + 5, i, str(v), va="center", fontsize=8)
ax4.tick_params(axis="y", labelsize=7)

# — 5. Groupes raciaux
ax5 = fig.add_subplot(gs[2, :2])
race_plot = race_counts.head(8)
bars5 = ax5.bar(race_plot.index, race_plot.values, color="#d85a30")
ax5.set_title("Groupes raciaux les plus étudiés", fontsize=11)
ax5.set_ylabel("Fréquence (toutes colonnes race)")
ax5.tick_params(axis="x", rotation=40, labelsize=8)
for spine in ["top", "right"]:
    ax5.spines[spine].set_visible(False)
for bar, val in zip(bars5, race_plot.values, strict=True):
    ax5.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 20,
        f"{val:,}",
        ha="center",
        fontsize=8,
    )

# — 6. Focus thématique combiné
ax6 = fig.add_subplot(gs[2, 2])
focus_cols = {
    "access_to_care": "Accès",
    "treatment_received": "Traitement",
    "health_outcome": "Résultat",
}
available = {k: v for k, v in focus_cols.items() if k in df.columns}
combos = {}
for col, label in available.items():
    combos[label] = df[col].sum()
ax6.bar(combos.keys(), combos.values(), color=["#3266ad", "#1d9e75", "#d85a30"])
ax6.set_title("Focus thématique principal", fontsize=11)
ax6.set_ylabel("Nombre d'articles")
for spine in ["top", "right"]:
    ax6.spines[spine].set_visible(False)

plt.savefig("health_disparities_analysis.png", dpi=150, bbox_inches="tight")
print("\n✓ Graphiques sauvegardés dans health_disparities_analysis.png")
plt.show()
