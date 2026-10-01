import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from utils import (
    load_data, FEATURES, TARGET, FIG_DIR, FIG_INDIV_DIR,
    save_and_show_table, save_fig, save_individual,
)

VAR_META = {
    "CO2": ("CO2 emissions",                   " ",        "Geophysical"),
    "EA":  ("Educational attainment",          " ",         "Social"),
    "GDP": ("GDP growth",                      " ",         "Economic"),
    "HE":  ("Health expenditure (per capita)", " ",       "Economic"),
    "NM":  ("Nurses and midwives",             " ", "Social"),
    "UE":  ("Unemployment rate",               " ",         "Economic"),
    "UP":  ("Urban population",                " ",         "Social"),
    "LE":  ("Life expectancy at birth",        " ",     "Target"),
}


def build_table_2(df):
    rows = []
    for col in FEATURES + [TARGET]:
        name, unit, cat = VAR_META[col]
        rows.append({
            "Category": cat,
            "Variable": name,
            "Abbrev.":  col,
            "Unit":     unit,
            "Min":      round(df[col].min(), 3),
            "Max":      round(df[col].max(), 3),
            "Mean":     round(df[col].mean(), 3),
            "Std.":     round(df[col].std(), 3),
        })
    return pd.DataFrame(rows)


def _one_violin_box(ax, values, color, title, ylabel):
    parts = ax.violinplot(values, showmeans=False, showmedians=False,
                          showextrema=False, widths=0.8)
    for pc in parts["bodies"]:
        pc.set_facecolor(color); pc.set_edgecolor("black"); pc.set_alpha(0.55)
    ax.boxplot(values, widths=0.18, patch_artist=True,
               boxprops=dict(facecolor="white", edgecolor="black"),
               medianprops=dict(color="black", linewidth=1.5),
               flierprops=dict(marker="o", markerfacecolor="red",
                               markersize=3, linestyle="none"),
               showfliers=True)
    ax.set_title(title, fontweight="bold")
    ax.set_xticks([])
    #ax.set_ylabel(ylabel, fontweight="bold")
    ax.set_ylabel(" ", fontweight="bold")
    ax.grid(axis="y", alpha=0.3, linestyle="--")


def figure_3(df):
    cols = FEATURES + [TARGET]
    palette = sns.color_palette("Set2", n_colors=len(cols))

    # Individual figures
    for c, color in zip(cols, palette):
        fig, ax = plt.subplots(figsize=(4.5, 4.0))
        _one_violin_box(ax, df[c].values, color, c, VAR_META[c][1])
        save_individual(fig, f"figure_03_violin_{c}")

    # Combined grid
    ncol = 4
    nrow = int(np.ceil(len(cols) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 3.0 * nrow))
    axes = axes.flatten()
    for i, (c, color) in enumerate(zip(cols, palette)):
        _one_violin_box(axes[i], df[c].values, color, c, VAR_META[c][1])
    for j in range(len(cols), len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("Distribution of input variables and target (LE).",
                 fontweight="bold")
    plt.tight_layout()
    save_fig(fig, "figure_01_data_distribution", also_individual=False)


def figure_4(df):
    cols = FEATURES + [TARGET]
    corr = df[cols].corr(method="pearson")

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                vmin=-1, vmax=1, square=True, linewidths=0.6,
                cbar_kws={"shrink": 0.8, "label": "Pearson r"},
                annot_kws={"fontweight": "bold"}, ax=ax)
    cbar = ax.collections[0].colorbar
    cbar.set_label("Pearson r", fontweight="bold")
    for lbl in cbar.ax.get_yticklabels():
        lbl.set_fontweight("bold")
    ax.set_title("Correlation heatmap of input variables and target.",
                 fontweight="bold", pad=12)
    plt.tight_layout()
    save_fig(fig, "supp_correlation_heatmap")

    r_target = corr[TARGET].drop(TARGET).abs().sort_values(ascending=False)
    print("\nTop absolute correlations with LE:")
    print(r_target.to_string(float_format=lambda v: f"{v:.3f}"))


def main():
    df = load_data()
    print(f"Loaded {df.shape[0]} rows x {df.shape[1]} columns.")
    tab = build_table_2(df)
    save_and_show_table(tab, "supp_descriptive_stats")
    figure_3(df)
    figure_4(df)


if __name__ == "__main__":
    main()
