"""
Manuscript outputs: Figure 2 (train/test distributions under Protocol A).
(Internal figure/table numbers in the docstring below are legacy; see docs/MANUSCRIPT_MAP.md.)
section_03_split_and_distribution.py  --  Train/test distributional consistency

Produces:
    Figure 5 : Violin-plot comparison of each feature between train and test
               (combined grid AND one file per variable in figures/individual/)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from utils import (
    load_data, FEATURES, TARGET, SPLIT_SEED,
    train_test_split_random, save_fig, save_individual,
)


def _panel(ax, df_combined, col):
    palette = {"Train": "#2EC4B6", "Test": "#FF9F1C"}
    sns.violinplot(data=df_combined, x="split", y=col, hue="split",
                   palette=palette, inner="quartile",
                   legend=False, ax=ax)
    ax.set_title(col, fontweight="bold")
    ax.set_xlabel(""); ax.set_ylabel("")
    ax.grid(axis="y", alpha=0.3, linestyle="--")


def figure_5(df):
    X = df[FEATURES].values.astype(float)
    y = df[TARGET].values.astype(float)
    Xtr, Xte, ytr, yte, _, _ = train_test_split_random(X, y, seed=SPLIT_SEED)
    print(f"Train: {len(ytr)} samples, Test: {len(yte)} samples.")

    combined = pd.concat([
        pd.DataFrame({**{f: Xtr[:, i] for i, f in enumerate(FEATURES)},
                      TARGET: ytr, "split": "Train"}),
        pd.DataFrame({**{f: Xte[:, i] for i, f in enumerate(FEATURES)},
                      TARGET: yte, "split": "Test"}),
    ], ignore_index=True)

    cols = FEATURES + [TARGET]

    # Individual
    for c in cols:
        fig, ax = plt.subplots(figsize=(4.5, 4.0))
        _panel(ax, combined, c)
        save_individual(fig, f"figure_05_split_{c}")

    # Combined
    ncol = 4
    nrow = int(np.ceil(len(cols) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 2.8 * nrow))
    axes = axes.flatten()
    for i, c in enumerate(cols):
        _panel(axes[i], combined, c)
    for j in range(len(cols), len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("Distributional consistency between train and test sets",
                 fontweight="bold")
    plt.tight_layout()
    save_fig(fig, "figure_02_train_test_distributions", also_individual=False)


def main():
    figure_5(load_data())


if __name__ == "__main__":
    main()
