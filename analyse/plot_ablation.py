"""
Plotting utilities for the feature ablation analysis.

Two chart types are produced:

1. `plot_single_language_ablation`
   One figure per (language, feature_config): a line chart showing the mean
   change in CE loss (ablated - baseline) for the target language corpus vs.
   all other language corpora, across all layers.

2. `plot_all_languages_grid`
   A 3x3 subplot grid (one cell per non-English language) that overlays the
   three feature_config variants (rank-#1, rank-#2, rank-#1+#2) so the reader
   can compare how many features need to be ablated to affect each language.

Both functions read the .npy files produced by `analyse/ablation.py`.
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator

from config import MainConfig


LANGUAGE_DISPLAY_NAMES: dict[str, str] = {
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "ja": "Japanese",
    "ko": "Korean",
    "pt": "Portuguese",
    "th": "Thai",
    "vi": "Vietnamese",
    "zh": "Chinese",
    "ar": "Arabic",
}

PLOT_STYLE: dict = {
    "font.size": 14,
    "font.weight": "bold",
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
}

# Colours for the three feature_config variants in the grid plot
FEATURE_CONFIG_COLOURS = ["tab:red", "tab:green", "tab:orange"]
FEATURE_CONFIG_MARKERS = ["o", "s", "^"]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _mean_ce_change_per_layer(
    ablated_ce_loss: np.ndarray,  # (n_layers, n_total_texts)
    baseline_ce_loss: np.ndarray,  # (n_total_texts,)
    text_slice: slice,
) -> np.ndarray:
    """
    Compute the mean change in CE loss (ablated - baseline) for a given subset
    of texts (identified by `text_slice`) across all layers.

    Returns a 1-D array of shape (n_layers,).
    """
    ce_change = ablated_ce_loss - baseline_ce_loss[np.newaxis, :]  # (n_layers, n_texts)
    return ce_change[:, text_slice].mean(axis=1)


def _complementary_slice(
    target_slice: slice,
    total: int,
) -> np.ndarray:
    """
    Return the indices of all texts *not* belonging to `target_slice`.
    Used to compute the CE change on all-other-languages.
    """
    all_indices = np.arange(total)
    target_indices = all_indices[target_slice]
    return np.setdiff1d(all_indices, target_indices)


# ---------------------------------------------------------------------------
# Public plot functions
# ---------------------------------------------------------------------------


def plot_single_language_ablation(
    model_name: str,
    target_language: str,
    start_idx: int,
    topk: int,
    all_languages: list[str],
    max_samples_per_language: int,
    layers: list[int],
) -> None:
    """
    Line chart: CE loss change when ablating `topk` features starting at rank
    `start_idx` for `target_language`, compared across the target language
    corpus vs. all other language corpora.

    Saved to:
        results/ablation/{model_name}/{target_language}/
            ce_change_{start_idx}_{topk}.png / .pdf
    """
    results_dir = f"results/ablation/{model_name}/{target_language}"
    ablated_path = os.path.join(
        results_dir, f"sae_ce_loss_all_layers_{start_idx}_{topk}.npy"
    )

    if not os.path.exists(ablated_path):
        print(f"  [skip] {ablated_path} not found.")
        return

    baseline_ce_loss = np.load(os.path.join(results_dir, "ori_ce_loss.npy"))
    ablated_ce_loss = np.load(ablated_path)  # (n_layers, n_total_texts)

    language_slice = _rebuild_language_slice(
        all_languages, max_samples_per_language, len(baseline_ce_loss)
    )

    target_ce_change = _mean_ce_change_per_layer(
        ablated_ce_loss, baseline_ce_loss, language_slice[target_language]
    )
    other_indices = _complementary_slice(
        language_slice[target_language], len(baseline_ce_loss)
    )
    other_ce_change = (ablated_ce_loss - baseline_ce_loss[np.newaxis, :])[
        :, other_indices
    ].mean(axis=1)

    target_display = LANGUAGE_DISPLAY_NAMES.get(
        target_language, target_language.upper()
    )
    rank_label = (
        f"rank #{start_idx + 1}"
        if topk == 1
        else f"ranks #{start_idx + 1}-#{start_idx + topk}"
    )

    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(10, 4))

        ax.plot(
            layers,
            target_ce_change,
            label=f"{target_display} corpus",
            linewidth=2,
            color="tab:blue",
        )
        ax.plot(
            layers,
            other_ce_change,
            label="All other corpora",
            linewidth=2,
            color="tab:orange",
            linestyle="--",
        )

        ax.set_title(f"Ablating {target_display} Features ({rank_label})")
        ax.set_xlabel("Layer")
        ax.set_ylabel("Change in CE Loss")
        ax.yaxis.set_major_locator(MultipleLocator(3))
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(
            ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.22), frameon=False
        )

        plt.tight_layout()
        stem = f"ce_change_{start_idx}_{topk}"
        fig.savefig(os.path.join(results_dir, f"{stem}.pdf"), bbox_inches="tight")
        fig.savefig(
            os.path.join(results_dir, f"{stem}.png"), bbox_inches="tight", dpi=150
        )
        plt.close(fig)

    print(f"  Saved: {results_dir}/{stem}.png")


def plot_all_languages_grid(
    model_name: str,
    target_language: str,
    feature_configs: list[list[int]],
    all_languages: list[str],
    max_samples_per_language: int,
    layers: list[int],
    non_english_languages: list[str] | None = None,
) -> None:
    """
    3x3 subplot grid: one panel per non-English language, each panel overlaying
    the CE loss change curves for each feature_config variant.

    This lets the reader compare how many features must be ablated to disrupt
    processing of each language in the corpus.

    Saved to:
        results/ablation/{model_name}/{target_language}/ce_change_grid.png / .pdf
    """
    results_dir = f"results/ablation/{model_name}/{target_language}"
    baseline_ce_loss = np.load(os.path.join(results_dir, "ori_ce_loss.npy"))
    language_slice = _rebuild_language_slice(
        all_languages, max_samples_per_language, len(baseline_ce_loss)
    )

    if non_english_languages is None:
        non_english_languages = [l for l in all_languages if l != "en"]

    n_panels = len(non_english_languages)
    n_cols = 3
    n_rows = (n_panels + n_cols - 1) // n_cols

    with plt.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(6 * n_cols, 6 * n_rows), sharey=True
        )
        axes_flat: list[Axes] = np.array(axes).flatten().tolist()

        # Collect line handles once for the shared legend
        legend_lines: list[Line2D] = []
        legend_labels: list[str] = []

        for panel_idx, panel_language in enumerate(non_english_languages):
            ax = axes_flat[panel_idx]
            panel_display = LANGUAGE_DISPLAY_NAMES.get(
                panel_language, panel_language.upper()
            )

            for config_idx, (start_idx, topk) in enumerate(feature_configs):
                ablated_path = os.path.join(
                    results_dir, f"sae_ce_loss_all_layers_{start_idx}_{topk}.npy"
                )
                if not os.path.exists(ablated_path):
                    continue

                ablated_ce_loss = np.load(ablated_path)
                ce_change = _mean_ce_change_per_layer(
                    ablated_ce_loss, baseline_ce_loss, language_slice[panel_language]
                )

                rank_label = (
                    f"Rank #{start_idx + 1} feature"
                    if topk == 1
                    else f"Rank #{start_idx + 1}-#{start_idx + topk} features"
                )
                colour = FEATURE_CONFIG_COLOURS[
                    config_idx % len(FEATURE_CONFIG_COLOURS)
                ]
                marker = FEATURE_CONFIG_MARKERS[
                    config_idx % len(FEATURE_CONFIG_MARKERS)
                ]

                (line,) = ax.plot(
                    layers,
                    ce_change,
                    label=rank_label,
                    linewidth=2,
                    color=colour,
                    marker=marker,
                    markersize=4,
                )
                if panel_idx == 0:
                    target_display = LANGUAGE_DISPLAY_NAMES.get(
                        target_language, target_language.upper()
                    )
                    legend_lines.append(line)
                    legend_labels.append(f"{rank_label} ({target_display})")

            ax.set_title(f"CE Loss for {panel_display}")
            ax.set_xlabel("Layer")
            ax.set_ylim(bottom=-0.5)
            ax.grid(True, linestyle="--", alpha=0.5)

        # Label y-axis only for leftmost panels
        for row in range(n_rows):
            axes_flat[row * n_cols].set_ylabel("Change in CE Loss")

        # Hide any unused panels
        for ax in axes_flat[n_panels:]:
            ax.set_visible(False)

        fig.legend(
            legend_lines,
            legend_labels,
            loc="lower center",
            ncol=len(feature_configs),
            bbox_to_anchor=(0.5, 0.02),
            frameon=False,
        )

        target_display = LANGUAGE_DISPLAY_NAMES.get(
            target_language, target_language.upper()
        )
        fig.suptitle(
            f"Feature Ablation: {target_display} Features",
            fontsize=16,
            fontweight="bold",
            y=1.01,
        )

        plt.tight_layout(rect=[0, 0.08, 1, 1])
        stem = "ce_change_grid"
        fig.savefig(os.path.join(results_dir, f"{stem}.pdf"), bbox_inches="tight")
        fig.savefig(
            os.path.join(results_dir, f"{stem}.png"), bbox_inches="tight", dpi=150
        )
        plt.close(fig)

    print(f"  Saved: {results_dir}/{stem}.png")


# ---------------------------------------------------------------------------
# Slice reconstruction helper (mirrors ablation.py ordering)
# ---------------------------------------------------------------------------


def _rebuild_language_slice(
    all_languages: list[str],
    max_samples_per_language: int,
    total_texts: int,
) -> dict[str, slice]:
    """
    Reconstruct the language -> slice mapping used when the flat CE loss array
    was created.  Assumes languages are sorted and capped at
    `max_samples_per_language` each, with the total summing to `total_texts`.
    """
    language_slice: dict[str, slice] = {}
    cursor = 0
    for lan in all_languages:
        available = min(max_samples_per_language, total_texts - cursor)
        language_slice[lan] = slice(cursor, cursor + available)
        cursor += available
        if cursor >= total_texts:
            break
    return language_slice


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def plot_ablation_results() -> None:
    """
    Plot the ablation results for the given configuration.
    """
    cfg = MainConfig.load("config.yaml").analyse
    ablation_cfg = cfg.ablation
    model_name = cfg.llm_path.split("/")[-1]
    layers = cfg.layers if cfg.layers else list(range(cfg.num_layers))

    # Determine the full language order as it was written to the .npy files.
    # Language ordering is always sorted(samples_by_language.keys()), matching the JSONL.
    with open(ablation_cfg.dataset_path, "r", encoding="utf-8") as f:
        language_set: set[str] = set()
        for line in f:
            language_set.add(json.loads(line)["lan"])
    all_languages = sorted(language_set)

    for target_language in ablation_cfg.target_languages:
        print(
            f"\nPlotting ablation results for {LANGUAGE_DISPLAY_NAMES.get(target_language, target_language)}..."
        )

        for start_idx, topk in ablation_cfg.feature_configs:
            plot_single_language_ablation(
                model_name=model_name,
                target_language=target_language,
                start_idx=start_idx,
                topk=topk,
                all_languages=all_languages,
                max_samples_per_language=ablation_cfg.max_samples_per_language,
                layers=layers,
            )

        plot_all_languages_grid(
            model_name=model_name,
            target_language=target_language,
            feature_configs=ablation_cfg.feature_configs,
            all_languages=all_languages,
            max_samples_per_language=ablation_cfg.max_samples_per_language,
            layers=layers,
        )

    print("\nAll ablation plots saved.")


if __name__ == "__main__":
    plot_ablation_results()
