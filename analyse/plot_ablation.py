"""
Plotting utilities for the feature ablation analysis.
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
    "arb": "Arabic",
    "vie": "Vietnamese",
    "sin": "Sinhala",
    "tel": "Telugu",
    "zsm": "Chinese",
    "som": "Somali",
    "tam": "Tamil",
    "yor": "Yoruba",
    "pan": "Panjabi",
    "guj": "Gujarati",
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
    3x3 subplot grid: one panel per non-English language, each panel overlaying the CE loss change curves for each feature_config variant.

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

        for row in range(n_rows):
            axes_flat[row * n_cols].set_ylabel("Change in CE Loss")

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


def plot_single_layer_bar(
    model_name: str,
    target_language: str,
    feature_configs: list[list[int]],
    all_languages: list[str],
    max_samples_per_language: int,
    layer_idx: int = 0,
) -> None:
    """
    This is the right visualisation when only one layer was run (e.g. layer 20),
    because the line-chart functions collapse to a single point in that case.

    Saved to:
        results/ablation/{model_name}/{target_language}/ce_change_bar.png / .pdf
    """
    results_dir = f"results/ablation/{model_name}/{target_language}"
    baseline_path = os.path.join(results_dir, "ori_ce_loss.npy")
    if not os.path.exists(baseline_path):
        print(f"  [skip] baseline not found for {target_language}")
        return

    baseline_ce_loss = np.load(baseline_path)
    language_slice = _rebuild_language_slice(
        all_languages, max_samples_per_language, len(baseline_ce_loss)
    )

    present_languages = [
        l for l in all_languages if language_slice[l].stop > language_slice[l].start
    ]

    n_langs = len(present_languages)
    n_configs = len(feature_configs)
    bar_width = 0.8 / n_configs
    x = np.arange(n_langs)

    target_display = LANGUAGE_DISPLAY_NAMES.get(
        target_language, target_language.upper()
    )

    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(max(10, n_langs * 0.9), 5))

        for config_idx, (start_idx, topk) in enumerate(feature_configs):
            ablated_path = os.path.join(
                results_dir, f"sae_ce_loss_all_layers_{start_idx}_{topk}.npy"
            )
            if not os.path.exists(ablated_path):
                print(f"  [skip] {ablated_path} not found.")
                continue

            ablated_ce_loss = np.load(ablated_path)  # (n_layers, n_total_texts)
            ablated_row = ablated_ce_loss[layer_idx]  # (n_total_texts,)
            ce_change = ablated_row - baseline_ce_loss  # (n_total_texts,)

            per_lang_delta = np.array(
                [ce_change[language_slice[l]].mean() for l in present_languages]
            )

            rank_label = (
                f"Rank #{start_idx + 1}"
                if topk == 1
                else f"Ranks #{start_idx + 1}–#{start_idx + topk}"
            )

            bar_colours = [
                (
                    "tab:blue"
                    if l == target_language
                    else FEATURE_CONFIG_COLOURS[
                        config_idx % len(FEATURE_CONFIG_COLOURS)
                    ]
                )
                for l in present_languages
            ]

            offset = (config_idx - (n_configs - 1) / 2) * bar_width
            bars = ax.bar(
                x + offset,
                per_lang_delta,
                width=bar_width,
                color=bar_colours,
                label=rank_label,
                edgecolor="white",
                linewidth=0.4,
            )

        display_labels = [
            LANGUAGE_DISPLAY_NAMES.get(l, l.upper()) for l in present_languages
        ]
        ax.set_xticks(x)
        ax.set_xticklabels(display_labels, rotation=35, ha="right", fontsize=11)

        tick_labels = ax.get_xticklabels()
        target_pos = (
            present_languages.index(target_language)
            if target_language in present_languages
            else -1
        )
        if target_pos >= 0:
            tick_labels[target_pos].set_color("tab:blue")
            tick_labels[target_pos].set_fontweight("bold")

        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_ylabel("ΔCE Loss (ablated - baseline)")
        ax.set_title(f"Ablation of {target_display} Features at Layer 20")
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.legend(loc="upper right", frameon=False)

        plt.tight_layout()
        stem = "ce_change_bar"
        fig.savefig(os.path.join(results_dir, f"{stem}.pdf"), bbox_inches="tight")
        fig.savefig(
            os.path.join(results_dir, f"{stem}.png"), bbox_inches="tight", dpi=150
        )
        plt.close(fig)

    print(f"  Saved: {results_dir}/{stem}.png")


def plot_ablation_results() -> None:
    """
    Plot the ablation results for the given configuration.
    """
    cfg = MainConfig.load("config.yaml").analyse
    ablation_cfg = cfg.ablation
    model_name = cfg.llm_path.split("/")[-1]
    layers = cfg.layers if cfg.layers else list(range(cfg.num_layers))

    with open(ablation_cfg.dataset_path, "r", encoding="utf-8") as f:
        language_set: set[str] = set()
        for line in f:
            language_set.add(json.loads(line)["lan"])
    all_languages = sorted(language_set)

    for target_language in ablation_cfg.target_languages:
        print(
            f"\nPlotting ablation results for {LANGUAGE_DISPLAY_NAMES.get(target_language, target_language)}..."
        )

        if len(layers) == 1:
            plot_single_layer_bar(
                model_name=model_name,
                target_language=target_language,
                feature_configs=ablation_cfg.feature_configs,
                all_languages=all_languages,
                max_samples_per_language=ablation_cfg.max_samples_per_language,
                layer_idx=0,
            )
        else:
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
