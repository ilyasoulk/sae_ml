import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_ranked_data(filepath: str, model_name: str) -> pd.DataFrame:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Could not find features file: {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    for layer, languages in data.items():
        layer_num = int(layer.replace("layer_", ""))
        for lan, features in languages.items():
            for rank, f in enumerate(features, start=1):
                u_target = f["u_target"]
                v_other = f["v_other"]

                rows.append({
                    "Model": model_name,
                    "Layer": layer_num,
                    "Language": lan,
                    "Rank": f"Rank #{rank}",
                    "Feature_ID": f["feature_idx"],
                    "U_Target": u_target,
                    "V_Other": v_other,
                    "Score (ν)": f["score"],
                    "Contamination_Ratio": v_other / (u_target + 1e-8),
                })
    return pd.DataFrame(rows)


def generate_plots():
    # 1. Load Configuration

    target_layer = 20
    out_dir = "results/multiling_sae"
    gemma_features_path = "top_features.json"
    custom_features_path = "top_features_custom.json"

    os.makedirs(out_dir, exist_ok=True)
    print(f"Generating plots for Layer {target_layer}...")

    # 2. Load Data
    df_gemma = load_ranked_data(gemma_features_path, "Gemma Scope")
    df_custom = load_ranked_data(custom_features_path, "Custom SAE (Aya)")

    # Filter for the target layer
    df_gemma_layer = df_gemma[df_gemma["Layer"] == target_layer]
    df_custom_layer = df_custom[df_custom["Layer"] == target_layer]

    if df_custom_layer.empty:
        raise ValueError(
            f"No data found for Layer {target_layer} in {cfg.custom_features_path}."
        )

    sns.set_theme(style="ticks", context="paper", font_scale=1.4)

    # ==========================================
    # PLOT 1: Hero Feature Reproduction (Custom SAE Only, Ranks 1-4)
    # ==========================================
    print("Generating Hero Feature Reproduction plot...")
    ranks_to_plot = ["Rank #1", "Rank #2", "Rank #3", "Rank #4"]
    df_reproduction = df_custom_layer[df_custom_layer["Rank"].isin(ranks_to_plot)]

    rank_palette = {
        "Rank #1": "#3175a1",  # Blue
        "Rank #2": "#e1812c",  # Orange
        "Rank #3": "#44923e",  # Green
        "Rank #4": "#c03d3e",  # Red
    }

    plt.figure(figsize=(14, 6))
    ax1 = sns.barplot(
        data=df_reproduction,
        x="Language",
        y="Score (ν)",
        hue="Rank",
        palette=rank_palette,
        edgecolor="black",
        linewidth=1.5,
    )

    plt.title(f"Layer {target_layer}", weight="bold", fontsize=18)
    plt.xlabel("Languages", weight="bold", fontsize=16)
    plt.ylabel("Values of $\\nu$", weight="bold", fontsize=16)

    sns.move_legend(
        ax1,
        "upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=4,
        title=None,
        frameon=False,
        fontsize=12,
    )

    sns.despine()
    plt.tight_layout()

    reproduction_path = os.path.join(
        out_dir, f"layer_{target_layer}_hero_feature_reproduction.png"
    )
    plt.savefig(reproduction_path, dpi=300, bbox_inches="tight")
    plt.close()

    # ==========================================
    # PLOT 2: Contamination Ratio Purity Comparison (Custom vs Gemma, Rank #1 Only)
    # ==========================================
    if not df_gemma_layer.empty:
        print("Generating Purity Comparison plot...")
        df_combined = pd.concat([df_gemma_layer, df_custom_layer], ignore_index=True)
        df_top1 = df_combined[df_combined["Rank"] == "Rank #1"]

        comparison_palette = {"Gemma Scope": "#e74c3c", "Custom SAE (Aya)": "#3498db"}

        plt.figure(figsize=(12, 6))
        ax2 = sns.barplot(
            data=df_top1,
            x="Language",
            y="Contamination_Ratio",
            hue="Model",
            palette=comparison_palette,
            edgecolor="black",
            linewidth=1.5,
        )

        plt.title(
            f"Rank #1 Feature Contamination Ratio per Language - Layer {target_layer}",
            weight="bold",
            fontsize=16,
        )
        plt.xlabel("Languages", weight="bold", fontsize=14)
        plt.ylabel(
            "Contamination Ratio ($v_{other} / u_{target}$)", weight="bold", fontsize=14
        )

        plt.axhline(0, color="black", linewidth=1.5)
        plt.legend(title="Model (Lower Bar = Purer Feature)")
        sns.despine()

        plt.tight_layout()
        purity_path = os.path.join(
            out_dir, f"layer_{target_layer}_purity_comparison.png"
        )
        plt.savefig(purity_path, dpi=300)
        plt.close()
    else:
        print(
            f"Warning: No Gemma Scope data found for Layer {target_layer}. Skipping Plot 2."
        )

    print(f"Successfully saved all plots to '{out_dir}/'")


if __name__ == "__main__":
    generate_plots()

