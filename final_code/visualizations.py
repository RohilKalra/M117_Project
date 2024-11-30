import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np


def create_visualizations(analysis, timestamp, output_dir):
    """Create and save visualizations of the results"""
    plt.style.use("seaborn-v0_8-darkgrid")
    sns.set_palette("husl")

    # Get number of positions
    positions = list(analysis["avg_similarity_by_position"].keys())
    n_positions = len(positions)

    # Adjust figure size based on number of positions
    base_width = 15
    base_height = 12
    scaling_factor = max(1, n_positions / 6)
    fig = plt.figure(
        figsize=(base_width * scaling_factor, base_height * scaling_factor)
    )

    # 1. Cosine Similarity by Position Plot (Bar)
    plt.subplot(2, 2, 1)
    similarities = list(analysis["avg_similarity_by_position"].values())

    bars = plt.bar(positions, similarities)
    plt.title("Average Cosine Similarity by Poison Position")
    plt.xlabel("Poison Position")
    plt.ylabel("Average Cosine Similarity")

    # Force integer x-axis ticks
    plt.gca().xaxis.set_major_locator(plt.MultipleLocator(1))

    font_size = min(10, 120 / n_positions)
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.3f}",
            ha="center",
            va="bottom",
            fontsize=font_size,
        )

    # 2. Impact Strength Plot
    plt.subplot(2, 2, 2)
    impact_strengths = list(analysis["impact_strength"].values())
    plt.plot(positions, impact_strengths, "ro-", linewidth=2, markersize=8)
    plt.title("Impact Strength by Poison Position")
    plt.xlabel("Poison Position")
    plt.ylabel("Embedding Distance")
    plt.grid(True, linestyle="--", alpha=0.7)

    # Force integer x-axis ticks
    plt.gca().xaxis.set_major_locator(plt.MultipleLocator(1))

    # 3. Similarity Trend
    plt.subplot(2, 2, 3)
    plt.plot(positions, similarities, "o-", linewidth=2, markersize=8)
    plt.title("Similarity Trend Across Positions")
    plt.xlabel("Poison Position")
    plt.ylabel("Average Cosine Similarity")
    plt.grid(True, linestyle="--", alpha=0.7)

    # Force integer x-axis ticks
    plt.gca().xaxis.set_major_locator(plt.MultipleLocator(1))

    # 4. Propagation Patterns Heatmap
    plt.subplot(2, 2, 4)
    propagation_data = []
    max_steps = 0

    # Prepare propagation data
    for pos in positions:
        if pos in analysis["propagation_patterns"]:
            pattern = analysis["propagation_patterns"][pos]
            propagation_data.append(pattern)
            max_steps = max(max_steps, len(pattern))

    if propagation_data:
        # Pad shorter patterns with NaN
        padded_data = np.array(
            [
                pattern + [np.nan] * (max_steps - len(pattern))
                for pattern in propagation_data
            ]
        )

        # Adjust font size for annotations based on matrix size
        annot_fontsize = min(10, 200 / (n_positions * max_steps))

        sns.heatmap(
            padded_data,
            cmap="YlOrRd",
            mask=np.isnan(padded_data),
            annot=True,
            fmt=".2f",
            xticklabels=[f"Step {i+1}" for i in range(max_steps)],
            yticklabels=[f"Pos {i}" for i in positions],
            annot_kws={"size": annot_fontsize},
        )
        plt.title("Propagation Patterns\n(Similarity at each step after poison)")

        # Rotate x-axis labels if many steps
        if max_steps > 6:
            plt.xticks(rotation=45)
    else:
        plt.text(0.5, 0.5, "No propagation data available", ha="center", va="center")
        plt.title("Propagation Patterns")

    # Adjust layout with larger spacing when scaling up
    plt.tight_layout(pad=2.0 * scaling_factor)

    # Save with adjusted DPI for larger numbers of positions
    dpi = min(300, 100 * scaling_factor)
    plt.savefig(
        os.path.join(output_dir, f"similarity_analysis_{timestamp}.png"),
        dpi=dpi,
        bbox_inches="tight",
    )
    plt.close()