"""
plot_results.py
===============
Run this on your LOCAL machine after downloading eval_results.json
and eval_turns.csv from the server.

    pip install matplotlib numpy pandas seaborn
    python plot_results.py eval_results.json eval_turns.csv

Generates 4 PDF figures ready for the IEEE paper.
"""

import json, sys, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── colour palette (matches IEEE paper style) ──────────────────────────────
COLORS = {
    "valence":       "#2196F3",
    "arousal":       "#FF9800",
    "stress":        "#F44336",
    "clarity":       "#4CAF50",
    "neutral":       "#9E9E9E",
    "flow":          "#4CAF50",
    "deescalation":  "#F44336",
    "clarification": "#FF9800",
}
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 300,
})


def load_data(json_path, csv_path):
    with open(json_path) as f:
        metrics = json.load(f)

    turns = metrics["raw_turns"]
    return metrics["metrics"], turns


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 1 (Paper Fig 2):  Emotional Trajectory  –  raw vs EMA
# ═══════════════════════════════════════════════════════════════════════════
def plot_emotional_trajectory(turns, out="fig2_emotional_trajectory.pdf"):
    """
    Shows raw and EMA for valence and stress across turns.
    Marks interaction mode switches with background shading.
    """
    # Use first session only for clarity
    first_session = turns[0]["session_id"]
    session_turns = [t for t in turns if t["session_id"] == first_session]

    x = [t["turn"] for t in session_turns]
    val_raw  = [t["valence_raw"]  for t in session_turns]
    val_ema  = [t["valence_ema"]  for t in session_turns]
    str_raw  = [t["stress_raw"]   for t in session_turns]
    str_ema  = [t["stress_ema"]   for t in session_turns]
    modes    = [t["interaction_mode"] for t in session_turns]

    fig, axes = plt.subplots(2, 1, figsize=(6.5, 4.5), sharex=True)

    mode_colors = {
        "deescalation":  "#FFCDD2",
        "clarification": "#FFE0B2",
        "flow":          "#C8E6C9",
        "neutral":       "#F5F5F5",
    }

    for ax, raw, ema, label, color in zip(
        axes,
        [val_raw, str_raw],
        [val_ema,  str_ema],
        ["Valence", "Stress"],
        [COLORS["valence"], COLORS["stress"]],
    ):
        # Background mode shading
        for i, mode in enumerate(modes):
            ax.axvspan(x[i] - 0.5, x[i] + 0.5, color=mode_colors.get(mode, "#F5F5F5"), alpha=0.4)

        ax.plot(x, raw, "o--", color=color, alpha=0.5, linewidth=1.2,
                markersize=4, label=f"{label} (raw)")
        ax.plot(x, ema, "o-",  color=color, linewidth=2.0,
                markersize=5, label=f"{label} (EMA)")
        ax.axhline(0, color="gray", linestyle=":", linewidth=0.8)
        ax.set_ylabel(label)
        ax.legend(loc="upper right", framealpha=0.8)
        ax.set_ylim(-1.05, 1.05)
        ax.grid(axis="y", alpha=0.3)

    axes[1].set_xlabel("Conversational Turn")

    # Mode legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=v, alpha=0.6, label=k)
                       for k, v in mode_colors.items()]
    fig.legend(handles=legend_elements, loc="lower center", ncol=4,
               fontsize=8, title="Interaction Mode", bbox_to_anchor=(0.5, -0.02))

    fig.suptitle("Emotional Trajectory: Raw vs. EMA-Smoothed State", y=1.01)
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    print(f"✅  Saved: {out}")
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 2 (Paper Fig 3):  Interaction Mode & Emotion Distribution
# ═══════════════════════════════════════════════════════════════════════════
def plot_distributions(metrics, out="fig3_distributions.pdf"):
    mode_data  = metrics["interaction_mode_distribution"]
    emo_data   = metrics["text_emotion_distribution"]

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.5))

    # ── Left: Interaction Mode Pie ──────────────────────────────────
    mode_labels = list(mode_data.keys())
    mode_vals   = [mode_data[m]["count"] for m in mode_labels]
    mode_cols   = [COLORS.get(m, "#9E9E9E") for m in mode_labels]
    wedges, texts, autotexts = axes[0].pie(
        mode_vals, labels=mode_labels, colors=mode_cols,
        autopct="%1.1f%%", startangle=90,
        textprops={"fontsize": 9}
    )
    axes[0].set_title("Interaction Mode Distribution")

    # ── Right: Top-8 Text Emotion Bar ──────────────────────────────
    top_emos  = list(emo_data.keys())[:8]
    top_pcts  = [emo_data[e]["pct"] for e in top_emos]
    bar_colors = plt.cm.tab10(np.linspace(0, 1, len(top_emos)))

    bars = axes[1].barh(top_emos[::-1], top_pcts[::-1], color=bar_colors[::-1])
    axes[1].set_xlabel("Frequency (%)")
    axes[1].set_title("Dominant Text Emotion Distribution")
    axes[1].set_xlim(0, max(top_pcts) * 1.25)
    for bar, pct in zip(bars, top_pcts[::-1]):
        axes[1].text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                     f"{pct:.1f}%", va="center", fontsize=8)
    axes[1].grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    print(f"✅  Saved: {out}")
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 3 (Paper Fig 4):  Trend Stability Score Bar Chart
# ═══════════════════════════════════════════════════════════════════════════
def plot_tss(metrics, out="fig4_tss.pdf"):
    tss = metrics["trend_stability_score"]
    axes_labels = list(tss.keys())
    tss_vals    = [tss[a] for a in axes_labels]
    bar_colors  = [COLORS.get(a, "#9E9E9E") for a in axes_labels]

    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    bars = ax.bar(axes_labels, tss_vals, color=bar_colors, edgecolor="white", linewidth=0.5)
    ax.set_ylim(0, 1.1)
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=0.8)
    ax.set_ylabel("Trend Stability Score (TSS)")
    ax.set_title("EMA Smoothing Effectiveness per Psychological Axis")
    for bar, val in zip(bars, tss_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    print(f"✅  Saved: {out}")
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 4 (Paper Fig 5):  Memory Importance Distribution
# ═══════════════════════════════════════════════════════════════════════════
def plot_memory(metrics, raw_memory_decisions, out="fig5_memory.pdf"):
    stored  = [d["importance"] for d in raw_memory_decisions if d["stored"]]
    skipped = [d["importance"] for d in raw_memory_decisions if not d["stored"]]

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2))

    # ── Left: Histogram of importance scores ──────────────────────
    if stored:
        axes[0].hist(stored,  bins=10, alpha=0.7, color="#4CAF50", label=f"Stored  (n={len(stored)})")
    if skipped:
        axes[0].hist(skipped, bins=10, alpha=0.7, color="#F44336", label=f"Skipped (n={len(skipped)})")
    axes[0].axvline(0.6, color="black", linestyle="--", linewidth=1.2, label="Threshold=0.6")
    axes[0].set_xlabel("Importance Score")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Memory Importance Score Distribution")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    # ── Right: Storage rate across turns (running average) ────────
    decisions = raw_memory_decisions
    if decisions:
        stored_flags = [1 if d["stored"] else 0 for d in decisions]
        running_msr  = np.cumsum(stored_flags) / np.arange(1, len(stored_flags)+1)
        turn_indices = list(range(1, len(running_msr)+1))
        axes[1].plot(turn_indices, running_msr, color="#2196F3", linewidth=2)
        axes[1].axhline(0.4, color="gray", linestyle=":", linewidth=1, label="Target MSR~0.4")
        axes[1].set_xlabel("Decision #")
        axes[1].set_ylabel("Cumulative Storage Rate")
        axes[1].set_title("Memory Storage Rate Over Turns")
        axes[1].set_ylim(0, 1.05)
        axes[1].legend(fontsize=8)
        axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    print(f"✅  Saved: {out}")
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════
# PRINT PAPER-READY TABLE VALUES
# ═══════════════════════════════════════════════════════════════════════════
def print_paper_tables(metrics):
    print("\n" + "═"*60)
    print("  PAPER-READY VALUES  (copy these into your LaTeX tables)")
    print("═"*60)

    tss = metrics["trend_stability_score"]
    print("\n── Table: Trend Stability Scores ──")
    for ax, v in tss.items():
        print(f"  {ax:10s}: TSS = {v:.3f}")

    m = metrics["memory"]
    print(f"\n── Table: Memory System ──")
    print(f"  Memory Storage Rate (MSR)       : {m['storage_rate_msr']:.2%}")
    print(f"  Total memories stored           : {m['stored_count']}")
    print(f"  Avg importance score (stored)   : {m['avg_importance_stored']:.3f}")
    print(f"  Avg importance score (skipped)  : {m['avg_importance_skipped']:.3f}")

    print(f"\n── Table: Interaction Mode Distribution ──")
    for mode, v in metrics["interaction_mode_distribution"].items():
        print(f"  {mode:15s}: {v['count']} turns ({v['pct']}%)")

    ax_stats = metrics["psychological_axis_stats"]
    print(f"\n── Table: Psychological Axis Statistics ──")
    print(f"  {'Axis':10s}  {'Mean':>8}  {'Std':>7}  {'Min':>7}  {'Max':>7}  {'TSS':>7}")
    for ax, v in ax_stats.items():
        print(f"  {ax:10s}  {v['mean_raw']:+8.3f}  {v['std_raw']:7.3f}  {v['min_raw']:7.3f}  {v['max_raw']:7.3f}  {v['tss']:7.4f}")

    sh = metrics["emotional_shift"]
    print(f"\n── Table: Emotional Shift Detection ──")
    print(f"  Shift rate   : {sh['shift_rate']:.2%}")
    print(f"  Total shifts : {sh['total_shifts']}")
    print(f"  By dimension : {sh['by_dimension']}")

    tm = metrics["trend_mode_usage"]
    print(f"\n── Table: Trend Mode Usage ──")
    print(f"  Instant mode : {tm['distribution']['instant']}%")
    print(f"  Trend mode   : {tm['distribution']['trend']}%")
    print(f"  Avg confidence: {tm['avg_confidence']:.4f}")
    print("═"*60 + "\n")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    json_path = sys.argv[1] if len(sys.argv) > 1 else "eval_results.json"
    csv_path  = sys.argv[2] if len(sys.argv) > 2 else "eval_turns.csv"

    if not os.path.exists(json_path):
        print(f"Error: {json_path} not found. Run eval/save endpoint first.")
        sys.exit(1)

    metrics, turns = load_data(json_path, csv_path)

    with open(json_path) as f:
        full_data = json.load(f)
    raw_memory_decisions = full_data.get("raw_memory_decisions", [])

    print_paper_tables(metrics)
    plot_emotional_trajectory(turns)
    plot_distributions(metrics)
    plot_tss(metrics)
    plot_memory(metrics, raw_memory_decisions)

    print("\n✅ All 4 figures generated. Add them to your paper with:")
    print("   \\includegraphics{fig2_emotional_trajectory.pdf}")
    print("   \\includegraphics{fig3_distributions.pdf}")
    print("   \\includegraphics{fig4_tss.pdf}")
    print("   \\includegraphics{fig5_memory.pdf}")