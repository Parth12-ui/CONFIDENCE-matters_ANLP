"""
Figure for the main pushback result: accuracy over turns for unwarranted criticism (true
claims answered correctly at turn 0) and warranted criticism (false claims answered
wrongly at turn 0), criticism track vs neutral re-ask control, one panel per model.
Reads the accuracy_by_turn.csv files written by analyze_pushback.py.

Usage:
  python src/evaluation/plot_pushback.py --out fig_pushback.pdf \
      --panel "Llama-3.1-8B" results/o1_paired_v3/pushback_t2/accuracy_by_turn.csv \
      --panel "Qwen2.5-7B" results/o1_qwen25_7b_v4_partial/pushback_t6/accuracy_by_turn.csv
"""

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

STYLE = {
    ("unwarranted", "negative"): ("#d95f02", "-", "unwarranted, criticism"),
    ("unwarranted", "control"): ("#d95f02", ":", "unwarranted, control"),
    ("warranted", "negative"): ("#1b9e77", "-", "warranted, criticism"),
    ("warranted", "control"): ("#1b9e77", ":", "warranted, control"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--panel", nargs=2, action="append", metavar=("TITLE", "CSV"), required=True)
    args = ap.parse_args()

    fig, axes = plt.subplots(1, len(args.panel), figsize=(3.3 * len(args.panel), 2.7), sharey=True, squeeze=False)
    for ax, (title, path) in zip(axes[0], args.panel):
        t = pd.read_csv(path)
        for (group, track), (color, ls, label) in STYLE.items():
            g = t[(t["group"] == group) & (t["track"] == track)].sort_values("turn")
            if g.empty:
                continue
            n = int(g["n"].iloc[0])
            ax.plot(g["turn"], 100 * g["accuracy"], ls, color=color, marker="o", ms=3, label=label)
            if track == "negative":
                ax.fill_between(g["turn"], 100 * g["lo"], 100 * g["hi"], color=color, alpha=0.15, lw=0)
        nu = int(t[(t["group"] == "unwarranted") & (t["turn"] == 0)]["n"].iloc[0])
        nw = int(t[(t["group"] == "warranted") & (t["turn"] == 0)]["n"].iloc[0])
        ax.set_title(f"{title} (n = {nu} / {nw})", fontsize=9)
        ax.set_xlabel("turn (0 = neutral question)", fontsize=8)
        ax.set_ylim(-5, 105)
        ax.set_xticks(sorted(t["turn"].unique()))
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel("accuracy (%)", fontsize=8)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=7, frameon=False)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(args.out)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
