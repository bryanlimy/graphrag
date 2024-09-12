import pickle
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import sem
from tqdm import tqdm

import plot
import utils

DPI = 300
INPUT_DIR = Path("./inputs/podcast")
RESULT_DIR = Path("./results/dynamic")
PLOT_DIR = Path("figures/community_level")

plot.set_font()

TICK_FONTSIZE = 8
LABEL_FONTSIZE = 9
TITLE_FONTSIZE = 10


def plot_selected_community_level(
    x: list[str],
    y: np.ndarray,
    qid: int = None,
    filename: str = None,
):
    figure, ax = plt.subplots(nrows=1, ncols=1, figsize=(3, 3), dpi=DPI)

    if y.ndim == 1:
        ax.bar(x, 100 * y / np.sum(y), color="dodgerblue", width=0.5)
    else:
        values = np.mean(100 * y / np.sum(y, axis=1)[:, None], axis=0)
        error_bars = sem(values, axis=0)
        ax.bar(x, values, color="dodgerblue", width=0.5, clip_on=False)
        ax.errorbar(
            x,
            values,
            yerr=error_bars,
            fmt="none",
            color="black",
            linewidth=1,
            clip_on=True,
        )

    x_ticks = np.arange(len(x), dtype=int)
    ax.set_xlim(x_ticks[0] - 0.5, x_ticks[-1] + 0.5)
    plot.set_xticks(
        ax,
        ticks=x_ticks,
        tick_labels=x,
        label="Community level",
        tick_fontsize=TICK_FONTSIZE,
        label_fontsize=LABEL_FONTSIZE,
        label_pad=0,
    )

    y_ticks = np.linspace(0, 100, 3)
    ax.set_ylim(y_ticks[0], y_ticks[-1])
    plot.set_yticks(
        ax,
        ticks=y_ticks,
        tick_labels=y_ticks.astype(int),
        label="Percentage (%)",
        tick_fontsize=TICK_FONTSIZE,
        label_fontsize=LABEL_FONTSIZE,
        label_pad=0,
    )
    plot.set_ticks_params(ax)

    ax.set_title(
        (
            "Overall selected communities"
            if qid is None
            else f"Query {qid} selected communities"
        ),
        fontsize=LABEL_FONTSIZE,
        pad=2,
    )
    sns.despine(ax=ax, trim=True)

    plot.save_figure(
        figure,
        filename=PLOT_DIR / (f"query{qid:03d}.jpg" if filename is None else filename),
        dpi=4 * DPI,
    )


def get_distribution(qid: int, report_df: pd.DataFrame):
    result_dir = RESULT_DIR / f"query{qid:03d}"
    if not result_dir.is_dir():
        return None, None
    with open(result_dir / "result.pkl", "rb") as file:
        result = pickle.load(file)

    relevant_communities = result["selection_result"]["relevant_communities"]

    options = sorted(report_df.level.unique())

    # get community levels
    levels = []
    for community in relevant_communities:
        community_df = report_df[report_df["community"] == community]
        levels.append(community_df.iloc[0].level)

    counter = dict(Counter(levels))

    x, y = options, np.zeros(len(options), dtype=np.float32)
    for i, level in enumerate(options):
        y[i] = counter.get(level, 0)
    return x, y


def main():
    report_df = pd.read_parquet(INPUT_DIR / "create_final_community_reports.parquet")

    options, distribution = None, []
    for qid in tqdm(utils.QUERIES.keys()):
        x, y = get_distribution(qid, report_df=report_df)
        if x is not None and y is not None:
            plot_selected_community_level(x=x, y=y, qid=qid)
            if options is None:
                options = x
            distribution.append(y)
    distribution = np.array(distribution)
    plot_selected_community_level(
        x=options, y=distribution, filename="overall_selected_communities.jpg"
    )

    print(f"saved plots to {PLOT_DIR}")


if __name__ == "__main__":
    main()
