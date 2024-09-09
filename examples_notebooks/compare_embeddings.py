import asyncio
import os
import pickle
from math import ceil
from math import floor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import MultipleLocator
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


import plot
import utils
from compute_community_embedding import embed_text

DPI = 300
INPUT_DIR = Path("./inputs/podcast")
RESULT_DIR = Path("./results/dynamic")
PLOT_DIR = Path("figures/embeddings")

plot.set_font()

TICK_FONTSIZE = 8
LABEL_FONTSIZE = 9
TITLE_FONTSIZE = 10


def get_embeddings():
    report_df = pd.read_parquet(INPUT_DIR / "create_final_community_reports.parquet")
    embedding_df = pd.read_parquet(INPUT_DIR / "community_embedding.parquet")
    return report_df, embedding_df


def plot_pairwise_similarity(embedding_df: pd.DataFrame, filename: Path):
    similarities = cosine_similarity(embedding_df.embedding.to_list())
    similarities = similarities.astype(np.float32)
    similarities = similarities[np.triu_indices_from(similarities, k=1)]

    df = pd.DataFrame({"similarity": similarities})
    figure, ax = plt.subplots(nrows=1, ncols=1, figsize=(3, 3), dpi=DPI)
    x_ticks = np.linspace(0, 1.0, 3)

    sns.histplot(
        df,
        x="similarity",
        bins=40,
        binrange=(x_ticks[0], x_ticks[-1]),
        color="black",
        stat="percent",
        fill=False,
        linewidth=1,
        clip_on=False,
        alpha=0.8,
        ax=ax,
    )
    ax.set_xlim(x_ticks[0], x_ticks[-1])
    ax.set_xticks(x_ticks, labels=np.round(x_ticks, 1), fontsize=9)
    ax.set_xlabel("Cosine similarity", fontsize=10, labelpad=0)
    y_ticks = np.linspace(0, 50, 3)
    ax.set_ylim(y_ticks[0], y_ticks[-1])
    ax.set_yticks(y_ticks, labels=y_ticks.astype(int), fontsize=9)
    ax.tick_params(axis="both", which="both", length=2, pad=1, width=0.8)
    ax.set_title("Community Embeddings\nPairwise Similarity", fontsize=10, pad=2)
    sns.despine(ax=ax, trim=True)

    filename.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(filename, dpi=DPI, bbox_inches="tight", pad_inches=0.02)
    plt.close(figure)


def get_community_decisions(qid: int) -> (list[str], np.ndarray):
    """Return decisions results in the same order as community"""
    result_dir = RESULT_DIR / f"query{qid:03d}"
    with open(result_dir / "result.pkl", "rb") as file:
        result = pickle.load(file)

    # extract decision results for each community that was queried
    communities = list(result["selection_result"]["outputs"].keys())
    decisions = []
    for community in communities:
        decision = list(result["selection_result"]["outputs"][community].values())
        decision = [int(v[0]) for l in decision for v in l]
        options, counts = np.unique(decision, return_counts=True)
        decisions.append(options[np.argmax(counts)])
    # communities = np.array(communities, dtype=int)
    decisions = np.array(decisions, dtype=int)
    # communities = report_df.loc[report_df["level"] == 0]["community"].to_list()
    # assert np.array_equal(
    #     communities,
    #     list(result["selection_result"]["outputs"].keys())[: len(communities)],
    # )
    # decisions = result["selection_result"]["decisions"][: len(communities)]
    # decisions = np.array([int(d) for d in decisions], dtype=int)
    return communities, decisions


def plot_community_decision(
    similarities: np.ndarray,
    decisions: np.ndarray,
    filename: Path,
    qid: int = None,
    title: str = None,
    scale_width: bool = True,
    add_p_value: bool = False,
):
    options = [1, 2, 3, 4, 5]
    data, widths = [], []
    for decision in options:
        idx = np.where(decisions == decision)[0]
        if len(idx) == 0:
            data.append([])
        else:
            data.append(similarities[idx])
        widths.append((len(idx) / len(decisions)) + 0.1)

    figure, ax = plt.subplots(nrows=1, ncols=1, figsize=(3, 2.5), dpi=DPI)

    x_ticks = np.arange(len(options), dtype=int)
    y_ticks = np.linspace(
        0.1 * floor(10 * np.min(similarities)),
        0.1 * ceil(10 * np.max(similarities)) if not add_p_value else 1.0,
        num=3,
    )

    bp = ax.boxplot(
        data,
        # notch=True,
        vert=True,
        positions=x_ticks,
        widths=widths if scale_width else 0.4,
        showfliers=False,
        showmeans=True,
        flierprops={"marker": "o", "markersize": 2, "alpha": 0.5},
        capprops={"clip_on": False},
        whiskerprops={"clip_on": False},
        meanprops={
            # "marker": "o",
            "markersize": 3,
            "markerfacecolor": "gold",
            "markeredgecolor": "none",
        },
        medianprops={
            "color": "red",
            # "zorder": 1,
            "solid_capstyle": "projecting",
        },
    )

    scatter_kw = {
        "s": 7,
        "marker": ".",
        "alpha": 0.3,
        "zorder": 0,
        "facecolors": "none",
        "edgecolors": "dodgerblue",
        "clip_on": False,
    }
    for i, option in enumerate(options):
        idx = np.where(decisions == option)[0]
        y = similarities[idx]
        x = np.random.normal(i, 0.06, size=len(y))
        ax.scatter(x, y, **scatter_kw)

    if add_p_value:
        max_height = max(np.max(data[0]), np.max(data[1])) + 0.05
        y_offset = 0.008
        plot.add_p_value(
            ax,
            x0=x_ticks[1],
            x1=x_ticks[2],
            y=max_height,
            y_offset=y_offset,
            array1=data[1],
            array2=data[2],
            fontsize=TICK_FONTSIZE,
        )
        plot.add_p_value(
            ax,
            x0=x_ticks[2],
            x1=x_ticks[3],
            y=max_height,
            y_offset=y_offset,
            array1=data[2],
            array2=data[3],
            fontsize=TICK_FONTSIZE,
        )
        plot.add_p_value(
            ax,
            x0=x_ticks[3],
            x1=x_ticks[4],
            y=max_height,
            y_offset=y_offset,
            array1=data[3],
            array2=data[4],
            fontsize=TICK_FONTSIZE,
        )
        for i in [1, 2, 3, 4]:
            text_height = plot.add_p_value(
                ax,
                x0=x_ticks[0],
                x1=x_ticks[i],
                y=max_height,
                y_offset=y_offset,
                array1=data[0],
                array2=data[i],
                fontsize=TICK_FONTSIZE,
            )
            max_height = max(max_height, text_height) + 0.02

    ax.set_xlim(x_ticks[0] - 0.6, x_ticks[-1] + 0.6)
    plot.set_xticks(
        ax,
        ticks=x_ticks,
        tick_labels=options,
        label="LLM rating (GPT-4o)",
        tick_fontsize=TICK_FONTSIZE,
        label_fontsize=LABEL_FONTSIZE,
        label_pad=0,
    )

    ax.set_ylim(y_ticks[0], y_ticks[-1])
    plot.set_yticks(
        ax,
        ticks=y_ticks,
        tick_labels=[f"{y:.2f}" for y in y_ticks],
        label="Cosine similarity",
        tick_fontsize=TICK_FONTSIZE,
        label_fontsize=LABEL_FONTSIZE,
        label_pad=0,
    )
    # ax.yaxis.set_minor_locator(MultipleLocator(0.01))

    plot.set_ticks_params(ax, length=2)
    ax.tick_params(axis="x", which="major", length=0)
    sns.despine(ax=ax, bottom=True, trim=True)

    if title is None:
        title = f"Cosine similarity between\nquery {qid} and community embeddings"
    ax.set_title(title, fontsize=LABEL_FONTSIZE, pad=6, linespacing=1)

    figure.tight_layout()
    plot.save_figure(figure, filename=filename, dpi=4 * DPI)


def compare_community_embedding_with_query(
    qid: int,
    embedding_df: pd.DataFrame,
):
    result_dir = RESULT_DIR / f"query{qid:03d}"
    if not result_dir.exists():
        return None, None

    # get communities that were queried and their decisions
    communities, decisions = get_community_decisions(qid)

    query_embedding = embed_text(utils.QUERIES[qid])

    # get community embeddings in the order of decisions
    community_embeddings = []
    for community in communities:
        embedding = embedding_df.loc[embedding_df["community"] == community]
        community_embeddings.append(embedding.iloc[0].embedding.astype(np.float32))

    # compare similarity between query and each community
    similarities = cosine_similarity([query_embedding] + community_embeddings)
    similarities = similarities[0, 1:].astype(np.float32)

    plot_community_decision(
        similarities=similarities,
        decisions=decisions,
        filename=PLOT_DIR / f"query{qid:03d}_community_decision.jpg",
        qid=qid,
    )

    return similarities, decisions


def main():
    report_df, embedding_df = get_embeddings()

    plot_pairwise_similarity(
        embedding_df,
        filename=PLOT_DIR / "pairwise_similarity.jpg",
    )

    similarities, decisions = [], []
    for qid in tqdm(utils.QUERIES.keys()):
        similarity, decision = compare_community_embedding_with_query(
            qid=qid,
            embedding_df=embedding_df,
        )
        if similarity is not None and decision is not None:
            similarities.extend(similarity)
            decisions.extend(decision)
    similarities = np.array(similarities, dtype=np.float32)
    decisions = np.array(decisions, dtype=np.float32)

    np.save("similarities.npy", similarities, allow_pickle=False)
    np.save("decisions.npy", decisions, allow_pickle=False)

    similarities = np.load("similarities.npy", allow_pickle=False)
    decisions = np.load("decisions.npy", allow_pickle=False)

    plot_community_decision(
        similarities=similarities,
        decisions=decisions,
        filename=PLOT_DIR / f"community_decision_similarity.jpg",
        title="Cosine similarity between\n query and community embeddings",
        scale_width=True,
        add_p_value=True,
    )

    print(f"saved plots to {PLOT_DIR}")


if __name__ == "__main__":
    main()
