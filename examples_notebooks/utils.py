import graphrag.index.graph.extractors.community_reports.schemas as schemas
import pandas as pd
import sys
from typing import List
from sklearn.metrics import confusion_matrix
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

QUERIES = {
    9: "Are there any common educational or career paths among the guests?",
    18: "How do guests generally perceive the impact of privacy laws on technology development?",
    19: "Do any tech leaders discuss the balance between innovation and ethical considerations?",
    26: "How do the predictions concerning technology trends differ between industry veterans and newcomers?",
    27: "How do tech leaders describe the influence of technology on everyday life?",
    34: "Are there conversations about digital divide and access to technology?",
    36: "Do the leaders speak about initiatives their companies have taken for societal benefits?",
    39: "Do any episodes focus on specific technological breakthroughs that have enhanced public services?",
    41: "Which guests share their experiences with tech initiatives in the education sector?",
    46: "Which episodes address the challenges faced in balancing user privacy with technological convenience?",
    49: "Which guests talk about the significance of company culture in driving technological advancements?",
    62: "How often do guests mention collaboration with other companies or industry rivals?",
    64: "What are some examples of industry-wide partnerships discussed in the podcast?",
    71: "Are there anecdotes about successful or unsuccessful pitches for tech-related funding?",
    75: "How do tech leaders describe the role of mentorship in their career journeys?",
    79: "What patterns in word choice are noticeable when leaders discuss industry challenges?",
    85: "How does the host's questioning style change when talking to leaders from different tech sectors?",
    97: "Retrieving data. Wait a few seconds and try to cut or copy again.",
    101: "What narrative structures do guests rely on when recounting the journey of their companies or own careers?",
    125: "What new markets or sectors do guests believe will be created by future technologies?",
}


def debug_mode() -> bool:
    """Return True if code being executed in PyCharm debug mode"""
    has_trace = hasattr(sys, "gettrace") and sys.gettrace() is not None
    has_breakpoint = sys.breakpointhook.__module__ != "sys"
    return has_trace or has_breakpoint


def compute_agreement(y1: List[str], y2: List[str]) -> float:
    """
    Return the level of agreement between y1 and y2.
    Agreement is defined as the percentage of decisions between the two raters are the same.

    perfect agreement: [0,0,0,0,0] vs [0,0,0,0,0], [1,1,1,1,1] vs [1,1,1,1,1]
    80% agreement: [0,0,0,0,1] vs [0,0,0,0,0], [0,0,0,0,1] vs [0,0,0,1,1]
    20% agreement: [0,0,0,0,1] vs [1,1,1,1,2]
    """
    assert len(y1) == len(y2)
    count = sum([min(y1.count(option), y2.count(option)) for option in set(y1)])
    return 100 * count / len(y1)


def cohen_kappa(y1: List[str], y2: List[str], labels: List[str]):
    """Return Cohen's Kappa inter-rater reliability score"""
    confusion = confusion_matrix(y1, y2, labels=labels)
    n_classes = confusion.shape[0]
    sum0 = np.sum(confusion, axis=0)
    sum1 = np.sum(confusion, axis=1)
    expected = np.outer(sum0, sum1) / np.sum(sum0)

    w_mat = np.ones([n_classes, n_classes], dtype=int)
    w_mat.flat[:: n_classes + 1] = 0

    k = np.sum(w_mat * confusion) / (np.sum(w_mat * expected) + 1e-8)
    return 1 - k


def plot_agreement(agreements: List[float], filename: Path = None):
    figure, ax = plt.subplots(nrows=1, ncols=1, figsize=(3, 2), dpi=240)

    df = pd.DataFrame({"agreements": agreements})
    sns.histplot(
        df,
        x="agreements",
        bins=20,
        binrange=(-1, 1),
        color="black",
        stat="probability",
        fill=False,
        linewidth=1,
        clip_on=False,
        alpha=0.8,
        ax=ax,
    )
    ax.axvline(
        x=0,
        color="black",
        alpha=0.3,
        linestyle="dotted",
        linewidth=1,
        zorder=-1,
    )
    x_ticks = np.linspace(-1.0, 1.0, 3)
    ax.set_xlim(x_ticks[0], x_ticks[-1])
    ax.set_xticks(x_ticks, labels=np.round(x_ticks, 1), fontsize=9)
    ax.set_xlabel("agreement score", fontsize=10, labelpad=0)
    y_ticks = np.linspace(0, 1, 3)
    ax.set_ylim(y_ticks[0], y_ticks[-1])
    ax.set_yticks(y_ticks, labels=(100 * y_ticks).astype(int), fontsize=9)
    ax.tick_params(axis="both", which="both", length=2, pad=1, width=0.8)
    sns.despine(ax=ax, trim=True)
    if filename is not None:
        filename.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(filename, dpi=240, bbox_inches="tight", pad_inches=0.02)
    else:
        plt.show()
    plt.close(figure)


def get_community_hierarchy(input_dir: str):
    node_df = pd.read_parquet(f"{input_dir}/create_final_nodes.parquet")

    community_df = (
        node_df.groupby([schemas.NODE_COMMUNITY, schemas.NODE_LEVEL])
        .agg({schemas.NODE_NAME: list})
        .reset_index()
    )
    community_levels = {}
    for _, row in community_df.iterrows():
        level = row[schemas.NODE_LEVEL]
        name = row[schemas.NODE_NAME]
        community = row[schemas.NODE_COMMUNITY]

        if community_levels.get(level) is None:
            community_levels[level] = {}
        community_levels[level][community] = name

    # get unique levels, sorted in ascending order
    levels = sorted(community_levels.keys())

    community_hierarchy = []

    for idx in range(len(levels) - 1):
        level = levels[idx]
        next_level = levels[idx + 1]
        current_level_communities = community_levels[level]
        next_level_communities = community_levels[next_level]

        for current_community in current_level_communities:
            current_entities = current_level_communities[current_community]

            # loop through next level's communities to find all the subcommunities
            entities_found = 0
            for next_level_community in next_level_communities:
                next_entities = next_level_communities[next_level_community]
                if set(next_entities).issubset(set(current_entities)):
                    community_hierarchy.append(
                        {
                            schemas.NODE_COMMUNITY: current_community,
                            schemas.COMMUNITY_LEVEL: level,
                            schemas.SUB_COMMUNITY: next_level_community,
                            schemas.SUB_COMMUNITY_SIZE: len(next_entities),
                        }
                    )

                    entities_found += len(next_entities)
                    if entities_found == len(current_entities):
                        break

    return pd.DataFrame(community_hierarchy)
