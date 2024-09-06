import pickle
import numpy as np
import pandas as pd
from typing import List
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


def plot_agreement(agreements: List[float], filename: Path = None):
    figure, ax = plt.subplots(nrows=1, ncols=1, figsize=(3, 3), dpi=240)

    df = pd.DataFrame({"agreements": agreements})
    sns.histplot(
        df,
        x="agreements",
        bins=20,
        binrange=(0, 100),
        color="black",
        # stat="probability",
        fill=False,
        linewidth=1,
        clip_on=False,
        alpha=0.8,
        ax=ax,
    )
    x_ticks = np.linspace(0, 100, 3)
    ax.set_xlim(x_ticks[0], x_ticks[-1])
    ax.set_xticks(x_ticks, labels=x_ticks.astype(int), fontsize=9)
    ax.set_xlabel("perfect agreement rate (%)", fontsize=10, labelpad=0)
    y_ticks = np.linspace(0, 10, 3)
    ax.set_ylim(y_ticks[0], y_ticks[-1])
    ax.set_yticks(y_ticks, labels=y_ticks.astype(int), fontsize=9)
    ax.set_ylabel("Count", fontsize=10, labelpad=0)
    ax.tick_params(axis="both", which="both", length=2, pad=1, width=0.8)
    sns.despine(ax=ax, trim=True)
    ax.set_title("Perfect agreement distribution\nover 20 queries", fontsize=10, pad=0)
    if filename is not None:
        filename.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(filename, dpi=240, bbox_inches="tight", pad_inches=0.02)
    else:
        plt.show()
    plt.close(figure)


def compute_agreement(y1: List[str], y2: List[str]) -> float:
    assert len(y1) == len(y2)
    count = sum([min(y1.count(option), y2.count(option)) for option in set(y1)])
    return count / len(y1)


def main():
    output_dir = Path("decisions/words")
    perfect_agreements = []

    unsure_rate = {"Y-N-U": [], "N-Y-U": []}

    for file in output_dir.glob("*.pkl"):
        qid = file.name[10:13]

        print(f"Query {qid}: {QUERIES[int(qid)]}")
        with open(file, "rb") as file:
            decisions = pickle.load(file)

        agreements = 0
        for community, decision in decisions.items():
            agreement = compute_agreement(decision[0], decision[1])
            if agreement != 1:
                print(
                    f"\tcommunity {community}\n"
                    f"\t\tagreement between Y-N and Y-N-U: {100*compute_agreement(decision[0], decision[2]):.0f}% \t"
                    f"\t\tnum. unsure: {decision[2].count('2')}\n"
                    f"\t\tagreement between N-Y and N-Y-U: {100*compute_agreement(decision[1], decision[3]):.0f}% \t"
                    f"\t\tnum. unsure: {decision[3].count('2')}"
                )
                unsure_rate["Y-N-U"].append(decision[2].count("2") / len(decision[2]))
                unsure_rate["N-Y-U"].append(decision[3].count("2") / len(decision[3]))
            else:
                agreements += 1

        perfect_agreement = 100 * agreements / len(decisions)
        print(
            f"\t{100*agreements / len(decisions):.0f}% of the communities have "
            f"perfect agreement between Y-N and N-Y.\n"
        )
        perfect_agreements.append(perfect_agreement)

    print(
        f"\nOverall perfect agreements: {np.mean(perfect_agreements):.0f}% "
        f"+/- {np.std(perfect_agreements):.02f}%"
    )
    print(
        f"Unsure rate\n"
        f'\tY-N and Y-N-U: {np.mean(unsure_rate["Y-N-U"])*100:.2f}% +/- {np.std(unsure_rate["Y-N-U"])*100:.2f}%\n'
        f'\tN-Y and N-Y-U: {np.mean(unsure_rate["N-Y-U"])*100:.02f}% +/- {np.std(unsure_rate["N-Y-U"])*100:.2f}%'
    )
    plot_agreement(
        perfect_agreements,
        filename=Path("figures/perfect_agreement_distribution.jpg"),
    )


if __name__ == "__main__":
    main()
