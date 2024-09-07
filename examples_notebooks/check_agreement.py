import pickle
import numpy as np
import pandas as pd
from typing import List
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

import utils


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


def main():
    output_dir = Path("decisions/words")
    perfect_agreements = []

    unsure_rate = {"Y-N-U": [], "N-Y-U": []}

    for file in output_dir.glob("*.pkl"):
        qid = file.name[10:13]

        print(f"Query {qid}: {utils.QUERIES[int(qid)]}")
        with open(file, "rb") as file:
            decisions = pickle.load(file)

        agreements = 0
        for community, decision in decisions.items():
            agreement = utils.compute_agreement(decision[0], decision[1])
            if agreement != 100:
                print(
                    f"\tcommunity {community}\n"
                    f"\t\tagreement between Y-N and Y-N-U: {utils.compute_agreement(decision[0], decision[2]):.0f}% \t"
                    f"\t\tnum. unsure: {decision[2].count('2')}\n"
                    f"\t\tagreement between N-Y and N-Y-U: {utils.compute_agreement(decision[1], decision[3]):.0f}% \t"
                    f"\t\tnum. unsure: {decision[3].count('2')}"
                )
                unsure_rate["Y-N-U"].append(decision[2].count("2") / len(decision[2]))
                unsure_rate["N-Y-U"].append(decision[3].count("2") / len(decision[3]))
            else:
                agreements += 1

        perfect_agreement = 100 * agreements / len(decisions)
        print(
            f"\t{perfect_agreement:.0f}% of the communities have "
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
