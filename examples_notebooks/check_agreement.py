import pickle
import numpy as np
import pandas as pd
from typing import List


def compute_agreement(y1: List[str], y2: List[str]) -> float:
    assert len(y1) == len(y2)
    count = sum([min(y1.count(option), y2.count(option)) for option in set(y1)])
    return count / len(y1)


def main():
    with open("decisions.pkl", "rb") as file:
        decisions = pickle.load(file)

    agreements = 0
    for community, decision in decisions.items():
        agreement = compute_agreement(decision[0], decision[1])
        if agreement != 1:
            print(
                f"community {community}\n"
                f"agreement between Y-N and Y-N-U: {100*compute_agreement(decision[0], decision[2]):.0f}% \t"
                f"num. unsure: {decision[2].count('2')}\n"
                f"agreement between N-Y and N-Y-U: {100*compute_agreement(decision[1], decision[3]):.0f}% \t"
                f"num. unsure: {decision[3].count('2')}\n"
            )
        else:
            agreements += 1

    print(
        f"{100*agreements / len(decisions):.0f}% of the communities have "
        f"perfect agreement between Y-N and N-Y."
    )


if __name__ == "__main__":
    main()
