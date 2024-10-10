import pickle
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import cohen_kappa_score

from run import QUERIES

OUTPUT_DIR = Path("results")


def load_result(output_dir: Path):
    results = {}
    for qid, query in QUERIES.items():
        with open(output_dir / f"qid{qid:03d}.pkl", "rb") as file:
            result = pickle.load(file)
        # sort ratings by community ID
        result["ratings"] = dict(sorted(result["ratings"].items()))
        results[qid] = result
    return results


def get_rating_distribution(ratings: list[int]):
    return dict(sorted(Counter(ratings).items()))


def calculate_agreement(ratings1: list[int], ratings2: list[int]):
    assert len(ratings1) == len(ratings2)
    # agreement = cohen_kappa_score(
    #     y1=ratings1,
    #     y2=ratings2,
    #     labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    # )
    agreement = np.sum(np.array(ratings1) == np.array(ratings2)) / len(ratings1)
    return agreement


def calculate_retrieval_rate(
    ground_truth: list[int], ratings: list[int], threshold: int = 1
):
    assert len(ground_truth) == len(ratings)
    g_count, r_count = 0, 0
    for i in range(len(ground_truth)):
        if ground_truth[i] >= threshold:
            g_count += 1
            if ratings[i] >= threshold:
                r_count += 1
    return r_count / g_count if g_count > 0 else np.nan


def compare(method1: Path, method2: Path):
    result1 = load_result(method1)
    result2 = load_result(method2)

    agreements = []
    retrieval_rates = []
    for qid in QUERIES.keys():
        ratings1 = list(result1[qid]["ratings"].values())
        ratings2 = list(result2[qid]["ratings"].values())
        agreement = calculate_agreement(ratings1=ratings1, ratings2=ratings2)
        agreements.append(agreement)
        retrieval_rate = calculate_retrieval_rate(
            ground_truth=ratings1, ratings=ratings2, threshold=1
        )
        if not np.isnan(retrieval_rate):
            retrieval_rates.append(retrieval_rate)
        # print(
        #     f"Query ({qid})"
        #     f"\tAgreement: {agreement:.04f}"
        #     f"\tRetrieval rate: {retrieval_rate:.04f}\n"
        #     f"{method1}: {get_rating_distribution(ratings1)}\n"
        #     f"{method2}: {get_rating_distribution(ratings2)}\n"
        # )

    # print(
    #     f"Average agreement between {method1} vs {method2}: "
    #     f"{np.mean(agreements) * 100:.02f}% +/- {np.std(agreements) * 100:.02f}"
    # )
    print(
        f"Average retrieval rate between {method2} against {method1}: "
        f"{np.mean(retrieval_rates) * 100:.02f}% +/- {np.std(retrieval_rates) * 100:.02f}\n\n"
    )


def main():
    # compare(method1="gpt-4o-full_content", method2="gpt-4o-summary")
    print("")
    compare(
        method1=OUTPUT_DIR / "0-to-10-reason-rating" / "gpt-4o-full_content",
        method2=OUTPUT_DIR / "0-to-10-rating" / "gpt-4o-full_content",
    )
    compare(
        method1=OUTPUT_DIR / "0-to-10-reason-rating" / "gpt-4o-full_content",
        method2=OUTPUT_DIR / "0-to-10-rating" / "gpt-4o-mini-full_content",
    )
    compare(
        method1=OUTPUT_DIR / "0-to-10-rating" / "gpt-4o-full_content",
        method2=OUTPUT_DIR / "0-to-10-rating" / "gpt-4o-mini-full_content",
    )
    compare(
        method1=OUTPUT_DIR / "0-to-10-reason-rating" / "gpt-4o-full_content",
        method2=OUTPUT_DIR / "0-to-10-reason-rating" / "gpt-4o-mini-full_content",
    )
    # compare(method1="gpt-4o-full_content", method2="gpt-4o-mini-summary")
    # compare(method1="gpt-4o-mini-full_content", method2="gpt-4o-mini-summary")


if __name__ == "__main__":
    main()
