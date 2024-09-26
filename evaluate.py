from pathlib import Path
import pickle

from collections import Counter

from run import QUERIES

OUTPUT_DIR = Path("results")


def main():
    for qid, query in QUERIES.items():
        print(f"Query ({qid}): {query}")
        filename = OUTPUT_DIR / "gpt-4o-full_content" / f"qid{qid:03d}.pkl"
        with open(filename, "rb") as file:
            result = pickle.load(file)
        print(
            f"rating distribution: {dict(sorted(Counter(result['ratings'].values()).items()))}"
        )


if __name__ == "__main__":
    main()
