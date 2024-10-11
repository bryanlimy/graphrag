import logging
import pickle
from pathlib import Path

from graphrag.query.cli import run_global_search
from graphrag.query.structured_search.global_search.search import GlobalSearchResult

# logging.basicConfig(level=logging.INFO, format="%(message)s")
#
# logging.getLogger("httpx").setLevel(logging.WARNING)

AP_NEWS_QUESTIONS = {
    "activity_global_question": {
        1: "Across the dataset, identify common diplomatic engagements.",
        2: "Across the dataset, what are the most controversial geopolitical tensions?",
        3: "Across the dataset, compare complementary political strategies and their impact on public opinion.",
        4: "Across the dataset, what are some complementary advocacy efforts for marginalized groups?",
        5: "Across the dataset, explain controversial regulatory actions.",
        6: "Across the dataset, identify controversial political strategies.",
        7: "Across the dataset, summarize typical contexts of human rights violations.",
        8: "Across the dataset, explain common policy changes.",
        9: "Across the dataset, summarize surprising impacts of legislative actions on public policy.",
        10: "Across the dataset, explain emerging trends in international policy changes.",
    },
    "data_global_question": {
        1: "Across the dataset, describe the common logistical challenges faced in complex operations.",
        2: "Across the dataset, summarize the common measures implemented to address health risks during extreme weather conditions.",
        3: "Across the dataset, describe the primary reasons for legal challenges against state regulatory frameworks.",
        4: "Across the dataset, describe prevalent challenges in implementing policy changes.",
        5: "Across the dataset, describe common legal arguments used to challenge new state laws.",
        6: "Across the dataset, what are the most common challenges faced by governments and organizations in addressing misinformation and public trust?",
        7: "Across the dataset, describe the common outcomes of state budget decisions on public sector employment and services.",
        8: "Across the dataset, describe how legal and public reactions typically manifest in response to state-level executive orders or actions.",
        9: "Across the dataset, what are the most common factors influencing budget allocation decisions in different states?",
        10: "Across the dataset, how do misinformation and public trust impact societal risks?",
    },
}

OUTPUT_DIR = Path("results") / "AP_news"


def estimate_cost(result: GlobalSearchResult, dynamic_search: bool) -> float:
    cost = 0
    for name, prompt_tokens in result.prompt_tokens.items():
        if name == "build_context" and dynamic_search:
            cost += prompt_tokens * (0.15 / 1_000_000)
        else:
            cost += prompt_tokens * (2.5 / 1_000_000)
    for name, output_tokens in result.output_tokens.items():
        if name == "build_context" and dynamic_search:
            cost += output_tokens * (0.6 / 1_000_000)
        else:
            cost += output_tokens * (10 / 1_000_000)
    return cost


def run_question(
    question_id: int,
    question: str,
    output_dir: Path,
    community_level: int | None,
    dynamic_selection: bool,
):
    print(f"Question {question_id} {question} (dynamic: {dynamic_selection})")
    filename = (
        output_dir
        / ("dynamic" if dynamic_selection else "fixed")
        / f"question_{question_id:02d}.pkl"
    )
    if filename.exists():
        return
    filename.parent.mkdir(parents=True, exist_ok=True)
    _, _, result = run_global_search(
        config_filepath=None,
        data_dir="examples_notebooks/inputs/AP",
        root_dir=str(Path.cwd()),
        community_level=community_level,
        dynamic_selection=dynamic_selection,
        response_type="Multiple Paragraphs",
        streaming=False,
        query=question,
    )
    with open(filename, "wb") as file:
        pickle.dump(result, file)
    cost = estimate_cost(result=result, dynamic_search=dynamic_selection)
    print(
        f"LLM calls: {sum(result.llm_calls.values())}, "
        f"prompt tokens: {sum(result.prompt_tokens.values())}, "
        f"output tokens: {sum(result.output_tokens.values())}, "
        f"estimated cost: ${cost:.02f}."
    )


def main():
    for question_theme, questions in AP_NEWS_QUESTIONS.items():
        output_dir = OUTPUT_DIR / question_theme
        for question_id, question in questions.items():
            run_question(
                question_id=question_id,
                question=question,
                output_dir=output_dir,
                community_level=None,
                dynamic_selection=True,
            )
            run_question(
                question_id=question_id,
                question=question,
                output_dir=output_dir,
                community_level=2,
                dynamic_selection=False,
            )


if __name__ == "__main__":
    main()
    # run_global_search(
    #     config_filepath=None,
    #     data_dir="examples_notebooks/inputs/podcast",
    #     root_dir=str(Path.cwd()),
    #     community_level=None,
    #     dynamic_selection=True,
    #     response_type="Multiple Paragraphs",
    #     streaming=False,
    #     query="Are there any common educational or career paths among the guests?",
    # )
    # run_global_search(
    #     config_filepath=None,
    #     data_dir="examples_notebooks/inputs/podcast",
    #     root_dir=str(Path.cwd()),
    #     community_level=None,
    #     dynamic_selection=True,
    #     response_type="Multiple Paragraphs",
    #     streaming=False,
    #     query="What are the main themes in this dataset?",
    # )
    # run_global_search(
    #     config_filepath=None,
    #     data_dir="examples_notebooks/inputs/operation dulce",
    #     root_dir=str(Path.cwd()),
    #     community_level=None,
    #     dynamic_selection=True,
    #     response_type="Multiple Paragraphs",
    #     streaming=False,
    #     query="What is Cosmic Vocalization and who are involved in it?",
    # )
    # run_global_search(
    #     config_filepath=None,
    #     data_dir="examples_notebooks/inputs/operation dulce",
    #     root_dir=str(Path.cwd()),
    #     community_level=4,
    #     dynamic_selection=False,
    #     response_type="Multiple Paragraphs",
    #     streaming=False,
    #     query="What is Cosmic Vocalization and who are involved in it?",
    # )
    # run_global_search(
    #     config_filepath=None,
    #     data_dir="examples_notebooks/inputs/AP",
    #     root_dir=str(Path.cwd()),
    #     community_level=None,
    #     dynamic_selection=True,
    #     response_type="Multiple Paragraphs",
    #     streaming=False,
    #     query="Across the dataset, how are accountability measures enforced for public officials involved in controversial incidents?",
    # )
    # run_global_search(
    #     config_filepath=None,
    #     data_dir="examples_notebooks/inputs/AP",
    #     root_dir=str(Path.cwd()),
    #     community_level=0,
    #     dynamic_selection=False,
    #     response_type="Multiple Paragraphs",
    #     streaming=False,
    #     query="Across the dataset, how are accountability measures enforced for public officials involved in controversial incidents?",
    # )
