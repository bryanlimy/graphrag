import logging
import pickle
from pathlib import Path

from graphrag.query.cli import run_global_search
from graphrag.query.structured_search.global_search.search import GlobalSearchResult

# logging.basicConfig(level=logging.INFO, format="%(message)s")
#
# logging.getLogger("httpx").setLevel(logging.WARNING)

DATA_GLOBAL_QUESTION = [
    "Across the dataset, describe the common logistical challenges faced in complex operations.",
    "Across the dataset, summarize the common measures implemented to address health risks during extreme weather conditions.",
    "Across the dataset, describe the primary reasons for legal challenges against state regulatory frameworks.",
    "Across the dataset, describe prevalent challenges in implementing policy changes.",
    "Across the dataset, describe common legal arguments used to challenge new state laws.",
    "Across the dataset, what are the most common challenges faced by governments and organizations in addressing misinformation and public trust?",
    "Across the dataset, describe the common outcomes of state budget decisions on public sector employment and services.",
    "Across the dataset, describe how legal and public reactions typically manifest in response to state-level executive orders or actions.",
    "Across the dataset, what are the most common factors influencing budget allocation decisions in different states?",
    "Across the dataset, how do misinformation and public trust impact societal risks?",
    "Across the dataset, describe the major geopolitical threats impacting international security.",
    "Across the dataset, compare how public health strategies vary in response to seasonal health challenges.",
    "Across the dataset, describe common factors contributing to health-related issues.",
    "Across the dataset, what are some common factors contributing to accidents and incidents?",
    "Across the dataset, explain the primary factors contributing to the spread of infectious diseases globally.",
    "Across the dataset, describe the important and urgent policy changes related to healthcare and education funding.",
    "Across the dataset, what measures are being implemented to mitigate health and safety risks?",
    "Across the dataset, how are accountability measures enforced for public officials involved in controversial incidents?",
    "Across the dataset, explain how political and economic decisions impact healthcare and education funding.",
    "Across the dataset, describe the common legal consequences for individuals convicted of federal sex abuse charges.",
    "Across the dataset, describe the effects of Medicaid expansion proposals on healthcare access and quality.",
    "Across the dataset, describe important and urgent changes in abortion laws and their impact on emergency medical care across different states.",
    "Across the dataset, explain how legal systems determine the severity of sentences for criminal activities.",
    "Across the dataset, identify common factors frequently identified as causes of explosion incidents.",
    "Across the dataset, describe how international alliances influence global security dynamics.",
    "Across the dataset, describe the emerging and developing ways governments are addressing public health and safety issues.",
    "Across the dataset, describe the important measures being implemented to address drug-related health crises.",
    "Across the dataset, how do legal and public reactions vary in response to government actions perceived as limiting constitutional rights?",
    "Across the dataset, describe how public health initiatives are evolving to tackle emerging health challenges.",
    "Across the dataset, describe how legal outcomes vary for law enforcement personnel involved in cases of misconduct.",
    "Across the dataset, identify prevalent causes of accidents during military exercises.",
    "Across the dataset, describe the major challenges in accessing new medical treatments or technologies.",
    "Across the dataset, what are the most common factors influencing public health and safety concerns?",
    "Across the dataset, describe the prevalent legislative challenges in passing significant acts or bills.",
    "Across the dataset, how is transparency balanced with privacy in the disclosure of personal health issues of high-ranking officials?",
    "Across the dataset, what are the common legal arguments used to challenge laws perceived to infringe on constitutional rights?",
    "Across the dataset, what are the trends in judicial decisions regarding the balance between public safety and individual constitutional rights?",
    "Across the dataset, describe how legislative changes in healthcare policy influence healthcare delivery and outcomes.",
    "Across the dataset, describe how state-level legislative changes reflect broader national trends in policy making.",
    "Across the dataset, summarize the important outcomes for workers who continue strikes beyond government-imposed deadlines.",
    "Across the dataset, describe the common sentences given for arson-related offenses.",
    "Across the dataset, describe the important measures being adopted to address health risks related to extreme weather conditions.",
    "Across the dataset, describe the common actions being implemented to improve public health systems.",
    "Across the dataset, describe the common actions being taken to investigate and prevent mass violence incidents.",
    "Across the dataset, describe the common trends in vaccination rates for major diseases.",
    "Across the dataset, identify common causes of transportation accidents.",
    "Across the dataset, describe actions being taken by governments and organizations to address systemic challenges.",
    "Across the dataset, describe the common actions being taken by governments to address international trade issues.",
    "Across the dataset, describe the important measures being taken to prevent fraud in government programs.",
    "Across the dataset, what are the common outcomes of legal proceedings involving public servants accused of misconduct?",
]
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
        i + 1: DATA_GLOBAL_QUESTION[i] for i in range(len(DATA_GLOBAL_QUESTION))
    },
}

OUTPUT_DIR = Path("results") / "AP_news"


def estimate_cost(result: GlobalSearchResult, dynamic_search: bool) -> float:
    """Estimate total cost of the search based on the number of prompt and output tokens.

    Cost per token data is from https://openai.com/api/pricing.
    """
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
        / (
            f"dynamic_{community_level}"
            if dynamic_selection
            else f"fixed_{community_level}"
        )
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
        f"\nLLM calls: {sum(result.llm_calls.values())}, "
        f"prompt tokens: {sum(result.prompt_tokens.values())}, "
        f"output tokens: {sum(result.output_tokens.values())}, "
        f"estimated cost: ${cost:.02f}.\n"
        f"--------------------------------------------------------\n"
    )


def main():
    for question_type, questions in AP_NEWS_QUESTIONS.items():
        if question_type != "data_global_question":
            continue
        output_dir = OUTPUT_DIR / question_type
        for question_id, question in questions.items():
            # run_question(
            #     question_id=question_id,
            #     question=question,
            #     output_dir=output_dir,
            #     community_level=None,
            #     dynamic_selection=True,
            # )
            # run_question(
            #     question_id=question_id,
            #     question=question,
            #     output_dir=output_dir,
            #     community_level=2,
            #     dynamic_selection=False,
            # )
            run_question(
                question_id=question_id,
                question=question,
                output_dir=output_dir,
                community_level=2,
                dynamic_selection=True,
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
