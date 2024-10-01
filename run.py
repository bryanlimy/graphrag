import asyncio
import logging
import os
import pickle
from pathlib import Path
from time import time

import pandas as pd
import tiktoken

from graphrag.model import Community, CommunityReport
from graphrag.query.context_builder.dynamic_community_selection import (
    DynamicCommunitySelection,
)
from graphrag.query.indexer_adapters import (
    read_indexer_communities,
    read_indexer_entities,
    read_indexer_reports,
)
from graphrag.query.llm.oai.chat_openai import ChatOpenAI
from graphrag.query.llm.oai.typing import OpenaiApiType

logging.basicConfig(level=logging.INFO, format="%(message)s")

logging.getLogger("httpx").setLevel(logging.WARNING)


INPUT_DIR = Path("examples_notebooks") / "inputs" / "podcast"
OUTPUT_DIR = Path("results")


def get_llm(llm_model: str = "gpt-4o"):
    api_key = os.environ["GRAPHRAG_LLM_API_KEY"]
    api_base = os.environ["GRAPHRAG_LLM_API_BASE"]
    api_version = "2024-02-15-preview"
    llm_init_params = {
        "api_key": api_key,
        "api_base": api_base,
        "api_version": api_version,
        "model": llm_model,
        "deployment_name": llm_model,
        "api_type": OpenaiApiType.AzureOpenAI,
        "max_retries": 50,
    }
    llm = ChatOpenAI(**llm_init_params)
    token_encoder = tiktoken.encoding_for_model(llm.model)
    print(f"Use LLM model {llm.model}.")
    return llm, token_encoder


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
    97: "How does the flow of the conversation affect the depth of the stories shared by the guests?",
    101: "What narrative structures do guests rely on when recounting the journey of their companies or own careers?",
    125: "What new markets or sectors do guests believe will be created by future technologies?",
}


async def global_search(
    qid: int,
    query: str,
    communities: list[Community],
    reports: list[CommunityReport],
    llm: ChatOpenAI,
    token_encoder: tiktoken.Encoding,
    use_summary: bool,
):
    dynamic_selector = DynamicCommunitySelection(
        community_reports=reports,
        communities=communities,
        llm=llm,
        token_encoder=token_encoder,
        keep_parent=True,
        use_summary=use_summary,
        concurrent_coroutines=4,
        rating_threshold=1,
    )

    start = time()
    _, llm_info = await dynamic_selector.select(query)
    end = time()

    result = {
        "ratings": llm_info["ratings"],
        "elapse": int(end - start),
        "llm_calls": llm_info["llm_calls"],
        "prompt_tokens": llm_info["prompt_tokens"],
        "output_tokens": llm_info["output_tokens"],
    }
    print(f'Elapse: {result["elapse"]}s')

    filename = (
        OUTPUT_DIR
        / f"{llm.model}-{'summary' if use_summary else 'full_content'}"
        / f"qid{qid:03d}.pkl"
    )
    filename.parent.mkdir(parents=True, exist_ok=True)
    with open(filename, "wb") as file:
        pickle.dump(result, file)


def main(llm_model: str, use_summary: bool):
    community_df = pd.read_parquet(INPUT_DIR / "create_final_communities.parquet")
    entity_df = pd.read_parquet(INPUT_DIR / "create_final_nodes.parquet")
    report_df = pd.read_parquet(INPUT_DIR / "create_final_community_reports.parquet")

    communities = read_indexer_communities(
        final_communities=community_df,
        final_nodes=entity_df,
        final_community_reports=report_df,
    )
    reports = read_indexer_reports(
        final_community_reports=report_df,
        final_nodes=entity_df,
        community_level=None,
        dynamic_selection=True,
    )

    llm, token_encoder = get_llm(llm_model=llm_model)
    for qid, query in QUERIES.items():
        print(f"Query ({qid}): {query}")
        asyncio.run(
            global_search(
                qid=qid,
                query=query,
                communities=communities,
                reports=reports,
                llm=llm,
                token_encoder=token_encoder,
                use_summary=use_summary,
            )
        )


def method(community_level: int):
    full_input_dir = Path("examples_notebooks") / "inputs" / "AP"
    entity_df = pd.read_parquet(full_input_dir / "create_final_nodes.parquet")
    report_df = pd.read_parquet(
        full_input_dir / "create_final_community_reports.parquet"
    )
    entity_embedding_df = pd.read_parquet(
        full_input_dir / "create_final_entities.parquet"
    )
    community_df = pd.read_parquet(full_input_dir / "create_final_communities.parquet")

    filename = full_input_dir / "community_tree.pkl"
    if filename.exists():
        with open(filename, "rb") as file:
            communities = pickle.load(file)
    else:
        communities = read_indexer_communities(community_df, entity_df, report_df)
    reports = read_indexer_reports(
        report_df, entity_df, community_level=community_level, dynamic_selection=True
    )
    entities = read_indexer_entities(
        entity_df, entity_embedding_df, community_level=community_level
    )

    llm, token_encoder = get_llm(llm_model="gpt-4o")
    dynamic_selector = DynamicCommunitySelection(
        community_reports=reports,
        communities=communities,
        llm=llm,
        token_encoder=token_encoder,
        keep_parent=True,
        use_summary=False,
        concurrent_coroutines=4,
        rating_threshold=1,
        start_with_root=False,
    )
    print(dynamic_selector.starting_communities)


if __name__ == "__main__":
    method(community_level=1)
    # main(llm_model="gpt-4o", use_summary=False)
    # main(llm_model="gpt-4o", use_summary=True)
    # main(llm_model="gpt-4o-mini", use_summary=True)
    # main(llm_model="gpt-4o-mini", use_summary=False)
