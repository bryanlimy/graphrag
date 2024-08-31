import os
import asyncio
import pandas as pd
import tiktoken
from collections import deque
import graphrag.index.graph.extractors.community_reports.schemas as schemas
from graphrag.query.indexer_adapters import read_indexer_entities, read_indexer_reports
from graphrag.query.llm.oai.chat_openai import ChatOpenAI
from graphrag.query.llm.oai.typing import OpenaiApiType
from graphrag.query.structured_search.global_search.community_context import (
    GlobalCommunityContext,
)
from graphrag.query.structured_search.global_search.search import GlobalSearch
from graphrag.query.llm.base import BaseLLM


# parquet files generated from indexing pipeline
INPUT_DIR = "./inputs/operation dulce"
COMMUNITY_REPORT_TABLE = "create_final_community_reports"
ENTITY_TABLE = "create_final_nodes"
ENTITY_EMBEDDING_TABLE = "create_final_entities"

# community level in the Leiden community hierarchy from which we will load the community reports
# higher value means we use reports from more fine-grained communities (at the cost of higher computation cost)
COMMUNITY_LEVEL = 2


def set_llm():
    api_key = os.getenv("GRAPHRAG_OPENAI_API_KEY")
    api_base = os.getenv("GRAPHRAG_OPENAI_API_BASE")
    api_version = "2024-02-15-preview"
    llm_model = "gpt-4o"
    llm_deployment_name = "gpt-4o"

    llm_init_params = {
        "api_key": api_key,
        "api_base": api_base,
        "api_version": api_version,
        "model": llm_model,
        "deployment_name": llm_deployment_name,
        "api_type": OpenaiApiType.AzureOpenAI,
        "max_retries": 50,
    }

    llm = ChatOpenAI(**llm_init_params)

    token_encoder = tiktoken.get_encoding("o200k_base")
    return llm, token_encoder


def get_community_hierarchy():
    node_df = pd.read_parquet(f"{INPUT_DIR}/create_final_nodes.parquet")

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


def fix_community_selection():
    entity_df = pd.read_parquet(f"{INPUT_DIR}/create_final_nodes.parquet")
    report_df = pd.read_parquet(f"{INPUT_DIR}/create_final_community_reports.parquet")
    entity_embedding_df = pd.read_parquet(f"{INPUT_DIR}/create_final_entities.parquet")

    reports = read_indexer_reports(report_df, entity_df, COMMUNITY_LEVEL)
    entities = read_indexer_entities(entity_df, entity_embedding_df, COMMUNITY_LEVEL)
    print(f"Total report count: {len(report_df)}")
    print(
        f"Report count after filtering by community level {COMMUNITY_LEVEL}: {len(reports)}"
    )
    return reports, entities, 0, 0, 0


SYSTEM_MESSAGE = """
You are a helpful assistant responsible for deciding whether the provided information is useful in answering a given question, even if it is only partially relevant.

Return "0" if the information is not relevant at all to the question.
Return "1" if the provided information is useful, helpful or relevant to the question.

#######
Information
{description}
######
Question
{question}
"""


def dynamic_community_selection(
    llm: BaseLLM, token_encoder: tiktoken.Encoding, query: str
):
    community_hierarchy = get_community_hierarchy()
    report_df = pd.read_parquet(f"{INPUT_DIR}/create_final_community_reports.parquet")

    print(f"Consider the question: {query}\n")

    queue = deque(report_df.loc[report_df["level"] == 0]["community"])

    LLM_calls, prompt_tokens, output_tokens = 0, 0, 0
    relevant_communities = []
    # find all community summary at level 0
    while queue:
        community = queue.popleft()
        report = report_df.loc[report_df["community"] == community]
        assert len(report) == 1, "Each community should only have one report"
        report = report.iloc[0]
        messages = [
            {
                "role": "system",
                "content": SYSTEM_MESSAGE.format(
                    description=report.full_content, question=query
                ),
            },
            {"role": "user", "content": query},
        ]
        prompt_tokens += len(token_encoder.encode(messages[0]["content"])) + len(
            token_encoder.encode(messages[1]["content"])
        )
        decision = asyncio.run(
            llm.agenerate(messages=messages, max_tokens=2000, temperature=0.0)
        )
        output_tokens += len(token_encoder.encode(decision))
        LLM_calls += 1
        print(
            f"Community {community} (level: {report.level})\n"
            f"Summary: {report.summary}\nRelevant: {decision}\n\n"
        )
        if decision[0] == "1":
            # TODO what should we do if one child is relevant but another is not? Should we keep the parent node or not in this case?
            if decision[0] == "1":
                relevant_communities.append(community)
            sub_communities = community_hierarchy.loc[
                community_hierarchy["community"] == community
            ]
            for _, community_df in sub_communities.iterrows():
                queue.append(community_df.sub_community)

    assert len(relevant_communities), f"Cannot find any relevant community reports"
    relevant_report_df = report_df.loc[
        report_df["community"].isin(relevant_communities)
    ]

    entity_df = pd.read_parquet(f"{INPUT_DIR}/create_final_nodes.parquet")
    entity_embedding_df = pd.read_parquet(f"{INPUT_DIR}/create_final_entities.parquet")

    reports = read_indexer_reports(relevant_report_df, entity_df, None)
    entities = read_indexer_entities(entity_df, entity_embedding_df, None)
    print(f"Total report count: {len(report_df)}")
    print(f"Report count after dynamic community selection: {len(reports)}")
    return reports, entities, LLM_calls, prompt_tokens, output_tokens


def main(use_dynamic_selection: bool = True):
    query = "What is the major conflict in this story and who are the protagonist and antagonist?"
    llm, token_encoder = set_llm()

    if use_dynamic_selection:
        reports, entities, LLM_calls, prompt_tokens, output_tokens = (
            dynamic_community_selection(
                llm=llm, token_encoder=token_encoder, query=query
            )
        )
    else:
        reports, entities, LLM_calls, prompt_tokens, output_tokens = (
            fix_community_selection()
        )

    context_builder = GlobalCommunityContext(
        community_reports=reports,
        entities=entities,
        # default to None if you don't want to use community weights for ranking
        token_encoder=token_encoder,
    )

    context_builder_params = {
        "use_community_summary": False,
        # False means using full community reports. True means using community short summaries.
        "shuffle_data": True,
        "include_community_rank": True,
        "min_community_rank": 0,
        "community_rank_name": "rank",
        "include_community_weight": True,
        "community_weight_name": "occurrence weight",
        "normalize_community_weight": True,
        "max_tokens": 12_000,
        # change this based on the token limit you have on your model (if you are using a model with 8k limit, a good setting could be 5000)
        "context_name": "Reports",
    }

    map_llm_params = {
        "max_tokens": 1000,
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
    }

    reduce_llm_params = {
        "max_tokens": 2000,
        # change this based on the token limit you have on your model (if you are using a model with 8k limit, a good setting could be 1000-1500)
        "temperature": 0.0,
    }

    search_engine = GlobalSearch(
        llm=llm,
        context_builder=context_builder,
        token_encoder=token_encoder,
        max_data_tokens=12_000,
        # change this based on the token limit you have on your model (if you are using a model with 8k limit, a good setting could be 5000)
        map_llm_params=map_llm_params,
        reduce_llm_params=reduce_llm_params,
        allow_general_knowledge=False,
        # set this to True will add instruction to encourage the LLM to incorporate general knowledge in the response, which may increase hallucinations, but could be useful in some use cases.
        json_mode=True,
        # set this to False if your LLM model does not support JSON mode.
        context_builder_params=context_builder_params,
        concurrent_coroutines=32,
        response_type="multiple paragraphs",
        # free form text describing the response type and format, can be anything, e.g. prioritized list, single paragraph, multiple paragraphs, multiple-page report
    )

    result = search_engine.search(query=query)

    print(result.response)

    # inspect number of LLM calls and tokens
    print(
        f"\n\nLLM calls: {result.llm_calls + LLM_calls}. "
        f"total prompt tokens: {result.prompt_tokens + prompt_tokens}. "
        f"total output tokens: {result.output_tokens + output_tokens}"
    )


if __name__ == "__main__":
    main()
