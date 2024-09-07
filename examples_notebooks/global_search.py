import asyncio
import os
from collections import deque
import numpy as np

import pandas as pd
import tiktoken
from graphrag.query.indexer_adapters import read_indexer_entities
from graphrag.query.indexer_adapters import read_indexer_reports
from graphrag.query.llm.base import BaseLLM
from graphrag.query.llm.oai.chat_openai import ChatOpenAI
from graphrag.query.llm.oai.typing import OpenaiApiType
from graphrag.query.structured_search.global_search.community_context import (
    GlobalCommunityContext,
)
from graphrag.query.structured_search.global_search.search import GlobalSearch

from time import time
from typing import List, Any, Dict


from pathlib import Path
from collections import Counter
from tqdm import tqdm
import pickle

import utils


# parquet files generated from indexing pipeline
# INPUT_DIR = "./inputs/operation dulce"
INPUT_DIR = "./inputs/podcast"
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


def fix_community_selection(report_df: pd.DataFrame):
    entity_df = pd.read_parquet(f"{INPUT_DIR}/create_final_nodes.parquet")
    entity_embedding_df = pd.read_parquet(f"{INPUT_DIR}/create_final_entities.parquet")

    reports = read_indexer_reports(report_df, entity_df, COMMUNITY_LEVEL)
    entities = read_indexer_entities(entity_df, entity_embedding_df, COMMUNITY_LEVEL)
    return reports, entities, {"llm_calls": 0, "prompt_tokens": 0, "output_tokens": 0}


def check_community_hierarchy(
    report_df: pd.DataFrame, community_hierarchy: pd.DataFrame
):
    dne = []
    for sub_community in community_hierarchy.sub_community.unique():
        if sub_community not in report_df.community.unique():
            dne.append(sub_community)
    print(f"Cannot find the following sub-communities in report_df: {sorted(dne)}\n")


MESSAGE_1 = """
You are a helpful assistant responsible for deciding whether the provided information is useful in answering a given question, even if it is only partially relevant.

Return NO if the provided information is not relevant at all to the question.
Return YES if the provided information is useful, helpful or relevant to the question.

#######
Information
{description}
######
Question
{question}
######
Return NO if the provided information is not relevant at all to the question.
Return YES if the provided information is useful, helpful or relevant to the question.
"""

MESSAGE_2 = """
You are a helpful assistant responsible for deciding whether the provided information is useful in answering a given question, even if it is only partially relevant.

Return YES if the provided information is useful, helpful or relevant to the question.
Return NO if the provided information is not relevant at all to the question.

#######
Information
{description}
######
Question
{question}
######
Return YES if the provided information is useful, helpful or relevant to the question.
Return NO if the provided information is not relevant at all to the question.
"""

MESSAGE_3 = """
You are a helpful assistant responsible for deciding whether the provided information is useful in answering a given question, even if it is only partially relevant.

Return NO if the provided information is not relevant at all to the question.
Return YES if the provided information is useful, helpful or relevant to the question.
Return UNSURE if you are unsure whether or not the provided information is relevant or helpful in answering the question. 

#######
Information
{description}
######
Question
{question}
######
Return NO if the provided information is not relevant at all to the question.
Return YES if the provided information is useful, helpful or relevant to the question.
Return UNSURE if you are unsure whether or not the provided information is relevant or helpful in answering the question. 
"""

MESSAGE_4 = """
You are a helpful assistant responsible for deciding whether the provided information is useful in answering a given question, even if it is only partially relevant.

Return YES if the provided information is useful, helpful or relevant to the question.
Return NO if the provided information is not relevant at all to the question.
Return UNSURE if you are unsure whether or not the provided information is relevant or helpful in answering the question. 

#######
Information
{description}
######
Question
{question}
######
Return YES if the provided information is useful, helpful or relevant to the question.
Return NO if the provided information is not relevant at all to the question.
Return UNSURE if you are unsure whether or not the provided information is relevant or helpful in answering the question. 
"""

MESSAGE_5 = """
You are a helpful assistant responsible for deciding whether the provided information 
is useful in answering a given question, even if it is only partially relevant.

On a scale from 1 to 5, please rate how relevant or helpful is the provided information in answering the question:
1 - Not relevant in any way to the question
2 - Potentially relevant to the question
3 - Relevant to the question
4 - Highly relevant to the question
5 - It directly answers to the question


#######
Information
{description}
######
Question
{question}
######
Please return the rating as a single value.
"""


def is_relevant(
    llm: BaseLLM,
    token_encoder: tiktoken.Encoding,
    query: str,
    report: pd.DataFrame,
    num_repeats: int = 1,
) -> (str, Dict[str, Any]):
    result = {
        "llm_calls": 0,
        "prompt_tokens": 0,
        "output_tokens": 0,
        "decisions": {},  # store the decision of the LLM, usually a single value.
        "outputs": {},  # store the raw output of the LLM
    }

    for i, message in enumerate([MESSAGE_5]):
        result["decisions"][i], result["outputs"][i] = [], []
        for repeat in range(num_repeats):
            messages = [
                {
                    "role": "system",
                    "content": message.format(
                        description=report.full_content, question=query
                    ),
                },
                {"role": "user", "content": query},
            ]

            if utils.debug_mode():
                decision = llm.generate(
                    messages=messages, max_tokens=2000, temperature=0.0
                )
            else:
                decision = asyncio.run(
                    llm.agenerate(messages=messages, max_tokens=2000, temperature=0.0)
                )

            result["decisions"][i].append(decision[0])
            result["outputs"][i].append(decision)

            result["llm_calls"] += 1
            result["prompt_tokens"] += len(
                token_encoder.encode(messages[0]["content"])
            ) + len(token_encoder.encode(messages[1]["content"]))
            result["output_tokens"] += len(token_encoder.encode(decision))

    # select the decision with the most votes
    options, counts = np.unique(
        [j for i in list(result["decisions"].values()) for j in i],
        return_counts=True,
    )
    decision = result["decision"] = options[np.argmax(counts)]

    if len(result["decisions"]) > 1:
        # info["agreement"] = utils.cohen_kappa(y1=decisions[0], y2=decisions[1], labels=["0", "1"])
        result["agreement"] = utils.compute_agreement(
            y1=result["decisions"][0], y2=result["decisions"][1]
        )

    return decision, result


def dynamic_community_selection(
    report_df: pd.DataFrame,
    llm: BaseLLM,
    token_encoder: tiktoken.Encoding,
    query: str,
    keep_parent: bool = False,
):
    community_tree = utils.get_community_hierarchy(INPUT_DIR)
    # check_community_hierarchy(report_df, community_tree)

    print(f"QUERY: {query}\n")

    start = time()

    queue = deque(report_df.loc[report_df["level"] == 0]["community"])

    results = {
        "llm_calls": 0,
        "prompt_tokens": 0,
        "output_tokens": 0,
        "decisions": [],
        "outputs": {},
        "agreements": [],
    }
    relevant_communities = set()

    while queue:
        community = queue.popleft()
        report = report_df.loc[report_df["community"] == community]
        assert (
            len(report) == 1
        ), f"Each community ({community}) should only have one report"

        report = report.iloc[0]

        decision, result = is_relevant(
            llm=llm,
            token_encoder=token_encoder,
            query=query,
            report=report,
            num_repeats=1,
        )

        results["llm_calls"] += result["llm_calls"]
        results["prompt_tokens"] += result["prompt_tokens"]
        results["output_tokens"] += result["output_tokens"]
        results["decisions"].append(decision)
        results["outputs"][community] = result["outputs"]
        if "agreement" in result:
            results["agreements"].append(result["agreement"])

        statement = f"Community {community} (level: {report.level}) {report.title}\n"

        append_communities = []
        if int(decision[0]) > 1:
            sub_communities = community_tree.loc[
                community_tree["community"] == community
            ].sub_community
            for sub_community in sub_communities:
                if sub_community not in report_df.community.unique():
                    statement += f"Cannot find community {sub_community} in report.\n"
                else:
                    queue.append(sub_community)
                    append_communities.append(sub_community)

            relevant_communities.add(community)

            # remove parent node since the current node is deemed relevant
            if not keep_parent:
                parent_community = community_tree.loc[
                    community_tree["sub_community"] == community
                ]
                if len(parent_community):
                    assert len(parent_community) == 1
                    relevant_communities.discard(parent_community.iloc[0].community)

        statement += f"Relevant: {decision}"
        if len(results["agreements"]):
            statement += f' (agreement: {np.mean(results["agreements"]):.0f}%)'
        if append_communities:
            statement += f" (add communities {append_communities} to queue)"
        statement += "\n"
        print(statement)

    assert len(relevant_communities), f"Cannot find any relevant community reports"
    relevant_report_df = report_df.loc[
        report_df["community"].isin(relevant_communities)
    ]

    end = time()

    print(f"Decision distribution: {Counter(results['decisions'])}")
    if len(results["agreements"]):
        print(f"Average agreement score: {np.mean(results['agreements']):.02f}.")
        utils.plot_agreement(
            results["agreements"], filename=Path("figures/agreements.jpg")
        )
    print(f"Elapse: {end - start:.0f}s\n")

    entity_df = pd.read_parquet(f"{INPUT_DIR}/create_final_nodes.parquet")
    entity_embedding_df = pd.read_parquet(f"{INPUT_DIR}/create_final_entities.parquet")

    reports = read_indexer_reports(relevant_report_df, entity_df, None)
    entities = read_indexer_entities(entity_df, entity_embedding_df, None)
    return reports, entities, results


def main(qid: int, use_dynamic_selection: bool = True):
    query = utils.QUERIES[qid]

    llm, token_encoder = set_llm()

    report_df = pd.read_parquet(f"{INPUT_DIR}/create_final_community_reports.parquet")

    if use_dynamic_selection:
        reports, entities, selection_result = dynamic_community_selection(
            report_df=report_df, llm=llm, token_encoder=token_encoder, query=query
        )
    else:
        reports, entities, selection_result = fix_community_selection(
            report_df=report_df
        )

    print(f"Total report count: {len(report_df)}")
    print(f"Report counts after dynamic community selection: {len(reports)}\n")

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
    llm_calls = result.llm_calls + selection_result["llm_calls"]
    prompt_tokens = result.prompt_tokens + selection_result["prompt_tokens"]
    output_tokens = result.output_tokens + selection_result["output_tokens"]
    print(
        f"\n\nLLM calls: {llm_calls}. "
        f"Total prompt tokens: {prompt_tokens}. "
        f"Total output tokens: {output_tokens}"
    )

    result_dir = Path(f"results/query{qid:03d}")
    result_dir.mkdir(parents=True, exist_ok=True)
    with open(result_dir / "response.md", "w") as file:
        response = f"### Query: {query}\n\n"
        response += (
            f"Decision distribution: {Counter(selection_result['decisions'])}\n\n"
        )
        response += f"Total report count: {len(report_df)}\n\n"
        response += (
            f"Report counts after dynamic community selection: {len(reports)}\n\n"
        )
        response += result.response
        response += f"\n\nLLM calls: {llm_calls}. Total prompt tokens: {prompt_tokens}. Total output tokens: {output_tokens}"
        file.write(response)
    with open(result_dir / "result.pkl", "wb") as file:
        pickle.dump(
            {
                "response": result.response,
                "llm_calls": llm_calls,
                "prompt_tokens": prompt_tokens,
                "output_tokens": output_tokens,
                "selection_result": selection_result,
            },
            file,
        )


def test_multi_query():
    report_df = pd.read_parquet(f"{INPUT_DIR}/create_final_community_reports.parquet")
    for qid, query in utils.QUERIES.items():
        llm, token_encoder = set_llm()
        _ = dynamic_community_selection(
            report_df=report_df.copy(deep=True),
            llm=llm,
            token_encoder=token_encoder,
            query=query,
        )


if __name__ == "__main__":
    for qid in [9, 18, 19]:
        main(qid=qid)
    # test_multi_query()
