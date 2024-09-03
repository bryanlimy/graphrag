import asyncio
import os
from collections import deque
import numpy as np
import graphrag.index.graph.extractors.community_reports.schemas as schemas
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
from sklearn.metrics import cohen_kappa_score

from typing import List
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter

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


def plot_agreement(kappa: List[float], filename: Path = None):
    figure, ax = plt.subplots(nrows=1, ncols=1, figsize=(3, 2), dpi=240)

    df = pd.DataFrame({"kappa": kappa})
    p = sns.histplot(
        df,
        x="kappa",
        bins=20,
        binrange=(-1, 1),
        color="black",
        stat="probability",
        fill=False,
        linewidth=1,
        clip_on=False,
        alpha=0.8,
        ax=ax,
    )
    ax.axvline(
        x=0,
        color="black",
        alpha=0.3,
        linestyle="dotted",
        linewidth=1,
        zorder=-1,
    )
    x_ticks = np.linspace(-1.0, 1.0, 3)
    ax.set_xlim(x_ticks[0], x_ticks[-1])
    ax.set_xticks(x_ticks, labels=np.round(x_ticks, 1), fontsize=9)
    ax.set_xlabel("cohen's kappa score", fontsize=10, labelpad=0)
    y_ticks = np.linspace(0, 1, 3)
    ax.set_ylim(y_ticks[0], y_ticks[-1])
    ax.set_yticks(y_ticks, labels=(100 * y_ticks).astype(int), fontsize=9)
    ax.tick_params(axis="both", which="both", length=2, pad=1, width=0.8)
    sns.despine(ax, trim=True)
    if filename is not None:
        filename.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(filename, dpi=240, bbox_inches="tight", pad_inches=0.02)
    else:
        plt.show()
    plt.close(figure)


MESSAGE_1 = """
You are a helpful assistant responsible for deciding whether the provided information is useful in answering a given question, even if it is only partially relevant.

Return "0" if the information is not relevant at all to the question.
Return "1" if the provided information is useful, helpful or relevant to the question.
Return "2" if the provided information is not sufficient to make a decision.

#######
Information
{description}
######
Question
{question}
######
Return "0" if the information is not relevant at all to the question.
Return "1" if the provided information is useful, helpful or relevant to the question.
Return "2" if the provided information is not sufficient to make a decision.
"""

MESSAGE_2 = """
You are a helpful assistant responsible for deciding whether the provided information is useful in answering a given question, even if it is only partially relevant.

Return "2" if the provided information is not sufficient to make a decision.
Return "1" if the provided information is useful, helpful or relevant to the question.
Return "0" if the information is not relevant at all to the question.

#######
Information
{description}
######
Question
{question}
######
Return "2" if the provided information is not sufficient to make a decision.
Return "1" if the provided information is useful, helpful or relevant to the question.
Return "0" if the information is not relevant at all to the question.
"""


def check_community_hierarchy(
    report_df: pd.DataFrame, community_hierarchy: pd.DataFrame
):
    dne = []
    for sub_community in community_hierarchy.sub_community.unique():
        if sub_community not in report_df.community.unique():
            dne.append(sub_community)
    print(f"Cannot find the following sub-communities in report_df: {sorted(dne)}\n")


def is_relevant(
    llm: BaseLLM,
    token_encoder: tiktoken.Encoding,
    query: str,
    report: pd.DataFrame,
    num_repeats: int = 1,
):
    info = {"LLM_calls": 0, "prompt_tokens": 0, "output_tokens": 0}

    decisions1, decisions2 = [], []
    for i in (0, 1):
        message = MESSAGE_1 if i == 0 else MESSAGE_2
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

            decision = asyncio.run(
                llm.agenerate(messages=messages, max_tokens=2000, temperature=0.0)
            )
            # decision = llm.generate(messages=messages, max_tokens=2000, temperature=0.0)

            if i % 2 == 0:
                decisions1.append(decision)
            else:
                decisions2.append(decision)

            info["LLM_calls"] += 1
            info["prompt_tokens"] += len(
                token_encoder.encode(messages[0]["content"])
            ) + len(token_encoder.encode(messages[1]["content"]))
            info["output_tokens"] += len(token_encoder.encode(decision))

    # select the decision with the most votes
    decisions = decisions1 + decisions2
    options, counts = np.unique(decisions, return_counts=True)
    decision = options[np.argmax(counts)]

    info["kappa"] = 1
    if not np.array_equal(decisions1, decisions2):
        info["kappa"] = cohen_kappa_score(decisions1, decisions2)

    return decision, info


def dynamic_community_selection(
    llm: BaseLLM,
    token_encoder: tiktoken.Encoding,
    query: str,
    keep_parent: bool = False,
):
    community_tree = get_community_hierarchy()
    report_df = pd.read_parquet(f"{INPUT_DIR}/create_final_community_reports.parquet")

    # check_community_hierarchy(report_df, community_tree)

    print(f"QUERY: {query}\n")

    queue = deque(report_df.loc[report_df["level"] == 0]["community"])

    LLM_calls, prompt_tokens, output_tokens, kappa_scores = 0, 0, 0, []
    relevant_communities = set()
    decisions = []
    while queue:
        community = queue.popleft()
        report = report_df.loc[report_df["community"] == community]
        assert (
            len(report) == 1
        ), f"Each community ({community}) should only have one report"

        report = report.iloc[0]

        decision, info = is_relevant(
            llm=llm,
            token_encoder=token_encoder,
            query=query,
            report=report,
            num_repeats=3,
        )

        LLM_calls += info["LLM_calls"]
        prompt_tokens += info["prompt_tokens"]
        output_tokens += info["output_tokens"]
        decisions.append(decision)
        kappa_scores.append(info["kappa"])

        statement = f"Community {community} (level: {report.level}) {report.title}\n"

        append_communities = []
        if decision[0] == "1":
            # TODO what should we do if one child is relevant but another is not? Should we keep the parent node or not in this case?
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

        statement += f"Relevant: {decision} (kappa: {info['kappa']:.02f})"
        if append_communities:
            statement += f" (add communities {append_communities} to queue)"
        statement += "\n"
        print(statement)

    assert len(relevant_communities), f"Cannot find any relevant community reports"
    relevant_report_df = report_df.loc[
        report_df["community"].isin(relevant_communities)
    ]

    print(f"Decision distribution: {Counter(decisions)}")
    print(f"Average cohen's kappa score: {np.mean(kappa_scores):.02f}.")
    plot_agreement(kappa_scores, filename=Path("figures/cohen_kappa_score.jpg"))

    entity_df = pd.read_parquet(f"{INPUT_DIR}/create_final_nodes.parquet")
    entity_embedding_df = pd.read_parquet(f"{INPUT_DIR}/create_final_entities.parquet")

    reports = read_indexer_reports(relevant_report_df, entity_df, None)
    entities = read_indexer_entities(entity_df, entity_embedding_df, None)
    print(f"Total report count: {len(report_df)}")
    print(f"Report count after dynamic community selection: {len(reports)}\n")
    return reports, entities, LLM_calls, prompt_tokens, output_tokens


def main(use_dynamic_selection: bool = True):
    # query = "What is the major conflict in this story and who are the protagonist and antagonist?"
    query = "Are there any common educational or career paths among the guests?"

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
