import os

import pandas as pd
import tiktoken

from graphrag.query.indexer_adapters import read_indexer_entities, read_indexer_reports
from graphrag.query.llm.oai.chat_openai import ChatOpenAI
from graphrag.query.llm.oai.typing import OpenaiApiType
from graphrag.query.structured_search.global_search.community_context import (
    GlobalCommunityContext,
)
from graphrag.query.structured_search.global_search.search import GlobalSearch

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


def load_community_report():
    entity_df = pd.read_parquet(f"{INPUT_DIR}/create_final_nodes.parquet")
    report_df = pd.read_parquet(f"{INPUT_DIR}/create_final_community_reports.parquet")
    entity_embedding_df = pd.read_parquet(f"{INPUT_DIR}/create_final_entities.parquet")

    reports = read_indexer_reports(report_df, entity_df, COMMUNITY_LEVEL)
    entities = read_indexer_entities(entity_df, entity_embedding_df, COMMUNITY_LEVEL)
    print(f"Total report count: {len(report_df)}")
    print(
        f"Report count after filtering by community level {COMMUNITY_LEVEL}: {len(reports)}"
    )
    return reports, entities


def main():
    query = "What is the major conflict in this story and who are the protagonist and antagonist?"
    llm, token_encoder = set_llm()

    entity_df = pd.read_parquet(f"{INPUT_DIR}/create_final_nodes.parquet")
    report_df = pd.read_parquet(f"{INPUT_DIR}/create_final_community_reports.parquet")
    entity_embedding_df = pd.read_parquet(f"{INPUT_DIR}/create_final_entities.parquet")

    reports = read_indexer_reports(report_df, entity_df, COMMUNITY_LEVEL)
    entities = read_indexer_entities(entity_df, entity_embedding_df, COMMUNITY_LEVEL)
    print(f"Total report count: {len(report_df)}")
    print(
        f"Report count after filtering by community level {COMMUNITY_LEVEL}: {len(reports)}"
    )

    SYSTEM_MESSAGE = """
    You are a helpful assistant responsible for deciding whether the provided information is relevant to a given question.
    Please answer YES, if the provided title and description are helpful in answering the question, even if only in parts. Answer NO otherwise. If the answer is NO, please state why the information is not relevant.
    
    Title: {title}
    Description: {description}
    """

    # find all community summary at level 0
    for _, report in report_df.loc[report_df["level"] == 0].iterrows():
        messages = [
            {
                "role": "system",
                "content": SYSTEM_MESSAGE.format(
                    title=report.title, description=report.full_content
                ),
            },
            {"role": "user", "content": query},
        ]
        decision = llm.generate(messages=messages, max_tokens=2000, temperature=0.0)
        print(decision)

    context_builder = GlobalCommunityContext(
        community_reports=reports,
        entities=entities,  # default to None if you don't want to use community weights for ranking
        token_encoder=token_encoder,
    )

    context_builder_params = {
        "use_community_summary": False,  # False means using full community reports. True means using community short summaries.
        "shuffle_data": True,
        "include_community_rank": True,
        "min_community_rank": 0,
        "community_rank_name": "rank",
        "include_community_weight": True,
        "community_weight_name": "occurrence weight",
        "normalize_community_weight": True,
        "max_tokens": 12_000,  # change this based on the token limit you have on your model (if you are using a model with 8k limit, a good setting could be 5000)
        "context_name": "Reports",
    }

    map_llm_params = {
        "max_tokens": 1000,
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
    }

    reduce_llm_params = {
        "max_tokens": 2000,  # change this based on the token limit you have on your model (if you are using a model with 8k limit, a good setting could be 1000-1500)
        "temperature": 0.0,
    }

    search_engine = GlobalSearch(
        llm=llm,
        context_builder=context_builder,
        token_encoder=token_encoder,
        max_data_tokens=12_000,  # change this based on the token limit you have on your model (if you are using a model with 8k limit, a good setting could be 5000)
        map_llm_params=map_llm_params,
        reduce_llm_params=reduce_llm_params,
        allow_general_knowledge=False,  # set this to True will add instruction to encourage the LLM to incorporate general knowledge in the response, which may increase hallucinations, but could be useful in some use cases.
        json_mode=True,  # set this to False if your LLM model does not support JSON mode.
        context_builder_params=context_builder_params,
        concurrent_coroutines=32,
        response_type="multiple paragraphs",  # free form text describing the response type and format, can be anything, e.g. prioritized list, single paragraph, multiple paragraphs, multiple-page report
    )

    result = search_engine.search(query=query)

    print(result.response)

    # inspect number of LLM calls and tokens
    print(
        f"\n\nLLM calls: {result.llm_calls}. "
        f"total prompt tokens: {result.prompt_tokens}. "
        f"total output tokens: {result.output_tokens}"
    )


if __name__ == "__main__":
    main()