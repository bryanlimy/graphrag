# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Algorithm to dynamically select relevant communities with respect to a query."""

import asyncio
import logging
from collections import Counter
from copy import deepcopy
from time import time

import tiktoken

from graphrag.model import Community, CommunityReport
from graphrag.query.context_builder.rate_relevancy import rate_relevancy
from graphrag.query.llm.oai.chat_openai import ChatOpenAI

log = logging.getLogger(__name__)

import os
from graphrag.query.llm.oai.typing import OpenaiApiType


def get_llm(llm_model: str = "gpt-4o"):
    api_key = os.environ["GRAPHRAG_OPENAI_API_KEY"]
    api_base = os.environ["GRAPHRAG_OPENAI_API_BASE"]
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
    print(f"Use {llm.model} in dynamic community selection")
    return llm, token_encoder


class DynamicCommunitySelection:
    """Dynamic community selection to select community reports that are relevant to the query.

    Any community report with a rating EQUAL or ABOVE the rating_threshold is considered relevant.
    """

    def __init__(
        self,
        community_reports: list[CommunityReport],
        communities: list[Community],
        llm: ChatOpenAI,
        token_encoder: tiktoken.Encoding,
        keep_parent: bool = False,
        num_repeats: int = 1,
        use_summary: bool = False,
        use_logit_bias: bool = True,
        concurrent_coroutines: int = 4,
        rating_threshold: int = 2,
    ):
        self.reports = {report.community_id: report for report in community_reports}
        # mapping from community to sub communities
        self.node2children = {
            community.id: (
                []
                if community.sub_community_ids is None
                else community.sub_community_ids
            )
            for community in communities
        }
        # mapping from community to parent community
        self.node2parent: dict[str, str] = {
            sub_community: community
            for community, sub_communities in self.node2children.items()
            for sub_community in sub_communities
        }
        # get all communities at level 0
        self.root_communities: list[str] = [
            community.id
            for community in communities
            if community.level == "0" and community.id in self.reports
        ]
        self.llm, self.token_encoder = get_llm(llm_model="gpt-4o")
        # self.llm = llm
        # self.token_encoder = token_encoder
        self.keep_parent = keep_parent
        self.num_repeats = num_repeats
        self.use_summary = use_summary
        self.llm_kwargs = {"temperature": 0.0, "max_tokens": 2}
        possible_ratings = [1, 2, 3, 4, 5]
        if use_logit_bias:
            # bias the output to the rating tokens
            self.llm_kwargs["logit_bias"] = {
                self.token_encoder.encode(str(token))[0]: 5
                for token in possible_ratings
            }
        self.semaphore = asyncio.Semaphore(concurrent_coroutines)
        if rating_threshold not in possible_ratings:
            raise ValueError("rating_threshold must be one of %s" % possible_ratings)
        # self.rating_threshold = rating_threshold
        self.rating_threshold = 1

    async def select(
        self, query: str
    ) -> tuple[list[CommunityReport], dict[str, int], dict[str, int]]:
        """
        Select relevant communities with respect to the query.

        Args:
            query: the query to rate against
        """
        start = time()
        queue = deepcopy(self.root_communities)  # start search from level 0 communities

        ratings = {}  # store the ratings for each community
        llm_info = {"llm_calls": 0, "prompt_tokens": 0, "output_tokens": 0}
        relevant_communities = set()
        while queue:
            gather_results = await asyncio.gather(
                *[
                    rate_relevancy(
                        query=query,
                        description=(
                            self.reports[community].summary
                            if self.use_summary
                            else self.reports[community].full_content
                        ),
                        llm=self.llm,
                        token_encoder=self.token_encoder,
                        num_repeats=self.num_repeats,
                        semaphore=self.semaphore,
                        **self.llm_kwargs,
                    )
                    for community in queue
                ]
            )

            communities_to_rate = []
            for community, result in zip(queue, gather_results, strict=True):
                rating = result["rating"]
                log.debug(
                    "dynamic community selection: community %s rating %s",
                    community,
                    rating,
                )
                ratings[community] = rating
                llm_info["llm_calls"] += result["llm_calls"]
                llm_info["prompt_tokens"] += result["prompt_tokens"]
                llm_info["output_tokens"] += result["output_tokens"]
                if rating >= self.rating_threshold:
                    relevant_communities.add(community)
                    # find children nodes of the current node and append them to the queue
                    # TODO check why some sub_communities are NOT in report_df
                    if community in self.node2children:
                        for sub_community in self.node2children[community]:
                            if sub_community in self.reports:
                                communities_to_rate.append(sub_community)
                            else:
                                log.info(
                                    "dynamic community selection: cannot find community %s in reports",
                                    sub_community,
                                )
                    # remove parent node if the current node is deemed relevant
                    if not self.keep_parent and community in self.node2parent:
                        relevant_communities.discard(self.node2parent[community])
            queue = communities_to_rate

        community_reports = [
            self.reports[community] for community in relevant_communities
        ]
        end = time()

        log.info(
            "dynamic community selection (took: %ss)\n"
            "\trating distribution %s\n"
            "\t%s out of %s community reports are relevant\n"
            "\tprompt tokens: %s, output tokens: %s",
            int(end - start),
            dict(sorted(Counter(ratings.values()).items())),
            len(relevant_communities),
            len(self.reports),
            llm_info["prompt_tokens"],
            llm_info["output_tokens"],
        )
        return community_reports, llm_info, ratings
