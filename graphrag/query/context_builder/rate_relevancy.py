# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Algorithm to rate the relevancy between a query and description text."""

import asyncio
import logging
from contextlib import nullcontext
from typing import Any

import numpy as np
import tiktoken

from graphrag.llm.openai.utils import try_parse_json_object
from graphrag.query.llm.oai.chat_openai import ChatOpenAI
from graphrag.query.llm.text_utils import num_tokens

log = logging.getLogger(__name__)

RATE_QUERY = """
---Role---
You are a helpful assistant responsible for deciding whether the provided information is useful in answering a given question, even if it is only partially relevant.

---Goal---

On a scale from 1 to 5, please rate how relevant or helpful is the provided information in answering the question:
1 - Not relevant in any way to the question
2 - Potentially relevant to the question
3 - Relevant to the question
4 - Highly relevant to the question
5 - It directly answers to the question

---Information---

{description}

---Question---

{question}

---Target response length and format---

Please response in the following JSON format with two entries:
- "reason": the reasoning of your rating, please include information that you have considered.
- "rating": the relevancy rating from 1 to 5.
{{
    "reason": str,
    "rating": int.
}}
"""


async def rate_relevancy(
    query: str,
    community_id: str,
    description: str,
    llm: ChatOpenAI,
    token_encoder: tiktoken.Encoding,
    num_repeats: int = 1,
    semaphore: asyncio.Semaphore | None = None,
    **llm_kwargs: Any,
) -> dict[str, Any]:
    """
    Rate the relevancy between the query and description on a scale of 1 to 5.

    A rating of 1 indicates the community is not relevant to the query and a rating of 5
    indicates the community directly answers the query.

    Args:
        query: the query (or question) to rate against
        description: the community description to rate, it can be the community
            title, summary, or the full content.
        llm: LLM model to use for rating
        token_encoder: token encoder
        num_repeats: number of times to repeat the rating process for the same community (default: 1)
        llm_kwargs: additional arguments to pass to the LLM model
        semaphore: asyncio.Semaphore to limit the number of concurrent LLM calls (default: None)
    """
    # if community_id in (
    #     "3",
    #     "9",
    #     "19",
    #     "61",
    #     "62",
    #     "133",
    #     "146",
    #     "150",
    #     "159",
    #     "168",
    #     "172",
    #     "203",
    #     "337",
    #     "348",
    #     "423",
    #     "464",
    #     "466",
    #     "781",
    #     "787",
    #     "829",
    #     "890",
    #     "921",
    #     "1014",
    #     "60",
    # ):
    #     print("here")
    llm_calls, prompt_tokens, output_tokens, ratings = 0, 0, 0, []
    messages = [
        {
            "role": "system",
            "content": RATE_QUERY.format(description=description, question=query),
        },
        {"role": "user", "content": query},
    ]
    for _ in range(num_repeats):
        async with semaphore if semaphore is not None else nullcontext():
            response = await llm.agenerate(messages=messages, **llm_kwargs)
        try:
            _, parsed_response = try_parse_json_object(response)
            ratings.append(parsed_response["rating"])
        except KeyError:
            # in case of json parsing error, default to rating 2 so the report is kept.
            # json parsing error should rarely happen.
            log.info("Error parsing json response, defaulting to rating 2")
            ratings.append(2)
        llm_calls += 1
        prompt_tokens += num_tokens(messages[0]["content"], token_encoder)
        output_tokens += num_tokens(response, token_encoder)
    # select the decision with the most votes
    options, counts = np.unique(ratings, return_counts=True)
    rating = int(options[np.argmax(counts)])
    return {
        "rating": rating,
        "ratings": ratings,
        "llm_calls": llm_calls,
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
    }
