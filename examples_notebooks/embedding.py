"""Util functions to embed a collection of text using OpenAI embedding model"""

import asyncio
import os
import uuid
from dataclasses import dataclass

from graphrag.model.text_unit import TextUnit
from graphrag.query.llm.base import BaseTextEmbedding
from graphrag.query.llm.oai.embedding import OpenAIEmbedding
from graphrag.query.llm.oai.typing import OpenaiApiType
from tqdm.asyncio import tqdm_asyncio


class TextEmbedder:
    def __init__(
        self, text_embedder: BaseTextEmbedding, concurrent_coroutines: int = 32
    ):
        self.text_embedder = text_embedder
        self.semaphore = asyncio.Semaphore(concurrent_coroutines)

    async def aembed_raw_text(self, text: str) -> list[float]:
        async with self.semaphore:
            return await self.text_embedder.aembed(text)

    async def aembed_text_unit(self, text_unit: TextUnit) -> TextUnit:
        text_unit.text_embedding = await self.aembed_raw_text(text_unit.text)
        return text_unit

    async def aembed_batch(
        self, text_units: list[TextUnit], batch_size: int | None = 1000
    ) -> list[TextUnit]:
        if batch_size is None:
            batch_size = len(text_units)

        results: list[TextUnit] = []
        for i in range(0, len(text_units), batch_size):
            batch = (
                text_units[i : i + batch_size]
                if i + batch_size < len(text_units)
                else text_units[i:]
            )
            embeddings = await tqdm_asyncio.gather(
                *[self.aembed_text_unit(input) for input in batch]
            )
            results.extend(embeddings)
        return results


if __name__ == "__main__":
    api_key = os.getenv("GRAPHRAG_OPENAI_API_KEY")
    api_base = os.getenv("GRAPHRAG_OPENAI_API_BASE")
    api_version = "2024-02-15-preview"
    llm_model = "gpt-4o"
    llm_deployment_name = "gpt-4o"
    embedding_model = "text-embedding-ada-002"
    embedding_deployment_name = embedding_model

    oai_embedder = OpenAIEmbedding(
        api_base=api_base,
        api_key=api_key,
        model=embedding_model,
        deployment_name=embedding_deployment_name,
        api_type=OpenaiApiType.AzureOpenAI,
        api_version=api_version,
        max_retries=50,
    )

    text_embedder = TextEmbedder(text_embedder=oai_embedder, concurrent_coroutines=32)
