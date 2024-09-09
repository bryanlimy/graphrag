"""Util functions to embed a collection of text using OpenAI embedding model"""

import asyncio
import os
from pathlib import Path

import numpy as np
import pandas as pd
from graphrag.model.text_unit import TextUnit
from graphrag.query.llm.base import BaseTextEmbedding
from graphrag.query.llm.oai.embedding import OpenAIEmbedding
from graphrag.query.llm.oai.typing import OpenaiApiType
from tqdm.asyncio import tqdm_asyncio

INPUT_DIR = "./inputs/podcast"


class TextEmbedder:
    def __init__(
        self,
        text_embedder: BaseTextEmbedding,
        concurrent_coroutines: int = 32,
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

    async def aembed_raw_text_batch(
        self, text: list[str], batch_size: int | None = 1000
    ) -> list[TextUnit]:
        if batch_size is None:
            batch_size = len(text)

        results: list[TextUnit] = []
        for i in range(0, len(text), batch_size):
            batch = text[i : i + batch_size] if i + batch_size < len(text) else text[i:]
            embeddings = await tqdm_asyncio.gather(
                *[self.aembed_raw_text(input) for input in batch]
            )
            results.extend(embeddings)
        return results


def get_text_embedder() -> TextEmbedder:
    api_key = os.getenv("GRAPHRAG_OPENAI_API_KEY")
    api_base = os.getenv("GRAPHRAG_OPENAI_API_BASE")
    api_version = "2024-02-15-preview"
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
    return TextEmbedder(text_embedder=oai_embedder, concurrent_coroutines=32)


def embed_text(text: str) -> np.ndarray:
    text_embedder = get_text_embedder()
    embedding = asyncio.run(text_embedder.aembed_raw_text(text))
    return np.array(embedding, dtype=np.float32)


def extract_community_embeddings():
    text_embedder = get_text_embedder()

    report_df = pd.read_parquet(f"{INPUT_DIR}/create_final_community_reports.parquet")

    communities = report_df.community.to_list()
    full_contents = report_df.full_content.to_list()

    embeddings = asyncio.run(text_embedder.aembed_raw_text_batch(full_contents))

    embedding_df = pd.DataFrame({"community": communities, "embedding": embeddings})

    filename = Path("inputs/podcast/community_embedding.parquet")
    embedding_df.to_parquet(filename)

    print(f"saved community embeddings to {filename}")


if __name__ == "__main__":
    extract_community_embeddings()
