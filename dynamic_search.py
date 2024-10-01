import logging
from pathlib import Path

from graphrag.query.cli import run_global_search

logging.basicConfig(level=logging.INFO, format="%(message)s")

logging.getLogger("httpx").setLevel(logging.WARNING)

if __name__ == "__main__":
    # run_global_search(
    #     config_filepath=None,
    #     data_dir="examples_notebooks/inputs/podcast",
    #     root_dir=str(Path.cwd()),
    #     community_level=1,
    #     dynamic_selection=True,
    #     response_type="Multiple Paragraphs",
    #     streaming=False,
    #     query="Are there any common educational or career paths among the guests?",
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
    run_global_search(
        config_filepath=None,
        data_dir="examples_notebooks/inputs/AP",
        root_dir=str(Path.cwd()),
        community_level=None,
        dynamic_selection=True,
        response_type="Multiple Paragraphs",
        streaming=False,
        query="Across the dataset, how are accountability measures enforced for public officials involved in controversial incidents?",
    )
