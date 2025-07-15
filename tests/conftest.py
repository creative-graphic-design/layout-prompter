import pathlib
import pickle
from typing import Dict, List

import datasets as ds
import pytest
from loguru import logger
from tqdm.auto import tqdm

from layout_prompter.datasets import (
    load_poster_layout,
    load_raw_poster_layout,
    load_raw_rico,
    load_rico25,
)
from layout_prompter.models import LayoutData


@pytest.fixture(autouse=True)
def print_newline() -> None:
    """Print a newline before each test to improve readability in test output."""
    print()


@pytest.fixture(scope="session")
def root_dir() -> pathlib.Path:
    return pathlib.Path(__file__).parents[1]


@pytest.fixture(scope="session")
def test_fixtures_dir(root_dir: pathlib.Path) -> pathlib.Path:
    """Return the directory for test fixtures."""
    return root_dir / "test_fixtures"


@pytest.fixture(scope="session")
def poster_layout_processed_data_path(test_fixtures_dir: pathlib.Path) -> pathlib.Path:
    """Return the path to the processed poster layout data directory."""
    processed_data_path = (
        test_fixtures_dir / "datasets" / "poster-layout" / "processed.pkl"
    )
    processed_data_path.parent.mkdir(parents=True, exist_ok=True)
    return processed_data_path


@pytest.fixture(scope="session")
def rico25_processed_data_path(test_fixtures_dir: pathlib.Path) -> pathlib.Path:
    """Return the path to the processed rico data directory."""
    processed_data_path = test_fixtures_dir / "datasets" / "rico25" / "processed.pkl"
    processed_data_path.parent.mkdir(parents=True, exist_ok=True)
    return processed_data_path


@pytest.fixture(scope="session")
def raw_hf_poster_layout_dataset() -> ds.DatasetDict:
    """Return the raw Hugging Face dataset for Poster Layout."""
    return load_raw_poster_layout()


@pytest.fixture(scope="session")
def raw_poster_layout_dataset() -> ds.DatasetDict:
    """Return the processed Hugging Face dataset for Poster Layout."""
    return load_poster_layout()


@pytest.fixture(scope="session")
def raw_hf_rico_dataset() -> ds.DatasetDict:
    """Return the raw Rico dataset."""
    return load_raw_rico()


@pytest.fixture(scope="session")
def raw_rico25_dataset() -> ds.DatasetDict:
    """Return the raw Rico 25 dataset."""
    return load_rico25()


@pytest.fixture(scope="session")
def poster_layout_dataset(
    raw_poster_layout_dataset: ds.DatasetDict,
    poster_layout_processed_data_path: pathlib.Path,
) -> Dict[str, List[LayoutData]]:
    """Load or process the Poster Layout dataset."""
    if poster_layout_processed_data_path.exists():
        logger.debug(
            f"Loading processed poster layout dataset from {poster_layout_processed_data_path}"
        )
        with poster_layout_processed_data_path.open("rb") as rf:
            return pickle.load(rf)

    # Convert Poster Layout dataset to LayoutData format
    layout_dataset = {
        split: [
            LayoutData.model_validate(data)
            for data in tqdm(
                raw_poster_layout_dataset[split],
                desc=f"Processing poster layout for {split}",
            )
        ]
        for split in raw_poster_layout_dataset
    }
    logger.debug(
        f"Saving processed poster layout dataset to {poster_layout_processed_data_path}",
    )
    with poster_layout_processed_data_path.open("wb") as wf:
        pickle.dump(layout_dataset, wf)

    return layout_dataset


@pytest.fixture(scope="session")
def rico25_dataset(
    raw_rico25_dataset: ds.DatasetDict,
    rico25_processed_data_path: pathlib.Path,
) -> Dict[str, List[LayoutData]]:
    """Load or process the Rico dataset."""
    if rico25_processed_data_path.exists():
        logger.debug(
            f"Loading processed rico dataset from {rico25_processed_data_path}"
        )
        with rico25_processed_data_path.open("rb") as rf:
            return pickle.load(rf)

    # Convert Rico dataset to LayoutData format
    layout_dataset = {
        split: [
            LayoutData.model_validate(data)
            for data in tqdm(
                raw_rico25_dataset[split],
                desc=f"Processing poster layout for {split}",
            )
        ]
        for split in raw_rico25_dataset
    }
    logger.debug(
        f"Saving processed rico dataset to {rico25_processed_data_path}",
    )
    with rico25_processed_data_path.open("wb") as wf:
        pickle.dump(layout_dataset, wf)

    return layout_dataset
