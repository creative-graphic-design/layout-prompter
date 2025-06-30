import pathlib
import pickle
from typing import Dict, List

import pytest
from layout_prompter.datasets import load_poster_layout, load_raw_poste_layout
from layout_prompter.models import LayoutData
from loguru import logger
from tqdm.auto import tqdm

import datasets as ds


@pytest.fixture
def root_dir() -> pathlib.Path:
    return pathlib.Path(__file__).parents[2]


@pytest.fixture
def test_fixtures_dir(root_dir: pathlib.Path) -> pathlib.Path:
    """Return the directory for test fixtures."""
    return root_dir / "test_fixtures"


@pytest.fixture
def processed_data_path(test_fixtures_dir: pathlib.Path) -> pathlib.Path:
    """
    Return the path to the processed data directory.
    This is used to store or access processed datasets.
    """
    processed_data_path = (
        test_fixtures_dir / "datasets" / "poster-layout" / "processed.pkl"
    )
    processed_data_path.parent.mkdir(parents=True, exist_ok=True)
    return processed_data_path


@pytest.fixture
def raw_hf_dataset() -> ds.DatasetDict:
    """Return the raw Hugging Face dataset for Poster Layout."""
    return load_raw_poste_layout()


@pytest.fixture
def hf_dataset() -> ds.DatasetDict:
    """Return the processed Hugging Face dataset for Poster Layout."""
    return load_poster_layout()


@pytest.fixture
def layout_dataset(
    hf_dataset: ds.DatasetDict, processed_data_path: pathlib.Path
) -> Dict[str, List[LayoutData]]:
    """Load or process the Poster Layout dataset.

    If the processed dataset exists, load it; otherwise, process the raw dataset.
    """
    if processed_data_path.exists():
        logger.debug(f"Loading processed dataset from {processed_data_path}")
        with processed_data_path.open("rb") as rf:
            return pickle.load(rf)

    layout_dataset = {
        split: [
            LayoutData.model_validate(data)
            for data in tqdm(hf_dataset[split], desc=f"Processing for {split}")
        ]
        for split in hf_dataset
    }

    logger.debug(f"Saving processed dataset to {processed_data_path}")
    with processed_data_path.open("wb") as wf:
        pickle.dump(layout_dataset, wf)

    return layout_dataset
