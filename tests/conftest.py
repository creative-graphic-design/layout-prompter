import datasets as ds
import pytest

from layout_prompter.datasets import load_poster_layout


@pytest.fixture
def raw_hf_dataset() -> ds.DatasetDict:
    return load_poster_layout(return_raw=True)


@pytest.fixture
def hf_dataset() -> ds.DatasetDict:
    return load_poster_layout()
