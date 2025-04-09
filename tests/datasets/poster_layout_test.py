import datasets as ds

from layout_prompter.datasets import load_poster_layout


def test_load_rico():
    dataset = load_poster_layout()
    assert isinstance(dataset, ds.DatasetDict)
