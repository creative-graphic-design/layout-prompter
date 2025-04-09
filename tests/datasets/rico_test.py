import datasets as ds

from layout_prompter.datasets import load_rico


def test_load_rico():
    dataset = load_rico()
    assert isinstance(dataset, ds.DatasetDict)
