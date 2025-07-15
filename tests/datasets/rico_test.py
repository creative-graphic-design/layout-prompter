import datasets as ds

from layout_prompter.datasets import load_raw_rico


def test_load_rico():
    dataset = load_raw_rico()
    assert isinstance(dataset, ds.DatasetDict)
