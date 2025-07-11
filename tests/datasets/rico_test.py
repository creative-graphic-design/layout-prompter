from layout_prompter.datasets import load_rico

import datasets as ds


def test_load_rico():
    dataset = load_rico()
    assert isinstance(dataset, ds.DatasetDict)
