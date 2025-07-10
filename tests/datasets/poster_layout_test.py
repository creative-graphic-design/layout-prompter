import datasets as ds

from layout_prompter.datasets import load_poster_layout


def test_load_poster_layout(
    expected_num_train: int = 9885,
    expected_num_test: int = 902,
):
    dataset = load_poster_layout()
    assert isinstance(dataset, ds.DatasetDict)

    assert dataset["train"].num_rows == expected_num_train
    assert dataset["test"].num_rows == expected_num_test
