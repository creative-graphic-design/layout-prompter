import datasets as ds


def load_raw_rico(
    dataset_name: str = "creative-graphic-design/Rico",
) -> ds.DatasetDict:
    dataset = ds.load_dataset(
        dataset_name,
        name="ui-screenshots-and-view-hierarchies",
    )
    assert isinstance(dataset, ds.DatasetDict)

    return dataset


def load_rico(dataset_name: str = "creative-graphic-design/Rico") -> ds.DatasetDict:
    dataset = load_raw_rico(dataset_name)
    raise NotImplementedError
