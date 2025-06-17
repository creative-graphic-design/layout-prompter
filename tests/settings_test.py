import datasets as ds
from typing import get_args

from layout_prompter.models.serialized_data import PosterClassNames
from layout_prompter.settings import PosterLayoutSettings


def test_poster_layout_settings(raw_dataset: ds.DatasetDict):
    settings = PosterLayoutSettings()

    assert settings.name == "poster-layout"
    assert settings.domain == "poster"
    assert settings.canvas_size.width == 102
    assert settings.canvas_size.height == 150
    assert settings.labels == list(get_args(PosterClassNames))

    # Check if the labels in the settings are the same as in the dataset
    actual_labels = (
        raw_dataset["train"].features["annotations"].feature["cls_elem"].names
    )
    actual_labels = list(filter(lambda label: label != "INVALID", actual_labels))
    assert actual_labels == settings.labels
