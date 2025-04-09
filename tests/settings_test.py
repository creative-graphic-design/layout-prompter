import datasets as ds

from layout_prompter.settings import PosterLayoutSettings, Rico25Settings


def test_rico25_settings(raw_dataset: ds.DatasetDict):
    settings = Rico25Settings()

    assert settings.name == "rico25"
    assert settings.domain == "android"
    assert settings.canvas_size.width == 90
    assert settings.canvas_size.height == 160
    assert settings.labels == [
        "text",
        "image",
        "icon",
        "list-item",
        "text-button",
        "toolbar",
        "web-view",
        "input",
        "card",
        "advertisement",
        "background-image",
        "drawer",
        "radio-button",
        "checkbox",
        "multi-tab",
        "pager-indicator",
        "modal",
        "on/off-switch",
        "slider",
        "map-view",
        "button-bar",
        "video",
        "bottom-navigation",
        "number-stepper",
        "date-picker",
    ]
    assert len(settings.labels) == 25  # Rico25 dataset has 25 labels

    breakpoint()

    # # Check if the labels in the settings are the same as in the dataset
    # actual_labels = (
    #     raw_dataset["train"].features["annotations"].feature["cls_elem"].names
    # )
    # actual_labels = list(filter(lambda label: label != "INVALID", actual_labels))
    # assert actual_labels == settings.labels


def test_poster_layout_settings(raw_dataset: ds.DatasetDict):
    settings = PosterLayoutSettings()

    assert settings.name == "poster-layout"
    assert settings.domain == "poster"
    assert settings.canvas_size.width == 102
    assert settings.canvas_size.height == 150
    assert settings.labels == ["text", "logo", "underlay"]

    # Check if the labels in the settings are the same as in the dataset
    actual_labels = (
        raw_dataset["train"].features["annotations"].feature["cls_elem"].names
    )
    actual_labels = list(filter(lambda label: label != "INVALID", actual_labels))
    assert actual_labels == settings.labels
