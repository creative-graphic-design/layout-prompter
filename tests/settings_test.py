from typing import get_args

from layout_prompter.models.serialized_data import PosterClassNames
from layout_prompter.settings import CanvasSize, PosterLayoutSettings, TaskSettings

import datasets as ds


def test_canvas_size():
    """Test CanvasSize model"""
    canvas = CanvasSize(width=100, height=200)
    assert canvas.width == 100
    assert canvas.height == 200


def test_canvas_size_edge_cases():
    """Test CanvasSize with edge cases"""
    # Zero dimensions
    canvas_zero = CanvasSize(width=0, height=0)
    assert canvas_zero.width == 0
    assert canvas_zero.height == 0

    # Large dimensions
    canvas_large = CanvasSize(width=10000, height=20000)
    assert canvas_large.width == 10000
    assert canvas_large.height == 20000


def test_poster_layout_settings():
    """Test PosterLayoutSettings without dataset dependency"""
    settings = PosterLayoutSettings()

    assert settings.name == "poster-layout"
    assert settings.domain == "poster"
    assert settings.canvas_size.width == 102
    assert settings.canvas_size.height == 150
    assert settings.labels == list(get_args(PosterClassNames))

    # Verify labels are the expected poster class names
    expected_labels = ["text", "logo", "underlay"]
    assert settings.labels == expected_labels


def test_poster_layout_settings_with_dataset(raw_hf_dataset: ds.DatasetDict):
    """Test PosterLayoutSettings against actual dataset"""
    settings = PosterLayoutSettings()

    # Check if the labels in the settings are the same as in the dataset
    actual_labels = (
        raw_hf_dataset["train"].features["annotations"].feature["cls_elem"].names
    )
    actual_labels = list(filter(lambda label: label != "INVALID", actual_labels))
    assert actual_labels == settings.labels


def test_poster_layout_settings_canvas_size():
    """Test canvas size properties"""
    settings = PosterLayoutSettings()
    canvas = settings.canvas_size

    assert isinstance(canvas, CanvasSize)
    assert canvas.width == 102
    assert canvas.height == 150


def test_poster_layout_settings_yaml_loading():
    """Test that settings are loaded from YAML file"""
    settings = PosterLayoutSettings()

    # These values should match what's in poster_layout.yaml
    assert settings.name == "poster-layout"
    assert settings.domain == "poster"
    assert settings.canvas_size.width == 102
    assert settings.canvas_size.height == 150


def test_task_settings_customise_sources():
    """Test the settings customise sources method"""
    sources = TaskSettings.settings_customise_sources(TaskSettings)
    assert len(sources) == 1
    # Should return a YamlConfigSettingsSource
    from pydantic_settings import YamlConfigSettingsSource

    assert isinstance(sources[0], YamlConfigSettingsSource)
