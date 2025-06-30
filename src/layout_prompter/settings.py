import pathlib
from typing import List, get_args

from pydantic import BaseModel
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

from layout_prompter.models.serialized_data import PosterClassNames


class CanvasSize(BaseModel):
    width: int
    height: int


class TaskSettings(BaseSettings):
    name: str
    domain: str
    canvas_size: CanvasSize
    labels: List[str]

    @classmethod
    def settings_customise_sources(
        cls, settings_cls: type[BaseSettings], *args, **kwargs
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (YamlConfigSettingsSource(settings_cls),)


class PosterLayoutSettings(TaskSettings):
    labels: List[str] = list(get_args(PosterClassNames))

    model_config = SettingsConfigDict(
        yaml_file=pathlib.Path(__file__).resolve().parents[2]
        / "settings"
        / "poster_layout.yaml",
    )


class Rico25Settings(TaskSettings):
    labels: List[str] = list(get_args(PosterClassNames))

    model_config = SettingsConfigDict(
        yaml_file=pathlib.Path(__file__).resolve().parents[2]
        / "settings"
        / "rico25.yaml",
    )
