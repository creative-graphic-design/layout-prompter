import pathlib

from pydantic import BaseModel
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)


class CanvasSize(BaseModel):
    width: int
    height: int


class TaskSettings(BaseSettings):
    name: str
    domain: str
    canvas_size: CanvasSize

    @classmethod
    def settings_customise_sources(
        cls, settings_cls: type[BaseSettings], *args, **kwargs
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (YamlConfigSettingsSource(settings_cls),)


class PosterLayoutSettings(TaskSettings):
    model_config = SettingsConfigDict(
        yaml_file=pathlib.Path(__file__).resolve().parents[2]
        / "settings"
        / "poster_layout.yaml",
    )
