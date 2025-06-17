from typing import Any, List, Literal, Protocol, Tuple, Union

from pydantic import BaseModel


class Coordinates(BaseModel):
    left: int
    top: int
    width: int
    height: int

    def to_tuple(self) -> Tuple[int, int, int, int]:
        return (self.left, self.top, self.width, self.height)


PosterClassNames = Literal[
    "text",
    "logo",
    "underlay",
]

Rico25ClassNames = Literal[
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


class LayoutSerializable(Protocol):
    """Protocol for objects that have serialized layout data."""

    layouts: List[Any]


# 汎用ベースクラス（後方互換性のため）
class SerializedData(BaseModel):
    class_name: Union[str, PosterClassNames, Rico25ClassNames]
    coord: Coordinates


class SerializedOutputData(BaseModel):
    layouts: List[SerializedData]


# 具体的なレイアウト専用クラス
class PosterLayoutSerializedData(SerializedData):
    class_name: PosterClassNames


class PosterLayoutSerializedOutputData(BaseModel):
    layouts: List[PosterLayoutSerializedData]


class Rico25SerializedData(SerializedData):
    class_name: Rico25ClassNames


class Rico25SerializedOutputData(BaseModel):
    layouts: List[Rico25SerializedData]
