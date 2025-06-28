from typing import Any, List, Literal, Tuple

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


class LayoutSerializedData(BaseModel):
    """Protocol for objects that have serialized layout data."""

    class_name: Any
    coord: Coordinates


class LayoutSerializedOutputData(BaseModel):
    """Protocol for objects that have serialized layout data."""

    layouts: List[Any]


class PosterLayoutSerializedData(LayoutSerializedData):
    """Serialized data for poster layouts."""

    class_name: PosterClassNames


class PosterLayoutSerializedOutputData(LayoutSerializedOutputData):
    """Serialized output data for poster layouts."""

    layouts: List[PosterLayoutSerializedData]


class Rico25SerializedData(LayoutSerializedData):
    """Serialized data for Rico25 layouts."""

    class_name: Rico25ClassNames


class Rico25SerializedOutputData(LayoutSerializedOutputData):
    """Serialized output data for Rico25 layouts."""

    layouts: List[Rico25SerializedData]
