from typing import Any, List, Literal, Protocol, Tuple

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


class LayoutSerializedData(Protocol):
    """Protocol for objects that have serialized layout data."""

    class_name: Any
    coord: Coordinates

    def model_dump(self): ...


class LayoutSerializedOutputData(Protocol):
    """Protocol for objects that have serialized layout data."""

    layouts: List[LayoutSerializedData]


class PosterLayoutSerializedData(BaseModel):
    class_name: PosterClassNames
    coord: Coordinates


class PosterLayoutSerializedOutputData(BaseModel):
    layouts: List[PosterLayoutSerializedData]


class Rico25SerializedData(BaseModel):
    class_name: Rico25ClassNames
    coord: Coordinates


class Rico25SerializedOutputData(BaseModel):
    layouts: List[Rico25SerializedData]
