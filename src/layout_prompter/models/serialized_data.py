from typing import List, Literal, Tuple

from pydantic import BaseModel


class Coordinates(BaseModel):
    left: int
    top: int
    width: int
    height: int

    def to_tuple(self) -> Tuple[int, int, int, int]:
        return (self.left, self.top, self.width, self.height)


class PosterLayoutSerializedData(BaseModel):
    class_name: Literal["text", "logo", "underlay"]
    coord: Coordinates


class PosterLayoutSerializedOutputData(BaseModel):
    layouts: List[PosterLayoutSerializedData]


class Rico25SerializedData(BaseModel):
    class_name: Literal[
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
    coord: Coordinates


class Rico25SerializedOutputData(BaseModel):
    layouts: List[Rico25SerializedData]
