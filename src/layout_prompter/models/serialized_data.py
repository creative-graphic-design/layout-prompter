from typing import Any, List, Literal

from pydantic import BaseModel

from .layout_data import Bbox

PosterClassNames = Literal[
    "text",
    "logo",
    "underlay",
]

Rico25ClassNames = Literal[
    "Text",
    "Image",
    "Icon",
    "Text Button",
    "List Item",
    "Input",
    "Background Image",
    "Card",
    "Web View",
    "Radio Button",
    "Drawer",
    "Checkbox",
    "Advertisement",
    "Modal",
    "Pager Indicator",
    "Slider",
    "On/Off Switch",
    "Button Bar",
    "Toolbar",
    "Number Stepper",
    "Multi-Tab",
    "Date Picker",
    "Map View",
    "Video",
    "Bottom Navigation",
]


class LayoutSerializedData(BaseModel):
    """Protocol for objects that have serialized layout data."""

    class_name: Any
    bbox: Bbox


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
