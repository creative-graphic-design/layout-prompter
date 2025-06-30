from .layout_data import LayoutData, ProcessedLayoutData
from .serialized_data import (
    Coordinates,
    LayoutSerializedData,
    LayoutSerializedOutputData,
    PosterLayoutSerializedData,
    PosterLayoutSerializedOutputData,
    Rico25SerializedData,
    Rico25SerializedOutputData,
)

# Aliases for backward compatibility
SerializedData = LayoutSerializedData
SerializedOutputData = LayoutSerializedOutputData

__all__ = [
    "LayoutData",
    "ProcessedLayoutData",
    "Coordinates",
    #
    # Base Protocols
    #
    "LayoutSerializedData",
    "LayoutSerializedOutputData",
    #
    # Backward compatibility aliases
    #
    "SerializedData",
    "SerializedOutputData",
    #
    # Poster Layout
    #
    "PosterLayoutSerializedData",
    "PosterLayoutSerializedOutputData",
    #
    # Rico-25 Layout
    #
    "Rico25SerializedData",
    "Rico25SerializedOutputData",
]
