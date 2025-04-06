from typing import Annotated, Literal

from PIL.Image import Image

PilImage = Annotated[Image, "Pillow Image"]

Task = Literal[
    "gen-t", "gen-ts", "gen-r", "completion", "refinement", "content", "text"
]
