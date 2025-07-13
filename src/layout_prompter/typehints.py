from typing import Annotated, Literal

from PIL.Image import Image

PilImage = Annotated[Image, "Pillow Image"]

Task = Literal[
    "gen-t",  # for generation conditioned on types
    "gen-ts",  # for generation conditioned on types with sizes
    "gen-r",  # for generation conditioned on relationships
    "completion",  # for completion aimed to complete layout from a set of specified elements
    "refinement",  # for refinement of a layout that applies local changes to the elements that need improvements while maintaining the original layout design
    "content",  # for generation conditioned on base image content
    "text",  # for text-to-layout generation
]
