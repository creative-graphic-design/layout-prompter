import math
from functools import cached_property
from typing import Generic, Optional, Sequence, Tuple, TypeVar

from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Self

from layout_prompter.typehints import PilImage
from layout_prompter.utils import base64_to_pil

CoordinateType = TypeVar("CoordinateType", int, float)


class CanvasSize(BaseModel):
    width: int
    height: int

    model_config = ConfigDict(
        frozen=True,  # for hashable CanvasSize
    )


class BaseBbox(BaseModel, Generic[CoordinateType]):
    """Generic base class for bounding boxes."""

    left: CoordinateType
    top: CoordinateType
    width: CoordinateType
    height: CoordinateType

    model_config = ConfigDict(
        frozen=True,
        strict=True,  # to ensure all fields are required
        extra="forbid",  # to prevent additional fields
        revalidate_instances="always",  # to revalidate on every instance creation
    )

    @property
    def right(self) -> CoordinateType:
        """Calculate the right coordinate of the bounding box."""
        return self.left + self.width

    @property
    def bottom(self) -> CoordinateType:
        """Calculate the bottom coordinate of the bounding box."""
        return self.top + self.height

    def to_ltwh(
        self,
    ) -> Tuple[CoordinateType, CoordinateType, CoordinateType, CoordinateType]:
        """Convert bounding box to left, top, width, height format."""
        return (self.left, self.top, self.width, self.height)

    def to_ltrb(
        self,
    ) -> Tuple[CoordinateType, CoordinateType, CoordinateType, CoordinateType]:
        """Convert bounding box to left, top, right, bottom format."""
        return (self.left, self.top, self.right, self.bottom)


class Bbox(BaseBbox[int]):
    """Bounding box in absolute pixel coordinates."""

    left: int = Field(
        ge=0,
        description="Left coordinate of the bounding box",
    )
    top: int = Field(
        ge=0,
        description="Top coordinate of the bounding box",
    )
    width: int = Field(
        ge=0,
        description="Width of the bounding box",
    )
    height: int = Field(
        ge=0,
        description="Height of the bounding box",
    )


class NormalizedBbox(BaseBbox[float]):
    left: float = Field(
        ge=0.0,
        le=1.0,
        description="Left coordinate of the normalized bounding box",
    )
    top: float = Field(
        ge=0.0,
        le=1.0,
        description="Top coordinate of the normalized bounding box",
    )
    width: float = Field(
        ge=0.0,
        le=1.0,
        description="Width of the normalized bounding box",
    )
    height: float = Field(
        ge=0.0,
        le=1.0,
        description="Height of the normalized bounding box",
    )

    def discretize(self, canvas_size: CanvasSize) -> Bbox:
        """Convert normalized bounding box to absolute pixel coordinates."""

        return Bbox(
            left=math.floor(self.left * canvas_size.width),
            top=math.floor(self.top * canvas_size.height),
            width=math.floor(self.width * canvas_size.width),
            height=math.floor(self.height * canvas_size.height),
        )


class LayoutData(BaseModel):
    idx: Optional[int] = Field(
        default=None,
        description="Index of the layout data",
    )

    bboxes: Optional[Sequence[NormalizedBbox]] = Field(
        description="List of bounding boxes in normalized coordinates"
    )
    labels: Optional[Sequence[str]] = Field(
        description="List of labels for the bounding boxes",
    )

    canvas_size: CanvasSize

    encoded_image: Optional[str]
    content_bboxes: Optional[Sequence[NormalizedBbox]]

    model_config = ConfigDict(
        frozen=True,
        strict=True,  # to ensure all fields are required
        extra="forbid",  # to prevent additional fields
        revalidate_instances="always",  # to revalidate on every instance creation
    )

    @model_validator(mode="after")
    def validate_bboxes_and_labels(self) -> Self:
        if self.bboxes is not None and self.labels is not None:
            assert len(self.bboxes) == len(self.labels), (
                "The number of bounding boxes must match the number of labels."
            )
        return self

    @cached_property
    def content_image(self) -> PilImage:
        """Get the content image from the encoded image."""
        assert self.encoded_image is not None, (
            "Encoded image must be provided to get content image."
        )
        return base64_to_pil(self.encoded_image)

    def is_content_aware(self) -> bool:
        """Check if the layout data is content-aware."""
        return self.encoded_image is not None or self.content_bboxes is not None


class ProcessedLayoutData(LayoutData):
    gold_bboxes: Sequence[NormalizedBbox] = Field(
        description="List of ground truth bounding boxes in normalized coordinates"
    )

    orig_bboxes: Sequence[NormalizedBbox] = Field(
        description="List of original bounding boxes in normalized coordinates"
    )
    orig_labels: Sequence[str] = Field(
        description="List of original labels for the bounding boxes",
    )
    orig_canvas_size: CanvasSize = Field(
        description="Original canvas size of the layout data"
    )

    discrete_bboxes: Optional[Sequence[Bbox]] = Field(
        description="List of discretized bounding boxes in normalized coordinates"
    )
    discrete_gold_bboxes: Optional[Sequence[Bbox]] = Field(
        description="List of discretized ground truth bounding boxes in normalized coordinates"
    )
    discrete_content_bboxes: Optional[Sequence[Bbox]] = Field(
        description="List of discretized content bounding boxes in normalized coordinates"
    )
