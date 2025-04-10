from functools import cached_property
from typing import Optional

import pydantic_numpy.typing as pnd
from pydantic import BaseModel

from layout_prompter.settings import CanvasSize
from layout_prompter.typehints import PilImage
from layout_prompter.utils import base64_to_pil


class LayoutData(BaseModel):
    bboxes: pnd.Np2DArray
    labels: pnd.NpNDArray
    canvas_size: CanvasSize

    encoded_image: Optional[str]
    encoded_saliency_map: Optional[str]
    content_bboxes: Optional[pnd.Np2DArray]

    @cached_property
    def content_image(self) -> PilImage:
        assert self.encoded_image is not None
        return base64_to_pil(self.encoded_image)

    @cached_property
    def saliency_map(self) -> PilImage:
        assert self.encoded_saliency_map is not None
        return base64_to_pil(self.encoded_saliency_map)

    def is_content_aware(self) -> bool:
        return self.encoded_image is not None and self.content_bboxes is not None


class ProcessedLayoutData(LayoutData):
    gold_bboxes: pnd.Np2DArray

    orig_bboxes: pnd.Np2DArray
    orig_labels: pnd.NpNDArray

    discrete_bboxes: Optional[pnd.Np2DArrayInt32]
    discrete_gold_bboxes: Optional[pnd.Np2DArrayInt32]
    discrete_content_bboxes: Optional[pnd.Np2DArrayInt32]
