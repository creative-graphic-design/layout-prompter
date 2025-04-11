import copy
from typing import Any, Union

import numpy as np
from langchain_core.runnables import Runnable
from langchain_core.runnables.config import RunnableConfig
from pydantic import BaseModel

from layout_prompter.models import LayoutData, ProcessedLayoutData
from layout_prompter.utils import decapsulate


class DiscretizeBboxes(BaseModel, Runnable):
    name: str = "discretize-bboxes"

    num_x_grid: int
    num_y_grid: int

    @property
    def max_x(self) -> int:
        return self.num_x_grid

    @property
    def max_y(self) -> int:
        return self.num_y_grid

    def discretize(self, bboxes: np.ndarray) -> np.ndarray:
        assert bboxes.shape[1] == 4, "bboxes should be of shape (N, 4)"

        clipped_bboxes = np.clip(bboxes, a_min=0.0, a_max=1.0)
        x1, y1, x2, y2 = decapsulate(clipped_bboxes)

        discrete_x1 = np.floor(x1 * self.max_x)
        discrete_y1 = np.floor(y1 * self.max_y)
        discrete_x2 = np.floor(x2 * self.max_x)
        discrete_y2 = np.floor(y2 * self.max_y)

        discrete_bboxes = np.stack(
            [discrete_x1, discrete_y1, discrete_x2, discrete_y2], axis=-1
        )

        return discrete_bboxes.astype(np.int32)

    def continuize(self, bboxes: np.ndarray) -> np.ndarray:
        x1, y1, x2, y2 = decapsulate(bboxes)
        cx1, cx2 = x1 / self.max_x, x2 / self.max_x
        cy1, cy2 = y1 / self.max_y, y2 / self.max_y
        continuize_bboxes = np.stack([cx1, cy1, cx2, cy2], axis=-1)
        return continuize_bboxes.astype(np.float32)

    def invoke(
        self,
        input: Union[LayoutData, ProcessedLayoutData],
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> ProcessedLayoutData:
        canvas_size = input.canvas_size
        bboxes, labels = copy.deepcopy(input.bboxes), copy.deepcopy(input.labels)
        content_bboxes = (
            copy.deepcopy(input.content_bboxes) if input.is_content_aware() else None
        )
        encoded_image = input.encoded_image if isinstance(input, LayoutData) else None

        gold_bboxes = (
            copy.deepcopy(input.bboxes)
            if isinstance(input, LayoutData)
            else input.gold_bboxes
        )
        orig_bboxes = (
            copy.deepcopy(gold_bboxes)
            if isinstance(input, LayoutData)
            else input.orig_bboxes
        )
        orig_labels = (
            copy.deepcopy(input.labels)
            if isinstance(input, LayoutData)
            else input.orig_labels
        )

        discrete_bboxes = self.discretize(bboxes)
        discrete_gold_bboxes = self.discretize(gold_bboxes)

        content_bboxes = (
            copy.deepcopy(input.content_bboxes) if input.is_content_aware() else None
        )
        discrete_content_bboxes = (
            self.discretize(content_bboxes) if content_bboxes is not None else None
        )

        return ProcessedLayoutData(
            bboxes=bboxes,
            labels=labels,
            gold_bboxes=gold_bboxes,
            encoded_image=encoded_image,
            content_bboxes=content_bboxes,
            discrete_bboxes=discrete_bboxes,
            discrete_gold_bboxes=discrete_gold_bboxes,
            discrete_content_bboxes=discrete_content_bboxes,
            orig_bboxes=orig_bboxes,
            orig_labels=orig_labels,
            canvas_size=canvas_size,
        )
