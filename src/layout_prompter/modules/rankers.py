from typing import Any, List, Optional

import numpy as np
from langchain_core.runnables import Runnable
from langchain_core.runnables.config import RunnableConfig
from pydantic import BaseModel, model_validator
from typing_extensions import Self

from layout_prompter.models import LayoutSerializedOutputData
from layout_prompter.utils import (
    compute_alignment,
    compute_overlap,
    convert_ltwh_to_ltrb,
)


class LayoutRanker(Runnable):
    """Base class for layout ranking algorithms."""

    def invoke(
        self,
        input: List[LayoutSerializedOutputData],
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> List[LayoutSerializedOutputData]:
        raise NotImplementedError


class LayoutPrompterRanker(BaseModel, LayoutRanker):
    name: str = "layout-prompter-ranker"

    lam_ali: float = 0.2
    lam_ove: float = 0.2
    lam_iou: float = 0.6

    @model_validator(mode="after")
    def check_lambda_params(self) -> Self:
        assert self.lam_ali + self.lam_ove + self.lam_iou == 1.0, self
        return self

    def invoke(
        self,
        input: List[LayoutSerializedOutputData],
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> List[LayoutSerializedOutputData]:
        metrics = []
        for data in input:
            bboxes = np.array([layout.coord.to_tuple() for layout in data.layouts])
            labels = np.array([layout.class_name for layout in data.layouts])

            bboxes = convert_ltwh_to_ltrb(bboxes)
            bboxes = bboxes[None, :, :]

            labels = labels[None, :]
            padmsk = np.ones_like(labels, dtype=bool)

            ali_score = compute_alignment(bboxes, padmsk)
            ove_score = compute_overlap(bboxes, padmsk)
            metrics.append((ali_score, ove_score))

        metrics_arr = np.array(metrics)

        min_vals = np.min(metrics_arr, axis=0, keepdims=True)
        max_vals = np.max(metrics_arr, axis=0, keepdims=True)

        scaled_metrics = (metrics_arr - min_vals) / (max_vals - min_vals)

        quality = (
            scaled_metrics[:, 0] * self.lam_ali + scaled_metrics[:, 1] * self.lam_ove
        )

        # Sort the input based on the quality scores
        sorted_input = sorted(zip(input, quality), key=lambda x: x[1])

        # The above data is a list of tuples of (input, query),
        # so in the end, only the first input is picked up and returned
        return [item[0] for item in sorted_input]
