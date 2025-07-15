import abc
from typing import Any, List, Optional, Tuple

import numpy as np
from langchain_core.runnables import RunnableSerializable
from langchain_core.runnables.config import RunnableConfig
from pydantic import model_validator
from typing_extensions import Self

from layout_prompter.models import LayoutSerializedOutputData
from layout_prompter.utils import (
    compute_alignment,
    compute_overlap,
)


class LayoutRanker(RunnableSerializable):
    """Base class for layout ranking algorithms."""

    @abc.abstractmethod
    def invoke(
        self,
        input: List[LayoutSerializedOutputData],
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> List[LayoutSerializedOutputData]:
        raise NotImplementedError


class LayoutPrompterRanker(LayoutRanker):
    name: str = "layout-prompter-ranker"

    lam_ali: float = 0.2
    lam_ove: float = 0.2
    lam_iou: float = 0.6

    @model_validator(mode="after")
    def check_lambda_params(self) -> Self:
        assert self.lam_ali + self.lam_ove + self.lam_iou == 1.0, self
        return self

    def calculate_metrics(self, data: LayoutSerializedOutputData) -> Tuple[float, ...]:
        if not data.layouts:
            raise ValueError("Cannot calculate metrics for empty layouts")
        bboxes = np.array([layout.bbox.to_ltrb() for layout in data.layouts])
        labels = np.array([layout.class_name for layout in data.layouts])

        bboxes, labels = bboxes[None, :, :], labels[None, :]
        padmsk = np.ones_like(labels, dtype=bool)

        ali_score = compute_alignment(bboxes, padmsk)
        ove_score = compute_overlap(bboxes, padmsk)
        return (ali_score, ove_score)

    def invoke(
        self,
        input: List[LayoutSerializedOutputData],
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> List[LayoutSerializedOutputData]:
        metrics_arr = np.array([self.calculate_metrics(data) for data in input])

        min_vals = np.min(metrics_arr, axis=0, keepdims=True)
        max_vals = np.max(metrics_arr, axis=0, keepdims=True)

        scaled_metrics = (metrics_arr - min_vals) / (max_vals - min_vals)

        # Calculate the quality score based on the weighted sum of the metrics
        quality = (
            scaled_metrics[:, 0] * self.lam_ali + scaled_metrics[:, 1] * self.lam_ove
        )

        # Sort the input based on the quality scores
        sorted_input = sorted(zip(input, quality), key=lambda x: x[1])

        # The above data is a list of tuples of (input, query),
        # so in the end, only the first input is picked up and returned
        return [item[0] for item in sorted_input]
