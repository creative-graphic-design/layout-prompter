import copy
from typing import Any, Union

from langchain_core.runnables import Runnable
from langchain_core.runnables.config import RunnableConfig
from loguru import logger

from layout_prompter.models import LayoutData, ProcessedLayoutData


class LexicographicSort(Runnable):
    name: str = "lexicographic-sort"

    def invoke(
        self,
        input: Union[LayoutData, ProcessedLayoutData],
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> ProcessedLayoutData:
        assert input.bboxes is not None and input.labels is not None

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

        # Extract left and top coordinates from bboxes
        assert input.bboxes is not None
        left, top, _, _ = input.bboxes.T

        # Get the indices of the sorted bboxes based on left and top coordinates
        sorter = zip(*sorted(enumerate(zip(top, left)), key=lambda c: c[1:]))
        sorted_indices = list(list(sorter)[0])

        # Sort bboxes and labels based on the sorted indices
        bboxes, labels = bboxes[sorted_indices], labels[sorted_indices]
        gold_bboxes = gold_bboxes[sorted_indices]

        # Return the processed layout data
        processed_data = ProcessedLayoutData(
            idx=input.idx,
            bboxes=bboxes,
            labels=labels,
            encoded_image=encoded_image,
            content_bboxes=content_bboxes,
            gold_bboxes=gold_bboxes,
            orig_bboxes=orig_bboxes,
            orig_labels=orig_labels,
            discrete_bboxes=None,
            discrete_gold_bboxes=None,
            discrete_content_bboxes=None,
            canvas_size=canvas_size,
        )
        logger.trace(f"{processed_data=}")
        return processed_data
