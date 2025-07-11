import copy
from typing import Any, Union

from langchain_core.runnables import RunnableSerializable
from langchain_core.runnables.config import RunnableConfig
from loguru import logger

from layout_prompter.models import CanvasSize, LayoutData, ProcessedLayoutData


class DiscretizeBboxes(RunnableSerializable):
    name: str = "discretize-bboxes"
    target_canvas_size: CanvasSize

    def invoke(
        self,
        input: Union[LayoutData, ProcessedLayoutData],
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> ProcessedLayoutData:
        assert input.bboxes is not None and input.labels is not None

        logger.trace(
            f"Discretize from {input.canvas_size=} to the {self.target_canvas_size=}"
        )

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
        orig_canvas_size = (
            input.orig_canvas_size
            if isinstance(input, ProcessedLayoutData)
            else input.canvas_size
        )

        discrete_bboxes = [
            bbox.discretize(canvas_size=self.target_canvas_size) for bbox in bboxes
        ]
        discrete_gold_bboxes = [
            bbox.discretize(canvas_size=self.target_canvas_size) for bbox in gold_bboxes
        ]

        content_bboxes = (
            copy.deepcopy(input.content_bboxes) if input.is_content_aware() else None
        )
        discrete_content_bboxes = (
            [
                bbox.discretize(canvas_size=self.target_canvas_size)
                for bbox in content_bboxes
            ]
            if content_bboxes is not None
            else None
        )

        processed_data = ProcessedLayoutData(
            idx=input.idx,
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
            orig_canvas_size=orig_canvas_size,
            canvas_size=self.target_canvas_size,
        )
        logger.trace(f"{processed_data=}")
        return processed_data
