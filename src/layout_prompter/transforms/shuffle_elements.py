import copy
from typing import Any, Union

from langchain_core.runnables.config import RunnableConfig
from loguru import logger

from layout_prompter.models.layout_data import LayoutData, ProcessedLayoutData
from layout_prompter.transforms import BaseTransform


class ShuffleElements(BaseTransform):
    name: str = "shuffle-elements"

    def invoke(
        self,
        input: Union[LayoutData, ProcessedLayoutData],
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> ProcessedLayoutData:
        assert input.bboxes is not None and input.labels is not None

        gold_bboxes = (
            copy.deepcopy(input.bboxes)
            if isinstance(input, LayoutData)
            else input.gold_bboxes
        )

        breakpoint()
        logger.trace(f"{processed_data=}")
        return processed_data
