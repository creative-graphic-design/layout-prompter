import copy
from typing import Any, List, Tuple, Union, cast

from langchain_core.runnables import Runnable
from langchain_core.runnables.config import RunnableConfig
from loguru import logger

from layout_prompter.models import LayoutData, NormalizedBbox, ProcessedLayoutData


class LabelDictSort(Runnable):
    name: str = "label-dict-sort"

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

        # Sort bboxes and labels based on the label dictionary
        combined = list(
            zip(
                bboxes,  # 0
                gold_bboxes,  # 1
                labels,  # 2 is the index for labels
            )
        )
        sorted_combined = sorted(
            combined,
            key=lambda x: x[2],  # 2 is the index for labels as you can see above
        )
        ## Unpack the sorted combined list
        bboxes, gold_bboxes, labels = cast(
            Tuple[List[NormalizedBbox], List[NormalizedBbox], List[str]],
            zip(*sorted_combined),
        )

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
