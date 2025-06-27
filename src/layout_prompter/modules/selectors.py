import random
from typing import List, Optional, Tuple

import cv2
import numpy as np
from langchain_core.example_selectors.base import BaseExampleSelector
from pydantic import BaseModel, model_validator
from typing_extensions import Self

from layout_prompter.models import ProcessedLayoutData
from layout_prompter.settings import CanvasSize


class LayoutSelector(BaseExampleSelector, BaseModel):
    examples: List[ProcessedLayoutData]
    canvas_size: CanvasSize
    num_prompt: int = 10
    candidate_size: Optional[int] = None
    is_shuffle: bool = True

    @model_validator(mode="after")
    def post_int(self) -> Self:
        if self.candidate_size is None:
            return self

        random.shuffle(self.examples)
        self.examples = self.examples[: self.candidate_size]

        return self

    def select_examples(  # type: ignore[override]
        self, input_variables: ProcessedLayoutData
    ) -> List[ProcessedLayoutData]:
        raise NotImplementedError

    def add_example(  # type: ignore[override]
        self,
        example: ProcessedLayoutData,
    ) -> None:
        self.examples.append(example)

    def _is_filter(self, data: ProcessedLayoutData) -> bool:
        discrete_gold_bboxes = data.discrete_gold_bboxes
        assert discrete_gold_bboxes is not None

        num = (discrete_gold_bboxes[:, 2:] == 0).sum().item()
        return bool(num)

    def _retrieve_examples(
        self, scores: List[Tuple[int, float]]
    ) -> List[ProcessedLayoutData]:
        scores = sorted(scores, key=lambda x: x[1], reverse=True)
        assert len(scores) == len(self.examples)

        candidates = []
        for idx, _ in scores:
            candidate = self.examples[idx]
            if not self._is_filter(candidate):
                candidates.append(candidate)
                if len(candidates) == self.num_prompt:
                    break

        if self.is_shuffle:
            random.shuffle(candidates)

        return candidates


class ContentAwareSelector(LayoutSelector):
    def _to_binary_image(self, content_bboxes: np.ndarray) -> np.ndarray:
        binary_image = np.zeros(
            (self.canvas_size.height, self.canvas_size.width),
            dtype=np.uint8,
        )
        content_bboxes = content_bboxes.tolist()
        for content_bbox in content_bboxes:
            left, top, width, height = content_bbox
            cv2.rectangle(
                binary_image,
                (left, top),
                (left + width, top + height),
                (255,),
                thickness=-1,
            )
        return binary_image

    def _calculate_iou(
        self, query: ProcessedLayoutData, candidate: ProcessedLayoutData
    ) -> float:
        query_content_bboxes = query.discrete_content_bboxes
        assert query_content_bboxes is not None
        query_binary = self._to_binary_image(query_content_bboxes)

        candidate_content_bboxes = candidate.discrete_content_bboxes
        assert candidate_content_bboxes is not None
        candidate_binary = self._to_binary_image(candidate_content_bboxes)

        intersection = cv2.bitwise_and(candidate_binary, query_binary)
        union = cv2.bitwise_or(candidate_binary, query_binary)
        iou = (np.sum(intersection) + 1) / (np.sum(union) + 1)
        return iou.item()

    def select_examples(  # type: ignore[override]
        self,
        input_variables: ProcessedLayoutData,
    ) -> List[ProcessedLayoutData]:
        scores = [
            (idx, self._calculate_iou(input_variables, candidate))
            for idx, candidate in enumerate(self.examples)
        ]
        return self._retrieve_examples(scores)
