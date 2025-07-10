import abc
import random
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pydantic_numpy.typing as pnd
from langchain_core.example_selectors.base import BaseExampleSelector
from loguru import logger
from pydantic import BaseModel, model_validator
from tqdm.auto import tqdm
from typing_extensions import Self, override

from layout_prompter.models import Bbox, CanvasSize, ProcessedLayoutData


class LayoutSelectorOutput(BaseModel):
    selected_examples: List[ProcessedLayoutData]


class LayoutSelector(BaseExampleSelector, BaseModel):
    examples: List[ProcessedLayoutData]
    num_prompt: int = 10
    candidate_size: Optional[int] = None
    is_shuffle: bool = True

    @model_validator(mode="after")
    def post_int(self) -> Self:
        if self.candidate_size is None:
            return self

        logger.debug(
            f"Selecting {self.candidate_size} candidates from {len(self.examples)} examples."
        )
        random.shuffle(self.examples)
        self.examples = self.examples[: self.candidate_size]

        return self

    @override
    @abc.abstractmethod
    def select_examples(  # type: ignore[override]
        self, input_variables: ProcessedLayoutData
    ) -> LayoutSelectorOutput:
        raise NotImplementedError

    @override
    def add_example(  # type: ignore[override]
        self,
        example: ProcessedLayoutData,
    ) -> None:
        self.examples.append(example)

    def _is_filter(self, data: ProcessedLayoutData) -> bool:
        """Filtering function to exclude data with bboxes that have width or height of 0."""
        discrete_gold_bboxes = data.discrete_gold_bboxes
        assert discrete_gold_bboxes is not None

        num_invalid_bboxes = sum(
            [(bbox.width == 0) + (bbox.height == 0) for bbox in discrete_gold_bboxes]
        )
        return num_invalid_bboxes > 0

    def _retrieve_examples(
        self, scores: List[Tuple[int, float]], return_indices: bool = False
    ) -> List[Tuple[int, ProcessedLayoutData]]:
        scores = sorted(scores, key=lambda x: x[1], reverse=True)
        assert len(scores) == len(self.examples)

        candidates: List[Tuple[int, ProcessedLayoutData]] = []
        for idx, _ in scores:
            candidate = self.examples[idx]
            if not self._is_filter(candidate):
                candidates.append((idx, candidate))
                if len(candidates) == self.num_prompt:
                    break

        if self.is_shuffle:
            random.shuffle(candidates)

        return candidates


class ContentAwareSelectorOutput(LayoutSelectorOutput):
    query_saliency_map: Optional[pnd.NpNDArray] = None
    candidate_saliency_maps: Optional[List[pnd.NpNDArray]] = None


def calculate_iou(
    query_saliency_map: np.ndarray, candidate_saliency_map: np.ndarray
) -> float:
    intersection = cv2.bitwise_and(query_saliency_map, candidate_saliency_map)
    union = cv2.bitwise_or(query_saliency_map, candidate_saliency_map)
    iou = (np.sum(intersection) + 1) / (np.sum(union) + 1)
    return iou.item()


class ContentAwareSelector(LayoutSelector):
    return_saliency_maps: bool = False

    def _to_binary_image(
        self, content_bboxes: Sequence[Bbox], canvas_size: CanvasSize
    ) -> np.ndarray:
        binary_image = np.zeros(
            (canvas_size.height, canvas_size.width),
            dtype=np.uint8,
        )
        for content_bbox in content_bboxes:
            cv2.rectangle(
                img=binary_image,
                pt1=(content_bbox.left, content_bbox.top),
                pt2=(content_bbox.right, content_bbox.bottom),
                color=(255,),
                thickness=-1,
            )
        return binary_image

    @override
    def select_examples(  # type: ignore[override]
        self,
        input_variables: ProcessedLayoutData,
    ) -> ContentAwareSelectorOutput:
        logger.debug(
            f"Selecting {self.num_prompt} candidates from {len(self.examples)} examples."
        )

        query = input_variables
        query_content_bboxes = query.discrete_content_bboxes
        assert query_content_bboxes is not None
        query_saliency_map = self._to_binary_image(
            query_content_bboxes, canvas_size=query.canvas_size
        )

        scores: List[Tuple[int, float]] = []
        candidate_saliency_maps: List[np.ndarray] = []

        it = tqdm(
            self.examples,
            desc="Calculating scores for selecting candidate examples",
        )
        for idx, candidate in enumerate(it):
            candidate_content_bboxes = candidate.discrete_content_bboxes
            assert candidate_content_bboxes is not None
            candidate_saliency_map = self._to_binary_image(
                candidate_content_bboxes, canvas_size=candidate.canvas_size
            )
            candidate_saliency_maps.append(candidate_saliency_map)

            iou = calculate_iou(
                query_saliency_map=query_saliency_map,
                candidate_saliency_map=candidate_saliency_map,
            )
            scores.append((idx, iou))

        candidates = self._retrieve_examples(
            scores, return_indices=self.return_saliency_maps
        )
        candidate_indices = [idx for idx, _ in candidates]
        candidate_examples = [example for _, example in candidates]

        return ContentAwareSelectorOutput(
            selected_examples=candidate_examples,
            query_saliency_map=query_saliency_map
            if self.return_saliency_maps
            else None,
            candidate_saliency_maps=[
                candidate_saliency_maps[idx] for idx in candidate_indices
            ]
            if self.return_saliency_maps
            else None,
        )
