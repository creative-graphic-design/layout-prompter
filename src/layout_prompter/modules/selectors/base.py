import abc
import random
from typing import List, Optional, Tuple

from langchain_core.example_selectors.base import BaseExampleSelector
from loguru import logger
from pydantic import BaseModel, model_validator
from typing_extensions import Self, override

from layout_prompter.models import ProcessedLayoutData


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
