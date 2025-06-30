import logging
import random
from typing import Any, Dict, List, Optional, Union, cast

import numpy as np
import pydantic_numpy.typing as pnd
from langchain_core.runnables.config import RunnableConfig

from layout_prompter.models import LayoutData, ProcessedLayoutData
from layout_prompter.transforms import (
    DiscretizeBboxes,
    LabelDictSort,
    LexicographicSort,
)

from .base import Processor, ProcessorConfig

logger = logging.getLogger(__name__)


class ContentAwareProcessorConfig(ProcessorConfig):
    """Configuration for ContentAwareProcessor."""


class ContentAwareProcessor(Processor):
    name: str = "content-aware-processor"

    max_element_numbers: int = 10

    # Store the possible labels from the training data.
    # During testing, randomly sample from this group for generation.
    _possible_labels: List[pnd.NpNDArray] = []

    def _process(self, layout_data: LayoutData) -> ProcessedLayoutData:
        assert isinstance(layout_data, LayoutData), (
            f"Input must be of type LayoutData. Got: {layout_data=}"
        )
        bboxes, labels = layout_data.bboxes, layout_data.labels
        is_train = bboxes is not None and labels is not None

        if is_train:
            assert labels is not None
            if len(labels) <= self.max_element_numbers:
                # Store the labels for generating the prompt
                self._possible_labels.append(labels)
        else:
            assert len(self._possible_labels) > 0, (
                "Please process the training data first."
            )
            # In the test data, bboxes and labels do not exist.
            # The labels are randomly sampled from the `possible_labels` obtained from the train data.
            # The bboxes are set below the sampled labels.
            labels = random.choice(self._possible_labels)
            bboxes = np.zeros((len(labels), 4))
            logger.debug(f"Sampled {labels=}")

            layout_data = layout_data.model_copy(
                update={"bboxes": bboxes, "labels": labels}
            )

        # Define the chain of preprocess transformations
        transform = LexicographicSort() | LabelDictSort() | DiscretizeBboxes()

        # Execute the transformations
        return transform.invoke(layout_data)
