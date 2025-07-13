import random
from typing import Any, List, Optional, Tuple, Union

from langchain_core.runnables import RunnableSerializable
from langchain_core.runnables.config import (
    RunnableConfig,
    ensure_config,
    get_callback_manager_for_config,
)
from loguru import logger
from pydantic import ConfigDict

from layout_prompter.models import (
    CanvasSize,
    LayoutData,
    NormalizedBbox,
    ProcessedLayoutData,
)
from layout_prompter.transforms import (
    DiscretizeBboxes,
    LabelDictSort,
    LexicographicSort,
)
from layout_prompter.utils import Configuration


class ProcessorConfig(Configuration):
    """Base Configuration for Processor."""


class Processor(RunnableSerializable):
    """Base class for all processors."""

    model_config = ConfigDict(
        frozen=True,  # for hashable Processor
    )


class ContentAwareProcessorConfig(ProcessorConfig):
    """Configuration for ContentAwareProcessor."""

    labels_for_generation: Optional[List[str]] = None


class ContentAwareProcessor(Processor):
    name: str = "content-aware-processor"

    target_canvas_size: CanvasSize
    max_element_numbers: int = 10

    # Store the possible labels from the training data.
    # During testing, randomly sample from this group for generation.
    _possible_labels: Tuple[Tuple[str, ...], ...] = tuple()  # type: ignore[assignment]

    def _invoke(
        self, layout_data: LayoutData, config: Optional[RunnableConfig] = None, **kwargs
    ) -> ProcessedLayoutData:
        conf = ContentAwareProcessorConfig.from_runnable_config(config)

        assert isinstance(layout_data, LayoutData), (
            f"Input must be of type LayoutData. Got: {type(layout_data)=}. "
            "If you want to preprocess multiple LayoutData (i.e., List[LayoutData]), "
            "please use the .batch method."
        )
        bboxes, labels = layout_data.bboxes, layout_data.labels
        is_train = bboxes is not None and labels is not None

        if is_train:
            assert labels is not None
            if len(labels) <= self.max_element_numbers:
                # Store the labels for generating the prompt
                self._possible_labels = self._possible_labels + (tuple(labels),)
        else:
            if conf.labels_for_generation is not None:
                # If labels_for_generation is provided, use it directly.
                labels = conf.labels_for_generation
                logger.debug(f"Using provided {labels=}")
            else:
                assert len(self._possible_labels) > 0, (
                    "Please process the training data first."
                )
                # In the test data, bboxes and labels do not exist.
                # The labels are randomly sampled from the `possible_labels` obtained from the train data.
                # The bboxes are set below the sampled labels.
                labels = list(random.choice(self._possible_labels))
                logger.debug(f"Sampled {labels=}")

            # Prepare empty bboxes for generation.
            bboxes = [
                NormalizedBbox(left=0.0, top=0.0, width=0.0, height=0.0)
                for _ in range(len(labels))
            ]

            # Overwrite layout_data with the new bboxes and labels
            layout_data = layout_data.model_copy(
                update={"bboxes": bboxes, "labels": labels}
            )

        # Define the chain of preprocess transformations
        chain = (
            LexicographicSort()
            | LabelDictSort()
            | DiscretizeBboxes(target_canvas_size=self.target_canvas_size)
        )

        # Execute the transformations
        processed_layout_data = chain.invoke(layout_data)
        assert isinstance(processed_layout_data, ProcessedLayoutData)

        return processed_layout_data

    def batch(
        self,
        inputs: List[LayoutData],
        config: Optional[Union[RunnableConfig, List[RunnableConfig]]] = None,
        *,
        return_exceptions: bool = False,
        **kwargs: Any | None,
    ) -> List[ProcessedLayoutData]:
        return super().batch(
            inputs, config, return_exceptions=return_exceptions, **kwargs
        )

    def invoke(
        self, input: LayoutData, config: Optional[RunnableConfig] = None, **kwargs: Any
    ) -> ProcessedLayoutData:
        config = ensure_config(config)
        callback_manager = get_callback_manager_for_config(config)
        run_manager = callback_manager.on_chain_start(
            serialized=None, inputs=input, name=self.name
        )

        try:
            processed_layout_data = self._invoke(
                layout_data=input, config=config, **kwargs
            )
        except Exception as err:
            run_manager.on_chain_error(err)
            raise err

        run_manager.on_chain_end(outputs=processed_layout_data)

        return processed_layout_data
