import abc
from typing import Any, Dict, List, Optional, Union, cast

from langchain_core.runnables import Runnable, RunnableConfig
from pydantic import BaseModel

from layout_prompter.models import LayoutData, ProcessedLayoutData
from layout_prompter.utils import Configuration


class ProcessorConfig(Configuration):
    """Base Configuration for Processor."""


class Processor(BaseModel, Runnable, metaclass=abc.ABCMeta):
    """Base class for all processors."""

    @abc.abstractmethod
    def _process(self, layout_data: LayoutData) -> ProcessedLayoutData:
        """Process a single LayoutData instance.

        Args:
            layout_data (LayoutData): The layout data to process.

        Returns:
            LayoutData: The processed layout data.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def _process_list(self, dataset: List[LayoutData]) -> List[LayoutData]:
        return [self._process(example) for example in dataset]

    def invoke(
        self,
        input: Union[Dict[str, List[LayoutData]], List[LayoutData], LayoutData],
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> Union[Dict[str, List[LayoutData]], List[LayoutData], LayoutData]:
        if isinstance(input, dict):
            return {
                split: cast(
                    List[LayoutData],
                    self._process_list(dataset),
                )
                for split, dataset in input.items()
            }
        elif isinstance(input, list):
            return self._process_list(input)

        elif isinstance(input, LayoutData):
            return self._process(input)

        else:
            raise ValueError(
                f"Unsupported input type: {type(input)}. "
                "Expected Dict[str, List[LayoutData]], List[LayoutData], or LayoutData."
            )


class ContentAgnosticProcessor(Processor):
    """Base class for content-agnostic processors."""
