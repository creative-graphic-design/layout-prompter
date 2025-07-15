from typing import Any, List, Optional, Union

from langchain_core.runnables import RunnableSerializable
from langchain_core.runnables.config import (
    RunnableConfig,
)
from pydantic import ConfigDict

from layout_prompter.models import (
    LayoutData,
    ProcessedLayoutData,
)
from layout_prompter.utils import Configuration


class ProcessorConfig(Configuration):
    """Base Configuration for Processor."""


class Processor(RunnableSerializable):
    """Base class for all processors."""

    model_config = ConfigDict(
        frozen=True,  # for hashable Processor
    )

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
