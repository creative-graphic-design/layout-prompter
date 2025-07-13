import abc
from typing import Any, List, Optional, Union

from langchain_core.runnables import RunnableSerializable
from langchain_core.runnables.config import (
    RunnableConfig,
    ensure_config,
    get_callback_manager_for_config,
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

    @abc.abstractmethod
    def _invoke(
        self,
        layout_data: LayoutData,
        config: Optional[RunnableConfig] = None,
        **kwargs,
    ) -> ProcessedLayoutData:
        raise NotImplementedError

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
