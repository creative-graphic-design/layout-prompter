import abc
from typing import Any, List, Optional, Union

from langchain_core.runnables import RunnableSerializable
from langchain_core.runnables.config import RunnableConfig

from layout_prompter.models import LayoutData, ProcessedLayoutData


class LayoutTransform(RunnableSerializable):
    @abc.abstractmethod
    def invoke(
        self,
        input: Union[LayoutData, ProcessedLayoutData],
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> Any:
        raise NotImplementedError

    def batch(
        self,
        inputs: Union[List[LayoutData], List[ProcessedLayoutData]],
        config: Optional[Union[RunnableConfig, List[RunnableConfig]]] = None,
        *,
        return_exceptions: bool = False,
        **kwargs: Any | None,
    ) -> List:
        return super().batch(
            inputs, config, return_exceptions=return_exceptions, **kwargs
        )
