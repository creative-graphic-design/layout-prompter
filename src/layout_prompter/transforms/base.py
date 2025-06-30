import abc
from typing import Any, Union

from langchain_core.runnables import Runnable
from langchain_core.runnables.config import RunnableConfig

from layout_prompter.models import LayoutData, ProcessedLayoutData


class BaseTransform(Runnable):
    @abc.abstractmethod
    def invoke(
        self,
        input: Union[LayoutData, ProcessedLayoutData],
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> ProcessedLayoutData:
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement the invoke method."
        )
