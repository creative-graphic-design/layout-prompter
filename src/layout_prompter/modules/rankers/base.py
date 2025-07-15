import abc
from typing import Any, List, Optional

from langchain_core.runnables import RunnableSerializable
from langchain_core.runnables.config import RunnableConfig

from layout_prompter.models import LayoutSerializedOutputData


class LayoutRanker(RunnableSerializable):
    """Base class for layout ranking algorithms."""

    @abc.abstractmethod
    def invoke(
        self,
        input: List[LayoutSerializedOutputData],
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> List[LayoutSerializedOutputData]:
        raise NotImplementedError
