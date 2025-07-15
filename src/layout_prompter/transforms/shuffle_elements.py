import copy
from typing import Any, Union

from langchain_core.runnables import RunnableSerializable
from langchain_core.runnables.config import RunnableConfig
from loguru import logger

from layout_prompter.models import CanvasSize, LayoutData, ProcessedLayoutData


class ShuffleElements(RunnableSerializable):
    name: str = "shuffle-elements"
