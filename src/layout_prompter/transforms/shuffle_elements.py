# import copy
# from typing import Any, Union


# from langchain_core.runnables.config import RunnableConfig
# from loguru import logger
# from layout_prompter.models import CanvasSize, LayoutData, ProcessedLayoutData
from .base import LayoutTransform


class ShuffleElements(LayoutTransform):
    name: str = "shuffle-elements"
