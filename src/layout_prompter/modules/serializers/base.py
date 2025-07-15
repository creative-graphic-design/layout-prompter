import json
from typing import Any, Final, List, Optional, Type

from langchain_core.prompt_values import ChatPromptValue
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_core.runnables import RunnableSerializable
from langchain_core.runnables.config import RunnableConfig
from loguru import logger
from pydantic import BaseModel, Field

from layout_prompter.models import (
    LayoutSerializedData,
    ProcessedLayoutData,
)
from layout_prompter.utils import Configuration

SYSTEM_PROMPT: Final[str] = """\
Please generate a layout based on the given information. You need to ensure that the generated layout looks realistic, with elements well aligned and avoiding unnecessary overlap.

# Preamble
## Task Description
{task_description}

## Layout Domain
{layout_domain} layout

## Canvas Size
{canvas_width}px x {canvas_height}px"""

UNK_TOKEN: Final[str] = "<unk>"


class LayoutSerializerConfig(Configuration):
    """Base class for all layout serializers."""

    input_schema: Type[LayoutSerializedData]


class LayoutSerializerInput(BaseModel):
    """Input for the layout serializer."""

    query: ProcessedLayoutData
    candidates: List[ProcessedLayoutData]


class LayoutSerializer(RunnableSerializable):
    task_type: str = Field(
        description="Type of the task to be performed.",
    )
    layout_domain: str = Field(
        description="Domain of the layout, e.g., 'poster', 'webpage'.",
    )
    unk_token: str = Field(
        description="Token to use for unknown elements.",
        default=UNK_TOKEN,
    )
    system_prompt: str = Field(
        description="System prompt to guide the layout generation.",
        default=SYSTEM_PROMPT,
    )

    # constraint_template: str = Field(
    #     description="Template for the content constraint.",
    # )

    add_index_token: bool = True
    add_sep_token: bool = True
    add_unk_token: bool = False

    def _convert_to_double_bracket(self, s: str) -> str:
        """Convert a string to double bracket format.

        When using `FewshotPromptTemplate`, if the data contains JSON format data as an example,
        it is recognized as a template and an error occurs. See the following issue:
        FewShotPromptTemplate bug on examples with JSON strings · Issue #4367 · langchain-ai/langchain https://github.com/langchain-ai/langchain/issues/4367.
        As this issue has been closed, it is difficult to expect any further action to be taken.
        Here, we will temporarily deal with this by escaping { and } into {{ and }}, referring to https://github.com/langchain-ai/langchain/issues/4367#issuecomment-1557528059.
        """
        return s.replace("{", "{{").replace("}", "}}")
