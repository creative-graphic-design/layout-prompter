import json
import logging
from dataclasses import dataclass, field
from typing import Any, Final, List, Optional, Type

from langchain_core.prompt_values import ChatPromptValue
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_core.runnables import Runnable
from langchain_core.runnables.config import RunnableConfig
from pydantic import BaseModel

from layout_prompter.models import (
    Coordinates,
    LayoutSerializedData,
    ProcessedLayoutData,
)

logger = logging.getLogger(__name__)

UNK_TOKEN: Final[str] = "<unk>"

SYSTEM_PROMPT: Final[str] = """\
Please generate a layout based on the given information. You need to ensure that the generated layout looks realistic, with elements well aligned and avoiding unnecessary overlap.

# Preamble
## Task Description
{task_description}

## Layout Domain
{layout_domain} layout

## Canvas Size
{canvas_width}px x {canvas_height}px"""

CONTENT_AWARE_CONSTRAINT: Final[str] = """\
# Constraints
## Content Constraint
{content_constraint}

## Element Type Constraint
{type_constraint}"""

SERIALIZED_LAYOUT: Final[str] = """\
# Serialized Layout
{serialized_layout}"""


class LayoutSerializerInput(BaseModel):
    query: ProcessedLayoutData
    candidates: List[ProcessedLayoutData]


@dataclass
class LayoutSerializer(Runnable):
    task_type: Optional[str] = field(
        default=None,
        metadata={
            "description": "Type of the task to be performed. This should be set in the subclass."
        },
    )
    layout_domain: Optional[str] = field(
        default=None,
        metadata={
            "description": 'Domain of the layout, e.g., "poster", "webpage". This should be set in the subclass.'
        },
    )
    unk_token: Final[str] = UNK_TOKEN

    system_prompt: Final[str] = SYSTEM_PROMPT
    constraint_template: Final[str] = CONTENT_AWARE_CONSTRAINT

    add_index_token: bool = True
    add_sep_token: bool = True
    add_unk_token: bool = False

    schema: Optional[Type[LayoutSerializedData]] = None

    def __post_init__(self) -> None:
        assert self.task_type is not None, (
            f"{self.task_type=} must be set in the subclass"
        )
        assert self.layout_domain is not None, (
            f"{self.layout_domain=} must be set in the subclass"
        )
        assert self.schema is not None, f"{self.schema=} must be set in the subclass"

    def _convert_to_double_bracket(self, s: str) -> str:
        """Convert a string to double bracket format.

        When using `FewshotPromptTemplate`, if the data contains JSON format data as an example,
        it is recognized as a template and an error occurs. See the following issue:
        FewShotPromptTemplate bug on examples with JSON strings · Issue #4367 · langchain-ai/langchain https://github.com/langchain-ai/langchain/issues/4367.
        As this issue has been closed, it is difficult to expect any further action to be taken.
        Here, we will temporarily deal with this by escaping { and } into {{ and }}, referring to https://github.com/langchain-ai/langchain/issues/4367#issuecomment-1557528059.
        """
        return s.replace("{", "{{").replace("}", "}}")


@dataclass
class ContentAwareSerializer(LayoutSerializer):
    task_type: str = (
        "content-aware layout generation\n"
        "Please place the following elements to avoid salient content, and underlay must be the background of text or logo."
    )
    name: str = "content-aware-serializer"

    def _get_content_constraint(self, data: ProcessedLayoutData) -> str:
        content_bboxes = data.discrete_content_bboxes
        assert content_bboxes is not None

        coordinates = []
        for content_bbox in content_bboxes:
            left, top, width, height = content_bbox
            coordinates.append(
                Coordinates(left=left, top=top, width=width, height=height)
            )
        content_constraint = json.dumps([c.model_dump() for c in coordinates])
        return self._convert_to_double_bracket(content_constraint)

    def _get_type_constraint(self, data: ProcessedLayoutData) -> str:
        assert data.labels is not None
        type_constraint = json.dumps(
            {idx: label for idx, label in enumerate(data.labels)}
        )
        return self._convert_to_double_bracket(type_constraint)

    def _get_serialized_layout(self, data: ProcessedLayoutData) -> str:
        assert data.labels is not None and data.discrete_bboxes is not None
        assert self.schema is not None, "Schema must be defined for serialization."

        labels, discrete_gold_bboxes = data.labels, data.discrete_bboxes

        serialized_data_list: List[LayoutSerializedData] = []
        for class_name, bbox in zip(labels, discrete_gold_bboxes):
            left, top, width, height = bbox
            serialized_data_dict = {
                "class_name": class_name,
                "coord": {
                    "left": left,
                    "top": top,
                    "width": width,
                    "height": height,
                },
            }
            serialized_data = self.schema(**serialized_data_dict)
            serialized_data_list.append(serialized_data)
        return json.dumps([d.model_dump() for d in serialized_data_list])

    def invoke(
        self,
        input: LayoutSerializerInput,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> ChatPromptValue:
        example_prompt = ChatPromptTemplate.from_messages(
            [
                CONTENT_AWARE_CONSTRAINT,
                SERIALIZED_LAYOUT,
            ]
        )
        example_prompt = ChatPromptTemplate.from_messages(
            [
                ("human", CONTENT_AWARE_CONSTRAINT),
                ("ai", SERIALIZED_LAYOUT),
            ]
        )
        examples = [
            {
                "content_constraint": self._get_content_constraint(candidate),
                "type_constraint": self._get_type_constraint(candidate),
                "serialized_layout": self._get_serialized_layout(candidate),
            }
            for candidate in input.candidates
        ]

        few_shot_prompt = FewShotChatMessagePromptTemplate(
            examples=examples,
            example_prompt=example_prompt,
        )
        system_prompt = SystemMessagePromptTemplate.from_template(
            template=self.system_prompt,
        )
        human_prompt = HumanMessagePromptTemplate.from_template(
            template=CONTENT_AWARE_CONSTRAINT
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                system_prompt,
                few_shot_prompt,
                human_prompt,
            ]
        )

        final_prompt = prompt.invoke(
            {
                "canvas_width": input.query.canvas_size.width,
                "canvas_height": input.query.canvas_size.height,
                "task_description": self.task_type,
                "layout_domain": self.layout_domain,
                "content_constraint": self._get_content_constraint(input.query),
                "type_constraint": self._get_type_constraint(input.query),
            }
        )
        assert isinstance(final_prompt, ChatPromptValue)
        logger.debug(final_prompt.to_messages())

        return final_prompt
