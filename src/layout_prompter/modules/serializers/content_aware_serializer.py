import json
from typing import Any, Final, Optional

from langchain_core.prompt_values import ChatPromptValue
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_core.runnables.config import RunnableConfig
from loguru import logger

from layout_prompter.models import (
    ProcessedLayoutData,
)

from .base import LayoutSerializer, LayoutSerializerConfig, LayoutSerializerInput

CONTENT_AWARE_CONSTRAINT: Final[str] = """\
# Constraints
## Content Constraint
{content_constraint}

## Element Type Constraint
{type_constraint}

# Serialized Layout"""

SERIALIZED_LAYOUT: Final[str] = """\
{serialized_layout}"""


class ContentAwareSerializer(LayoutSerializer):
    task_type: str = (
        "content-aware layout generation\n"
        "Please place the following elements to avoid salient content, and underlay must be the background of text or logo."
    )
    name: str = "content-aware-serializer"

    def _get_content_constraint(self, data: ProcessedLayoutData) -> str:
        content_bboxes = data.discrete_content_bboxes
        assert content_bboxes is not None

        content_constraint = json.dumps([bbox.model_dump() for bbox in content_bboxes])
        return self._convert_to_double_bracket(content_constraint)

    def _get_type_constraint(self, data: ProcessedLayoutData) -> str:
        assert data.labels is not None
        type_constraint = json.dumps(
            {idx: label for idx, label in enumerate(data.labels)}
        )
        return self._convert_to_double_bracket(type_constraint)

    def invoke(
        self,
        input: LayoutSerializerInput,
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> ChatPromptValue:
        logger.debug(f"Invoking ContentAwareSerializer with input: {input}")

        conf = LayoutSerializerConfig.from_runnable_config(config)

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
                "serialized_layout": self._get_serialized_layout(
                    candidate, schema=conf.input_schema
                ),
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

        for message in final_prompt.to_messages():
            logger.debug(message.pretty_repr())

        return final_prompt
