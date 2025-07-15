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
    LayoutSerializedData,
    ProcessedLayoutData,
)

from .base import LayoutSerializer, LayoutSerializerConfig, LayoutSerializerInput

GEN_TYPE_CONSTRAINT: Final[str] = """\
# Constraints
## Element Type Constraint
{type_constraint}

# Serialized Layout"""

SERIALIZED_LAYOUT: Final[str] = """\
{serialized_layout}"""


class GenTypeSerializer(LayoutSerializer):
    task_type: str = "generation conditioned on given element types"
    name: str = "gen-type-serializer"

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
    ) -> Any:
        conf = LayoutSerializerConfig.from_runnable_config(config)

        example_prompt = ChatPromptTemplate.from_messages(
            [
                GEN_TYPE_CONSTRAINT,
                SERIALIZED_LAYOUT,
            ]
        )
        example_prompt = ChatPromptTemplate.from_messages(
            [
                ("human", GEN_TYPE_CONSTRAINT),
                ("ai", SERIALIZED_LAYOUT),
            ]
        )
        examples = [
            {
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
            template=GEN_TYPE_CONSTRAINT
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
                "type_constraint": self._get_type_constraint(input.query),
            }
        )
        assert isinstance(final_prompt, ChatPromptValue)

        for message in final_prompt.to_messages():
            logger.debug(message.pretty_repr())

        return final_prompt
