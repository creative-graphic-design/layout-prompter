import json
from typing import Any, Final, List

from langchain_core.prompt_values import ChatPromptValue
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_core.runnables import Runnable
from langchain_core.runnables.config import RunnableConfig
from pydantic import BaseModel, field_validator

from layout_prompter.models import Coordinates, ProcessedLayoutData, SerializedData
from layout_prompter.utils.image import base64_to_pil, pil_to_base64

SYSTEM_PROMPT: Final[str] = """\
Please generate a layout based on the given information. You need to ensure that the generated layout looks realistic, with elements well aligned and avoiding unnecessary overlap.

# Preamble
## Task Description
{task_description}

## Layout Domain
{layout_domain} layout

## Canvas Size
{canvas_width}px x {canvas_height}px
"""

CONTENT_AWARE_CONSTRAINT: Final[str] = """\
# Constraints
## Content Constraint
{content_constraint}

## Element Type Constraint
{type_constraint}
"""

SERIALIZED_LAYOUT: Final[str] = """\
# Serialized Layout
{serialized_layout}
"""


class LayoutSerializerInput(BaseModel):
    query: ProcessedLayoutData
    candidates: List[ProcessedLayoutData]


class LayoutSerializer(BaseModel, Runnable):
    TASK_TYPE: str = ""
    UNK_TOKEN: Final[str] = "<unk>"

    system_prompt: str = SYSTEM_PROMPT
    add_index_token: bool = True
    add_sep_token: bool = True
    add_unk_token: bool = False

    @field_validator("TASK_TYPE", mode="after")
    def validate_task_type(cls, value: str) -> str:
        assert value != "", "TASK_TYPE should be set in the subclass"
        return value

    def _convert_to_double_bracket(self, s: str) -> str:
        """Convert a string to double bracket format.

        When using `FewshotPromptTemplate`, if the data contains JSON format data as an example,
        it is recognized as a template and an error occurs. See the following issue:
        FewShotPromptTemplate bug on examples with JSON strings · Issue #4367 · langchain-ai/langchain https://github.com/langchain-ai/langchain/issues/4367.
        As this issue has been closed, it is difficult to expect any further action to be taken.
        Here, we will temporarily deal with this by escaping { and } into {{ and }}, referring to https://github.com/langchain-ai/langchain/issues/4367#issuecomment-1557528059.
        """
        return s.replace("{", "{{").replace("}", "}}")


class ContentAwareSerializer(LayoutSerializer):
    TASK_TYPE: str = (
        "content-aware layout generation\n"
        "Please place the following elements to avoid salient content, and underlay must be the background of text or logo."
    )
    name: str = "content-aware-serializer"

    layout_domain: str

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
        type_constraint = json.dumps(
            {idx: label for idx, label in enumerate(data.labels)}
        )
        return self._convert_to_double_bracket(type_constraint)

    def _get_serialized_layout(self, data: ProcessedLayoutData) -> str:
        assert len(data.labels) == len(data.discrete_gold_bboxes)

        serialized_data_list = []
        for class_name, bbox in zip(data.labels, data.discrete_gold_bboxes):
            left, top, width, height = bbox

            serialized_data = SerializedData(
                class_name=class_name,
                coord=Coordinates(left=left, top=top, width=width, height=height),
            )
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
                (
                    "human",
                    [
                        CONTENT_AWARE_CONSTRAINT,
                        # {
                        #     "type": "image_url",
                        #     "image_url": {
                        #         "url": "data:image/png;base64,{content_image}",
                        #     },
                        # },
                        # {
                        #     "type": "image_url",
                        #     "image_url": {
                        #         "url": "data:image/jpeg;base64,{saliency_map}",
                        #     },
                        # },
                    ],
                ),
                ("ai", SERIALIZED_LAYOUT),
            ]
        )

        candidate_content_images = [c.content_image.copy() for c in input.candidates]
        for content_image in candidate_content_images:
            content_image.thumbnail((256, 256))

        candidate_saliency_maps = [c.saliency_map.copy() for c in input.candidates]
        for saliency_map in candidate_saliency_maps:
            saliency_map.thumbnail((256, 256))

        examples = [
            {
                "content_constraint": self._get_content_constraint(candidate),
                "type_constraint": self._get_type_constraint(candidate),
                "serialized_layout": self._get_serialized_layout(candidate),
                # "content_image": pil_to_base64(content_image),
                # "saliency_map": pil_to_base64(saliency_map),
            }
            for content_image, saliency_map, candidate in zip(
                candidate_content_images, candidate_saliency_maps, input.candidates
            )
        ]

        few_shot_prompt = FewShotChatMessagePromptTemplate(
            examples=examples,
            example_prompt=example_prompt,
        )

        # system_prompt = SystemMessagePromptTemplate.from_template(
        #     template=SYSTEM_PROMPT,
        # )
        # human_prompt = HumanMessagePromptTemplate.from_template(
        #     template=CONTENT_AWARE_CONSTRAINT
        # )

        # prompt = ChatPromptTemplate.from_messages(
        #     [
        #         system_prompt,
        #         few_shot_prompt,
        #         human_prompt,
        #     ]
        # )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    SYSTEM_PROMPT,
                ),
                few_shot_prompt,
                (
                    "human",
                    [
                        CONTENT_AWARE_CONSTRAINT,
                        # {
                        #     "type": "image_url",
                        #     "image_url": {
                        #         "url": "data:image/png;base64,{content_image}",
                        #     },
                        # },
                        # {
                        #     "type": "image_url",
                        #     "image_url": {
                        #         "url": "data:image/png;base64,{saliency_map}",
                        #     },
                        # },
                    ],
                ),
            ]
        )

        query_content_image = input.query.content_image.copy()
        query_content_image.thumbnail((256, 256))

        query_saliency_map = input.query.saliency_map.copy()
        query_saliency_map.thumbnail((256, 256))

        final_prompt = prompt.invoke(
            {
                "canvas_width": input.query.canvas_size.width,
                "canvas_height": input.query.canvas_size.height,
                "task_description": self.TASK_TYPE,
                "layout_domain": self.layout_domain,
                "content_constraint": self._get_content_constraint(input.query),
                "type_constraint": self._get_type_constraint(input.query),
                # "content_image": pil_to_base64(query_content_image),
                # "saliency_map": pil_to_base64(query_saliency_map),
            }
        )

        # breakpoint()

        assert isinstance(final_prompt, ChatPromptValue)
        return final_prompt
