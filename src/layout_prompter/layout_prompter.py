from dataclasses import dataclass
from typing import (
    Any,
    List,
    Optional,
    Type,
    cast,
)

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable
from langchain_core.runnables.config import RunnableConfig

from layout_prompter.models import LayoutSerializedOutputData, ProcessedLayoutData
from layout_prompter.modules.rankers import LayoutRanker
from layout_prompter.modules.selectors import LayoutSelector
from layout_prompter.modules.serializers import LayoutSerializer, LayoutSerializerInput
from layout_prompter.utils import Configuration


class LayoutPrompterConfiguration(Configuration):
    """Configuration for LayoutPrompter."""

    num_return: int = 10


@dataclass
class LayoutPrompter(Runnable):
    selector: LayoutSelector
    serializer: LayoutSerializer
    llm: BaseChatModel
    ranker: LayoutRanker
    schema: Type[LayoutSerializedOutputData]

    def invoke(
        self,
        input: ProcessedLayoutData,
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> Any:
        # Load configuration
        conf = LayoutPrompterConfiguration.from_runnable_config(config)

        # Get candidates based on the input query
        candidates = self.selector.select_examples(input)

        # Define the input for the serializer based on the input query and selected candidates
        serializer_input = LayoutSerializerInput(query=input, candidates=candidates)

        # Construct the few-shot layout examples as prompt messages
        messages = self.serializer.invoke(input=serializer_input)

        # Generate batched layouts
        structured_llm = self.llm.with_structured_output(self.schema)
        outputs = cast(
            List[LayoutSerializedOutputData],
            structured_llm.batch([messages] * conf.num_return),
        )

        # Rank the generated layouts
        ranked_outputs = self.ranker.invoke(outputs)

        return ranked_outputs
