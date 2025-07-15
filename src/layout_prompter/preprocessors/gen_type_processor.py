from typing import Optional

from langchain_core.runnables.config import RunnableConfig

from layout_prompter.models import ProcessedLayoutData
from layout_prompter.models.layout_data import LayoutData
from layout_prompter.transforms import (
    DiscretizeBboxes,
    LabelDictSort,
    LexicographicSort,
)

from .base import Processor


class GenTypeProcessor(Processor):
    name: str = "gen-type-processor"

    def _invoke(
        self,
        layout_data: LayoutData,
        config: Optional[RunnableConfig] = None,
        **kwargs,
    ) -> ProcessedLayoutData:
        # Define the chain of preprocess transformations
        chain = (
            LexicographicSort()
            | LabelDictSort()
            | DiscretizeBboxes(target_canvas_size=self.target_canvas_size)
        )

        # Invoke the chain with the provided layout data
        processed_layout_data = chain.invoke(layout_data)
        assert isinstance(processed_layout_data, ProcessedLayoutData)

        return processed_layout_data
