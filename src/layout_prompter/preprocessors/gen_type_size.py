from layout_prompter.models import ProcessedLayoutData
from layout_prompter.models.layout_data import LayoutData
from layout_prompter.transforms import (
    DiscretizeBboxes,
    LabelDictSort,
    LexicographicSort,
    ShuffleElements,
)

from .base import ContentAgnosticProcessor, ProcessorConfig


class GenTypeSizeProcessorConfig(ProcessorConfig):
    """Configuration for GenTypeSizeProcessor.

    Configuration for GenTypeSizeProcessor, which is used for layout generation tasks that specify the type (also known as category; i.e., logo, text, and image) and the size of layout elements.
    """


class GenTypeSizeProcessor(ContentAgnosticProcessor):
    name: str = "gen-type-size-processor"

    def _process(self, layout_data: LayoutData) -> ProcessedLayoutData:
        # - short_by_pos=False
        # - shuffle_before_sort_by_label=True
        # - sort_by_pos_before_sort_by_label=False
        transform = LexicographicSort() | LabelDictSort() | DiscretizeBboxes()
        return transform.invoke(layout_data)
