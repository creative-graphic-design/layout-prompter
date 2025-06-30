from layout_prompter.models import ProcessedLayoutData
from layout_prompter.models.layout_data import LayoutData
from layout_prompter.transforms import (
    DiscretizeBboxes,
    LabelDictSort,
    LexicographicSort,
)

from .base import ContentAgnosticProcessor, ProcessorConfig


class GenTypeProcessorConfig(ProcessorConfig):
    """Configuration for GenTypeProcessor.

    Configuration for GenTypeProcessor, which is used for layout generation tasks that specify the type (also known as category) of layout elements such as logo, text, and image.
    """


class GenTypeProcessor(ContentAgnosticProcessor):
    name: str = "gen-type-processor"

    sort_by_pos: bool = False
    shuffle_before_sort_by_label: bool = False
    sort_by_pos_before_sort_by_label: bool = False

    def _process(self, layout_data: LayoutData) -> ProcessedLayoutData:
        # - short_by_pos=False
        # - shuffle_before_sort_by_label=False
        # - sort_by_pos_before_sort_by_label=False
        transform = LexicographicSort() | LabelDictSort() | DiscretizeBboxes()
        return transform.invoke(layout_data)
