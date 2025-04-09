import datasets as ds

from layout_prompter.preprocessors import ContentAwareProcessor
from layout_prompter.settings import PosterLayoutSettings
from layout_prompter.utils.testing import LayoutPrompterTestCase


class TestContentAwareProcessor(LayoutPrompterTestCase):
    def test_content_aware_processor(self, raw_dataset: ds.DatasetDict):
        settings = PosterLayoutSettings()

        processor = ContentAwareProcessor(
            canvas_size=settings.canvas_size,
        )
        raw_dataset = processor.invoke(
            raw_dataset,
            config={"configurable": {"num_proc": 32}},
        )
        assert isinstance(raw_dataset, ds.DatasetDict)

        dataset_dir = self.FIXTURES_ROOT / "datasets" / "poster-layout" / "processed"

        if not dataset_dir.exists():
            dataset_dir.mkdir(parents=True, exist_ok=True)
            raw_dataset.save_to_disk(dataset_dir)
