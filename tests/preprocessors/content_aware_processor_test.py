import datasets as ds

from layout_prompter.preprocessors import ContentAwareProcessor
from layout_prompter.settings import PosterLayoutSettings
from layout_prompter.utils.testing import LayoutPrompterTestCase


class TestContentAwareProcessor(LayoutPrompterTestCase):
    def test_content_aware_processor(self, hf_dataset: ds.DatasetDict):
        settings = PosterLayoutSettings()

        processor = ContentAwareProcessor(
            canvas_size=settings.canvas_size,
        )
        processed_dataset = processor.invoke(
            hf_dataset,
            config={"configurable": {"num_proc": 32}},
        )
        assert isinstance(hf_dataset, ds.DatasetDict)

        dataset_dir = self.FIXTURES_ROOT / "datasets" / "poster-layout" / "processed"

        if not dataset_dir.exists():
            dataset_dir.mkdir(parents=True, exist_ok=True)
            processed_dataset.save_to_disk(dataset_dir)
