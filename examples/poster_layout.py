import argparse
import pathlib
from typing import List, cast

from langchain.chat_models import init_chat_model
from langchain.smith.evaluation.progress import ProgressBarCallback
from tqdm.auto import tqdm

from layout_prompter import LayoutPrompter
from layout_prompter.datasets import load_poster_layout
from layout_prompter.models import (
    LayoutData,
    PosterLayoutSerializedData,
    PosterLayoutSerializedOutputData,
    ProcessedLayoutData,
)
from layout_prompter.modules import (
    ContentAwareSelector,
    ContentAwareSerializer,
    LayoutPrompterRanker,
)
from layout_prompter.preprocessors import ContentAwareProcessor
from layout_prompter.settings import PosterLayoutSettings
from layout_prompter.transforms import DiscretizeBboxes
from layout_prompter.utils.workers import get_num_workers
from layout_prompter.visualizers import ContentAwareVisualizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Layout Prompter")
    parser.add_argument(
        "--num_prompt",
        type=int,
        default=10,
        help="Number of prompts to generate",
    )
    parser.add_argument(
        "--model-provider",
        type=str,
        default="openai",
        help="Model provider to use",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="gpt-4o",
        help="Model ID to use",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of workers for processing",
    )
    parser.add_argument(
        "--save-dir",
        type=pathlib.Path,
        default=pathlib.Path(__file__).parent.resolve() / "generated" / "content_aware",
        help="Directory to save generated images",
    )
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    # Load the settings for Poster Layout dataset
    settings = PosterLayoutSettings()
    # Load the dataset
    hf_dataset = load_poster_layout()

    # Convert the Hugging Face dataset to a dictionary of LayoutData
    dataset = {
        split: [
            LayoutData.model_validate(data)
            for data in tqdm(hf_dataset[split], desc=f"Processing for {split}")
        ]
        for split in hf_dataset
    }
    tng_dataset, tst_dataset = dataset["train"], dataset["test"]

    # Define the content-aware processor
    processor = ContentAwareProcessor()

    # Process the training dataset to generate candidate examples
    candidate_examples = cast(
        List[ProcessedLayoutData],
        processor.batch(
            inputs=tng_dataset,
            config={
                "max_concurrency": args.num_workers or get_num_workers(),
                "callbacks": [ProgressBarCallback(total=len(tng_dataset))],
            },
        ),
    )

    # Select a random test example or use a fixed index for reproducibility
    # idx = random.choice(range(len(dataset["test"])))
    idx = 443
    test_data = tst_dataset[idx]

    # Process the test data
    inference_example = cast(ProcessedLayoutData, processor.invoke(input=test_data))

    # Define the discretizer for bounding boxes
    bbox_discretizer = DiscretizeBboxes()

    # Apply the bbox discretizer to candidate examples and test data
    candidate_examples = cast(
        List[ProcessedLayoutData],
        bbox_discretizer.batch(
            candidate_examples,
            config={"configurable": {"target_canvas_size": settings.canvas_size}},
        ),
    )
    inference_example = bbox_discretizer.invoke(
        inference_example,
        config={"configurable": {"target_canvas_size": settings.canvas_size}},
    )

    layout_prompter = LayoutPrompter(
        selector=ContentAwareSelector(
            num_prompt=args.num_prompt,
            examples=candidate_examples,
        ),
        serializer=ContentAwareSerializer(
            layout_domain=settings.domain,
        ),
        llm=init_chat_model(
            model_provider=args.model_provider,
            model=args.model_id,
        ),
        ranker=LayoutPrompterRanker(),
    )
    outputs = layout_prompter.invoke(
        input=inference_example,
        config={
            "configurable": {
                "input_schema": PosterLayoutSerializedData,
                "output_schema": PosterLayoutSerializedOutputData,
            }
        },
    )

    visualizer = ContentAwareVisualizer(
        canvas_size=settings.canvas_size, labels=settings.labels
    )
    visualizer_config = {
        "resize_ratio": 2.0,
        "bg_image": inference_example.content_image.copy(),  # Copy the background image to avoid race conditions in batch processing
        "content_bboxes": inference_example.discrete_content_bboxes,
    }

    save_dir = args.save_dir
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving visualizations to {save_dir}")

    visualizations = visualizer.batch(
        outputs.ranked_outputs,
        config={
            "configurable": visualizer_config,
            "max_concurrency": args.num_workers,
        },
    )
    for i, visualization in enumerate(visualizations):
        visualization.save(save_dir / f"{idx=},{i=}.png")


if __name__ == "__main__":
    main(args=parse_args())
