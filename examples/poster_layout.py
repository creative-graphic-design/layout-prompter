import argparse
import pathlib
from typing import List, cast

from langchain.chat_models import init_chat_model
from layout_prompter import LayoutPrompter
from layout_prompter.datasets import load_poster_layout
from layout_prompter.models import (
    LayoutData,
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
from layout_prompter.visualizers import ContentAwareVisualizer
from tqdm.auto import tqdm


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
        "--save-dir",
        type=pathlib.Path,
        default=pathlib.Path(__file__).parent.resolve() / "generated" / "content_aware",
        help="Directory to save generated images",
    )
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    settings = PosterLayoutSettings()
    hf_dataset = load_poster_layout()

    dataset = {
        split: [
            LayoutData.model_validate(data)
            for data in tqdm(hf_dataset[split], desc=f"Processing for {split}")
        ]
        for split in hf_dataset
    }

    processor = ContentAwareProcessor()
    candidate_examples = cast(
        List[ProcessedLayoutData], processor.invoke(input=dataset["train"])
    )
    # inference_examples = processor.invoke(input=dataset["test"])

    # idx = random.choice(range(len(dataset["test"])))
    idx = 443
    inference_example = cast(
        ProcessedLayoutData, processor.invoke(input=dataset["test"][idx])
    )

    layout_prompter = LayoutPrompter(
        selector=ContentAwareSelector(
            num_prompt=args.num_prompt,
            canvas_size=settings.canvas_size,
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
        schema=PosterLayoutSerializedOutputData,
    )
    outputs = layout_prompter.invoke(input=inference_example)

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
        outputs, config={"configurable": visualizer_config}
    )
    for i, visualization in enumerate(visualizations):
        visualization.save(save_dir / f"{idx=},{i=}.png")


if __name__ == "__main__":
    main(args=parse_args())
