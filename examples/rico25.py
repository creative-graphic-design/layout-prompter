import argparse
import pathlib
from typing import get_args

from layout_prompter.datasets import load_raw_rico
from layout_prompter.settings import Rico25Settings
from layout_prompter.typehints import Task


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Layout Prompter")
    parser.add_argument(
        "--task",
        type=Task,
        choices=get_args(Task),
        default="gen-t",
        help="Task to perform",
    )
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
    settings = Rico25Settings()
    hf_dataset = load_rico25()


if __name__ == "__main__":
    main(args=parse_args())
