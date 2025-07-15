from typing import Tuple

import datasets as ds
import numpy as np
from loguru import logger

from layout_prompter.models import LayoutData
from layout_prompter.settings import Rico25Settings
from layout_prompter.utils import normalize_bboxes


def _filter_empty_bboxes(example):
    return len([bbox for child in example["children"] for bbox in child["bounds"]]) > 0


def _filter_too_many_bboxes(example, max_elements: int = 10):
    return (
        len([bbox for child in example["children"] for bbox in child["bounds"]])
        <= max_elements
    )


def load_raw_rico(
    dataset_name: str = "creative-graphic-design/Rico",
    dataset_type: str = "ui-screenshots-and-hierarchies-with-semantic-annotations",
) -> ds.DatasetDict:
    # Load the RICO dataset
    dataset = ds.load_dataset(
        dataset_name,
        name=dataset_type,
    )
    assert isinstance(dataset, ds.DatasetDict)
    return dataset


def load_rico25(
    dataset_name: str = "creative-graphic-design/Rico",
    dataset_type: str = "ui-screenshots-and-hierarchies-with-semantic-annotations",
    num_proc: int = 32,
    max_elements: int = 10,
) -> ds.DatasetDict:
    # Load the RICO settings
    settings = Rico25Settings()

    # Load the RICO dataset
    dataset = load_raw_rico(
        dataset_name=dataset_name,
        dataset_type=dataset_type,
    )

    dataset = dataset.filter(
        _filter_empty_bboxes,
        desc="Filter out empty bboxes",
        num_proc=num_proc,
    )
    dataset = dataset.filter(
        _filter_too_many_bboxes,
        fn_kwargs={"max_elements": max_elements},
        desc="Filter by max elements",
        num_proc=num_proc,
    )

    train_feature = dataset["train"].features
    train_children_feature = train_feature["children"].feature
    component_labeler = train_children_feature.feature["component_label"]

    def convert_to_layout_data(example):
        # Get the canvas size
        W, H = example["bounds"][2:]

        # Get the children associated with the example
        children = example["children"]

        # # Get bboxes from children and filter out invalid ones
        bboxes = np.array(
            [bbox for child in children for bbox in child["bounds"]],
        )

        # Get labels from children
        labels = [
            component_labeler.int2str(label_id)
            for child in children
            for label_id in child["component_label"]
        ]

        # Ensure bboxes and labels have the same length
        assert len(bboxes) == len(labels)

        # Convert bboxes to (left, top, width, height) format
        bboxes[:, 2] -= bboxes[:, 0]
        bboxes[:, 3] -= bboxes[:, 1]

        # Normalize bboxes
        bboxes = normalize_bboxes(bboxes=bboxes, w=W, h=H)

        # Get the canvas size as a dictionary
        canvas_size = settings.canvas_size.model_dump()

        data = {
            "bboxes": [
                {
                    "left": bbox[0],
                    "top": bbox[1],
                    "width": bbox[2],
                    "height": bbox[3],
                }
                for bbox in bboxes.tolist()
            ],
            "labels": labels,
            "canvas_size": canvas_size,
            "encoded_image": None,
            "content_bboxes": None,
        }

        try:
            # Ensure the data conforms to the `LayoutData` model
            assert LayoutData.model_validate(data)
        except Exception as err:
            logger.trace(f"Data validation failed: {err}. Data: {example=}. ")
            return None

        return data

    dataset = dataset.map(
        convert_to_layout_data,
        desc="Convert RICO dataset to LayoutData format",
        remove_columns=dataset.column_names["train"],
        num_proc=num_proc,
    )

    logger.debug(dataset)

    return dataset
