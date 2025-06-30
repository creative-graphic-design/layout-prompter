import logging

import numpy as np

import datasets as ds
from layout_prompter.models import LayoutData
from layout_prompter.settings import PosterLayoutSettings
from layout_prompter.transforms import SaliencyMapToBboxes
from layout_prompter.utils import normalize_bboxes, pil_to_base64

logger = logging.getLogger(__name__)


def _filter_empty_data(example):
    anns = example.get("annotations")
    is_test = anns is None

    if not is_test:
        return len(anns["cls_elem"]) > 0
    else:
        return is_test  # Always return True for test data


def _filter_empty_content_bboxes(example):
    return example["content_bboxes"] is not None


def load_raw_poste_layout(
    dataset_name: str = "creative-graphic-design/PKU-PosterLayout",
) -> ds.DatasetDict:
    # Load the PosterLayout dataset
    dataset = ds.load_dataset(
        dataset_name,
        verification_mode="no_checks",
    )
    assert isinstance(dataset, ds.DatasetDict)
    return dataset


def load_poster_layout(
    dataset_name: str = "creative-graphic-design/PKU-PosterLayout",
    filter_threshold: int = 100,
    num_proc: int = 32,
) -> ds.DatasetDict:
    # Load the PosterLayout settings
    settings = PosterLayoutSettings()

    # Load the PosterLayout dataset
    dataset = load_raw_poste_layout(dataset_name)

    # Apply filtering to remove invalid data
    dataset = dataset.filter(
        _filter_empty_data,
        desc="Filter out empty data",
        num_proc=num_proc,
    )

    # Get the mapping from label ids to label names
    train_features = dataset["train"].features
    train_annotation_features = train_features["annotations"].feature
    id2label = train_annotation_features["cls_elem"].int2str

    # Define the saliency map to bboxes transformation
    saliency_map_to_bboxes = SaliencyMapToBboxes(threshold=filter_threshold)

    def convert_to_layout_data_format(example):
        anns = example["annotations"]
        is_train = anns is not None

        content_image = example["inpainted_poster"] if is_train else example["canvas"]
        encoded_image = pil_to_base64(content_image)

        saliency_map = example["pfpn_saliency_map"]
        map_w, map_h = saliency_map.size

        if is_train:
            bboxes = np.array(anns["box_elem"])
            labels = np.array(list(map(id2label, anns["cls_elem"])))
            assert len(bboxes) == len(labels)

            # Convert bboxes to [x, y, w, h] format
            bboxes[:, 2] -= bboxes[:, 0]
            bboxes[:, 3] -= bboxes[:, 1]

            # Normalize bboxes
            bboxes = normalize_bboxes(bboxes=bboxes, w=map_w, h=map_h)
        else:
            bboxes, labels = None, None

        # Get the content bboxes from the saliency map
        content_bboxes = saliency_map_to_bboxes.invoke(saliency_map)

        if content_bboxes is not None:
            # Normalize content bboxes
            content_bboxes = normalize_bboxes(bboxes=content_bboxes, w=map_w, h=map_h)

        # Get the canvas size as a dictionary
        canvas_size = settings.canvas_size.model_dump()

        data = {
            "bboxes": bboxes,
            "labels": labels,
            "canvas_size": canvas_size,
            "encoded_image": encoded_image,
            "content_bboxes": content_bboxes,
        }
        assert LayoutData.model_validate(data)
        return data

    dataset = dataset.map(
        convert_to_layout_data_format,
        desc="Convert to LayoutData format",
        remove_columns=dataset.column_names["train"],
        num_proc=num_proc,
    )

    dataset = dataset.filter(
        _filter_empty_content_bboxes,
        desc="Filter out empty content bboxes",
        num_proc=num_proc,
    )

    logger.debug(dataset)

    return dataset
