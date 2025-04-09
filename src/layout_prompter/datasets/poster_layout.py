import logging

import datasets as ds

logger = logging.getLogger(__name__)


def _filter_empty_data(example):
    anns = example.get("annotations")
    is_test = anns is None

    if not is_test:
        return len(anns["cls_elem"]) > 0
    else:
        return is_test  # Always return True for test data


def _convert_id_to_label(example, id2label):
    ann = example["annotations"]
    is_test = ann is None

    if is_test:
        # There is no annotation in the test set
        return example

    # Convert label ids to label names
    cls_elem = [id2label(label_id) for label_id in ann["cls_elem"]]
    ann["cls_elem"] = cls_elem

    return example


def load_poster_layout(
    dataset_name: str = "creative-graphic-design/PKU-PosterLayout",
    num_proc: int = 32,
    return_raw: bool = False,
) -> ds.DatasetDict:
    dataset = ds.load_dataset(
        dataset_name,
        verification_mode="no_checks",
    )
    assert isinstance(dataset, ds.DatasetDict)

    if return_raw:
        # Return the raw dataset without any processing
        return dataset

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

    # Convert the cls_elem column, which corresponds to class labels,
    # from ClassLabel to string
    train_annotation_features["cls_elem"] = ds.Value("string")

    # Apply label conversion to the dataset
    dataset = dataset.map(
        _convert_id_to_label,
        fn_kwargs={"id2label": id2label},
        features=train_features,  # Override the features to use the updated cls_elem
        desc="Apply label conversion",
        num_proc=num_proc,
    )

    # Rename and remove the columns to match the expected format
    dataset = dataset.rename_columns(
        {
            "inpainted_poster": "content_image",
            "pfpn_saliency_map": "saliency_map",
        }
    )
    dataset = dataset.remove_columns(
        [
            "original_poster",
            "basnet_saliency_map",
        ]
    )
    # Execute cases that must be processed individually for each split
    dataset["train"] = dataset["train"].remove_columns("canvas")
    dataset["test"] = (
        dataset["test"]
        .remove_columns("content_image")
        .rename_columns({"canvas": "content_image"})
    )

    logger.debug(dataset)

    return dataset
