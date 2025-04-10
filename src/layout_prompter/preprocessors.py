import random
from collections import defaultdict
from typing import Any, Dict, List, Optional

import datasets as ds
import numpy as np
import pydantic_numpy.typing as pnd
from langchain_core.runnables import Runnable
from langchain_core.runnables.config import RunnableConfig
from pydantic import BaseModel
from tqdm.auto import tqdm

from layout_prompter.models import LayoutData
from layout_prompter.settings import CanvasSize
from layout_prompter.transforms import (
    DiscretizeBboxes,
    LabelDictSort,
    LexicographicSort,
    SaliencyMapToBboxes,
)
from layout_prompter.utils import normalize_bboxes, pil_to_base64


class Processor(BaseModel, Runnable):
    canvas_size: CanvasSize


class ContentAwareProcessor(Processor):
    name: str = "content-aware-processor"

    filter_threshold: int = 100
    max_element_numbers: int = 10

    # Store the possible labels from the training data.
    # During testing, randomly sample from this group for generation.
    _possible_labels: List[pnd.NpNDArray] = []

    @property
    def saliency_map_to_bboxes(self) -> SaliencyMapToBboxes:
        return SaliencyMapToBboxes(threshold=self.filter_threshold)

    def _process_train_data(
        self,
        encoded_image: str,
        encoded_saliency_map: str,
        anns: Dict[str, Any],
        map_w: int,
        map_h: int,
        content_bboxes: np.ndarray,
    ) -> LayoutData:
        bboxes = np.array(anns["box_elem"])
        labels = np.array(anns["cls_elem"])

        # Convert bboxes to [x, y, w, h] format
        bboxes[:, 2] -= bboxes[:, 0]
        bboxes[:, 3] -= bboxes[:, 1]

        # Normalize bboxes
        bboxes = normalize_bboxes(bboxes=bboxes, w=map_w, h=map_h)

        if len(labels) <= self.max_element_numbers:
            # Store the labels for generating the prompt
            self._possible_labels.append(labels)

        return LayoutData(
            bboxes=bboxes,
            labels=labels,
            content_bboxes=content_bboxes,
            encoded_image=encoded_image,
            encoded_saliency_map=encoded_saliency_map,
            canvas_size=self.canvas_size,
        )

    def _process_test_data(
        self,
        encoded_image: str,
        encoded_saliency_map: str,
        content_bboxes: np.ndarray,
    ) -> LayoutData:
        if len(self._possible_labels) == 0:
            raise RuntimeError("Please process the training data first.")

        # Randomly sample from the possible labels
        labels = np.array(random.choice(self._possible_labels))

        # Create empty bboxes for the test data
        bboxes = np.zeros((len(labels), 4))

        return LayoutData(
            bboxes=bboxes,
            labels=labels,
            content_bboxes=content_bboxes,
            encoded_saliency_map=encoded_saliency_map,
            encoded_image=encoded_image,
            canvas_size=self.canvas_size,
        )

    def _process(self, example) -> Optional[Dict[str, Any]]:
        saliency_map = example["saliency_map"]
        map_w, map_h = saliency_map.size

        content_bboxes = self.saliency_map_to_bboxes.invoke(saliency_map)
        if content_bboxes is None:
            # If the saliency map cannot recognize the bbox, exclude it as invalid example
            return None

        content_bboxes = normalize_bboxes(bboxes=content_bboxes, w=map_w, h=map_h)

        # Convert the content image to base64 string
        content_image = example["content_image"]
        encoded_image = pil_to_base64(content_image)
        encoded_saliency_map = pil_to_base64(saliency_map)

        anns = example["annotations"]
        is_train = anns is not None

        # Process the training or test data
        layout_data = (
            self._process_train_data(
                anns=anns,
                map_w=map_w,
                map_h=map_h,
                encoded_image=encoded_image,
                encoded_saliency_map=encoded_saliency_map,
                content_bboxes=content_bboxes,
            )
            if is_train
            else self._process_test_data(
                encoded_image=encoded_image,
                encoded_saliency_map=encoded_saliency_map,
                content_bboxes=content_bboxes,
            )
        )

        # Define the chain of preprocess transformations
        chain = (
            LexicographicSort()
            | LabelDictSort()
            | DiscretizeBboxes(
                num_x_grid=self.canvas_size.width,
                num_y_grid=self.canvas_size.height,
            )
        )
        return chain.invoke(layout_data).model_dump()

    def invoke(
        self,
        input: ds.DatasetDict,
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> ds.DatasetDict:
        processed_dataset = defaultdict(lambda: defaultdict(list))  # type: ignore[var-annotated]

        for split in input:
            # e.g, train_dataset, valid_dataset, test_dataset
            split_dataset = input[split]

            for example in tqdm(split_dataset, desc=f"Processing for {split}"):
                # Apply preprocessing
                processed_example = self._process(example)

                if processed_example is None:
                    # If the saliency map cannot recognize the bbox,
                    # exclude it as invalid example
                    continue

                # Store the processed example as columnar data
                for k, v in processed_example.items():
                    processed_dataset[split][k].append(v)

        return ds.DatasetDict(
            {k: ds.Dataset.from_dict(v) for k, v in processed_dataset.items()}
        )
