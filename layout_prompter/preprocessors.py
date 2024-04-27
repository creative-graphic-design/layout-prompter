from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Type, TypedDict

import cv2
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as T
from pandas import DataFrame

from layout_prompter.datasets import LayoutDataset
from layout_prompter.transforms import (
    AddCanvasElement,
    AddGaussianNoise,
    AddRelation,
    CLIPTextEncoderTransform,
    DiscretizeBoundingBox,
    LabelDictSort,
    LexicographicSort,
    SaliencyMapToBBoxes,
    ShuffleElements,
)
from layout_prompter.typehint import LayoutData, TextToLayoutData
from layout_prompter.utils import clean_text

if TYPE_CHECKING:
    from layout_prompter.typehint import ProcessedLayoutData

__all__ = [
    "Processor",
    "GenTypeProcessor",
    "GenTypeSizeProcessor",
    "GenRelationProcessor",
    "CompletionProcessor",
    "RefinementProcessor",
    "ContentAwareProcessor",
    "TextToLayoutProcessor",
]


@dataclass
class ProcessorMixin(object):
    dataset: LayoutDataset
    return_keys: Optional[Tuple[str, ...]] = None

    metadata: Optional[pd.DataFrame] = field(repr=False, default=None)

    def __post_init__(self) -> None:
        assert self.return_keys is not None

    def __call__(self, data: LayoutData) -> ProcessedLayoutData:
        raise NotImplementedError


@dataclass
class Processor(ProcessorMixin):
    sort_by_pos: Optional[bool] = None
    shuffle_before_sort_by_label: Optional[bool] = None
    sort_by_pos_before_sort_by_label: Optional[bool] = None

    transform_functions: Optional[List[nn.Module]] = None

    def __post_init__(self) -> None:
        conds = (
            self.sort_by_pos,
            self.shuffle_before_sort_by_label,
            self.sort_by_pos_before_sort_by_label,
        )
        if not any(conds):
            raise ValueError(
                "At least one of sort_by_pos, shuffle_before_sort_by_label, "
                "or sort_by_pos_before_sort_by_label must be True."
            )

        self.transform_functions = self._config_base_transform()

    @property
    def transform(self) -> T.Compose:
        return T.Compose(self.transform_functions)

    def _config_base_transform(self) -> List[nn.Module]:
        transform_functions: List[nn.Module] = []
        if self.sort_by_pos:
            transform_functions.append(LexicographicSort())
        else:
            if self.shuffle_before_sort_by_label:
                transform_functions.append(ShuffleElements())
            elif self.sort_by_pos_before_sort_by_label:
                transform_functions.append(LexicographicSort())
            transform_functions.append(LabelDictSort(self.dataset.id2label))
        transform_functions.append(
            DiscretizeBoundingBox(
                num_x_grid=self.dataset.canvas_width,
                num_y_grid=self.dataset.canvas_height,
            )
        )
        return transform_functions

    def __call__(self, data: LayoutData) -> ProcessedLayoutData:
        assert self.transform is not None and self.return_keys is not None
        _data = self.transform(copy.deepcopy(data))
        return {k: _data[k] for k in self.return_keys}  # type: ignore


@dataclass
class GenTypeProcessor(Processor):
    return_keys: Tuple[str, ...] = (
        "labels",
        "bboxes",
        "gold_bboxes",
        "discrete_bboxes",
        "discrete_gold_bboxes",
    )
    sort_by_pos: bool = False
    shuffle_before_sort_by_label: bool = False
    sort_by_pos_before_sort_by_label: bool = True


@dataclass
class GenTypeSizeProcessor(Processor):
    return_keys: Tuple[str, ...] = (
        "labels",
        "bboxes",
        "gold_bboxes",
        "discrete_bboxes",
        "discrete_gold_bboxes",
    )
    sort_by_pos: bool = False
    shuffle_before_sort_by_label: bool = True
    sort_by_pos_before_sort_by_label: bool = False


@dataclass
class GenRelationProcessor(Processor):
    return_keys: Tuple[str, ...] = (
        "labels",
        "bboxes",
        "gold_bboxes",
        "discrete_bboxes",
        "discrete_gold_bboxes",
        "relations",
    )
    sort_by_pos: bool = False
    shuffle_before_sort_by_label: bool = False
    sort_by_pos_before_sort_by_label: bool = True
    relation_constrained_discrete_before_induce_relations: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        assert self.transform_functions is not None

        self.transform_functions = self.transform_functions[:-1]
        if self.relation_constrained_discrete_before_induce_relations:
            self.transform_functions.append(
                DiscretizeBoundingBox(
                    num_x_grid=self.dataset.canvas_width,
                    num_y_grid=self.dataset.canvas_height,
                )
            )
            self.transform_functions.append(
                AddCanvasElement(discrete_fn=self.transform_functions[-1])
            )
            self.transform_functions.append(AddRelation())
        else:
            self.transform_functions.append(AddCanvasElement())
            self.transform_functions.append(AddRelation())
            self.transform_functions.append(
                DiscretizeBoundingBox(
                    num_x_grid=self.dataset.canvas_width,
                    num_y_grid=self.dataset.canvas_height,
                )
            )


@dataclass
class CompletionProcessor(Processor):
    return_keys: Tuple[str, ...] = (
        "labels",
        "bboxes",
        "gold_bboxes",
        "discrete_bboxes",
        "discrete_gold_bboxes",
    )
    sort_by_pos: bool = True
    shuffle_before_sort_by_label: bool = False
    sort_by_pos_before_sort_by_label: bool = False


@dataclass
class RefinementProcessor(Processor):
    return_keys: Tuple[str, ...] = (
        "labels",
        "bboxes",
        "gold_bboxes",
        "discrete_bboxes",
        "discrete_gold_bboxes",
    )

    sort_by_pos: bool = False
    shuffle_before_sort_by_label: bool = False
    sort_by_pos_before_sort_by_label: bool = True

    gaussian_noise_mean: float = 0.0
    gaussian_noise_std: float = 0.01
    train_bernoulli_beta: float = 1.0

    def __post_init__(self) -> None:
        super().__post_init__()
        assert self.transform_functions is not None

        self.transform_functions = [
            AddGaussianNoise(
                mean=self.gaussian_noise_mean,
                std=self.gaussian_noise_std,
                bernoulli_beta=self.train_bernoulli_beta,
            )
        ] + self.transform_functions


@dataclass
class ContentAwareProcessor(Processor):
    return_keys: Tuple[str, ...] = (
        "idx",
        "labels",
        "bboxes",
        "gold_bboxes",
        "content_bboxes",
        "discrete_bboxes",
        "discrete_gold_bboxes",
        "discrete_content_bboxes",
    )

    metadata: Optional[DataFrame] = None

    sort_by_pos: bool = False
    shuffle_before_sort_by_label: bool = False
    sort_by_pos_before_sort_by_label: bool = True
    filter_threshold: int = 100
    max_element_numbers: int = 10
    original_width: float = 513.0
    original_height: float = 750.0

    possible_labels: List[torch.Tensor] = field(default_factory=list)

    @property
    def saliency_map_to_bboxes(self) -> SaliencyMapToBBoxes:
        return SaliencyMapToBBoxes(threshold=self.filter_threshold)

    def _normalize_bboxes(self, bboxes):
        bboxes = bboxes.float()
        bboxes[:, 0::2] /= self.original_width
        bboxes[:, 1::2] /= self.original_height
        return bboxes

    def __call__(  # type: ignore[override]
        self,
        filename: str,
        idx: int,
        split: str,
    ) -> Optional[Dict[str, torch.Tensor]]:
        saliency_map = cv2.imread(filename)
        content_bboxes = self.saliency_map_to_bboxes(saliency_map)
        if len(content_bboxes) == 0:
            return None

        content_bboxes = self._normalize_bboxes(content_bboxes)

        if split == "train":
            assert self.metadata is not None
            _metadata = self.metadata[
                (self.metadata["poster_path"] == f"train/{idx}.png")
                & (self.metadata["cls_elem"] > 0)
            ]
            labels = torch.tensor(list(map(int, _metadata["cls_elem"])))
            bboxes = torch.tensor(list(map(eval, _metadata["box_elem"])))
            if len(labels) == 0:
                return None

            bboxes[:, 2] -= bboxes[:, 0]
            bboxes[:, 3] -= bboxes[:, 1]
            bboxes = self._normalize_bboxes(bboxes)
            if len(labels) <= self.max_element_numbers:
                self.possible_labels.append(labels)

            data = {
                "idx": idx,
                "labels": labels,
                "bboxes": bboxes,
                "content_bboxes": content_bboxes,
            }
        else:
            if len(self.possible_labels) == 0:
                raise RuntimeError("Please process training data first")

            labels = random.choice(self.possible_labels)
            data = {
                "idx": idx,
                "labels": labels,
                "bboxes": torch.zeros((len(labels), 4)),  # dummy
                "content_bboxes": content_bboxes,
            }

        return super().__call__(data)  # type: ignore


class TextToLayoutProcessorOutput(TypedDict):
    text: str
    embedding: torch.Tensor
    labels: torch.Tensor
    discrete_gold_bboxes: torch.Tensor
    discrete_bboxes: torch.Tensor


@dataclass
class TextToLayoutProcessor(ProcessorMixin):
    return_keys: Tuple[str, ...] = (
        "labels",
        "bboxes",
        "text",
        "embedding",
    )
    text_encode_transform: CLIPTextEncoderTransform = CLIPTextEncoderTransform()

    def _scale(self, original_width, elements_):
        elements = copy.deepcopy(elements_)
        ratio = self.dataset.canvas_width / original_width
        for i in range(len(elements)):
            elements[i]["position"][0] = int(ratio * elements[i]["position"][0])
            elements[i]["position"][1] = int(ratio * elements[i]["position"][1])
            elements[i]["position"][2] = int(ratio * elements[i]["position"][2])
            elements[i]["position"][3] = int(ratio * elements[i]["position"][3])
        return elements

    def __call__(  # type: ignore[override]
        self,
        data: TextToLayoutData,
    ) -> TextToLayoutProcessorOutput:
        text = clean_text(data["text"])

        embedding = self.text_encode_transform(
            clean_text(data["text"], remove_summary=True)
        )
        original_width = data["canvas_width"]
        elements = data["elements"]
        elements = self._scale(original_width, elements)
        elements = sorted(elements, key=lambda x: (x["position"][1], x["position"][0]))

        labels = [self.dataset.label2id[element["type"]] for element in elements]
        labels_tensor = torch.tensor(labels)
        bboxes = [element["position"] for element in elements]
        bboxes_tensor = torch.tensor(bboxes)

        return {
            "text": text,
            "embedding": embedding,
            "labels": labels_tensor,
            "discrete_gold_bboxes": bboxes_tensor,
            "discrete_bboxes": bboxes_tensor,
        }


PROCESSOR_MAP: Dict[str, Type[ProcessorMixin]] = {
    "gen-t": GenTypeProcessor,
    "gen-ts": GenTypeSizeProcessor,
    "gen-r": GenRelationProcessor,
    "completion": CompletionProcessor,
    "refinement": RefinementProcessor,
    "content": ContentAwareProcessor,
    "text": TextToLayoutProcessor,
}


def create_processor(
    dataset: LayoutDataset, task: str, metadata: Optional[pd.DataFrame] = None
) -> ProcessorMixin:
    processor_cls: Type[ProcessorMixin] = PROCESSOR_MAP[task]
    processor = processor_cls(
        dataset=dataset,
        metadata=metadata,
    )
    return processor
