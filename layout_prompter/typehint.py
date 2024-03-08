from typing import Tuple, TypedDict

import torch


class LayoutData(TypedDict):
    name: str
    bboxes: torch.Tensor
    labels: torch.Tensor
    canvas_size: Tuple[float, float]
    filtered: bool


class ProcessedLayoutData(TypedDict):
    name: str
    bboxes: torch.Tensor
    labels: torch.Tensor
    gold_bboxes: torch.Tensor
    discrete_bboxes: torch.Tensor
    discrete_gold_bboxes: torch.Tensor

    content_bboxes: torch.Tensor
    discrete_content_bboxes: torch.Tensor

    canvas_size: Tuple[float, float]


class Prompt(TypedDict):
    system_prompt: str
    user_prompt: str
