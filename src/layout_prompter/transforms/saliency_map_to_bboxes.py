from typing import Any, List, Optional, Tuple, Union

import cv2
import numpy as np
from langchain_core.runnables import Runnable
from pydantic import BaseModel

from layout_prompter.typehints import PilImage


class SaliencyMapToBboxes(BaseModel, Runnable):
    name: str = "saliency-map-to-bboxes"

    threshold: int = 100
    min_side: int = 80
    min_area: int = 6000
    is_filter_small_bboxes: bool = True

    def is_small_bbox(self, bbox: Union[List[int], Tuple[int, int, int, int]]) -> bool:
        assert len(bbox) == 4, f"bbox must be a list or tuple of 4 integers; got {bbox}"
        return any(
            [
                all([bbox[2] <= self.min_side, bbox[3] <= self.min_side]),
                bbox[2] * bbox[3] < self.min_area,
            ]
        )

    def get_filtered_bboxes(self, contours: Tuple[np.ndarray]) -> np.ndarray:
        bboxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if self.is_filter_small_bboxes and self.is_small_bbox([x, y, w, h]):
                continue

            bboxes.append([x, y, w, h])
        bboxes = sorted(bboxes, key=lambda x: (x[1], x[0]))
        return np.array(bboxes)

    def invoke(  # type: ignore[override]
        self,
        input: PilImage,
        **kwargs: Any,
    ) -> Optional[np.ndarray]:
        assert input.mode == "L", "saliency map must be grayscale image"
        saliency_map_gray = np.array(input)

        _, thresholded_map = cv2.threshold(
            saliency_map_gray, self.threshold, 255, cv2.THRESH_BINARY
        )
        contours, _ = cv2.findContours(
            thresholded_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        bboxes = self.get_filtered_bboxes(contours)  # type: ignore[arg-type]

        return bboxes if len(bboxes) != 0 else None
