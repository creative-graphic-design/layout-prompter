import numpy as np
from layout_prompter.transforms import SaliencyMapToBboxes

import datasets as ds


def test_saliency_map_to_bboxes(raw_dataset: ds.DatasetDict):
    saliency_map = raw_dataset["train"][0]["pfpn_saliency_map"]

    transformer = SaliencyMapToBboxes(threshold=100)
    bboxes = transformer.invoke(saliency_map)

    assert isinstance(bboxes, np.ndarray)
