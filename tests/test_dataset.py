import pandas as pd
import torch
from PIL import Image

from dataset import DefectDataset


def make_csv(tmp_path):
    """Create one fake grey image and a CSV that points to it."""
    img_path = tmp_path / "img.jpg"
    Image.new("L", (200, 200), color=128).save(img_path)
    csv = tmp_path / "list.csv"
    pd.DataFrame({"path": [str(img_path)], "label": [3], "class": ["pitted_surface"]}).to_csv(csv, index=False)
    return csv


def test_output_shape_and_label(tmp_path):
    ds = DefectDataset(make_csv(tmp_path), train=False)
    img, label = ds[0]
    assert len(ds) == 1
    assert img.shape == (3, 224, 224)
    assert label == 3


def test_eval_transform_is_deterministic(tmp_path):
    ds = DefectDataset(make_csv(tmp_path), train=False)
    a, _ = ds[0]
    b, _ = ds[0]
    assert torch.equal(a, b)
