import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
from torchvision import models


def build_model():
    m = models.mobilenet_v3_small(weights=None)
    m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, 6)
    return m.eval()


def test_output_has_6_classes():
    out = build_model()(torch.randn(2, 3, 224, 224))
    assert out.shape == (2, 6)


def test_onnx_export_matches_pytorch(tmp_path):
    model = build_model()
    dummy = torch.randn(1, 3, 224, 224)
    path = tmp_path / "m.onnx"
    torch.onnx.export(model, dummy, str(path), input_names=["image"],
                      output_names=["scores"], dynamo=False, opset_version=17)
    with torch.no_grad():
        expected = model(dummy).numpy()
    got = ort.InferenceSession(str(path)).run(None, {"image": dummy.numpy()})[0]
    assert np.abs(expected - got).max() < 1e-4
