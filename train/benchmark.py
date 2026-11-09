import time
import numpy as np
import torch
import torch.nn as nn
import onnxruntime as ort
import openvino as ov
from torchvision import models

from dataset import DefectDataset

WARMUP = 20
RUNS = 300


def measure(run_once):
    """Run a function many times and return p50 and p95 in milliseconds."""
    for _ in range(WARMUP):
        run_once()
    times = []
    for _ in range(RUNS):
        start = time.perf_counter()
        run_once()
        times.append((time.perf_counter() - start) * 1000)
    return np.percentile(times, 50), np.percentile(times, 95)


def main():
    # One real test image, batch of 1
    img = DefectDataset("data/split/test.csv", train=False)[0][0].unsqueeze(0)
    img_np = img.numpy()

    # 1. PyTorch
    model = models.mobilenet_v3_small(weights=None)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 6)
    model.load_state_dict(torch.load("models/best_lr0.0003.pth"))
    model.eval()
    def run_torch():
        with torch.no_grad():
            model(img)

    # 2. ONNX Runtime FP32
    sess = ort.InferenceSession("models/defect.onnx")
    def run_ort():
        sess.run(None, {"image": img_np})

    # 3 + 4. OpenVINO FP32 and INT8
    core = ov.Core()
    ov_fp32 = core.compile_model("models/defect.onnx", "CPU")
    ov_int8 = core.compile_model("models/defect_int8_ov.xml", "CPU")
    def run_ov_fp32():
        ov_fp32(img_np)
    def run_ov_int8():
        ov_int8(img_np)

    results = [
        ("PyTorch FP32", measure(run_torch)),
        ("ONNX Runtime FP32", measure(run_ort)),
        ("OpenVINO FP32", measure(run_ov_fp32)),
        ("OpenVINO INT8", measure(run_ov_int8)),
    ]

    print(f"{'Engine':<20}{'p50 (ms)':>10}{'p95 (ms)':>10}")
    for name, (p50, p95) in results:
        print(f"{name:<20}{p50:>10.2f}{p95:>10.2f}")


if __name__ == "__main__":
    main()