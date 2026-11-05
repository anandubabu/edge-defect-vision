import os
import numpy as np
import onnxruntime as ort
from torch.utils.data import DataLoader
from onnxruntime.quantization import (
    quantize_static, CalibrationDataReader, QuantFormat, QuantType)

from dataset import DefectDataset


# Feeds ~100 training images to the quantizer, one at a time
class CalibReader(CalibrationDataReader):
    def __init__(self, n=100):
        ds = DefectDataset("data/split/train.csv", train=False)
        self.images = [ds[i][0].unsqueeze(0).numpy() for i in range(n)]
        self.pos = 0

    def get_next(self):
        if self.pos >= len(self.images):
            return None                      # tells the quantizer "no more images"
        img = self.images[self.pos]
        self.pos += 1
        return {"image": img}


# Runs an ONNX file on the 270 test images and returns accuracy
def onnx_accuracy(path):
    sess = ort.InferenceSession(path)
    loader = DataLoader(DefectDataset("data/split/test.csv", train=False), batch_size=32)
    correct, total = 0, 0
    for images, labels in loader:
        scores = sess.run(None, {"image": images.numpy()})[0]
        preds = scores.argmax(axis=1)
        correct += (preds == labels.numpy()).sum()
        total += len(labels)
    return correct / total


def size_mb(path):
    total = os.path.getsize(path)
    if os.path.exists(path + ".data"):       # weights may sit in a second file
        total += os.path.getsize(path + ".data")
    return total / 1e6


def main():
    quantize_static(
        "models/defect_prep.onnx",
        "models/defect_int8.onnx",
        CalibReader(),
        quant_format=QuantFormat.QDQ,
        per_channel=True, reduce_range=True,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QUInt8,
        op_types_to_quantize=["Conv", "Gemm"],
    )
    print("saved models/defect_int8.onnx")

    for name, path in [("FP32", "models/defect.onnx"),
                       ("PREP", "models/defect_prep.onnx"),
                       ("INT8", "models/defect_int8.onnx")]:
        print(f"{name}: size {size_mb(path):.2f} MB, test acc {onnx_accuracy(path):.4f}")


if __name__ == "__main__":
    main()