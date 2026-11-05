import os
import nncf
import openvino as ov
from torch.utils.data import DataLoader

from dataset import DefectDataset


# NNCF gives us one batch from the DataLoader; we return only the image as NumPy
def transform_fn(batch):
    images, _ = batch
    return images.numpy()


# Test-set accuracy of an OpenVINO model
def ov_accuracy(compiled):
    loader = DataLoader(DefectDataset("data/split/test.csv", train=False), batch_size=32)
    correct, total = 0, 0
    for images, labels in loader:
        scores = compiled(images.numpy())[0]
        preds = scores.argmax(axis=1)
        correct += (preds == labels.numpy()).sum()
        total += len(labels)
    return correct / total


def main():
    core = ov.Core()
    model = core.read_model("models/defect.onnx")   # your FP32 ONNX model

    # 300 training images for calibration (the "measuring" step)
    calib_loader = DataLoader(DefectDataset("data/split/train.csv", train=False),
                              batch_size=1, shuffle=True)
    calib_ds = nncf.Dataset(calib_loader, transform_fn)

    # Quantize to INT8 (NNCF keeps sensitive layers safe automatically)
    q_model = nncf.quantize(model, calib_ds, subset_size=300)
    ov.save_model(q_model, "models/defect_int8_ov.xml")
    print("saved models/defect_int8_ov.xml")

    # Compare FP32 and INT8 on the test set
    for name, m in [("FP32", model), ("INT8", q_model)]:
        compiled = core.compile_model(m, "CPU")
        print(f"{name}: test acc {ov_accuracy(compiled):.4f}")

    print(f"INT8 size: {os.path.getsize('models/defect_int8_ov.bin') / 1e6:.2f} MB")


if __name__ == "__main__":
    main()