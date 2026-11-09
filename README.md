# Edge Defect Vision

Steel surface defect classifier: trained in PyTorch, exported to ONNX, quantized to INT8 with OpenVINO, and run from a C++17 app on a laptop CPU.
**Status:** nearly complete. Tests/CI and final polish in progress.

## Results at a glance
- **Test accuracy:** 98.9% (FP32) / 99.3% (INT8), on 270 unseen images
- **Model size:** 6.1 MB → 1.87 MB with INT8 (3.3× smaller, no accuracy loss)
- **Speed:** 10.4 ms (PyTorch) → 2.0 ms (ONNX Runtime) per image, p50, on a 2017 i5 laptop CPU
- **C++ app:** classifies images with the INT8 model, 20/20 correct on a test check

## Dataset
NEU surface defect database: 1,800 grayscale images, 6 classes, 300 per class.
Split 70 / 15 / 15 (train / val / test), stratified, fixed seed 42.

![Samples](docs/samples.png)

- **Crazing:** fine network of cracks across the surface
- **Inclusion:** small dark streaks or spots
- **Patches:** large dark blobs
- **Pitted surface:** many tiny dots
- **Rolled-in scale:** rough, smeared texture
- **Scratches:** long straight bright lines

## Training (A2)
- Model: MobileNetV3-Small, pretrained on ImageNet, last layer replaced with 6 outputs (transfer learning)
- Optimizer: Adam, batch size 32, 10 epochs, CPU only (Surface Pro 5, i5-7300U)
- Compared 3 learning rates in MLflow:

| Learning rate | Best val accuracy | Notes |
|---|---|---|
| 0.003 | 0.96 | least stable |
| 0.001 | 1.000 | dipped to 0.967 at epoch 8 |
| 0.0003 | 1.000 | stable at 1.000 from epoch 7, **chosen** |

## Evaluation (A3): test set, used once
**Test accuracy: 98.9% (267 / 270)**

![Confusion matrix](docs/confusion_matrix.png)

Findings:
1. All 3 errors involve crazing: 2 patches images were predicted as crazing, and 1 crazing image as pitted surface.
2. Inclusion, rolled-in scale and scratches were classified perfectly.
3. Validation accuracy was 100% but test accuracy 98.9%. The small drop shows why a separate, untouched test set is needed.

## ONNX export (A4)
- Exported with `torch.onnx.export` (input `image`, output `scores`, dynamic batch size).
- Verified with ONNX Runtime: on the same input, the ONNX output matches PyTorch within **2e-6** (max absolute difference).

## INT8 quantization (A5)
| Model | Size | Test accuracy |
|---|---|---|
| FP32 (ONNX) | 6.1 MB | 98.9% |
| INT8, ONNX Runtime static (best of 5 attempts) | 1.9 MB | 37% ❌ |
| INT8, OpenVINO NNCF | 1.87 MB | 99.3% ✅ |

ONNX Runtime's static INT8 failed on MobileNetV3 (tried MinMax and percentile calibration, Conv-only quantization, `reduce_range`, and the legacy exporter). OpenVINO NNCF handles the sensitive layers and gives no measurable accuracy loss. (99.3% vs 98.9% is one image out of 270, i.e. no real change.)

## C++ inference app (A6)
A C++17 program that loads the INT8 model with OpenVINO, preprocesses the image with OpenCV exactly like the Python pipeline (RGB, resize 224×224, scale, normalize, HWC→CHW), and prints the predicted class with its probability.

```
$ ./build/classify ../models/defect_int8_ov.xml ../data/raw/.../inclusion_163.jpg
inclusion (97.0147%)
```

`check20.sh` runs the app on the first 20 test images: **20 / 20 correct**.

## Speed benchmark (A7)
Surface Pro 5 (i5-7300U, 2017), batch size 1, 300 runs after 20 warm-up runs.

| Engine | p50 (ms) | p95 (ms) |
|---|---|---|
| PyTorch FP32 | 10.4 | 14.0 |
| ONNX Runtime FP32 | 2.0 | 3.3 |
| OpenVINO FP32 | 2.4 | 3.7 |
| OpenVINO INT8 | 2.6 | 4.3 |

- Exporting to ONNX gives a ~5× speedup over PyTorch.
- INT8 is 3.3× smaller with no accuracy loss, but not faster on this CPU: the i5-7300U has no VNNI instructions for fast 8-bit math. On VNNI-capable CPUs or NPUs, INT8 is typically faster.


## How to run

**Python (training and evaluation):**
```
python train/split.py
python train/show_samples.py
python train/train.py
python train/evaluate.py
python train/export_onnx.py
python train/quantize_ov.py
python train/benchmark.py
```

**C++ app** (needs OpenVINO and OpenCV installed):
```
source ~/openvino/setupvars.sh
cd cpp
cmake -S . -B build
cmake --build build
./build/classify ../models/defect_int8_ov.xml <image.jpg>
```

## Tech stack
Python · PyTorch · torchvision · scikit-learn · MLflow · ONNX · ONNX Runtime · OpenVINO / NNCF · C++17 · OpenCV · CMake