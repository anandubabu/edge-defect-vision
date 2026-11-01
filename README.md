# Edge Defect Vision

Steel surface defect classifier: trained in PyTorch, then quantized and run from C++ on a laptop CPU.
**Status:** in progress — data pipeline done.

## Dataset
NEU surface defect database: 1,800 grayscale images, 6 classes, 300 per class.
Split 70 / 15 / 15 (train / val / test), stratified, fixed seed 42.

![Samples](docs/samples.png)

[One line per class: how it looks.]

## Progress
- [x] A1 – Data split, dataset class, sample grid
- [ ] A2 – Training
- [ ] A3 – Evaluation
- [ ] A4–A6 – ONNX export, INT8, C++ app
- [ ] A7–A9 – Benchmark, CI, final README

## How to run
    python train/split.py
    python train/show_samples.py