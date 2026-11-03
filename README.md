# Edge Defect Vision

Steel surface defect classifier: trained in PyTorch, then quantized and run from C++ on a laptop CPU.
**Status:** in progress — data pipeline done.

## Dataset
NEU surface defect database: 1,800 grayscale images, 6 classes, 300 per class.
Split 70 / 15 / 15 (train / val / test), stratified, fixed seed 42.

![Samples](docs/samples.png)

[One line per class: how it looks.]



## How to run
    python train/split.py
    python train/show_samples.py
    
## Training (A2)
- Model: MobileNetV3-Small, pretrained on ImageNet, last layer replaced with 6 outputs (transfer learning)
- Optimizer: Adam, batch size 32, 10 epochs, CPU only (Surface Pro, i5-7300U)
- Compared 3 learning rates in MLflow:

| Learning rate | Best val accuracy | Notes |
|---|---|---|
| 0.003 | .96 | not the best |
| 0.001 | 1.000 | dipped to 0.967 at epoch 8 |
| 0.0003 | 1.000 | stable at 1.000 from epoch 7 — **chosen** |

Validation set: 270 images. Final test-set score follows in A3.   
    
