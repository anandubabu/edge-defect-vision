import numpy as np
import torch
import torch.nn as nn
import onnxruntime as ort
from torchvision import models

def main():
    # Recreate the trained classifier with the same six-class output layer.
    model=models.mobilenet_v3_small(weights=None)
    model.classifier[-1]=nn.Linear(model.classifier[-1].in_features, 6)
    model.load_state_dict(torch.load("models/best_lr0.0003.pth"))
    model.eval()

    # Use the model's expected input shape as the ONNX export example.
    dummy=torch.randn(1, 3, 224, 224)

    # Keep the batch dimension flexible so the exported model can process batches.
    torch.onnx.export(
        model,
        dummy,
        "models/defect.onnx",
        input_names=["image"],
        output_names=["scores"],
        dynamic_axes={"image": {0: "batch"}, "scores": {0: "batch"}},
        dynamo=False,
        opset_version=17,
    )
    print("Saved models/defect.onnx")

    # Compare PyTorch and ONNX outputs to verify that the export preserves results.
    with torch.no_grad():
        torch_out=model(dummy).numpy()
        sess=ort.InferenceSession("models/defect.onnx")
        onnx_out=sess.run(None,{"image":dummy.numpy()})[0]
        diff=np.abs(torch_out-onnx_out).max()
        print("Max diffrence :",diff)

if __name__=="__main__":
    main()