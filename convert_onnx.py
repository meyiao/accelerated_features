import os
import torch
from modules.xfeat import XFeat
from modules.xfeat import XFeatModel
import onnxruntime as ort

# os.environ['CUDA_VISIBLE_DEVICES'] = '' #Force CPU, comment for GPU

# set the model to evaluation mode
net = XFeatModel().eval()

# Random input
x = torch.randn(1, 3, 640, 640)

# export to ONNX
torch.onnx.export(net, x, "xfeat.onnx", verbose=True,
                  input_names=['input'],
                  output_names=['output_feats', "output_keypoints", "output_heatmap"],
                  opset_version=11)

print("ONNX model saved as xfeat.onnx")

# check the onnx model with onnxruntime
ort_session = ort.InferenceSession("xfeat.onnx")
print("ONNX model loaded successfully")

outputs = ort_session.run(None, {"input": x.numpy()})

# pytorch model outputs
torch_outputs = net(x)

# compare the outputs
for i in range(len(outputs)):
    print(f"onnx output shape {i}: {outputs[i].shape}")
    print(f"torch output shape {i}: {torch_outputs[i].shape}")
    print(f"Output {i} comparison: {torch.allclose(torch_outputs[i], torch.tensor(outputs[i]))}")
    print(f'Output {i} max diff: {torch.max(torch.abs(torch_outputs[i] - torch.tensor(outputs[i])))}')
    print("\n")
