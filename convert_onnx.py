import os
import torch
from modules.xfeat import XFeat
from modules.xfeat import XFeatModel

# os.environ['CUDA_VISIBLE_DEVICES'] = '' #Force CPU, comment for GPU

# set the model to evaluation mode
net = XFeatModel().eval()

# Random input
x = torch.randn(1, 3, 720, 720)

# export to ONNX
torch.onnx.export(net, x, "xfeat.onnx", verbose=True,
                  input_names=['input'],
                  output_names=['output_feats', "output_keypoints", "output_heatmap"],
                  opset_version=11)

print("ONNX model saved as xfeat.onnx")