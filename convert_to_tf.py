import onnx
from onnx2pytorch import ConvertModel
import torchinfo
import torch
#from onnx_tf.backend import prepare

model_path = 'models/u2net.onnx'
output_path = 'models/u2net_tf.pb'
onnx_model = onnx.load(model_path)  # load onnx model
pytorch_model = ConvertModel(onnx_model)
pytorch_model.eval()
data = torch.randn(1, 3, 320, 320)
output = pytorch_model(data)

# print (pytorch_model)
# tf_rep = prepare(onnx_model)  # prepare tf representation
# tf_rep.export_graph(output_path)  # export the model

#torchinfo.summary(pytorch_model)