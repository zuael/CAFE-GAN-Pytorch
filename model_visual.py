# 针对有网络模型，但还没有训练保存 .pth 文件的情况
import netron
import torch.onnx
from torch.autograd import Variable
from torchvision.models import resnet18  # 以 resnet18 为例

myNet = resnet18()  # 实例化 resnet18
x = torch.randn(16, 3, 40, 40)  # 随机生成一个输入
modelData = "./demo.pth"  # 定义模型数据保存的路径
# modelData = "./demo.onnx"  # 有人说应该是 onnx 文件，但我尝试 pth 是可以的 
torch.onnx.export(myNet, x, modelData)  # 将 pytorch 模型以 onnx 格式导出并保存
netron.start(modelData)  # 输出网络结构

#  针对已经存在网络模型 .pth 文件的情况
import netron

modelData = "./demo.pth"  # 定义模型数据保存的路径
netron.start(modelData)  # 输出网络结构