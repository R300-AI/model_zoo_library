import torch
from torchsummary import summary

from modelzoo_pytorch.models.alexnet import AlexNet

model = AlexNet()
model.eval()

dummy_input = torch.randn(1, 3, 224, 224)
summary(model, dummy_input.shape[1:])
