import shutil, os

def Build_Folder(path):
  if os.path.exists(path):
    shutil.rmtree(path); os.makedirs(path)
  else:
    os.makedirs(path)

import torch, ai_edge_torch, logging

class Benchmark():
  def __init__(self, model):
    self.name = model.__class__.__name__

    logging.info(self.name)


  def Convert_to_TFLite(self, model):
    # Convert
    sample_input = torch.randn(1, 3, 224, 224)
    edge_model = ai_edge_torch.convert(model, sample_input)

    # Inference in Python
    #output = edge_model(*sample_input)

    edge_model.export('./models/resnet.tflite')
    logging.info('Convert toTFLite successed.')

import torchvision

resnet18 = torchvision.models.resnet18(torchvision.models.ResNet18_Weights.IMAGENET1K_V1).eval()
tester = Benchmark(resnet18)
tester.Convert_to_TFLite()
