from .package import import_or_install

class TorchScript_Pofiler():
    """
    Chipsets for General Benchmark: [cpu, gpu]
    Chipsets for Genio Benchmark: [cpu]
    """
    def __init__(self, model_path, chipset):  
      import_or_install('torch')
      import torch
      self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

      self.model = torch.jit.load(model_path)
      self.model.to(self.device)
      self.model.eval()
      self.log = f"【TorchScript Runtime】\n - Model: {model_path}\n - Device: {self.device}\n"

    def run(self, input_size):   #@input_size: [None, int]
      from torch.profiler import profile, record_function, ProfilerActivity
      self.log += f" - Input Size: {input_size}\n"
      print(self.log)
        
      inputs = torch.from_numpy(input_size).float()
      with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, profile_memory=True, with_flops=True) as prof:
        with record_function(""):
          with torch.inference_mode():
            model(inputs)
      print(prof.key_averages().table(sort_by="cpu_time_total"))

class ONNX_Profiler():  
    """
    Chipsets for General Benchmark: [cpu]
    Chipsets for Genio Benchmark: [cpu]
    """
    def __init__(self, model_path, chipset):   #@chipset: [cpu, gpu]
      import_or_install('onnx_tool')
      
      self.model = onnx_tool.Model(model_path, {'constant_folding': True, 'verbose': True, 'if_fixed_branch': 'else', 'fixed_topk': 0})
      self.log = f"【ONNX Runtime】\n - Model: {model_path}\n"

    def run(self, input_size):   #@input_size: [None, int]
      print(self.log)

      inputs = torch.from_numpy(input_size).float()
      self.model.graph.graph_reorder_nodes()
      self.model.graph.shape_infer({'data': inputs.shape})
      self.model.graph.profile()
      print(self.model.graph.print_node_map())

class TFLite_Profiler():  
    """
    Chipsets for General Benchmark: [cpu]
    Chipsets for Genio Benchmark: [cpu, gpu, apu]
    """
    def __init__(self, model_path, chipset):   #@chipset: [cpu, gpu]
      import_or_install('onnx_tool')
      
      self.model = onnx_tool.Model(model_path, {'constant_folding': True, 'verbose': True, 'if_fixed_branch': 'else', 'fixed_topk': 0})
      self.log = f"【ONNX Runtime】\n - Model: {model_path}\n"

    def run(self, input_size):   #@input_size: [None, int]
      print(self.log)

      inputs = torch.from_numpy(input_size).float()
      self.model.graph.graph_reorder_nodes()
      self.model.graph.shape_infer({'data': inputs.shape})
      self.model.graph.profile()
      print(self.model.graph.print_node_map())

def TFLite_Profiler(self, inputs, model_path):
  import_or_install('tflite_runtime')
  import numpy as np
  import tflite_runtime.interpreter as tflite
  import time

  interpreter = tflite.Interpreter(model_path = model_path)
  interpreter.allocate_tensors()
  input_details, output_details = interpreter.get_input_details(), interpreter.get_output_details()
  
  inputs = np.zeros(input_details[0]['shape'], dtype=np.float32)
  start_point = time.time()
  for _ in range(10):
    interpreter.set_tensor(input_details[0]["index"], inputs)
    interpreter.invoke()
    interpreter.get_tensor(output_details[0]["index"])
  print((time.time()-start_point) * 100, 'ms')


def TFLite_ArmNN_Profiler(self, inputs, model_path):
  import_or_install('tflite_runtime')
  import numpy as np
  import tflite_runtime.interpreter as tflite
  import time
  
  BACKENDS = "GpuAcc"  #"GpuAcc,CpuAcc,CpuRef"
  DELEGATE_PATH = "/home/ubuntu/armnn/libarmnnDelegate.so.29"
  
  
  interpreter = tflite.Interpreter(model_path = model_path, experimental_delegates = [tflite.load_delegate(library = DELEGATE_PATH, options = {"backends":BACKENDS, "logging-severity": "info"})])
  interpreter.allocate_tensors()
  input_details, output_details = interpreter.get_input_details(), interpreter.get_output_details()
  
  inputs = np.zeros(input_details[0]['shape'], dtype=np.float32)
  start_point = time.time()
  for _ in range(10):
    interpreter.set_tensor(input_details[0]["index"], inputs)
    interpreter.invoke()
    interpreter.get_tensor(output_details[0]["index"])
  print((time.time()-start_point) * 100, 'ms')





