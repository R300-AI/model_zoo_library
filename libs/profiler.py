from .package import import_or_install

class CPU_Benchmarks():
  def __init__(self, inputs, model_path):
    if model_path.endswith('.onnx'):
      self.ONNX_Profiler(inputs, model_path)
    else:
      if model_path.endswith('.pt'):
        self.TorchScript_Pofiler(inputs, model_path)
      else:
        self.TFLite_Profiler(inputs, model_path)

  def TFLite_Profiler(self, inputs, model_path):
    import_or_install('tflite_runtime')
    import numpy as np
    import tflite_runtime.interpreter as tflite
    import time
    
    BACKENDS = "CpuAcc"  #"GpuAcc,CpuAcc,CpuRef"
    DELEGATE_PATH = "/home/ubuntu/armnn/libarmnnDelegate.so.29"
    
    
    interpreter = tflite.Interpreter(model_path = model_path, experimental_delegates = [tflite.load_delegate(library = DELEGATE_PATH, 
                                                                                                                             options = {"backends":BACKENDS, "logging-severity": "info"})])
    interpreter.allocate_tensors()
    input_details, output_details = interpreter.get_input_details(), interpreter.get_output_details()
    
    inputs = np.zeros(input_details[0]['shape'], dtype=np.float32)
    start_point = time.time()
    for _ in range(10):
      interpreter.set_tensor(input_details[0]["index"], inputs)
      interpreter.invoke()
      interpreter.get_tensor(output_details[0]["index"])
    print((time.time()-start_point) * 100, 'ms')

  def ONNX_Profiler(self, inputs, model_path):
    import_or_install('onnx_tool')

    m = onnx_tool.Model(model_path, {'constant_folding': True, 'verbose': True, 'if_fixed_branch': 'else', 'fixed_topk': 0})
    m.graph.graph_reorder_nodes()
    m.graph.shape_infer({'data': inputs.shape})
    m.graph.profile()
    print(m.graph.print_node_map())

  def TorchScript_Pofiler(self, inputs, model_path):
      import_or_install('torch')
    
      from torch.profiler import profile, record_function, ProfilerActivity
      model = torch.jit.load(model_path)
      model.eval()
      inputs = torch.from_numpy(inputs).float()

      with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, profile_memory=True, with_flops=True) as prof:
        with record_function(""):
          with torch.inference_mode():
            model(inputs)
      print(prof.key_averages().table(sort_by="cpu_time_total"))

class GPU_Benchmarks():
  def __init__(self, inputs, model_path):
  
    if model_path.endswith('.tflite'):
      self.TFLite_Profiler(inputs, model_path)
    else:
      print('require .tflite model')

  def TFLite_Profiler(self, inputs, model_path):
    import_or_install('tflite_runtime')
    import numpy as np
    import tflite_runtime.interpreter as tflite
    import time
    
    BACKENDS = "GpuAcc,CpuAcc,CpuRef"
    DELEGATE_PATH = "./libarmnnDelegate.so.29"
    
    
    interpreter = tflite.Interpreter(model_path = model_path, experimental_delegates = [tflite.load_delegate(library = DELEGATE_PATH, 
                                                                                                                             options = {"backends":BACKENDS, "logging-severity": "info"})])
    interpreter.allocate_tensors()
    input_details, output_details = interpreter.get_input_details(), interpreter.get_output_details()
    
    inputs = np.zeros(input_details[0]['shape'], dtype=np.float32)
    
    start_point = time.time()
    for _ in range(10):
      interpreter.set_tensor(input_details[0]["index"], inputs)
      interpreter.invoke()
      interpreter.get_tensor(output_details[0]["index"])
    print((time.time()-start_point) * 100, 'ms')
