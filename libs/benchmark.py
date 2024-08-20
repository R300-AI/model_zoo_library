class General_Benchmark(chipset, model_path, inputs=None):
  def __init__():
    ext = model_path.split('.')[-1]
    engine = {'cpu': ['.pt', '.onnx', '.tflite'], 'gpu': ['.pt'], 'apu': ['.onnx']}
    assert ext in engine[chipset], f"{ext} format are not support for {chipset}."

    self.profiler = [delegate(chipset, model_path) for format in engine[chipset] if format==ext]
    self.profiler.run(model_path)

def delegate(chipset):
  if model_path.endswith('.pt'):
    from profiler import TorchScript_Pofiler
    return TorchScript_Pofiler(model_path, chipset)
  
  elif model_path.endswith('.onnx'):
    from profiler import ONNX_Profiler
    return ONNX_Profiler(model_path, chipset)
  
  elif model_path.endswith('.tflite'):
    from profiler import TFLite_Profiler
    return TFLite_Profiler(model_path, chipset)
