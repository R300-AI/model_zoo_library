class General_Benchmark():
  def __init__(self, chipset, model_path, input_size=None):
    ext = model_path.split('.')[-1]
    engine = {'cpu': ['.pt', '.onnx', '.tflite'], 'gpu': ['.pt'], 'apu': ['.onnx']}
    assert ext in engine[chipset], f"{ext} format are not support for {chipset}."

    self.profiler = [delegate(chipset, model_path) for format in engine[chipset] if format==ext]
    self.profiler.run(input_size)

class Genio_Benchmark():
  def __init__(self, chipset, model_path, input_size=None):
    ext = model_path.split('.')[-1]
    engine = {'cpu': ['.pt', '.onnx', '.tflite'], 'gpu': ['.pt'], 'apu': ['.onnx']}
    assert ext in engine[chipset], f"{ext} format are not support for {chipset}."

    self.profiler = [delegate(chipset, model_path) for format in engine[chipset] if format==ext]
    self.profiler.run(input_size)

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
