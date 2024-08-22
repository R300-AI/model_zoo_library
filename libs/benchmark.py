class General_Benchmark():
  def __init__(self, chipset, model_path, input_size=None):
    ext = model_path.split('.')[-1]
    engine = {'cpu': ['engine', 'onnx', 'tflite'], 'gpu': ['engine'], 'apu': ['onnx']}
    assert ext in engine[chipset], f"{ext} format are not support for {chipset}."

    self.profiler = [self.delegates(chipset, model_path) for format in engine[chipset] if format==ext][0]
    self.profiler.run(input_size)

  def delegates(self, chipset, model_path):
    if model_path.endswith('.engine'):
      from .runtime import TensorRT_Runtime
      return TensorRT_Runtime(model_path, chipset)
    
    elif model_path.endswith('.onnx'):
      from .runtime import ONNX_Runtime
      return ONNX_Runtime(model_path, chipset)
    
    elif model_path.endswith('.tflite'):
      from .runtime import ArmNN_TFLite_Runtime
      return ArmNN_TFLite_Runtime(model_path, chipset)
      
class Genio_Benchmark():
  def __init__(self, chipset, model_path, input_size=None):
    ext = model_path.split('.')[-1]
    engine = {'cpu': ['engine', 'onnx', 'tflite'], 'gpu': ['tflite'], 'apu': ['onnx', 'tflite']}
    assert ext in engine[chipset], f"{ext} format are not support for {chipset}."

    self.profiler = [self.delegate(chipset, model_path) for format in engine[chipset] if format==ext]
    self.profiler.run(input_size)

  def delegates(self, chipset, model_path):
    if model_path.endswith('.engine'):
      from .runtime import TensorRT_Runtime
      return TensorRT_Runtime(model_path, chipset)
    
    elif model_path.endswith('.onnx'):
      from .runtime import ONNX_Runtime
      return ONNX_Runtime(model_path, chipset)
    
    elif model_path.endswith('.tflite'):
      from .runtime import ArmNN_TFLite_Runtime
      return ArmNN_TFLite_Runtime(model_path, chipset)
