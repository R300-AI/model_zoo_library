class Custom_IPC():
  def __init__(self, chipset, model_path, profiling):
    ext = model_path.split('.')[-1]
    metric = {'cpu': ['engine', 'onnx', 'tflite'], 'gpu': ['engine'], 'apu': ['onnx']}
    assert ext in metric[chipset], f"{ext} format are not support for {chipset} on {type(self).__name__}, please varify the config in --platform."

    self.profiler = [self.delegates(chipset, model_path, profiling) for format in metric[chipset] if format==ext][0]
    self.profiler.run()

  def delegates(self, chipset, model_path, profiling):
    if model_path.endswith('.engine'):
      from .engine import TensorRT_Interpreter
      return TensorRT_Interpreter(model_path, chipset, profiling)
    
    elif model_path.endswith('.onnx'):
      from .engine import ONNX_Interpreter
      return ONNX_Interpreter(model_path, chipset, profiling)
    
    elif model_path.endswith('.tflite'):
      from .engine import ArmNN_TFLite_Interpreter
      return ArmNN_TFLite_Interpreter(model_path, chipset, profiling)
      
class Genio_EVK():
  def __init__(self, chipset, model_path, profiling):
    ext = model_path.split('.')[-1]
    metric = {'cpu': ['engine', 'onnx', 'tflite'], 'gpu': ['tflite'], 'apu': ['onnx', 'tflite']}
    assert ext in metric[chipset], f"{ext} format are not support for {chipset} on {type(self).__name__}, please varify the config in --platform."

    self.profiler = [self.delegates(chipset, model_path, profiling) for format in metric[chipset] if format==ext][0]
    self.profiler.run()

  def delegates(self, chipset, model_path, profiling):
    if model_path.endswith('.engine'):
      from .engine import TensorRT_Interpreter
      return TensorRT_Interpreter(model_path, chipset, profiling)
    
    elif model_path.endswith('.onnx'):
      from .engine import ONNX_Interpreter
      return ONNX_Interpreter(model_path, chipset, profiling)
    
    elif model_path.endswith('.tflite'):
      from .engine import ArmNN_TFLite_Interpreter
      return ArmNN_TFLite_Interpreter(model_path, chipset, profiling)
