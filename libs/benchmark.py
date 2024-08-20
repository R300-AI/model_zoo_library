from profiler import TorchScript_Pofiler, ONNX_Profiler, TFLite_Profiler

class Benchmarks():
  def Benchmarks(processor, model_path, inputs=None):
    processor = processor.lower()
    assert processor in ['cpu', 'gpu'], "--processor should be 'cpu' or 'gpu."
    assert model_path.split('.')[-1] in ['pt', 'onnx', 'tflite'], "--model_path should be 'pt', 'onnx' or 'tflite' format."
    
    if model_path.endswith('.pt'):
      self.TorchScript_Pofiler(processor, model_path)
    
    elif model_path.endswith('.onnx'):
      self.ONNX_Profiler(processor, model_path)
    
    elif model_path.endswith('.tflite'):
      self.TFLite_Profiler(processor, model_path)
