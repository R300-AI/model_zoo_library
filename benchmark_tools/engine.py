from .tools import VARIFY_PACKAGE_INSTALLED, GET_LOGGER
import time, os
import numpy as np

class ONNX_Interpreter():  
    def __init__(self, model_path, chipset, profiling):
      VARIFY_PACKAGE_INSTALLED('onnx')
      VARIFY_PACKAGE_INSTALLED('onnxruntime')
      import onnx
      import onnxruntime as ort

      self.logger = GET_LOGGER()
      self.logger.info('【 ONNX Runtime 】')

      self.input_shape = np.array([d.dim_value for d in onnx.load(model_path).graph.input[0].type.tensor_type.shape.dim])
      self.logger.info(f"Input Details: {self.input_shape}")

      if profiling == True:
        VARIFY_PACKAGE_INSTALLED('onnx_tool')
        import onnx_tool
        sess = onnx_tool.Model(model_path, {'constant_folding': True, 'verbose': True, 'if_fixed_branch': 'else', 'fixed_topk': 0})
        sess.graph.graph_reorder_nodes()
        sess.graph.shape_infer({'data': self.input_shape})
        sess.graph.profile()
        self.logger.info(sess.graph.print_node_map())

      self.model = ort.InferenceSession(model_path)

    def run(self):
      iter, total_time = 10, 0
      dtype = {'tensor(float)': np.float32}
      print(self.model.get_inputs()[0].type)
      inputs, x = self.model.get_inputs()[0].name, np.zeros(self.input_shape, dtype=dtype[self.model.get_inputs()[0].type])
      for _ in range(iter):
        start_point = time.time()
        self.model.run(None, {inputs: x})
        total_time += time.time()-start_point
      self.logger.info(f'Latency: {round(total_time * 1000 / iter, 1)} ms')

class ArmNN_TFLite_Interpreter():
    def __init__(self, model_path, chipset, profiling):
      if profiling == True:
        VARIFY_PACKAGE_INSTALLED('tensorflow')
        import tensorflow.lite as tflite
        from tensorflow.lite.experimental import load_delegate
      else:
        VARIFY_PACKAGE_INSTALLED('tflite-runtime')
        import tflite_runtime.interpreter as tflite
        from tflite_runtime.interpreter import load_delegate

      # Initialize
      self.logger = GET_LOGGER()
      self.logger.info('【 ArmNN TFLite Runtime 】')
      if chipset == 'cpu':
        self.model = tflite.Interpreter(model_path = model_path)
          
      elif chipset == 'gpu':
        os.system("sh ./libs/install_pyarmnn.sh")
        
        self.library = "/home/ubuntu/armnn/libarmnnDelegate.so.29"
        self.model = tflite.Interpreter(model_path = model_path, 
                                         experimental_delegates = [load_delegate(library = self.library, options = {"backends": self.auto_backend(model_path, tflite, load_delegate),
                                         									    "logging-severity": "info"})])

      input_details, output_details = self.model.get_input_details(), self.model.get_output_details()
      self.logger.info(f"Input Details: {str(input_details[0]['shape'])} ({str(input_details[0]['dtype'])})")

      # Profiling
      if profiling == True:
        VARIFY_PACKAGE_INSTALLED('silabs-mltk')
        from mltk.core import profile_model
        self.logger.info(profile_model(model_path, return_estimates=True))
      else:
        self.logger.info('profiling are disable.')

      self.logger.info('initial successed.')
      
    def auto_backend(self, model_path, tflite, load_delegate):
      backends = ['CpuAcc', 'GpuAcc', 'GpuAcc,CpuAcc']
      fastest_backend, minimum_latency = None, None
      for backend in backends:
        print(backend)
        model = tflite.Interpreter(model_path = model_path, 
                                    experimental_delegates = [load_delegate(library = self.library, options = {"backends": backend, "logging-severity": "info"})])
        model.allocate_tensors()
        
        input, output = model.tensor(model.get_input_details()[0]["index"]), model.tensor(model.get_output_details()[0]["index"])
        input().fill(3.); model.invoke()
        start_point = time.time()
        input().fill(3.); model.invoke()
        latency = time.time()-start_point

        if fastest_backend == None:
          fastest_backend = backend; minimum_latency = latency
        elif latency < latency:
          fastest_backend = backend; minimum_latency = latency

      return fastest_backend

    def run(self):
      self.model.allocate_tensors()

      iter = 10
      input, output = self.model.tensor(self.model.get_input_details()[0]["index"]), self.model.tensor(self.model.get_output_details()[0]["index"])
      start_point = time.time()
      for _ in range(iter):
        input().fill(3.)
        self.model.invoke()
      self.logger.info(f'Latency: {round((time.time() - start_point) * 1000 / iter, 1)} ms')
      
