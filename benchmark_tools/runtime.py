from .tools import VARIFY_PACKAGE_INSTALLED, GET_LOGGER

class ONNX_Runtime():  
    def __init__(self, model_path, chipset):   #@chipset: [cpu, gpu]
      VARIFY_PACKAGE_INSTALLED('onnx_tool')
      VARIFY_PACKAGE_INSTALLED('onnxruntime')
      import onnx_tool, onnx
      import numpy as np
      import onnxruntime as ort

      self.logger = GET_LOGGER()
      self.logger.info('【 ONNX Runtime 】')

      self.input_shape = np.array([d.dim_value for d in onnx.load(model_path).graph.input[0].type.tensor_type.shape.dim])
      self.logger.info(f"Input Details: {self.input_shape}")
      
      sess = onnx_tool.Model(model_path, {'constant_folding': True, 'verbose': True, 'if_fixed_branch': 'else', 'fixed_topk': 0})
      sess.graph.graph_reorder_nodes()
      sess.graph.shape_infer({'data': self.input_shape})
      sess.graph.profile()
      self.logger.info(sess.graph.print_node_map())

      self.model = ort.InferenceSession(model_path)

    def run(self, input_size):   #@input_size: [None, int]
      import numpy as np
      import time
      iter, total_time = 10, 0
      inputs, x = self.model.get_inputs()[0].name, np.zeros(self.input_shape, dtype=np.float32)
      for _ in range(iter):
        start_point = time.time()
        self.model.run(None, {inputs: x})
        total_time += time.time()-start_point
      self.logger.info(f'Latency: {round(total_time * 1000 / iter, 1)} ms')

class ArmNN_TFLite_Runtime():
    def __init__(self, model_path, chipset):
      VARIFY_PACKAGE_INSTALLED('silabs-mltk')
      from mltk.core import profile_model
      import tensorflow as tf

      self.logger = GET_LOGGER()
      self.logger.info('【 ArmNN TFLite Runtime 】')

      # Loading
      if chipset == 'cpu':
        self.model = tf.lite.Interpreter(model_path = model_path)
          
      elif chipset == 'gpu':
        self.library = "/home/ubuntu/armnn/libarmnnDelegate.so.29"
        self.model = tf.lite.Interpreter(model_path = model_path, 
                                         experimental_delegates = [tf.lite.experimental.load_delegate(library = self.library, options = {"backends": self.auto_backend(model_path), "logging-severity": "info"})])

      # Auto/Manual Profiling 
      try:
         self.logger.info(profile_model(model_path, return_estimates=True))
      except:
         input_details, output_details = self.model.get_input_details(), self.model.get_output_details()
         self.logger.info(f"Input Details: {str(input_details[0]['shape'])} ({str(input_details[0]['dtype'])})")
      
      self.logger.info('initial successed.')
      
    def auto_backend(self, model_path):
      backends = ['CpuAcc', 'GpuAcc', 'GpuAcc,CpuAcc']
      fastest_backend, minimum_latency = None, None
      for backend in backends:
        model = tf.lite.Interpreter(model_path = model_path, 
                                    experimental_delegates = [tf.lite.experimental.load_delegate(library = self.library, options = {"backends": backend, "logging-severity": "info"})])
        model.allocate_tensors()
        
        input, output = model.tensor(model.get_input_details()[0]["index"]), model.tensor(model.get_output_details()[0]["index"])
        start_point = time.time()
        input().fill(3.); model.invoke()
        latency = time.time()-start_point

        if fastest_backend == None:
          fastest_backend = backend; minimum_latency = latency
        elif latency < latency:
          fastest_backend = backend; minimum_latency = latency

      return fastest_backend

    def run(self, input_size):   #@input_size: [None, int]
      import time
      self.model.allocate_tensors()

      iter, total_time = 3, 0
      input, output = self.model.tensor(self.model.get_input_details()[0]["index"]), self.model.tensor(self.model.get_output_details()[0]["index"])
      for _ in range(iter):
        start_point = time.time()
        input().fill(3.)
        self.model.invoke()
        total_time += time.time()-start_point
      self.logger.info(f'Latency: {round(total_time * 1000 / iter, 1)} ms')
