class CPU_Benchmarks(model_path):
  def __init__(self, inputs, model_path):
    if imodel_path.endswith('.onnx'):
      ONNX_Profiler(inputs, model_path)
    elif imodel_path.endswith('.py'):
      raise Warning("ONNX Format are recommanded to CPU benchmarks.")
      TorchScript_Pofiler(inputs, model_path)
    else:
      raise Warning("ONNX Format are recommanded to CPU benchmarks.")

  def TFLite_Profiler(self, inputs, model_path):
    raise Warning("Profiling TFLite Format are not support yet.")

  def ONNX_Profiler(self, inputs, model_path):
    import onnx_tool
    m = onnx_tool.Model(model_path, {'constant_folding': True, 'verbose': True, 'if_fixed_branch': 'else', 'fixed_topk': 0})
    m.graph.graph_reorder_nodes()
    m.graph.shape_infer({'data': inputs.shape})
    m.graph.profile()
    print(m.graph.print_node_map())

  def TorchScript_Pofiler(self, inputs, model_path):
      from torch.profiler import profile, record_function, ProfilerActivity
      model = torch.jit.load(model_path)
      model.eval()
      inputs = torch.from_numpy(inputs).float()

      with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, profile_memory=True, with_flops=True) as prof:
        with record_function(""):
          with torch.inference_mode():
            model(inputs)
      print(prof.key_averages().table(sort_by="cpu_time_total"))