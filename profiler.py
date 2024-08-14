from torch.profiler import profile, record_function, ProfilerActivity
import onnx_tool, argparse



def TorchScript_Profiler(model_path, inputs):
    # load model in TorchScript format.
    model = torch.jit.load(model_path)
    model.eval()

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, profile_memory=True, with_flops=True) as prof:
      with record_function(benchmark):
        with torch.inference_mode():
          model(inputs)
    print(prof.key_averages().table(sort_by="cpu_time_total"))

def ONNX_Profiler(model_path, inputs):
  m = onnx_tool.Model(model_path, {'constant_folding': True, 'verbose': True, 'if_fixed_branch': 'else', 'fixed_topk': 0})
  m.graph.graph_reorder_nodes()
  m.graph.shape_infer({'data': inputs.numpy().shape})
  m.graph.profile()
  print(m.graph.print_node_map())

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model_path", default=None, type=str, help="The name of the model in ./models folder")
parser.add_argument("-i", "--input_size", default=None, type=list, help="The shape of inputs")
args = parser.parse_args()

if __name__ == '__main__':
  import platform
  print(platform.system(), platform.processor(), platform.platform(), platform.machine(), platform.version(), platform.uname())
  if args.model_path.endwith('.pt'):
    TorchScript_Profiler(args.model_path, args.input_size)
  elif args.model_path.endwith('.onnx'):
    TorchScript_Profiler(args.model_path, args.input_size)
  else:
    raise "Benchmark Format are not support yet."
