from torch.profiler import profile, record_function, ProfilerActivity
import onnx_tool, argparse, torch
import numpy as np

def TorchScript_Profiler(model_path, inputs):
    # load model in TorchScript format.
    model = torch.jit.load(model_path)
    model.eval()

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, profile_memory=True, with_flops=True) as prof:
      with record_function(""):
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
parser.add_argument("-b", "--batch_size", default=None, type=int, help="Batch size when running")
parser.add_argument("-i", "--input_size", default=None, type=str, help="The String of input size")

args = parser.parse_args()

if __name__ == '__main__':
  inputs = torch.from_numpy(np.zeros(np.array(args.input_size.split(',')).astype(int))).float()
  
  if args.model_path.endswith('.pt'):
    TorchScript_Profiler(args.model_path, inputs)
  elif args.model_path.endswith('.onnx'):
    ONNX_Profiler (args.model_path, inputs)
  else:
    raise "Benchmark Format are not support yet."
