import argparse
import numpy as np
from libs.profiler import CPU_Benchmarks, GPU_Benchmarks

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", default=None, type=str, required=True, help="The path to the candidate model. (e.g. ./../*.onnx, ./../*.tflite)")
parser.add_argument("-p", "--processor", default=None, type=str, required=True,choices=['CPU', 'GPU'] , help="choice benchmark HW target.")
parser.add_argument("-i", "--input_size", default=None, type=str, required=True , help="the shape of inputs. (e.g. '1, 3, 244, 244')")
args = parser.parse_args()

if __name__ == '__main__':
  inputs = np.random.rand(*np.array(args.input_size.split(',')).astype(int))
  
  if args.engine == 'CPU':
    CPU_Benchmarks(inputs, args.model)
  elif args.engine == 'ArmNN':
    GPU_Benchmarks(inputs, args.model)

