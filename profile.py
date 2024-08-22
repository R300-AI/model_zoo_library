import argparse
import numpy as np
from benchmark_tools.platform import Custom_IPC, Genio_EVK

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", default=None, type=str, required=True, help="The path to the candidate model. (e.g. ./../*.onnx, ./../*.tflite)")
parser.add_argument("-p", "--platform", default='custom', type=str, choices=['custom', 'genio'] , help=".")
parser.add_argument("-c", "--chipset", default='cpu', type=str, choices=['cpu', 'gpu', 'apu'] , help=".")
parser.add_argument("-i", "--input_size", default=None, type=str , help="the shape of inputs. (e.g. '1, 3, 244, 244')")
args = parser.parse_args()

if __name__ == '__main__':
  # inputs = np.random.rand(*np.array(args.input_size.split(',')).astype(int))
  if args.platform.lower() == 'custom':
    Custom_IPC(args.chipset.lower(), args.model, args.input_size)
  elif args.platform.lower() == 'genio':
    Genio_EVK(args.chipset.lower(), args.model, args.input_size)

