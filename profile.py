import argparse
import numpy as np
from libs.profiler import General_Benchmark, Genio_Benchmark

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", default=None, type=str, required=True, help="The path to the candidate model. (e.g. ./../*.onnx, ./../*.tflite)")
parser.add_argument("-p", "--processor", default='general', type=str, choices=['genio'] , help=".")
parser.add_argument("-c", "--chipset", default=None, type=str, required=True,choices=['CPU', 'GPU', 'APU'] , help=".")
parser.add_argument("-i", "--input_size", default=None, type=str, required=True , help="the shape of inputs. (e.g. '1, 3, 244, 244')")
args = parser.parse_args()

if __name__ == '__main__':
  # inputs = np.random.rand(*np.array(args.input_size.split(',')).astype(int))
  if args.processor.lower() == 'general':
    General_Benchmark(args.chipset.lower(), args.model, args.input_size)
  elif args.processor.lower() == 'genio':
    Genio_Benchmark(args.chipset.lower(), args.model, args.input_size)

