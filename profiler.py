import argparse, torch
import numpy as np
from libs.profiler import TorchScript_Profiler, ONNX_Profiler

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", default=None, type=str, required=True, help="The path to the candidate model. (e.g. ./../*.onnx, ./../*.tflite)")
parser.add_argument("-t", "--engine", default=None, type=str, required=True,choices=['CPU', 'ArmNN', 'MDLA', 'VP'] , help="choice benchmark HW target.")
parser.add_argument("-i", "--input_size", default=None, type=str, required=True , help="the shape of inputs. (e.g. '1, 3, 244, 244')")
args = parser.parse_args()

if __name__ == '__main__':
  inputs = np.random.rand(*args.input_size.astype(int))
  
  if args.engine == 'cpu':
    Profile_CPU(inputs, args.model)

