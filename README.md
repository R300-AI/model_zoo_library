# model_zoo_library
## Model Zoo
### MediaTek Genio
* **Supported Operation System and Accelerators**
  
  |         OS       | ArmNN<br><sup>(Cortex-CPU, Mali-GPU)  | NeuronPilot<br><sup>(MDLA, VP)  |          PCIe Plugins          |
  |         ----     |         --------------------          |       -------------------       |      -------------------       |
  |      Yocto       |        :white_check_mark:             |       :white_check_mark:        |                                |
  |      Ubuntu      |       :black_square_button:           |       :white_check_mark:        |  `Hailo-8`                     |

  *Compatibility between the engine and the operating system.*

**Step1.** Connect to internet and clone this repository to your device.
  ```
  git clone https://github.com/R300-AI/model_zoo_library.git
  cd model_zoo_library

  pip install -r requirements.txt
  ```

**Step2.** Select and download eagered pre-trained model from [ITRI-AI-Hub/Model-Zoo](https://github.com/R300-AI/ITRI-AI-Hub/tree/main/Model-Zoo).

**Step3.** Run this command to get benchmark reports. (engine)
  ```
  python3 profile.py --model <path-to-your-model>.onnx --engine cpu --input_size '1, 3, 244, 244'
  ```

## Demo Template
### ArmNN
#### Python
  ```python
  import numpy as np
  import tflite_runtime.interpreter as tflite
  
  # Load TFLite model and allocate tensors.
  # (if you are using the complete tensorflow package you can find load_delegate in tf.experimental.load_delegate)
  armnn_delegate = tflite.load_delegate( library="<path-to-armnn-binaries>/libarmnnDelegate.so",
                                         options={"backends": "CpuAcc,GpuAcc,CpuRef", "logging-severity":"info"})
  # Delegates/Executes all operations supported by Arm NN to/with Arm NN
  interpreter = tflite.Interpreter(model_path="<your-armnn-repo-dir>/delegate/python/test/test_data/mock_model.tflite", 
                                   experimental_delegates=[armnn_delegate])
  interpreter.allocate_tensors()
  
  # Get input and output tensors.
  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()
  
  # Test model on random input data.
  input_shape = input_details[0]['shape']
  input_data = np.array(np.random.random_sample(input_shape), dtype=np.uint8)
  interpreter.set_tensor(input_details[0]['index'], input_data)
  
  interpreter.invoke()
  
  # Print out result
  output_data = interpreter.get_tensor(output_details[0]['index'])
  print(output_data)
  ```

## Acknowledgement
### Add more tools
