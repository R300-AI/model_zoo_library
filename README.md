# Usage
**Step1.** Connect to internet and clone this repository to your device.
  ```
  git clone https://github.com/R300-AI/model_zoo_library.git
  cd model_zoo_library
  ```

**Step2.** Select and download eagered pre-trained model from [ITRI-AI-Hub/Model-Zoo](https://github.com/R300-AI/ITRI-AI-Hub/tree/main/Model-Zoo).

## Deploy to MediaTek Genios

* **Supported Operation System and Accelerators**
  
  |         OS       | ArmNN<br><sup>(Cortex-CPU, Mali-GPU)  | NeuronPilot<br><sup>(MDLA, VP)  |          PCIe Plugins          |
  |         ----     |         --------------------          |       -------------------       |      -------------------       |
  |      Yocto       |        :white_check_mark:             |       :white_check_mark:        |                                |
  |      Ubuntu      |          :white_check_mark:           |       :black_square_button:     |  `Hailo-8`                     |

  *Compatibility between the engine and the operating system.*
  
### Ubuntu

## Deploy to General Computers
  * with CPU only
  ```
  python3 profile.py --model <path-to-your-model>.pt --input_size '1, 3, 244, 244'
  # python3 profile.py --model <path-to-your-model>.onnx
  # python3 profile.py --model <path-to-your-model>.tflite
  ```
  * with NVIDIA-GPU (*NOT prepared yet.*)
  ```
  python3 profile.py --model <path-to-your-model>.pt --processor genio --chipset gpu --input_size '1, 3, 244, 244'
  ```



## Model Zoo Demo Template
### ArmNN Runtime
* [Python](https://github.com/R300-AI/model_zoo_library/blob/main/template/armnn.py)


## Bacnbone
  |    Framework     |                Model Sets             | 
  |         ----     |         --------------------          |  
  | PyTorch/Vision   |  `ResNets`, `SSDs`                    | 
  
## Acknowledgement

This repository is mainly maintained by the EOSL-R300, and we would be grateful for your contribution to help us add more supported benchmarks and hardware.
