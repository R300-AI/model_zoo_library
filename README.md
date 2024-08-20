## Deploy to MediaTek Genio

* **Supported Operation System and Accelerators**
  
  |         OS       | ArmNN<br><sup>(Cortex-CPU, Mali-GPU)  | NeuronPilot<br><sup>(MDLA, VP)  |          PCIe Plugins          |
  |         ----     |         --------------------          |       -------------------       |      -------------------       |
  |      Yocto       |        :white_check_mark:             |       :white_check_mark:        |                                |
  |      Ubuntu      |          :white_check_mark:           |       :black_square_button:     |  `Hailo-8`                     |

  *Compatibility between the engine and the operating system.*
  
### Ubuntu
**Step1.** Connect to internet and clone this repository to your device.
  ```
  git clone https://github.com/R300-AI/model_zoo_library.git
  cd model_zoo_library

  sudo ln ${LD_LIBRARY_PATH}/libarmnnDelegate.so.29.0 libarmnnDelegate.so.29
  sudo ln ${LD_LIBRARY_PATH}/libarmnn.so.33.0 libarmnn.so.33
  ```

**Step2.** Select and download eagered pre-trained model from [ITRI-AI-Hub/Model-Zoo](https://github.com/R300-AI/ITRI-AI-Hub/tree/main/Model-Zoo).

**Step3.** Run this command to get benchmark reports. (engine)
  ```
  python3 profile.py --model <path-to-your-model>.onnx --engine CPU --input_size '1, 3, 244, 244'
  ```

## Demo Template
### ArmNN Runtime
* [Python](https://github.com/R300-AI/model_zoo_library/blob/main/template/armnn.py)


## Bacnbone
  |    Framework     |                Model Sets             | 
  |         ----     |         --------------------          |  
  | PyTorch/Vision   |  `ResNets`, `SSDs`                    | 
  
## Acknowledgement
### Add more tools
