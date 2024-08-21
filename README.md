# model_zoo_library: A Hardware Performance Benchmarking Toolkit

## Workflow

**Step1.** Connect to internet and clone this repository to your device.
  ```bash
  git clone https://github.com/R300-AI/model_zoo_library.git
  cd model_zoo_library
  ```

**Step2.** Download eagered pre-trained model from [ITRI-AI-Hub/Model-Zoo](https://github.com/R300-AI/ITRI-AI-Hub/tree/main/Model-Zoo).

**Step3.** Run.
  ```bash
  python3 profile.py --model <path-to-your-model>.pt
  ```
> **Params**: <br>
> * `--model`: <br>
> * `--platform`: <br>
> * `--chipset`: <br>

## Supported Metrics

* **MediaTek Genios**
  
  |         OS       | ArmNN<br><sup>(Cortex-CPU, Mali-GPU)  | NeuronPilot<br><sup>(MDLA, VP)  |          PCIe Plugins          |
  |         ----     |         --------------------          |       -------------------       |      -------------------       |
  |      Yocto       |        :white_check_mark:             |       :white_check_mark:        |                                |
  |      Ubuntu      |          :white_check_mark:           |       :black_square_button:     |  `Hailo-8`                     |

  *Compatibility between the engine and the operating system.*

## Model Zoo Demo Template
### ArmNN Runtime
* [Python](https://github.com/R300-AI/model_zoo_library/blob/main/template/armnn.py)
  
## Acknowledgement

This repository is mainly maintained by the EOSL-R300, and we would be grateful for your contribution to help us add more supported hardware benchmarks.
