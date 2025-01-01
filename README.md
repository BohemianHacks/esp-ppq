![Banner](https://oss.sensetime.com/20210820/9212d4b51db2e186dc39095b9e01cd3a/ccaf7b3f572fbe398f0d42e24435fc59.jpg)

## PPL Quantization Tool 0.6.6 (PPL Quantization Tool)

PPQ is a scalable, high-performance neural network quantification tool for industrial applications.

Neural network quantization, as a commonly used neural network acceleration scheme, has been widely used since 2016. Compared with neural network pruning and architecture search, network quantification is more versatile and has higher industrial practical value. Especially for end-side chips, in scenarios where on-chip area and power consumption are limited, we always want to convert all floating-point operations into fixed-point operations. The value of quantization technology is that floating-point operations and memory access are very expensive, and it relies on complex floating-point operators and high memory access bandwidth. If we can use lower bit-width fixed-point arithmetic to approximate floating-point results within an acceptable range, this will give us significant advantages in chip circuit design, system power consumption, system latency and throughput.

We are in the wave of the times. Artificial intelligence based on neural networks is developing rapidly. Technologies such as image recognition, image super-resolution, content generation, and model reconstruction are changing our lives. Along with this, is the ever-changing model structure, which becomes the first difficulty before model quantification and deployment. In order to handle complex structures, we designed a complete computational graph logic structure and graph scheduling logic. These efforts enable PPQ to parse and modify complex model structures, automatically determine the quantized and non-quantized areas in the network, and allow users to modify the scheduling logic. Take manual control.

Network quantification and performance optimization are serious engineering issues. We hope that users can participate in the network quantification and deployment process and participate in the performance optimization of neural networks. To this end, we provide corresponding deployment-related learning materials in Github, and deliberately emphasize the flexibility of the interface in software design. In our continuous attempts and explorations, we abstracted the logical type of quantizer, which is responsible for initializing quantization strategies on different hardware platforms, and allows users to customize the quantization bit width of each operator and each tensor in the network. Quantification granularity and calibration algorithms, etc. We have reorganized the quantification logic into 27 independent quantification optimization processes (Quantization Optimization Pass). PPQ users can arbitrarily combine the optimization processes according to their needs to complete highly flexible quantification tasks. As a user of PPQ, you can add and modify all optimization processes according to your needs, and explore new boundaries of quantitative technology.

This is a framework designed to handle complex quantification tasks - PPQ's execution engine is specially designed for quantification. As of PPQ 0.6.6 version, the software has a total of 99 common Onnx operator execution logic built-in, and natively supports the execution process. Quantitative simulation operations in . PPQ can complete the inference and quantification of the Onnx model without Onnxruntime. As part of the architecture design, we allow users to register new operator implementations for PPQ using Python + Pytorch or C++ / Cuda, and new logic can also replace existing operator implementation logic. PPQ allows the same operator to have different execution logic on different platforms, thereby supporting running simulations on different hardware platforms. With the help of the customized execution engine and the high-performance implementation of PPQ Cuda Kernel, PPQ has extremely significant performance advantages and can often complete quantification tasks with amazing efficiency.

The development of PPQ is closely related to the inference framework, which allows us to understand many details of hardware inference and strictly control hardware simulation errors. Thanks to the joint efforts of many open source workers at home and abroad, PPQ currently supports working with multiple inference frameworks such as TensorRT, OpenPPL, Openvino, ncnn, mnn, Onnxruntime, Tengine, Snpe, GraphCore, Metax, etc., and has pre-made corresponding quantizers and Export logic. PPQ is a highly extensible model quantization framework. With the functions in ppq.lib, you can extend the quantization capabilities of PPQ to other possible hardware and inference libraries. We look forward to working with you to bring artificial intelligence to thousands of households.

#### In the 0.6.6 version update, we bring you these features:
 1. [FP8 Quantization Specification](https://zhuanlan.zhihu.com/p/574825662), PPQ now supports FP8 [Quantification Simulation and Training] (https://github.com/ openppl-public/ppq/blob/master/ppq/samples/fp8_sample.py)
 2. [PFL basic class library](https://github.com/openppl-public/ppq/blob/master/ppq/samples/yolo6_sample.py), PPQ now provides a more basic set of api functions to help you complete More flexible quantification
 3. More powerful [graph pattern matching](https://github.com/openppl-public/ppq/blob/master/ppq/IR/search.py) and [graph fusion function](https://github .com/openppl-public/ppq/blob/master/ppq/IR/morph.py)
 4. Onnx-based model [QAT](https://github.com/openppl-public/ppq/blob/master/ppq/samples/QAT/imagenet.py) functions
 5. Brand new [TensorRT](https://github.com/openppl-public/ppq/blob/master/md_doc/deploy_trt_by_OnnxParser.md) quantization and export logic
 6. The world’s largest quantitative model library [OnnxQuant](https://github.com/openppl-public/ppq/tree/master/ppq/samples/QuantZoo)
 7. Other unknown software features

### Installation (Installation method)

1. Install CUDA from [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit-archive)

2. Install Complier

```bash
apt-get install ninja-build # for debian/ubuntu user
yum install ninja-build # for redhat/centos user
```

For Windows User:

 (1) Download ninja.exe from [https://github.com/ninja-build/ninja/releases](https://github.com/ninja-build/ninja/releases), add it to Windows PATH.

 (2) Install Visual Studio 2019 from [https://visualstudio.microsoft.com](https://visualstudio.microsoft.com/zh-hans/).

 (3) Add your C++ compiler to Windows PATH Environment, if you are using Visual Studio, it should be like "C:\Program Files\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.16.27023\bin\ Hostx86\x86"

 (4) Update PyTorch version to 1.10+.

3.Install PPQ

```bash
git clone https://github.com/openppl-public/ppq.git
cdppq
pip install -r requirements.txt
python setup.py install
```

* Install PPQ from our docker image (optional):

```bash
docker pull stephen222/ppq:ubuntu18.04_cuda11.4_cudnn8.4_trt8.4.1.5

docker run -it --rm --ipc=host --gpus all --mount type=bind,source=your custom path,target=/workspace stephen222/ppq:ubuntu18.04_cuda11.4_cudnn8.4_trt8.4.1.5 /bin /bash

git clone https://github.com/openppl-public/ppq.git
cdppq
export PYTHONPATH=${PWD}:${PYTHONPATH}
```

* Install PPQ using pip (optional):

```bash
python3 -m pip install ppq
```

### Learning Path (learning route)

#### Basic usage and sample scripts of PPQ
| | **Description Introduction** | **Link Link** |
| :-: | :- | :-: |
| 01 | Model Quantization | [onnx](https://github.com/openppl-public/ppq/blob/master/ppq/samples/Tutorial/quantize.py), [caffe](https://github.com /openppl-public/ppq/blob/master/ppq/samples/quantize_caffe_model.py), [pytorch](https://github.com/openppl-public/ppq/blob/master/ppq/samples/quantize_torch_model.py) |
| 02 | Executor | [executor](https://github.com/openppl-public/ppq/blob/master/ppq/samples/Tutorial/execute.py) |
| 03 | Error Analysis | [analyser](https://github.com/openppl-public/ppq/blob/master/ppq/samples/Tutorial/analyse.py) |
| 04 | Calibrator | [calibration](https://github.com/openppl-public/ppq/blob/master/ppq/samples/Tutorial/calibration.py) |
| 05 | Network fine-tuning | [finetune](https://github.com/openppl-public/ppq/blob/master/ppq/samples/Tutorial/finetune.py) |
| 06 | Network dispatch | [dispatch](https://github.com/openppl-public/ppq/blob/master/ppq/samples/Tutorial/dispatch.py) |
| 07 | Best Practice | [Best Practice](https://github.com/openppl-public/ppq/blob/master/ppq/samples/Tutorial/bestPractice.py) |
| | | |
| 08 | Target platform | [platform](https://github.com/openppl-public/ppq/blob/master/ppq/sampl
