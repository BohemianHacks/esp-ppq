"""PPQ Core Data Structure Abstraction PPQ.
"""

import time  # for hash generation
from enum import Enum
from typing import Any, Iterable, List

import torch

from .common import EXPORT_OVERLAPPED_CONFIG
from .defs import ppq_warning
from .storage import Serializable

MAX_RECURSION_DEPTH = 10000
import sys

sys.setrecursionlimit(MAX_RECURSION_DEPTH)


class QuantizationVisibility(Enum):
    FORCE_EXPORT       = 1
    EXPORT_WHEN_ACTIVE = 2
    INTERNAL           = 3


class NetworkFramework(Enum):
    PPL     = 1
    ONNX    = 2
    CAFFE   = 3
    NXP     = 4
    NATIVE  = 5


class TargetPlatform(Enum):
    """TargetPlatform is a core abstraction of PPQ framework, it defines
    "platform" as an attribute of an operation. Platform attribute of an
    operation indicates where this operation is going to be deployed. This
    feature enables PPQ to simulate inter-device computing.

    Platform attribute also tells PPQ how to quantize an operation, and how to execute it.
        ATTENTION: Different platform might bring different behaviour of a same operation.
        ATTENTION: Operation which is assigned to an non-quantizible platform will never be quantized.

    There are several supported platforms for PPQ now,
        however you are supposed to be aware of some particular platforms here:

    SHAPE_OR_INDEX is a virtual platform, however it is an EXTREMELY IMPORTANT components in PPQ.
        Dispatch an operation to platform SHAPE_OR_INDEX means this operation is SOI-related,
        it processes a SOI tensor and gives a processed SOI, all calculation of this operation must be sent to CPU
            (or any platform capable for calculating this.) when deploy.

        An operation with SHAPE_OR_INDEX platform assigned will never be quantized regardless of its type.
        It is a crucial feature for quantizing network that contains SOI-related operation. (Shufflenet etc.)

        By default, PPQ automatically detects all SOI-related operations, and dispatch them to SHAPE_OR_INDEX platform.
        To understand how this feature works, see also: ppq.sche

    UNSPECIFIED is a virtual platform, all operations are sent to this platform once they were created.
        Quantizer then dispatches them towards desired platform through its quantization logic.
    """
    MNN_INT8      = 100
    TRT_INT8      = 101
    TRT_FP8       = 105
    NCNN_INT8     = 102
    OPENVINO_INT8 = 103
    TENGINE_INT8  = 104
    ASC_INT8      = 106
    
    PPL_CUDA_INT8 = 201
    PPL_CUDA_INT4 = 202
    PPL_CUDA_FP16 = 203
    PPL_CUDA_MIX  = 204

    PPL_DSP_INT8  = 301
    SNPE_INT8     = 302
    PPL_DSP_TI_INT8 = 303
    QNN_DSP_INT8  = 304

    HOST_INT8 = 401

    NXP_INT8  = 501
    FPGA_INT8 = 502

    ESPDL_INT8 = 551
    ESPDL_INT16 = 552
    ESPDL_S3_INT8 = 553
    ESPDL_S3_INT16 = 554
    ESPDL_H_PRE_INT16 = 555
    ESPDL_S3_H_PRE_INT16 = 556

    RKNN_INT8 = 601

    METAX_INT8_C = 701 # channel wise
    METAX_INT8_T = 702 # tensor wise
    
    HEXAGON_INT8  = 801
    GRAPHCORE_FP8 = 901

    FP32 = 0
    FP16 = 1
    BF16 = 2
    FP8  = 3
    INT8 = 4
    # SHAPE-OR-INDEX
    SOI = -1
    # initial state
    UNSPECIFIED   = -2
    # boundary op
    BOUNDARY      = -3
    # just used for calling exporter
    ONNX          = -4
    CAFFE         = -5
    NATIVE        = -6
    ONNXRUNTIME   = -7
    # THIS IS A DUUMY PLATFORM JUST FOR CREATING YOUR OWN EXTENSION.
    EXTENSION     = -10086

    @ classmethod
    def is_quantized_platform(cls, platform) -> bool:
        # removed since PPQ 0.6.6
        return platform in {
            cls.PPL_DSP_INT8, cls.PPL_DSP_TI_INT8, cls.QNN_DSP_INT8, cls.TRT_INT8, cls.NCNN_INT8, cls.NXP_INT8,
            cls.SNPE_INT8, cls.PPL_CUDA_INT8, cls.PPL_CUDA_INT4, cls.EXTENSION, cls.PPL_CUDA_MIX, cls.RKNN_INT8,
            cls.METAX_INT8_C, cls.METAX_INT8_T, cls.OPENVINO_INT8, cls.FPGA_INT8, cls.TENGINE_INT8, 
            cls.FP8, cls.GRAPHCORE_FP8, cls.TRT_FP8, cls.ASC_INT8, cls.UNSPECIFIED, cls.INT8, cls.MNN_INT8}


class RoundingPolicy(Enum):
    """RoundingPolicy is a core setting for PPQ quantization calculation. It
    defines rounding behaviour inside quantization calculation.

    Formula: quant(x) = clip(round(x / scale, RoundingPolicy), -128, 127)

    PPQ Supports 7 different rounding policies now.
    Take a look at https://en.wikipedia.org/wiki/Rounding

    ATTENTION: RoundingPolicy greatly affects PPQ executor behaviour in some cases,
        to get a correct result from PPQ executor,
        make sure your RoundingPolicy is the same as your hardware.
    """
    ROUND_HALF_EVEN            = 0
    ROUND_HALF_UP              = 1
    ROUND_HALF_DOWN            = 2
    ROUND_HALF_TOWARDS_ZERO    = 3
    ROUND_HALF_FAR_FORM_ZERO   = 4
    ROUND_TO_NEAR_INT          = 5
    ROUND_UP                   = 6
    ROUND_DOWN                 = 7


class QuantizationProperty(Enum):
    """QuantizationProperty is a core abstraction for PPQ quantization
    calculation. QuantizationProperty and QuantizationPolicy together build a
    bitmap to describe quantization policy.

    A QuantizationPolicy instance contains multiple QuantizationProperty,
        QuantizationPolicy is used in PPQ (alone with other configuration) to describe how a tensor is quantized.

    During simulating, executor will quantize tensor corresponding to its QuantizationPolicy.
        (QuantizationPolicy is included by TensorQuantizationConfig)

    There are 8 different quantization property(s) supported by PPQ now.

        PER_TENSOR: Also known as per-layer quantization, which mean all parameters of this layer share the same scale and offset.
            (For Convulution layer and Gemm layer which has bias, bias layer will be negative quantized, they do not have a valid scale)

        PER_CHANNEL: parameters are quantized alone channel axis, each channel has a stand-alone scale and offset.

        LINEAR: Linear quantization, follow formula: quant(x) = clip(round(x / scale))

        FLOATING: Low precision float quantization, FP8, BF16, FP16.

        SYMMETRICAL: Symmetrical quantization, offset is deactivated in this mode.

        ASYMMETRICAL: Asymmetrical quantization, offset is activated in this mode.

        POWER_OF_2: Power-of-2 quantization, scale must be pow(2, k) in this mode.

        DYNAMIC: Dynamic Activation Quantization, scale is computed on the fly.

    ATTENTION: Not all combinations of all 8 QuantizationProperty are valid, see QuantizationPolicy.__check_valid
    ATTENTION: QuantizationPolicy is read-only, user can only assign its value when created, the only interface of
        QuantizationPolicy is function QuantizationPolicy.has_property.
    """
    PER_TENSOR   = 0x00000001
    PER_CHANNEL  = 0x00000002
    LINEAR       = 0x00000004
    FLOATING     = 0x00000008
    SYMMETRICAL  = 0x00000010
    ASYMMETRICAL = 0x00000020
    POWER_OF_2   = 0x00000040
    DYNAMIC      = 0x00000080

    def __or__(self, other: int) -> int:
        return self.value + other

    def __ror__(self, other: int) -> int:
        return self.value + other

    def __and__(self, other: int) -> int:
        return self.value & other

    def __rand__(self, other: int) -> int:
        return self.value & other

    def __radd__(self, other: int) -> int:
        return self.value + other

    def __add__(self, other: int) -> int:
        return self.value + other

    def __sub__(self, other: int) -> int:
        return self - (self.value & other)

    def __rsub__(self, other: int) -> int:
        return other - (self.value & other)


class QuantizationPolicy:
    """QuantizationPolicy is a core abstraction for PPQ quantization
    calculation. QuantizationProperty and QuantizationPolicy together build a
    bitmap to describe quantization policy.

    A QuantizationPolicy instance contains multiple QuantizationProperty,
        QuantizationPolicy is used in PPQ (alone with other configuration) to describe how a tensor is quantized.

    During simulating, executor will quantize tensor corresponding to its QuantizationPolicy.
        (QuantizationPolicy is included by TensorQuantizationConfig)

    There are 8 different quantization property(s) supported by PPQ now.

        PER_TENSOR: Also known as per-layer quantization, which mean all parameters of this layer share the same scale and offset.
            (For Convulution layer and Gemm layer which has bias, bias layer will be negative quantized, they do not have a valid scale)

        PER_CHANNEL: parameters are quantized alone channel axis, each channel has a stand-alone scale and offset.

        LINEAR: Linear quantization, follow formula: quant(x) = clip(round(x / scale))

        EXPONENTIAL: Exponential quantization, not yet used.

        SYMMETRICAL: Symmetrical quantization, offset is deactivated in this mode.

        ASYMMETRICAL: Asymmetrical quantization, offset is activated in this mode.

        POWER_OF_2: Power-of-2 quantization, scale must be pow(2, k) in this mode.

        DYNAMIC: Dynamic Activation Quantization, scale is computed on the fly.

    ATTENTION: Not all combinations of all 8 QuantizationProperty are valid, see QuantizationPolicy.__check_valid
    ATTENTION: QuantizationPolicy is read-only, user can only assign its value when created, the only interface of
        QuantizationPolicy is function QuantizationPolicy.has_property.
    """
    def __init__(self, policy: int) -> None:
        if not QuantizationPolicy.__check_valid(policy):
            raise ValueError(
                'invalid quantization pattern, valid partterns are listed in '
                'ppq.core.OperationQuantizationPolicy.__check_valid'
            )
        self._policy = policy

    def has_property(self, property: QuantizationProperty) -> bool:
        return (self._policy & property.value) != 0

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, QuantizationPolicy):
            raise TypeError('Can only compare QuantizationPolicy object '
                            'with another QuantizationPolicy object.')
        return self._policy == o._policy

    @ classmethod
    def __check_valid(cls, policy):
        return policy in {
            # Standard Int Quantization
            QuantizationProperty.ASYMMETRICAL | QuantizationProperty.LINEAR | QuantizationProperty.PER_CHANNEL,
            QuantizationProperty.ASYMMETRICAL | QuantizationProperty.LINEAR | QuantizationProperty.PER_TENSOR,
            QuantizationProperty.SYMMETRICAL | QuantizationProperty.LINEAR | QuantizationProperty.PER_CHANNEL,
            QuantizationProperty.SYMMETRICAL | QuantizationProperty.LINEAR | QuantizationProperty.PER_TENSOR,
            QuantizationProperty.ASYMMETRICAL | QuantizationProperty.LINEAR | QuantizationProperty.PER_TENSOR | QuantizationProperty.POWER_OF_2,
            QuantizationProperty.SYMMETRICAL | QuantizationProperty.LINEAR | QuantizationProperty.PER_TENSOR | QuantizationProperty.POWER_OF_2,
            QuantizationProperty.ASYMMETRICAL | QuantizationProperty.LINEAR | QuantizationProperty.PER_CHANNEL | QuantizationProperty.POWER_OF_2,
            QuantizationProperty.SYMMETRICAL | QuantizationProperty.LINEAR | QuantizationProperty.PER_CHANNEL | QuantizationProperty.POWER_OF_2,
            
            # Low Precision Float Quantization
            # QuantizationProperty.SYMMETRICAL | QuantizationProperty.FLOATING | QuantizationProperty.PER_CHANNEL,
            # QuantizationProperty.SYMMETRICAL | QuantizationProperty.FLOATING | QuantizationProperty.PER_TENSOR,
            # QuantizationProperty.ASYMMETRICAL | QuantizationProperty.FLOATING | QuantizationProperty.PER_CHANNEL,
            # QuantizationProperty.ASYMMETRICAL | QuantizationProperty.FLOATING | QuantizationProperty.PER_TENSOR,
            QuantizationProperty.SYMMETRICAL | QuantizationProperty.FLOATING | QuantizationProperty.PER_CHANNEL | QuantizationProperty.POWER_OF_2,
            QuantizationProperty.SYMMETRICAL | QuantizationProperty.FLOATING | QuantizationProperty.PER_TENSOR | QuantizationProperty.POWER_OF_2,
            # QuantizationProperty.ASYMMETRICAL | QuantizationProperty.FLOATING | QuantizationProperty.PER_CHANNEL | QuantizationProperty.POWER_OF_2,
            # QuantizationProperty.ASYMMETRICAL | QuantizationProperty.FLOATING | QuantizationProperty.PER_TENSOR | QuantizationProperty.POWER_OF_2,

            # Dynamic Activation Quantization
            QuantizationProperty.SYMMETRICAL | QuantizationProperty.LINEAR | QuantizationProperty.PER_TENSOR | QuantizationProperty.DYNAMIC,
            QuantizationProperty.ASYMMETRICAL | QuantizationProperty.LINEAR | QuantizationProperty.PER_TENSOR | QuantizationProperty.DYNAMIC,
            QuantizationProperty.SYMMETRICAL | QuantizationProperty.LINEAR | QuantizationProperty.PER_CHANNEL | QuantizationProperty.DYNAMIC,
            QuantizationProperty.ASYMMETRICAL | QuantizationProperty.LINEAR | QuantizationProperty.PER_CHANNEL | QuantizationProperty.DYNAMIC,
            QuantizationProperty.SYMMETRICAL | QuantizationProperty.LINEAR | QuantizationProperty.PER_TENSOR | QuantizationProperty.DYNAMIC | QuantizationProperty.POWER_OF_2,
            QuantizationProperty.ASYMMETRICAL | QuantizationProperty.LINEAR | QuantizationProperty.PER_TENSOR | QuantizationProperty.DYNAMIC | QuantizationProperty.POWER_OF_2,
            QuantizationProperty.SYMMETRICAL | QuantizationProperty.LINEAR | QuantizationProperty.PER_CHANNEL | QuantizationProperty.DYNAMIC | QuantizationProperty.POWER_OF_2,
            QuantizationProperty.ASYMMETRICAL | QuantizationProperty.LINEAR | QuantizationProperty.PER_CHANNEL | QuantizationProperty.DYNAMIC | QuantizationProperty.POWER_OF_2,
        }

    def to_dict(self) -> dict:
        """return a dictionary to describe this policy.

        nothing funny.
        """
        return {
            property.name: self.has_property(property)
            for property in QuantizationProperty
        }


class QuantizationStates(Enum):
    """QuantizationStates is a core data structure for PPQ quantization.
    QuantizationStates tells whether a quantization configuration is activated.

    ATTENTION: Changes of QuantizationState will greatly affect execution result.

    For a TensorQuantizationConfig instance, there are 9 available quantization states now.
    Only when state is ACTIVATED or NEGATIVE, corresponding tensor will be quantized during the execution.

    Here we give a brief description of each quantization state:

        INITIAL: given when TensorQuantizationConfig is created, is an initial state of all quantization configuration.

        PASSIVE_INIT: for particular parameter like bias of GEMM(Convolution) and padding value of Pad. Usually it
        does not have an independent quantization scale and offset, while gets quantized with other tensor's configuration.
            For GEMM and Convolution, there bias will be quantized with input scale * weight scale.
            For padding value and clip value, it shares the same scale with its input.
        Those parameters will have a PASSIVE_INIT state when created.

        OVERLAPPED: state OVERLAPPED means there is someone else takes control of current tensor,
        and overlapped tensor quantization configuration will be ignored by optimization algorithms and executor.

        Graph fusion always generate overlapped quantization, for a typical conv - relu fusion,
        the output quantization of convolution will be overlapped by the output tensor of relu.
        State OVERLAPPED cares only about quantization behaviour that cross layers.

        ACTIVATE: means corresponding tensor is ready to be quantized with its configuration.

        PASSIVE: means corresponding tensor is ready to be quantized with its configuration.
            (however its configuration is not stand alone, its scale and offset depends on someone else.)

        BAKED: means corresponding tensor has been pre-quantized, its value can directly
            go forward without quantization.
    """
    INITIAL       = 1 # Quantization parameters have just been initialized, the current config is not valid, and the data cannot be used
    ACTIVATED     = 4 # Indicates that the current config is valid
    BAKED         = 2 # Only for parameter quantization, indicating that the parameters have been statically quantized, the current config is not valid, and the data can be used directly
    OVERLAPPED    = 3 # Indicates that this input is not quantized, and the current quantization information is overwritten by the parent quantization information

    PASSIVE_INIT  = 6 # Indicates that this input is passively quantized and has just been initialized and cannot be used
    PASSIVE       = 5 # Indicates that this input is passively quantized, such as bias, clip value, etc., and the passive quantization parameters use the quantization information of other TQCs to complete the quantization
    PASSIVE_BAKED = 7 # Passive quantization and static quantization, the current config is not valid, and the data can be used directly
    FP32          = 8 # Indicates that this input is not quantized
    
    SOI           = -1 # Legacy State
    DEQUANTIZED   = -2 # Legacy State
    DEACTIVED     = -3 # Legacy State

    @ classmethod
    def is_activated(cls, state)->bool:
        return state in {QuantizationStates.ACTIVATED, QuantizationStates.PASSIVE}

    @ classmethod
    def can_export(cls, state) -> bool:
        return state not in {QuantizationStates.INITIAL, QuantizationStates.PASSIVE_INIT, 
                             QuantizationStates.DEQUANTIZED, QuantizationStates.DEACTIVED}


class TensorQuantizationConfig(Serializable):
    """
## TensorQuantizationConfig (Tensor Quantization Control Structure)
PPQ uses the quantization control structure to describe quantization behavior, which is defined in ppq.core.quant. As of PPQ 0.6.6, this structure consists of 15 different attributes. We will introduce you to the design concept of this core data structure.

### 1. QuantizationPolicy Quantization Policy
In TensorQuantizationConfig, the first thing to be mentioned is TQC.policy, which is a QuantizationPolicy object.
The policy attribute is used to describe the quantization rules. A complete quantization policy is composed of multiple quantization properties (QuantizationProperty); in PPQ, we currently support 8 different quantization properties. You can use the following properties to combine to form a custom quantization rule:

1. PER_TENSOR: Quantization is completed in units of Tensor, and each Tensor uses a scale and offset information.

2. PER_CHANNEL: Quantization is performed in units of channels, and each channel uses a scale and offset information.

3. LINEAR: Linear quantization. Common INT8 and INT16 are both linear quantization. There is no exponent bit in the representation of linear quantization.

4. FLOATING: Floating-point quantization, including FP8 E4M3, FP8 E5M2, FP16, BF16 and other formats. In floating-point quantization, the data consists of a base and an exponent.

5. SYMMETRICAL: Symmetric quantization. Offset is not enabled in quantization calculation.

6. ASYMMETRICAL: Asymmetric quantization. Offset is enabled in quantization calculation to complete quantization offset.

7. POWER_OF_2: The scale value must be an integer power of 2. This quantization behavior is more common on the end side and floating-point quantization.

8. DYNAMIC: Enable dynamic quantization strategy. For each batch of data, scale and offset will be dynamically calculated and updated.

The following figure explains the difference between floating point quantization and linear quantization:

![image](https://user-images.githubusercontent.com/43309460/199235366-1e83ed97-0731-4e1d-abeb-b7121e3d2a94.png)

### 2. Linear quantization and related properties

Linear quantization allows the following properties to be combined:

QuantizationProperty.ASYMMETRICAL | QuantizationProperty.LINEAR | QuantizationProperty.PER_CHANNEL,
QuantizationProperty.ASYMMETRICAL | QuantizationProperty.LINEAR | QuantizationProperty.PER_TENSOR,
QuantizationProperty.SYMMETRICAL | QuantizationProperty.LINEAR | QuantizationProperty.PER_CHANNEL,
QuantizationProperty.SYMMETRICAL | QuantizationProperty.LINEAR | QuantizationProperty.PER_TENSOR, QuantizationProperty.ASYMMETRICAL | QuantizationProperty.LINEAR | QuantizationProperty.PER_TENSOR | QuantizationProperty.POWER_OF_2, QuantizationProperty.SYMMETRICAL | QuantizationProperty.LINEAR | QuantizationProperty.PER_TENSOR | QuantizationProperty.POWER_OF_2, QuantizationProperty.ASYMMETRICAL | QuantizationProperty.LINEAR | QuantizationProperty.PER_CHANNEL | QuantizationProperty.POWER_OF_2, QuantizationProperty.SYMMETRICAL | QuantizationProperty.LINEAR | QuantizationProperty.PER_CHANNEL | QuantizationProperty.POWER_OF_2,

Linear quantization is the most commonly used numerical quantization method. Sometimes we also call it uniform quantization. In linear quantization, the calculation method of the quantization operation is:

- Unscaled FP32 = (FP32 / scale) - offset
- INT8 = Clip(Round(Unscale FP32), quant_min, quant_max)
- Dequantized FP32 = (INT8 + offset) * scale

The behavior of the Round function is determined by the TQC.rounding(RoundingPolicy) property. PPQ supports 7 different rounding policies, among which ROUND_HALF_EVEN is the most common rounding policy. For a detailed discussion of rounding policies, please refer to https://en.wikipedia.org/wiki/Rounding

quant_min, quant_max are determined by the TQC.quant_min, TQC.quant_max properties respectively. For linear quantization, they are integers, usually [-128, 127]. Some frameworks use [-127, 127] as the cutoff value, which is advantageous in some scenarios, but [-127, 127] is not allowed to be used as the cutoff value in the Q/DQ operator definition of Onnx.

PPQ can simulate any bit width quantization of 1-32 bits, but for deployment purposes, it is not recommended to use a configuration other than 8 bits. Users should be aware that high bit width quantization may cause the scale to be too small, resulting in floating point underflow.

### 3. Floating point quantization and related properties

Floating point quantization allows the following properties to be combined:

QuantizationProperty.SYMMETRICAL | QuantizationProperty.FLOATING | QuantizationProperty.PER_CHANNEL | QuantizationProperty.POWER_OF_2,

QuantizationProperty.SYMMETRICAL | QuantizationProperty.FLOATING | QuantizationProperty.PER_TENSOR | QuantizationProperty.POWER_OF_2,

In floating point quantization, the calculation method of the quantization function is:

- Unscaled FP32 = (FP32 / scale)

- FP8 = Convert(Unscale FP32, quant_min, quant_max)

- Dequantized FP32 = FP8 * scale

The Convert function has complex behavior, and its conversion process is divided into three different cases:

- When Unscaled FP32 is greater than quant_max, or less than quant_min, truncation is performed directly
- When the Unscaled FP32 amplitude is greater than the minimum value that FP8 can express, the extra base bits need to be removed and the base needs to be rounded
- When the Unscaled FP32 data is less than the minimum value that normalized FP8 can express, the floating point overflows. At this time, we calculate FP8 = Round(Unscaled FP32 / FP8_min) * FP8_min

FP8_min is the minimum value that unnormalized FP8 can express. For the FP8 E4M3 standard, the maximum value that can be expressed is 448.0 and the minimum value is -448.0.

quant_min, quant_max are determined by the TQC.quant_min, TQC.quant_max attributes respectively. For FLOATING quantization, we introduce a new attribute TQC.exponent_bits(int). Use this property to specify how many bits of the total bit width are used to represent the exponent (correspondingly, the base bits are the total bit width-exponent bits-1).

In floating-point quantization, the selection of scale factors has little effect on the quantization effect, so users can use the constant calibration strategy (see ppq.quantization.observer) to set all scale factors to 1.

For specific details about floating-point quantization, please refer to [this article](https://zhuanlan.zhihu.com/p/574825662)

### 4. Other quantization control properties

1. TQC.num_of_bits(int): quantization bit width, for INT8, FP8 quantization, the quantization bit width is 8. For INT16, FP16 quantization, the quantization bit width is 16.

2. TQC.state(QuantizationStates): quantization state. There are currently 8 different quantization states in PPQ. This property greatly enriches the semantics of PPQ quantization information, allowing us to control quantization behavior more flexibly. This attribute can be used to switch the quantization/unquantization state; perform quantization joint fixed point; perform parameter baking.

3. TQC.channel_axis(int): quantization axis. For PER_CHANNEL quantization, use this attribute to specify along which dimension to expand the quantization. If Per-tensor quantization is performed, this attribute is ignored and the user can set it to None.

4. TQC.observer_algorithm(str): observer algorithm, where observer is an object used to determine scale and offset. Use this attribute to specify what type of observer to use to determine scale and offset.

5. TQC.dominator(TensorQuantizationConfig): A pointer to the parent quantization information. In PPQ, TQC and TQC are not independent, and there can be a parent-child relationship between them. All child quantization information shares scale and offset with parent quantization information

6. TQC.visiblity(QuantizationVisibility): Export visibility. Use this property to tell the ppq exporter whether to export the current TQC.

### 5. Initialization of quantization control structure

TensorQuantizationConfig is the core data structure in PPQ. It is always created by Quantizer object:

# The following code creates the corresponding Tensor Quantization Config for a specified operator
quantizer = PFL.Quantizer(platform=TargetPlatform.TRT_FP8, graph=graph) # Get the quantizer corresponding to TRT_FP8
quantizer.quantize_operation(op_name = op.name, platform = dispatching[op.name])

In PPQ, the responsibility of Quantizer is to initialize their quantization control structure for operators. Different quantizers will create control structures according to different rules. For example, the quantizer corresponding to TRT_FP8 will only create quantization information for Conv and Gemm operators, requiring their inputs to be quantized in a symmetric-floating-point-per-channel manner. The quantizer corresponding to DSP_INT8 creates quantization information for almost all operators, requiring them to be quantized in an asymmetric-linear-per-tensor manner.

Users can manually create quantization control structures using the interface in ppq.lib:

# Create a default linear quantization control structure (symmetric, per-tensor)

from ppq.lib import LinearQuantiz
![Quantization State](https://user-images.githubusercontent.com/43309460/199236632-ec69ca29-9900-4875-8299-a196546d0dde.png)
    """
    def __init__(
        self,
        policy: QuantizationPolicy,
        rounding: RoundingPolicy  = RoundingPolicy.ROUND_HALF_EVEN,
        num_of_bits: int          = 8,
        quant_min: int            = -127,
        quant_max: int            = 128,
        exponent_bits: int        = 0,
        scale: Any                = None,
        offset: Any               = None,
        observer_algorithm: str   = None,
        detail: Any               = None,
        channel_axis: int         = None,
        visibility: QuantizationVisibility = QuantizationVisibility.EXPORT_WHEN_ACTIVE,
        state: QuantizationStates = QuantizationStates.INITIAL
    ):
        """Create a PPQ Tensor Quantization Configuration Instance.

        Args:
            policy (QuantizationPolicy):
                Quantization policy instance which defines the quantization behaviour from marco view.

            rounding (RoundingPolicy): Rounding policy used in quantization.

            num_of_bits (int): Quantization fraction bits. (2 < num_of_bits < 32)
            
            exponent_bits (int): Quantization exponent bits. (0 < num_of_bits < 8)
                For Int8 Quantization, num_of_bits = 8 and exponent_bits = 0
                For FP8 Quantization, num_of_bits = 4 and exponent_bits = 4

            quant_min (int): An integer value represents the upper bound(inclusive) of quantized value.

            quant_max (int): An integer value represents the lower bound(inclusive) of quantized value.

            scale (Any):
                Scale of quantized value, for per-tensor quantization policy, we use a single float as its scale,
                while for per-channel quantization policy, it will be an array that contains scales for each channel.

            offset (Any): Quantization offset for ASYMMETRICAL quantization policy,
                it will be set as 0 in SYMMETRICAL quantization schema.

            observer_algorithm (str): A string represents an observing algorithm for this tensor.
                PPQ support 'kl', 'minmax' observer now.

            detail (Any, optional): Only used by PPQ internal logic, detail is used to store some internal data,
                you are not supposed to use it.

            channel_axis (int, optional): Only used in PER_CHANNEL quantization, channel index.
        
            visiblity (Visiblity): visiblity is the attribute that controls export logic.

            Currently, there are 3 Visiblity level in PPQ:
            if Visiblity == FORCE_EXPORT, ppq exporter will export this TQC 
                ignoring state check(even if current TQC has been overrlapped).
            if Visiblity == EXPORT_WHEN_ACTIVD, ppq exporter will export this TQC only when it has been actived.
            if Visiblity == INTERNAL, This TQC will not be exported.

            state (QuantizationStates, optional):
                Defaults to QuantizationStates.INITIAL, see QuantizationStates for more detail.
        """

        assert num_of_bits <= 32, 'Cannot quantize a tensor with more than 32 bits.'
        assert num_of_bits >= 2, 'Cannot quantize a tensor with less than 2 bits.'
        assert exponent_bits <= 8, 'Cannot quantize a tensor with more than 8 bits exponent(fp32 overflow).'
        assert exponent_bits >= 0, 'Cannot quantize a tensor with less than 0 bits exponent.'
        
        self._policy = policy
        self._exponent_bits = exponent_bits
        self._num_of_bits = num_of_bits
        self._scale = scale
        self._offset = offset
        self.state = state
        self._rounding = rounding
        self._quant_min = quant_min
        self._quant_max = quant_max
        self._channel_axis = channel_axis
        self.observer_algorithm = observer_algorithm
        self.detail = {} if detail is None else detail
        self._dominator = self # union-find
        self._hash = self.__create_hash()
        self._visibility = visibility
        super().__init__()

    def can_export(self, export_overlapped: bool = EXPORT_OVERLAPPED_CONFIG) -> bool:
        if self.visibility == QuantizationVisibility.INTERNAL: 
            return False
        type_check  = isinstance(self.scale, torch.Tensor) and isinstance(self.offset, torch.Tensor)
        valid_states = {QuantizationStates.BAKED, QuantizationStates.PASSIVE_BAKED}

        if export_overlapped: 
            valid_states.add(QuantizationStates.OVERLAPPED)
        state_check = QuantizationStates.is_activated(self.state) or self.state in valid_states

        if (state_check or self.visibility == QuantizationVisibility.FORCE_EXPORT):
            if type_check: return True
        return False

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, TensorQuantizationConfig):
            raise TypeError('Can only compare TensorQuantizationConfig object '
                            'with another TensorQuantizationConfig object.')
        return self._hash == o._hash

    def __str__(self) -> str:
        return f'PPQ TensorQuantizationConfig({self.__hash__()})'

    _hash_seed = int(time.time())
    @ staticmethod
    def __create_hash():
        TensorQuantizationConfig._hash_seed = (
            0x343FD * TensorQuantizationConfig._hash_seed + 0x269EC3) % (2 << 31)
        return TensorQuantizationConfig._hash_seed

    def __hash__(self) -> int:
        return self._hash

    def is_same_scheme(self, o: object) -> bool:
        if not isinstance(o, TensorQuantizationConfig):
            raise TypeError('Can only compare TensorQuantizationConfig object '
                            'with another TensorQuantizationConfig object.')
        return (self.quant_max == o.quant_max and 
                self.quant_min == o.quant_min and 
                self.policy == o.policy and 
                self.num_of_bits == o.num_of_bits and
                self.exponent_bits == o.exponent_bits and
                self.channel_axis == o.channel_axis and
                self.rounding == o.rounding)

    @ property
    def dominated_by(self):
        """dominated_by is a crucial feature for tensor quantization
        configuration in PPQ. This property is actually maintained by union-
        find set data structure.

        Every tensor quantization configuration(A) is created with dominated_by = self, and only when
            it is overlapped by other configuration(B), it shall set A.dominated_by = B.
            Setting A.dominated_by = B also makes A, B as a quantization group.
            (quantization state of A is always set as OVERLAPPED here)

        So to say every tensor quantization configuration with dominated_by != self is overrlaped by
            other quantization configuration. When a tensor quantization configuration is overlapped,
            it means this tensor is already been quantized with another quantization configuration,
            and there is no need to be quantized with this configuration anymore.

        PPQ use this property to find root configuration for each configuration group,

        Returns:
            [TensorQuantizationConfig]: root configuration of this quantization group.

        ATTENTION: This configuration is invalid when self.dominated_by != self.
        """
        if self._dominator == self:
            return self
        else:
            root = self._dominator.dominated_by
            self._dominator = root
            return root

    @ dominated_by.setter
    def dominated_by(self, o):
        assert isinstance(o, TensorQuantizationConfig), (
            'Can only set this attribute with another tensor config.')
        if o._hash == self._hash:
            raise ValueError('Error with TQC.dominated_by = o: o must not equal to TQC its self.')
        root, dominator = self.dominated_by, o.dominated_by
        if self == dominator:
            raise ValueError('Can not Assign Dominator like this, '
                             'Circular reference was detected. Son TQC can not dominate its Father.')
        assert isinstance(root, TensorQuantizationConfig)
        if dominator != root:
            root._dominator = dominator
            self._dominator = dominator
            root.state = QuantizationStates.OVERLAPPED
            self.state = QuantizationStates.OVERLAPPED

    @ property
    def master_by(self):
        if self._dominator == self:
            return self
        else:
            root = self._dominator.dominated_by
            self._dominator = root
            return root

    @ master_by.setter
    def master_by(self, master):
        if not isinstance(master, TensorQuantizationConfig):
            raise TypeError(f'Error with TQC.master_by(o): o must be another Tensor Quantization Config, '
                            f'however {type(master)} was given.')
        if master._hash == self._hash:
            raise ValueError('Error with TQC.dominated_by = o: o must not equal to TQC its self.')
        self._dominator = master
        if master.scale is not None and master.offset is not None:
            self.state   = QuantizationStates.PASSIVE
        else: self.state = QuantizationStates.PASSIVE_INIT

    def is_revisable(self):
        return (self.dominated_by == self and self.state in {
            QuantizationStates.ACTIVATED,
            QuantizationStates.FP32,
            QuantizationStates.FP32,
            QuantizationStates.INITIAL,
            QuantizationStates.FP32,
            QuantizationStates.PASSIVE,
            QuantizationStates.PASSIVE_INIT
        })

    @ property
    def visibility(self) -> QuantizationVisibility:
        """ Export Visibility of this TQC.
        
        * QuantizationVisibility.EXPORT_WHEN_ACTIVE - Export this TQC when it is active.
        
        * QuantizationVisibility.FORCE_EXPORT - Force Export this TQC.
        
        * QuantizationVisibility.INTERNAL - Never Export this TQC.
        
        """
        return self._visibility

    @ visibility.setter
    def visibility(self, visiblity: QuantizationVisibility):
        self._visibility = visiblity

    @ property
    def scale(self) -> torch.Tensor:
        """ Get Quantization Scale of this TQC.
        
        If current TQC is dominated by other, return father TQC's scale instead.
        """
        if self.dominated_by == self: return self._scale
        else: return self.dominated_by.scale

    @ property
    def offset(self) -> torch.Tensor:
        """ Get Quantization Offset of this TQC.
        
        If current TQC is dominated by other, return father TQC's offset instead.
        """
        if self.dominated_by == self: return self._offset
        else: return self.dominated_by.offset

    @ property
    def policy(self) -> QuantizationPolicy:
        """ Get Quantization Policy of this TQC. """
        return self._policy

    @ property
    def num_of_bits(self) -> int:
        """ Get bit-width of this TQC. """
        return self._num_of_bits

    @ property
    def rounding(self) -> RoundingPolicy:
        """ Get Rounding Policy of this TQC. """
        return self._rounding

    @ property
    def quant_min(self) -> int:
        """ Get minimum quant value of this TQC. """
        return self._quant_min

    @ property
    def quant_max(self) -> int:
        """ Get maximum quant value of this TQC. """
        return self._quant_max

    @ property
    def exponent_bits(self) -> int:
        """ Get exponent bit-width of current TQC. 
        
        num_of_bits = exponent_bits + mantissa_bits
        """
        return self._exponent_bits

    @ property
    def mantissa_bits(self) -> int:
        """ Get mantissa bit-width of current TQC. 
        
        num_of_bits = exponent_bits + mantissa_bits
        """
        # there is one bit for sign.
        return self.num_of_bits - self._exponent_bits - 1
    
    @ property
    def channel_axis(self) -> int:
        """ Get Quantization Axis, For Per-tensor Quantization, it returns None. """
        return self._channel_axis

    @ scale.setter
    def scale(self, value: Any):
        if not self.is_revisable():
            raise PermissionError(
                'Can not change scale of this tensor quantization configuration now. '
                'It has been overlapped or has an inactive state. '
                'Due to it is not a active config, any change of this configuration is not allowed.'
            )
        else:
            self._scale = value

    @ offset.setter
    def offset(self, value: Any):
        if not self.is_revisable():
            raise PermissionError(
                'Can not change offset of this tensor quantization configuration now. '
                'It has been overlapped or has an inactive state. '
                'Due to it is not a active config, any change of this configuration is not allowed.'
            )
        else:
            self._offset = value

    @ policy.setter
    def policy(self, policy: QuantizationPolicy):
        self._policy = policy

    @ num_of_bits.setter
    def num_of_bits(self, bits: int):
        self._num_of_bits = bits

    @ rounding.setter
    def rounding(self, policy: RoundingPolicy):
        self._rounding = policy

    @ quant_min.setter
    def quant_min(self, min: int):
        self._quant_min = min

    @ quant_max.setter
    def quant_max(self, max: int):
        self._quant_max = max

    @ exponent_bits.setter
    def exponent_bits(self, bits: int):
        if not self.policy.has_property(QuantizationProperty.FLOATING):
            raise PermissionError(
                'Can not change property: exponent bits for this TQC. '
                'self.policy.has_property(QuantizationProperty.FLOATING) == False.')
        self._exponent_bits = bits

    @ channel_axis.setter
    def channel_axis(self, channel_axis: int):
        if not self.policy.has_property(QuantizationProperty.PER_CHANNEL):
            raise PermissionError(
                'Can not change property: quantization channel axis for this TQC. '
                'self.policy.has_property(QuantizationProperty.PER_CHANNEL) == False.')
        self._channel_axis = channel_axis

    def copy(self):
        """Create a tensor config from this one, keep policy and state
        unchanged.

        if there is an non-empty scale and offset, they will be cloned too.
        """
        scale, offset = None, None
        if self.scale is not None:
            if isinstance(self.scale, torch.Tensor):
                scale = self.scale.clone()
            else: scale = self.scale
        if self.offset is not None:
            if isinstance(self.offset, torch.Tensor):
                offset = self.offset.clone()
            else: offset = self.offset
        config = TensorQuantizationConfig(
            policy=self.policy,
            rounding=self.rounding,
            num_of_bits=self.num_of_bits,
            quant_min=self.quant_min,
            quant_max=self.quant_max,
            scale=scale, offset=offset,
            observer_algorithm=self.observer_algorithm,
            detail=self.detail.copy(),
            state=self.state,
            exponent_bits=self.exponent_bits,
            channel_axis=self.channel_axis,
            visibility=self.visibility
        )
        if self.state == QuantizationStates.OVERLAPPED:
            config._dominator = self._dominator
        return config


class ChannelwiseTensorQuantizationConfig(TensorQuantizationConfig):
    """ Legacy Class Since PPQ 0.6.6, Use TensorQuantizationConfig Instead. """
    def __init__(self,
        policy: QuantizationPolicy, rounding:RoundingPolicy,
        num_of_bits: int, exponent_bits: int, 
        quant_min: int, quant_max: int,
        scale: Any, offset: Any, observer_algorithm: str,
        state: QuantizationStates, channel_axis: int, detail: dict = {}
    ):
        ppq_warning('ChannelwiseTensorQuantizationConfig is now obsolescent(Since PPQ 0.6.6), '
                    'use TensorQuantizationConfig Instead.')
        if policy.has_property(QuantizationProperty.PER_TENSOR):
            raise TypeError('Can not assign QuantizationProperty.PER_TENSOR policy '
                'to a Channel-wise Tensor Quantization Config instance.')
        super().__init__(
            policy=policy, num_of_bits=num_of_bits,
            quant_min=quant_min, quant_max=quant_max, scale=scale, offset=offset,
            observer_algorithm=observer_algorithm, detail=detail, state=state,
            rounding=rounding, exponent_bits=exponent_bits
        )
        self.channel_axis = channel_axis

    @ classmethod
    def convert_from_tensor_config(cls,
        convert_from: TensorQuantizationConfig,
        scale: Iterable = None,
        offset: Iterable = None,
        channel_axis: int = 1,
    ):
        if scale is None: scale = convert_from.scale
        if offset is None: offset = convert_from.offset
        this = ChannelwiseTensorQuantizationConfig(
            policy=convert_from.policy,
            num_of_bits=convert_from.num_of_bits,
            quant_min=convert_from.quant_min,
            quant_max=convert_from.quant_max,
            scale=scale, offset=offset,
            observer_algorithm=convert_from.observer_algorithm,
            detail=convert_from.detail.copy(),
            state=convert_from.state,
            channel_axis=channel_axis,
            rounding=convert_from.rounding,
            exponent_bits=convert_from.exponent_bits
        )
        return this

    def copy(self):
        config = super().copy()
        return self.convert_from_tensor_config(
            config, scale=config.scale, offset=config.offset,
            channel_axis=self.channel_axis)


class OperationQuantizationConfig(Iterable):
    """OperationQuantizationConfig serves as a collection of tensor
    quantization configuration.

    See TensorQuantizationConfig for more information.
    """
    def __init__(
        self,
        input_quantization_configs: List[TensorQuantizationConfig] = None,
        output_quantization_configs: List[TensorQuantizationConfig] = None,
        is_positive_quant_op: bool = True
    ):
        """Create an operation quantization configuration.

        Args:
            input_quantization_configs (List[TensorQuantizationConfig], optional):
                a list contains all configuration of all input variables.

            output_quantization_configs (List[TensorQuantizationConfig], optional):
                a list contains all configuration of all output variables.

            ATTENTION: whether a variable is gonna to be quantized or not, it must have a quantization configuration.

            is_positive_quant_op (bool, optional): [description]. Defaults to True.
                some operations are passively quantized, such as Maxpooling, Padding.
                For those operations, set this property as False, PPQ will use this property to optimize your graph.
        """
        self.input_quantization_config     = self.__check_famliy_config(input_quantization_configs)
        self.output_quantization_config    = self.__check_famliy_config(output_quantization_configs)
        self.is_active_quant_op = is_positive_quant_op

    def export(self) -> str:
        raise Exception('Implement this first')

    def __check_famliy_config(self, famliy_configs):
        for famliy_config in famliy_configs:
            if not isinstance(famliy_config, TensorQuantizationConfig):
                raise TypeError(
                    f'You are trying to set famliy quantization config of {str(self)}, ' \
                    f'However your input is invalid, except one TensorQuantizationConfig object, ' \
                    f'while a {type(famliy_config)} was given.'
                )
        return famliy_configs

    def __str__(self) -> str:
        return f'Inputs config: {self.input_quantization_config}, '\
            f'Outputs config {self.output_quantization_config}'

    def __iter__(self) -> TensorQuantizationConfig:
        return (self.input_quantization_config + self.output_quantization_config).__iter__()

    def copy(self):
        """Create an operation config from this one, keep policy and state
        unchanged.

        if this one has an non-empty scale or offset, they will be cloned too.
        """
        return OperationQuantizationConfig(
            input_quantization_configs=[_.copy() for _ in self.input_quantization_config],
            output_quantization_configs=[_.copy() for _ in self.output_quantization_config],
            is_positive_quant_op=self.is_active_quant_op
        )
