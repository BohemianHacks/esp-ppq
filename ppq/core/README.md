## PPQ.Core (PPQ core definition)
You are browsing the core data structure definition of PPQ. The files in this directory describe the underlying logic of the PPQ software:

1. ppq.core.common: ppq predefined constants. Users can modify this file to configure the corresponding functions of the software.

2. ppq.core.config: ppq basic definition, including version number, software name, etc.

3. ppq.core.data: ppq basic data type, including data type conversion of pytorch.Tensor, numpy.ndarray.

4. ppq.core.defs: ppq metatype and global tool function definition.

5. ppq.core.ffi: ppq application programming interface, including the logic of calling c/c++, cuda code.

6. ppq.core.quant: ppq core quantitative data structure definition [very important].

7. ppq.core.storage: Definitions of ppq persistence operations.

## TensorQuantizationConfig (Tensor Quantization Control Structure)
PPQ uses the quantization control structure to describe quantization behavior, which is defined in ppq.core.quant. As of PPQ 0.6.6, the structure consists of 15 different attributes. We will introduce you to the design concept of this core data structure.

### 1. QuantizationPolicy Quantization Strategy
In TensorQuantizationConfig, the first thing to be mentioned is TQC.policy, which is a QuantizationPolicy object.
The policy attribute is used to describe the quantization rules. A complete quantization policy is composed of multiple quantization properties (QuantizationProperty). In PPQ, we currently support 8 different quantization properties. You can use the following attributes to combine and form custom quantization rules:

1. PER_TENSOR: Quantization is completed in units of Tensor. Each Tensor uses a scale and offset information.

2. PER_CHANNEL: Quantization is completed in units of Channel. Each Channel uses a scale and offset information.

3. LINEAR: Linear quantization. Common INT8 and INT16 are both linear quantization. There is no exponent bit in the linear quantization representation.

4. FLOATING: Floating point quantization, including FP8 E4M3, FP8 E5M2, FP16, BF16 and other formats. In floating point quantization, the data consists of two parts: base and exponent.

5. SYMMETRICAL: Symmetric quantization, offset is not enabled in quantization calculation.

6. ASYMMETRICAL: Asymmetric quantization, offset is enabled in quantization calculation to complete quantization offset.

7. POWER_OF_2: Limit the scale value to an integer power of 2. This quantization behavior is more common in end-side and floating-point quantization.

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
quantizer = PFL.Quantizer(platform=TargetPlatform.TRT_FP8, graph=graph) # Get TRT_FP8
