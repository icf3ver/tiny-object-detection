# Models

FRC_model.tflite : Not compiled for edgetpu

FRC_model_edgetpu.tflite : Mostly quantized 224x224 model 
compiled to run on the edgetpu. (No postprocessing)

# Brief Description

Backbone: mobilenetv2

Current speed on edgetpu: ~50ms per 224 x 224 frame (Subject to change)

speed without edgetpu: ~9000ms per 224 x 224 frame

Models compiled with min runtime v13 edgetpu_compiler v16.0.384591198

# Edgetpu Compiled Model Difficulties:

3 Operators remain unmapped and currently run on the CPU. 
Once fixed the model should run even faster.

# Current Compile

```
Input model: FRC_model.tflite
Input size: 8.05MiB
Output model: FRC_model_edgetpu.tflite
Output size: 14.17MiB
On-chip memory used for caching model parameters: 7.14MiB
On-chip memory remaining for caching model parameters: 1.75KiB
Off-chip memory used for streaming uncached model parameters: 6.63MiB
Number of Edge TPU subgraphs: 1
Total number of operations: 141
Operation log: FRC_model_edgetpu.log

Model successfully compiled but not all operations are supported by the Edge TPU. A percentage of the model will instead run on the CPU, which is slower. If possible, consider updating your model to use only operations supported by the Edge TPU. For details, visit g.co/coral/model-reqs.
Number of operations that will run on Edge TPU: 138
Number of operations that will run on CPU: 3

Operator                       Count      Status

RESHAPE                        1          Operation is otherwise supported, but not mapped due to some unspecified limitation
CONCATENATION                  1          Operation is otherwise supported, but not mapped due to some unspecified limitation
QUANTIZE                       1          Operation is otherwise supported, but not mapped due to some unspecified limitation
... All other operators were mapped successfully. See FRC_model_edgetpu.log.

```
