# Connected_Toilet_Sensors

'metadata': the 527 class names trained with the PANNs model

'Models': trained models

'utils': functions called by 'inference_PANNs.py'

## Inference codes
### 'PyTorch_ONNX_inference.py': 

Summary: Sound classification inference code, pre-trained (PANNs) MobileNetV1 model in ONNX format

Calls: metadata/, Models/


### 'inference_PANNs.py': 

Summary: Sound classification inference code, pre-trained (PANNs) MobileNetV1 model

Calls: metadata/, Models/, utils/

### 'inference_PANNs_retrained.py':
Summary: Sound classification inference code, re-trained MobileNetV1 model

Calls: Models/, utils/
