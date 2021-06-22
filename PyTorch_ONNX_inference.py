import onnxruntime as ort
import numpy as np

# the following libraries are not required at inference time, they are used for reading the test audio file
import librosa

# the following libraries are not required at inference time, they are used for read in the 527 sound classes used in the pre-trained model
import csv

required_length = 1 # required audio clip length (second)

################ 1. Load test audio clip ###############################################
sample_rate = 32000

audio_path = '../Dataset/Audio_Data/PuneBlock1Seat3_ev01_20200818160112_sw4_l10.wav'

(waveform, _) = librosa.core.load(audio_path, sr=sample_rate, mono=True)
# make test video to meet the required length
required_samples = required_length*sample_rate
waveform = waveform.repeat(int(np.ceil(required_samples/len(waveform))))[:required_samples] #  duplicate waveform and cut off extra length

# device = torch.device('cpu')
waveform = waveform[None, :]  # (1, audio_length)
# waveform = move_data_to_device(waveform, device)



################# 2. load 527 labels ###############################
with open('metadata/class_labels_indices.csv', 'r') as f:
    reader = csv.reader(f, delimiter=',')
    lines = list(reader)

labels = []
ids = []    # Each label has a unique id such as "/m/068hy"
for i1 in range(1, len(lines)):
    id = lines[i1][1]
    label = lines[i1][2]
    ids.append(id)
    labels.append(label)


################ 3. load pre-trained model and inference #######################
model_name = 'Models/PANNs/ONNX/MobileNetV1_1s.onnx'

# start an onnx inference session
ort_session = ort.InferenceSession(model_name)

input_name = ort_session.get_inputs()[0].name
# print("Input name  :", input_name)
# input_shape = ort_session.get_inputs()[0].shape
# print("Input shape :", input_shape)
# input_type = ort_session.get_inputs()[0].type
# print("Input type  :", input_type)
output_name = ort_session.get_outputs()[0].name
# print("Output name  :", output_name)
# output_shape = ort_session.get_outputs()[0].shape
# print("Output shape :", output_shape)
# output_type = ort_session.get_outputs()[0].type
# print("Output type  :", output_type)

# inference
outputs = ort_session.run([output_name],  {input_name: waveform.astype(np.float32)})

##################### 4. extract output #######################
sorted_indexes = np.argsort(outputs[0][0])[::-1]

# Print audio tagging top probabilities
for k in range(10):
    print('{}: {:.3f}'.format(np.array(labels)[sorted_indexes[k]], outputs[0][0][sorted_indexes[k]]))

print('The tested audio clip is classified as of sound: ' + labels[sorted_indexes[0]])