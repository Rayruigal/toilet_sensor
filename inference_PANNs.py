import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], 'utils'))
import numpy as np
import argparse
import librosa

import torch

from models import *
from pytorch_utils import move_data_to_device
import config

def audio_tagging(args):
    """Inference audio tagging result of an audio clip.
    """

    # Arugments & parameters
    window_size = args.window_size
    hop_size = args.hop_size
    mel_bins = args.mel_bins
    fmin = args.fmin
    fmax = args.fmax
    model_type = args.model_type
    checkpoint_path = args.checkpoint_path
    audio_path = args.audio_path
    device = torch.device('cuda') if args.cuda and torch.cuda.is_available() else torch.device('cpu')

    sample_rate = config.sample_rate
    classes_num = config.classes_num
    labels = config.labels

    # Model
    Model = eval(model_type)
    model = Model(sample_rate=sample_rate, window_size=window_size, 
        hop_size=hop_size, mel_bins=mel_bins, fmin=fmin, fmax=fmax, 
        classes_num=classes_num)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])

    # # Parallel
    # print('GPU number: {}'.format(torch.cuda.device_count()))
    # model = torch.nn.DataParallel(model)

    # if 'cuda' in str(device):
    #     model.to(device)
    
    # Load audio
    (waveform, _) = librosa.core.load(audio_path, sr=sample_rate, mono=True)

    waveform = waveform[None, :]    # (1, audio_length)
    waveform = move_data_to_device(waveform, device)

    # Forward
    with torch.no_grad():
        model.eval()
        batch_output_dict = model(waveform, None)

    clipwise_output = batch_output_dict['clipwise_output'].data.cpu().numpy()[0]
    """(classes_num,)"""

    sorted_indexes = np.argsort(clipwise_output)[::-1]

    print('The audio clip is labeled as of sound: ' + labels[sorted_indexes[0]] + '\n')

    # Print audio tagging top probabilities
    print('The top 10 most likely sound categories, with the according probability scores, are:')
    for k in range(10):
        print('{}: {:.3f}'.format(np.array(labels)[sorted_indexes[k]], 
            clipwise_output[sorted_indexes[k]]))

    # # Print embedding
    # if 'embedding' in batch_output_dict.keys():
    #     embedding = batch_output_dict['embedding'].data.cpu().numpy()[0]
    #     print('embedding: {}'.format(embedding.shape))

    return clipwise_output, labels


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Example of parser. ')
    subparsers = parser.add_subparsers(dest='mode')

    parser_at = subparsers.add_parser('audio_tagging')
    parser_at.add_argument('--window_size', type=int, default=1024)
    parser_at.add_argument('--hop_size', type=int, default=320)
    parser_at.add_argument('--mel_bins', type=int, default=64)
    parser_at.add_argument('--fmin', type=int, default=50)
    parser_at.add_argument('--fmax', type=int, default=14000)
    parser_at.add_argument('--model_type', type=str, required=True)
    parser_at.add_argument('--checkpoint_path', type=str, required=True)
    parser_at.add_argument('--audio_path', type=str, required=True)
    parser_at.add_argument('--cuda', action='store_true', default=False)

    args = parser.parse_args()

    args.mode = 'audio_tagging'


    args.window_size=1024
    args.hop_size=320
    args.mel_bins=64
    args.fmin=50
    args.fmax=14000
    args.model_type= "MobileNetV1" # "Cnn14" #
    args.checkpoint_path="Models/PANNs/MobileNetV1_mAP=0.389.pth" # Cnn14_mAP=0.431.pth" #
    args.audio_path="../Dataset/Audio_Data/PuneBlock1Seat3_ev01_20200818160112_sw4_l10.wav"
    args.cuda=False

    audio_tagging(args)
