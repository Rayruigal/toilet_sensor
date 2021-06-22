import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], 'utils'))
import numpy as np
import argparse
import librosa

import torch

from models import *
from pytorch_utils import move_data_to_device


class Transfer_Cnn14(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin,
                 fmax, classes_num, freeze_base):
        """Classifier for a new task using pretrained Cnn14 as a sub module.
        """
        super(Transfer_Cnn14, self).__init__()
        audioset_classes_num = 527

        self.base = Cnn14(sample_rate, window_size, hop_size, mel_bins, fmin,
                          fmax, audioset_classes_num)

        # Transfer to another task layer
        self.fc_transfer = nn.Linear(2048, classes_num, bias=True)

        if freeze_base:
            # Freeze AudioSet pretrained layers
            for param in self.base.parameters():
                param.requires_grad = False

        self.init_weights()

    def init_weights(self):
        init_layer(self.fc_transfer)

    def load_from_pretrain(self, pretrained_checkpoint_path):
        checkpoint = torch.load(pretrained_checkpoint_path, map_location='cpu')
        self.base.load_state_dict(checkpoint['model'])

    def forward(self, input, mixup_lambda=None):
        """Input: (batch_size, data_length)
        """
        output_dict = self.base(input, mixup_lambda)
        embedding = output_dict['embedding']

        clipwise_output = torch.log_softmax(self.fc_transfer(embedding), dim=-1)
        output_dict['clipwise_output'] = clipwise_output

        return output_dict


class Transfer_MobileNetV1(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin,
                 fmax, classes_num, freeze_base):
        """Classifier for a new task using pretrained Cnn14 as a sub module.
        """
        super(Transfer_MobileNetV1, self).__init__()
        audioset_classes_num = 527

        self.base = MobileNetV1(sample_rate, window_size, hop_size, mel_bins, fmin,
                                fmax, audioset_classes_num)

        # Transfer to another task layer
        self.fc_transfer = nn.Linear(1024, classes_num, bias=True)

        if freeze_base:
            # Freeze AudioSet pretrained layers
            for param in self.base.parameters():
                param.requires_grad = False

        self.init_weights()

    def init_weights(self):
        init_layer(self.fc_transfer)

    def load_from_pretrain(self, pretrained_checkpoint_path):
        checkpoint = torch.load(pretrained_checkpoint_path, map_location='cpu')
        self.base.load_state_dict(checkpoint['model'])

    def forward(self, input, mixup_lambda=None):
        """Input: (batch_size, data_length)
        """
        output_dict = self.base(input, mixup_lambda)
        embedding = output_dict['embedding']

        clipwise_output = torch.log_softmax(self.fc_transfer(embedding), dim=-1)
        output_dict['clipwise_output'] = clipwise_output

        return output_dict

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

    sample_rate = args.sample_rate
    classes_num = 6
    labels = ['diar', 'flush', 'pee', 'solid', 'others', 'silence'] # ['pee', 'solid', 'flush', 'diar'] # ['diar', 'flush', 'pee', 'solid']

    # Model
    Model = eval(model_type)
    model = Model(sample_rate=sample_rate, window_size=window_size, 
        hop_size=hop_size, mel_bins=mel_bins, fmin=fmin, fmax=fmax, 
        classes_num=classes_num, freeze_base = 0)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)

    # Load audio
    (waveform, _) = librosa.core.load(audio_path, sr=sample_rate, mono=True)

    waveform = waveform[None, :]    # (1, audio_length)
    waveform = move_data_to_device(waveform, device)

    window_len = args.window_len
    step_len = int(window_len/2)
    for index_clip in range(waveform.shape[1]//step_len):
        if index_clip*step_len+window_len < waveform.shape[1]:
            waveform_tmp = waveform[:,index_clip*step_len:index_clip*step_len+window_len]
            # Forward
            with torch.no_grad():
                model.eval()
                batch_output_dict = model(waveform_tmp, None)

            clipwise_output = batch_output_dict['clipwise_output'].data.cpu().numpy()[0]
            """(classes_num,)"""

            sorted_indexes = np.argsort(clipwise_output)[::-1]

            print('The [' + str((index_clip*step_len)/sample_rate) + ' : ' + str((index_clip*step_len+window_len)/sample_rate) + '] th seconds are labeled as of sound: ' + labels[sorted_indexes[0]] + '\n')

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
    args.sample_rate = 22050 # config.sample_rate #
    args.window_len = 1*args.sample_rate # second(s) #
    args.hop_size=320
    args.mel_bins=64
    args.fmin=50
    args.fmax=14000
    # args.model_type= "MobileNetV1" # "Cnn14" #
    args.model_type = 'Transfer_MobileNetV1'
    # args.checkpoint_path="Models/PANNs/Pre-trained/MobileNetV1_mAP=0.389.pth" # Cnn14_mAP=0.431.pth" #
    args.checkpoint_path = "Models/PANNs/Re-trained/Transfer_MobileNetV1_6classes.pth"
    args.audio_path="../../Dataset/Audio_Data/PuneBlock1Seat3_ev01_20200818160207_sw4_l18.wav" #/PuneBlock1Seat3_ev01_20200818160112_sw4_l10.wav" #/PuneBlock1Seat3_ev01_20200819225155_sw2_l45.wav" # /PuneBlock1Seat3_ev01_20200819161815_sw2_l46.wav"#/PuneBlock1Seat3_ev01_20200819111215_sw2_l37.wav" #
    args.cuda=False

    audio_tagging(args)
