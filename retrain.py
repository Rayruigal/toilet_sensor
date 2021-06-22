import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], 'utils'))
import numpy as np
import argparse
# import h5py
import math
import time
import logging
import matplotlib.pyplot as plt

import torch
torch.backends.cudnn.benchmark=True
torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
 
# from utilities import get_filename
from models import *
import config

import librosa
from pytorch_utils import move_data_to_device

from sklearn.utils.class_weight import compute_class_weight
from keras.utils import to_categorical
import pandas as pd

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

import pickle

import torch.onnx

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    # classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    # when results does not include all training classes
    classes_index_nan = list(set(np.arange(len(classes))) - set(list(np.array(y_true)) + list(np.array(y_pred))))
    for i in range(len(classes_index_nan), 0, -1):
        del classes[classes_index_nan[i - 1]]

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

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

def read_audio_data(sample_rate,device,labels):
    dir = '/dataP/ruh/Cough/AudioFiles/ForcedCoughs_cuted_windows/650ms/'
    lst_file = 'lst_all'
    text_file = open(dir + lst_file, "r")
    lines = text_file.readlines()
    # Load audio
    label = []
    data_x = []
    len_all = []
    userIDs = []
    for i in range(len(lines)):
        audio_file = lines[i].split(' ')[0]
        userIDs.append(audio_file.split('/')[0])
        label_tmp = lines[i].split(' ')[1].replace('\n','')
        label.append(label_tmp)
        (waveform, _) = librosa.core.load(dir + audio_file, sr=sample_rate, mono=True)

        waveform = waveform[None, :]    # (1, audio_length)
        waveform = move_data_to_device(waveform, device)
        data_x.append(waveform)
        len_all.append(waveform.shape[1])
        print('Read in file: ' + audio_file)
    len_max = max(len_all)
    for i in range(len(len_all)):
        if len(data_x[i][0]) < len_max:
            data_x[i] = torch.cat((data_x[i],move_data_to_device(torch.zeros(1,len_max-len(data_x[i][0])), device)), dim=1) # pad all files with zeros, so that all files share the same length

    data_y = []
    for i in range(len(label)):
        # data_y.append(np.eye(len(labels))[labels[label_gt[i]]]) # on-hot-encoder
        data_y.append(labels[label[i]])
    data_y = torch.tensor(data_y)

    data = [(data_x[i], data_y[i]) for i in range(0, len(data_y))]

    return  data, userIDs

def train(args):#, data_train):

    # Arugments & parameters
    window_size = args.window_size
    hop_size = args.hop_size
    mel_bins = args.mel_bins
    fmin = args.fmin
    fmax = args.fmax
    model_type = args.model_type
    pretrained_checkpoint_path = args.pretrained_checkpoint_path
    freeze_base = args.freeze_base
    device = 'cuda' if (args.cuda and torch.cuda.is_available()) else 'cpu'
    # userID = args.userID
    #
    # print('Working on user: ' + str(userID))

    sample_rate = args.sample_rate
    classes_num = config.classes_num
    pretrain = True if pretrained_checkpoint_path else False

    labels = {'snort': 0, 'snore_nose': 1, 'cough_explosive': 2, 'cough_soft': 3, 'wheeze': 4, 'snore_throat': 5,
              'clearing_throat': 6}

    file = open('/home/ruh/work/projects/Toilet_sensor/Code/Connected_Toilet_Sensors/Preprocessed_data/Data_6classes.pickle','rb')
    data = pickle.load(file)
    labels = list(set(data['train_gt']))

    if 'cuda' in device:
        logging.info('Using GPU.')
    else:
        logging.info('Using CPU. Set --cuda flag to use GPU.')

    # Model
    Model = eval(model_type)
    model = Model(sample_rate, window_size, hop_size, mel_bins, fmin, fmax, 
        len(labels), freeze_base)

    # Load pretrained model
    if pretrain:
        logging.info('Load pretrained model from {}'.format(pretrained_checkpoint_path))
        model.load_from_pretrain(pretrained_checkpoint_path)

    print('Load pretrained model successfully!')

    print('Transferred the model to the new task.')

    # # Parallel
    # print('GPU number: {}'.format(torch.cuda.device_count()))
    # model.base = torch.nn.DataParallel(model)

    if 'cuda' in device:
        print("using cuda")
        model.to(device)

    # Train and Test model
    criterion = torch.nn.CrossEntropyLoss()
    # optimizer_ft = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0., amsgrad=True)


    # input = data_x[0]
    # for i in range(1,len(data_x)):
    #     input = torch.cat((input,data_x[i]))
    # data_x = input

    model = train_model(model, data, model_type, criterion, optimizer, device, num_epochs=500)



        # # Print embedding
        # if 'embedding' in batch_output_dict.keys():
        #     embedding = batch_output_dict['embedding'].data.cpu().numpy()[0]
        #     print('embedding: {}'.format(embedding.shape))

    # file_result.close()

    print('Finished reading in data.')



def train_model(model, data, model_type, criterion, optimizer, device, num_epochs=50):

    # since = time.time()
    # file = open('/home/ruh/work/projects/Toilet_sensor/Code/Connected_Toilet_Sensors/Preprocessed_data/Data.pickle','rb')
    # data = pickle.load(file)

    data_train_tmp = data['train_audio']
    labels = data['labels']
    y_train = []
    data_train_x = []
    data_train_y= []
    for i in range(len(data['train_gt'])):
        label_tmp = [x for x in range(len(labels)) if data['train_gt'][i]==labels[x]]
        y_train.append(label_tmp[0])
        data_train_x.append(torch.tensor(data_train_tmp[i].reshape(1,-1)))
        data_train_y.append(torch.tensor(label_tmp[0]))
    data_train = [(data_train_x[i], data_train_y[i]) for i in range(0, len(data_train_y))]


    data_test_tmp = data['test_audio']
    y_test = []
    data_test_x = []
    data_test_y= []
    for i in range(len(data['test_gt'])):
        label_tmp = [x for x in range(len(labels)) if data['test_gt'][i]==labels[x]]
        y_test.append(label_tmp[0])
        data_test_x.append(torch.tensor(data_test_tmp[i].reshape(1,-1)))
        data_test_y.append(torch.tensor(label_tmp[0]))
    data_test = [(data_test_x[i], data_test_y[i]) for i in range(0, len(data_test_y))]

    # balanced sample on unbalanced training dataset
    class_sample_count = np.array([len(np.where(np.array(y_train) == t)[0]) for t in range(len(labels))])
    weight = sum(class_sample_count) / class_sample_count
    samples_weight = np.array([weight[t] for t in y_train])
    samples_weight = torch.from_numpy(samples_weight)
    samples_weigth = samples_weight.double()
    sampler = torch.utils.data.WeightedRandomSampler(samples_weight, int(np.max(class_sample_count))* len(labels)) # sample "number of labels" times the number of samples of the largest number category, so that each sample have a chance to be sampled
    trainset = torch.utils.data.DataLoader(data_train, batch_size=32, sampler=sampler)  # , shuffle = True)

    # trainset = torch.utils.data.DataLoader(data_train, batch_size=32, shuffle = True)
    history_loss = []
    history_accuracy_train = []
    history_accuracy_test = []
    for epoch in range(num_epochs):
        predicted_class = []
        real_class = []
        model.train()
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        loss_epoch = 0
        num_batches = 0
        for data_batch in trainset:
            num_batches += 1
            X, y = data_batch
            optimizer.zero_grad()
            output = model(X.view(-1,X.shape[-1]))
            y = move_data_to_device(y, device)
            loss = criterion(output['clipwise_output'], y)
            loss.backward()
            optimizer.step()
            loss_epoch += loss.data
        loss_epoch = loss_epoch / num_batches
        print(epoch, loss_epoch)
        history_loss.append(loss_epoch)

        correct_train = 0
        total_train = 0
        model.eval()
        with torch.no_grad():
            for data_batch in trainset:
                X, y = data_batch
                output = model(X.view(-1,X.shape[-1]))
                for idx, i  in enumerate(output['clipwise_output'].data.cpu()):
                    if torch.argmax(i) == y[idx]:
                        correct_train += 1
                    total_train += 1
        history_accuracy_train_tmp = round(correct_train/total_train, 3)
        history_accuracy_train.append(history_accuracy_train_tmp)
        # print("Accuracy train: " + str(history_accuracy_train_tmp))

        # inference and write results to text file
        correct_test = 0
        total_test = 0
        testset = torch.utils.data.DataLoader(data_test, batch_size=1, shuffle=False)
        model.eval()
        with torch.no_grad():
            for data_batch in testset:
                X, y = data_batch
                output = model(X.view(-1,X.shape[-1]))
                for idx, i  in enumerate(output['clipwise_output'].data.cpu()):
                    if torch.argmax(i) == y[idx]:
                        correct_test += 1
                    total_test += 1
                    predicted_class.append(torch.argmax(i))
                    real_class.append(y[idx])
                    # file_result.write(y[idx], )
        history_accuracy_test_tmp = round(correct_test / total_test, 3)
        history_accuracy_test.append(history_accuracy_test_tmp)
        # print("Accuracy test of user " + str(user) + ' is: ' + str(history_accuracy_test_tmp))

    f = open('Results/' + model_type + '/Prediction_results_retrain_22050_22050.pickle', 'wb')
    pickle.dump([real_class, predicted_class], f)
    f.close()

    # save model to ONNX
    model.eval()
    Input = torch.randn(1, 22050, requires_grad=True)
    Output = model(Input)
    torch.onnx.export(model, Input, "save_tmp_onnx", export_params=True,
                      opset_version=10,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable lenght axes
                                'output' : {0 : 'batch_size'}})

    # load saved onnx model to test
    import onnx
    onnx_model = onnx.load("save_tmp_onnx")
    onnx.checker.check_model(onnx_model)
    # run saved model to test
    import onnxruntime
    ort_session = onnxruntime.InferenceSession("save_tmp_onnx")
    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(Input)}
    ort_outs = ort_session.run(None, ort_inputs)
    # compare ONNX Runtime and PyTorch results
    np.testing.assert_allclose(to_numpy(Output), ort_outs[0], rtol=1e-03, atol=1e-05)
    print("Exported model has been tested with ONNXRuntime, and the result looks good!")

    # save model
    model_path = 'Models/PANNs/Re-trained/' + model_type + '.pth'
    torch.save(model.state_dict(), model_path)

    plt.plot(history_loss)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Learning history')
    plt.grid()
    plt.savefig('Results/LH_loss_retrain_ETHwindow_' + model_type + '_' + str(user) + '.png')
    plt.close()

    plt.plot(history_accuracy_train, label = 'Train accuracy')
    plt.plot(history_accuracy_test, label = 'Test accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Learning history')
    plt.grid()
    plt.legend()
    plt.savefig('../results/LH_accuracy_retrain_ETHwindow_' + model_type + '_' + str(user) + '.png')
    plt.close()

    accuracy = accuracy_score(real_class, predicted_class)
    F1 = f1_score(real_class, predicted_class, average='weighted')
    plot_confusion_matrix(real_class, predicted_class, classes=list(labels), title=model_type)
    plt.savefig('../results/retrain_ETHwindow_' + model_type + '_' + str(user) + '.png')
    plot_confusion_matrix(real_class, predicted_class, classes=list(labels), title=model_type, normalize=True)
    plt.savefig('../results/retrain_ETHwindow_' + model_type + '_' + str(user) + '_norm.png')

    print('Accuracy test of user ' + str(user) + ' is: ' + str(accuracy))
    print('F1:' + str(F1))
    # print('Accuracy per-user: ' + accuracy_per_user)



    #     for i in range(len(data_test)):
    #         X, y = data_test
    #         print('processing the ' + str(i) + 'th audio clip... \n')
    #         # Forward
    #         with torch.no_grad():
    #             model.eval()
    #             batch_output_dict = model(X[i], None)
    #
    #         clipwise_output = batch_output_dict['clipwise_output'].data.cpu().numpy()[0]
    #         """(classes_num,)"""
    #
    #         sorted_indexes = np.argsort(clipwise_output)[:,-1]
    #         predict.append()
    #
    #         # Print audio tagging top probabilities
    #         for k in range(len(y)):
    #             # print('{}: {:.3f}'.format(np.array(labels)[sorted_indexes[k]],
    #             #                           clipwise_output[sorted_indexes[k]]))
    #             # if label_gt[i] == 'snore':
    #             #     file_result.write(label_gt[i] + ' Inf ' + np.array(labels)[sorted_indexes[0]] + '\n')
    #             #     break
    #             if label_gt[i] in labels[sorted_indexes[k]].lower():
    #                 file_result.write(label_gt[i] + ' ' + str(k)+ ' ' + np.array(labels)[sorted_indexes[0]] + '\n')
    #                 predict.append(np.array(labels)[sorted_indexes[0]])
    #                 if k == 0:
    #                     accuracy += 1
    #                 break
    # accuracy = accuracy/len(label_gt)

        #     train_loss = 0
        #
        #     with torch.set_grad_enabled(True):
        #         outputs = model(X, None)
        #         loss = criterion(outputs['clipwise_output'], y)
        #
        #     loss.backward()
        #     optimizer.step()
        #
        #     train_loss += loss.item() * X.size(0)
        #
        # print('{} Loss: {:.4f}'.format('train', train_loss)) #  / dataset_sizes['train']))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    return model



if __name__ == '__main__':



    parser = argparse.ArgumentParser(description='Retrain whole network. ')
    subparsers = parser.add_subparsers(dest='mode')

    # parser.add_argument(
    #     "--model_type",
    #     type=str,
    #     default='Transfer_MobileNetV1',
    #     metavar="N",
    #     help="model selection",
    # )#model_type="Transfer_MobileNetV1" # "Transfer_Cnn14"
    #
    # parser.add_argument(
    #     "--userID",
    #     type=int,
    #     default='11',
    #     metavar="N",
    #     help="userID",
    # )

    # Train
    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('--window_size', type=int, required=True)
    parser_train.add_argument('--hop_size', type=int, required=True)
    parser_train.add_argument('--mel_bins', type=int, required=True)
    parser_train.add_argument('--fmin', type=int, required=True)
    parser_train.add_argument('--fmax', type=int, required=True)
    parser_train.add_argument('--model_type', type=str, required=True)
    parser_train.add_argument('--pretrained_checkpoint_path', type=str)
    parser_train.add_argument('--freeze_base', action='store_true', default=False)
    parser_train.add_argument('--cuda', action='store_true', default=False)

    # Parse arguments
    args = parser.parse_args()
    # args.filename = get_filename(__file__)

    args.mode = 'train'

    args.model_type = 'Transfer_MobileNetV1'

    if args.mode == 'train':

        args.window_size=1024
        args.sample_rate = 22050
        args.hop_size=320
        args.mel_bins=64
        args.fmin=50
        args.fmax=14000

        if args.model_type == 'Transfer_MobileNetV1':
            args.pretrained_checkpoint_path = "Models/PANNs/Pre-trained/MobileNetV1_mAP=0.389.pth"  # Cnn14_mAP=0.431.pth"
        if args.model_type == 'Transfer_Cnn14':
            args.pretrained_checkpoint_path = "Models/PANNs/Pre-trained/Cnn14_mAP=0.431.pth"  # Cnn14_mAP=0.431.pth"
        args.freeze_base = False
        # args.audio_path="/home/ruh/work/projects/Covid19/Codes/AudioAnalytics/AudioFiles/ForcedCoughs/11/20170629_103440.wav"
        args.cuda=False
        args.learning_rate = 5e-4

        train(args)



    else:
        raise Exception('Error argument!')