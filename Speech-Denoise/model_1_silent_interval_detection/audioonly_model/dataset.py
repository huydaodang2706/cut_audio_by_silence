import functools
import glob
import json
import math
import os
import random
import time
from itertools import groupby
from tqdm import tqdm

# python3 -m pip install --user imageio
import imageio
import librosa
import numpy as np
from pathlib import Path
import pylab
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import WeightedRandomSampler
from joblib import Parallel, delayed

from common import EXPERIMENT_NAME, PROJECT_ROOT, PHASE_TESTING, PHASE_TRAINING, PHASE_PREDICTION
from tools import *
from utils import ensure_dir, get_parent_dir, get_basename_no_ext
from visualization import draw_waveform, draw_spectrum
import cv2
import pandas as pd


DATA_ROOT = os.path.join(PROJECT_ROOT, "../", "dataset")
DEBUG_OUT = os.path.join(DATA_ROOT, "debug_dataset_output", EXPERIMENT_NAME)
NUM_DATA = 6000    # 100000
SILENT_CONSECUTIVE_FRAMES = 1
CLIP_FRAMES = 60
RANDOM_SEED = 10
PRED_RANDOM_SEED = 100
JSON_PARTIAL_NAME = '_TEDx1.json'

DATA_REQUIRED_SR = 16000
DATA_REQUIRED_FPS = 30.0
DATA_MAX_AUDIO_SAMPLES = int(math.floor(CLIP_FRAMES / DATA_REQUIRED_FPS * DATA_REQUIRED_SR))
# DATA_MAX_AUDIO_SAMPLES = 200
NOISE_MAX_LENGTH_IN_SECOND = DATA_MAX_AUDIO_SAMPLES / DATA_REQUIRED_SR * 1.5  # times 1.5 to be safe

SNRS = [-10, -7, -3, 0, 3, 7, 10]

NOISE_SRC_ROOT_TRAIN = os.path.join(DATA_ROOT, "noise_data_DEMAND", "train_noise")
NOISE_SRC_ROOT_TEST = os.path.join(DATA_ROOT, "noise_data_DEMAND", "test_noise")

AUDIOSET_NOISE_SRC_TRAIN = os.path.join(DATA_ROOT, "audioset_noises_balanced_train")
AUDIOSET_NOISE_SRC_EVAL = os.path.join(DATA_ROOT, "audioset_noises_balanced_eval")


# Functions
##############################################################################
def get_dataloader(phase, max_audio_length=None, batch_size=4, num_workers=4, csv_file=None):
    print('Mode:', phase)

    num_data = NUM_DATA if phase == PHASE_TRAINING else NUM_DATA // 10
    is_shuffle = phase == PHASE_TRAINING

    # dataset
    dataset = AudioVisualAVSpeechMultipleVideoDataset(phase, num_data, CLIP_FRAMES, SILENT_CONSECUTIVE_FRAMES,\
        max_audio_length=max_audio_length, csv_file=csv_file)

    # dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=is_shuffle,\
        num_workers=num_workers, pin_memory=True, worker_init_fn=np.random.seed())

    return dataloader


# datasets
##############################################################################
class AudioVisualAVSpeechMultipleVideoDataset(Dataset):
    def __init__(self, phase, num_samples, clip_frames, consecutive_frames, max_audio_length, csv_file=None, n_fft=510, hop_length=160, win_length=400, X_col='audio_path', y_col='label'):
        print('========== DATASET CONSTRUCTION ==========')
        print('Initializing dataset...')
        super(AudioVisualAVSpeechMultipleVideoDataset, self).__init__()
        self.phase = phase

        self.clip_frames = clip_frames
        self.consecutive_frames = consecutive_frames
        self.num_samples = num_samples

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.max_audio_length = max_audio_length 

        print('Loading data...')
        
        self.df = pd.read_csv(csv_file, delimiter=';')
        self.X_col = X_col
        self.y_col = y_col
       
        
        print('Generating data items...')
        print('========== SUMMARY ==========')
        print('Mode:', phase)
        print('Num samples:', self.num_samples)
        print('Data frames:', self.clip_frames)
        print('Consecutive frames:', self.consecutive_frames)
        print('Max noise length in seconds:', NOISE_MAX_LENGTH_IN_SECOND)
        print('Max audio samples per data:', DATA_MAX_AUDIO_SAMPLES)
        print('n_fft: {}'.format(self.n_fft))
        print('hop_length: {}'.format(self.hop_length))
        print('win_length: {}'.format(self.win_length))

    def __len__(self):
        return len(self.df)

    def __get_input(self, path, max_audio_length):
        snd, sr = librosa.load(path, sr=DATA_REQUIRED_SR)
        if max_audio_length is not None:
            snd = librosa.util.fix_length(snd, max_audio_length*sr)
        snd = audio_normalize(snd)

        audio = snd
        mixed_sig_stft = fast_stft(audio, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length)
        mixed_sig_stft = torch.tensor(mixed_sig_stft.transpose((2, 0, 1)), dtype=torch.float32)
        
        return mixed_sig_stft

    def __get_output(self, label_string, audio_length=None):
        raw_label = json.loads(label_string)
        raw_label = [0 if x==2 else x for x in raw_label]
        # print("This is raw label:",raw_label)
        # if num_classes == 3:
        #     for i, x in enumerate(raw_label):
        #         if x == 0:
        #         # intermidiate silence
        #             class_label = numpy.array([[1, 0, 0]])
        #         elif x == 1:
        #         # speech 
        #             class_label = numpy.array([[0, 1, 0]])
        #         elif x == 2:
        #         # final silence 
        #             class_label = numpy.array([[0, 0, 1]])
            
        #     label = None
        #     if i == 0:
        #         label = class_label
        #     else:
        #         label = numpy.append(label, class_label, axis=0)

        # else: 
        # binary classification
        # label = json.loads(f['label'])
        # if audio_length is not None:
        #     for i in range(audio_length - len(raw_label)):
        #         raw_label.append(0)
        #         # Append silence at the end
        if audio_length is not None:
            if audio_length > len(raw_label):
                for i in range(audio_length - len(raw_label)):
                    raw_label.append(0)
            # Append silence at the end
            else:
                for i in range(len(raw_label)- audio_length):
                    raw_label.pop()

        # if audio_length < 600:
        #     print("Length of label:  th1 ", len(raw_label))
        # if len(raw_label) < 600:
        #     print("Length of label:  th2 ", len(raw_label))
        # print("Length of label ::::",len(raw_label))
        # label = numpy.array(raw_label)
        # label = numpy.expand_dims(label, axis=0)
        label = torch.tensor(raw_label, dtype=torch.float32)
        return label


    def __getitem__(self, index):
        # Item is the object to iterate
        # item[0]: video clip index
        # item[1]: data first bit's index in video clip
        # item[2]: data bit stream
        # item[3]: audio_path
        # item[4]: framerate
        item = self.df.iloc[index]
        audio_path = item[self.X_col]
        label_string = item[self.y_col]

        audio = self.__get_input(audio_path, max_audio_length=self.max_audio_length)
        label = self.__get_output(label_string, audio.shape[-1])
        
        #Chi cat lay audio co do dai 6s, neu thieu thi padding them 
        return {
            # "frames": frames,
            "label": label,
            "audio": audio,
        }




def test():
    print('In test')
    data_csv_path='/home3/huydd/cut_by_mean/GLDNN_EOU_detection/data/train_youtube_huy_gan.csv'
    # dataloader = get_dataloader(PHASE_TRAINING, batch_size=1, num_workers=20, dataset_json=data_json_path)
    dataloader = get_dataloader(PHASE_TESTING, batch_size=10, num_workers=1,csv_file=data_csv_path)
    # dataloader = get_dataloader(PHASE_PREDICTION, batch_size=8, num_workers=0)
    for i, data in enumerate(dataloader):
        print('================================================================')
        print('batch index:', i)
        # print('data[\'frames\'].size():', data['frames'].size())
        # print('data[\'bitstream\']',data['audio'])
        print('data[\'label\'].size():', data['label'].size())
        # print('data[\'label\']:', data['label'])
        print('data[\'audio\'].size():', data['audio'].size())
        # print('min-max: ({}, {})'.format(torch.min(data['frames']).numpy().squeeze(),\
        #     torch.max(data['frames']).numpy().squeeze()))
        print('================================================================')

        # fig = plt.figure()
        # fig.subplots_adjust(left=0.2, right=0.95, hspace=0.3)
        # ax1 = fig.add_subplot(2, 1, 1)
        # librosa.display.waveplot(data['audio'].numpy().squeeze(), sr=DATA_REQUIRED_SR, ax=ax1)
        # ax2 = fig.add_subplot(2, 1, 2)
        # s_spec = librosa.stft(data['audio'].numpy().squeeze(), n_fft=510, hop_length=158, win_length=400)
        # librosa.display.specshow(librosa.amplitude_to_db(np.abs(s_spec), ref=np.max), y_axis='log', x_axis='time', sr=DATA_REQUIRED_SR, ax=ax2)
        # plt.savefig('audio.jpg')
        # plt.close(fig)
        # librosa.output.write_wav('audio.wav', data['audio'].numpy().squeeze(), DATA_REQUIRED_SR)
        # exit()

        # if i >= 0:
        #     exit()


if __name__ == "__main__":
    test()
