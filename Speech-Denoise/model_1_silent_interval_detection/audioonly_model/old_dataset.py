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
def get_dataloader(phase, batch_size=4, num_workers=4, snr_idx=None, dataset_json=None, clean_audio=False):
    print('Mode:', phase)

    num_data = NUM_DATA if phase == PHASE_TRAINING else NUM_DATA // 10
    is_shuffle = phase == PHASE_TRAINING

    # dataset
    dataset = AudioVisualAVSpeechMultipleVideoDataset(phase, num_data, CLIP_FRAMES, SILENT_CONSECUTIVE_FRAMES,\
        snr_idx, dataset_json=dataset_json, clean_audio=clean_audio)

    # dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=is_shuffle,\
        num_workers=num_workers, pin_memory=True, worker_init_fn=np.random.seed())

    return dataloader


# datasets
##############################################################################
class AudioVisualAVSpeechMultipleVideoDataset(Dataset):
    def __init__(self, phase, num_samples, clip_frames, consecutive_frames, snr_idx, dataset_json=None, clean_audio=False, n_fft=510, hop_length=160, win_length=400):
        print('========== DATASET CONSTRUCTION ==========')
        print('Initializing dataset...')
        super(AudioVisualAVSpeechMultipleVideoDataset, self).__init__()
        self.phase = phase
        if dataset_json is None:
            self.dataset_json = os.path.join(DATA_ROOT, phase + JSON_PARTIAL_NAME)
            # self.data_json = os.path.join(DATA_ROOT, phase + JSON_PARTIAL_NAME).split('.json')[0] + '_from_training.json'
        else:
            self.dataset_json = dataset_json
        self.clip_frames = clip_frames
        self.consecutive_frames = consecutive_frames
        self.clean_audio = clean_audio
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length

        print('Loading data...')
        with open(self.dataset_json, 'r') as fp:
            info = json.load(fp)
            # print("Infor:",info)
        self.dataset_path = info['dataset_path']
        self.num_files = info['num_videos']
        self.files = info['files']
        # print("Files:", info["files"])
        self.snr_idx = snr_idx
        # Set clean audio = False for test
        # print("Files:",self.files)
       
        
        print('Generating data items...')
        # list of tuples (video_index, start_frame, bit_stream, center_label)
        # [(0, 450, '000011111111111', 1.0), ..., (79, 18349, '000111100000000', 0.0)]
        # self.items = create_sample_list_from_indices(self.files, num_samples, self.clip_frames)
        self.items = create_sample_list_from_indices(self.files, \
            clip_frames=self.clip_frames, \
            silent_consecutive_frames=self.consecutive_frames, \
            random_seed=RANDOM_SEED,audio_length=601)
        if phase == PHASE_TESTING:
            self.items = create_sample_list_from_indices(self.files, \
                # num_samples=len(self.items)//10, \
                clip_frames=self.clip_frames, \
                silent_consecutive_frames=self.consecutive_frames, \
                random_seed=RANDOM_SEED,audio_length=601)
        elif phase == PHASE_PREDICTION:
            self.items = create_sample_list_from_indices(self.files, clip_frames=self.clip_frames,\
                silent_consecutive_frames=self.consecutive_frames, random_seed=RANDOM_SEED, pred=True)
        # print("Self.items:",self.items)
        self.num_samples = len(self.items)
        # print("Number of samples: items length:",self.num_samples)
        # print("Items:",self.items)
        # self.num_samples = num_samples

       
        print('========== SUMMARY ==========')
        print('Mode:', phase)
        print('Dataset JSON:', self.dataset_json)
        print('Dataset path:', self.dataset_path)
        print('Num samples:', self.num_samples)
        print('Data frames:', self.clip_frames)
        print('Consecutive frames:', self.consecutive_frames)
        print('Max noise length in seconds:', NOISE_MAX_LENGTH_IN_SECOND)
        print('Max audio samples per data:', DATA_MAX_AUDIO_SAMPLES)
        print('n_fft: {}'.format(self.n_fft))
        print('hop_length: {}'.format(self.hop_length))
        print('win_length: {}'.format(self.win_length))

    def __getitem__(self, index):
        # Item is the object to iterate
        # item[0]: video clip index
        # item[1]: data first bit's index in video clip
        # item[2]: data bit stream
        # item[3]: audio_path
        # item[4]: framerate

        item = self.items[index]
   
        file_info_dict = self.files[item[0]]
   
        try:
            # Get labels
            labels = torch.tensor(item[-1], dtype=torch.float32)
   
            # Get audio chunck
            # snd: audio time series, sr: sample rate
            snd, sr = librosa.load(item[3], sr=DATA_REQUIRED_SR)
            # print("snd shape:", snd.shape)
            # snd = librosa.util.normalize(snd)
            ### Normalize signal to be between -1 -> 1
            snd = audio_normalize(snd)

            audio = snd
   
            mixed_sig_stft = fast_stft(audio, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length)
            mixed_sig_stft = torch.tensor(mixed_sig_stft.transpose((2, 0, 1)), dtype=torch.float32)
   
        except Exception as e:
            # print(e)
            raise RuntimeError

        return {
            # "frames": frames,
            "label": labels,
            "audio": mixed_sig_stft,
        }

    def __len__(self):
        return len(self.items)


def test():
    print('In test')
    data_json_path='/home3/huydd/huydd/data_with_noise/val_data/result_json/val.json'
    # dataloader = get_dataloader(PHASE_TRAINING, batch_size=1, num_workers=20, dataset_json=data_json_path)
    dataloader = get_dataloader(PHASE_TESTING, batch_size=1, num_workers=1,dataset_json=data_json_path)
    # dataloader = get_dataloader(PHASE_PREDICTION, batch_size=8, num_workers=0)
    for i, data in enumerate(dataloader):
        print('================================================================')
        print('batch index:', i)
        # print('data[\'frames\'].size():', data['frames'].size())
        # print('data[\'bitstream\']',data['audio'])
        print('data[\'label\'].size():', data['label'].size())
        print('data[\'label\']:', data['label'])
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
