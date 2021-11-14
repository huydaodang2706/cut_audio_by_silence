import argparse
import os
from re import L
import librosa
import collections
from scipy.io.wavfile import write
from pydub import AudioSegment

import numpy as np
import torch 

from networks import get_network
from tools import *
from transform import *

DATA_REQUIRED_SR = 16000
FRAME_SIZE = 0.01 # giay
SIGMOID_THRESHOLD = 0.5

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def predict_chunk(model, chunk, gpu):
    audio = audio_normalize(chunk)
    mixed_sig_stft = fast_stft(audio, n_fft=510, hop_length=160, win_length=400)
    mixed_sig_stft = torch.tensor(mixed_sig_stft.transpose((2,0,1)),dtype=torch.float32)
    audio_input = mixed_sig_stft.unsqueeze(0)

    len_audio = audio.shape[0] # samplerate
    num_frame = int((len_audio/DATA_REQUIRED_SR)/FRAME_SIZE)
    # print('Len audio: ', len_audio)

    # print("Audio_input:",audio_input.shape)
    if torch.cuda.device_count() >= 1 and gpu:
        audio_input = audio_input.cuda()

    output = model(audio_input)
    pred_labels = (torch.sigmoid(output).detach().cpu().numpy() >= SIGMOID_THRESHOLD).astype(np.float32)
    frames = pred_labels[0].tolist()
    frames = list(map(int, frames))
    
    frames = frames[0:num_frame]
    # print('Output shape: ', len(frames) )
    return frames
 
    # print("Silence frames:",frames)
def predict_single_audio_file(model, file_path, chunk_size=400, gpu=False, window_size=100):
    sound, sr = librosa.load(file_path, sr=DATA_REQUIRED_SR)
    sound_len = sound.shape[0]
    chunk_len = int(chunk_size * FRAME_SIZE * DATA_REQUIRED_SR)
    stride_len = chunk_len - int(window_size * FRAME_SIZE * DATA_REQUIRED_SR)
    n_block = int((sound_len-chunk_len)/stride_len + 1) + 1
    frame_get = int(window_size/2)
    print(n_block)
    # process each block
    frames = []
    for x in range(n_block):
        start = x*stride_len
        end = start + chunk_len
        # print(type(start))
        # print(type(end))
        if x < (n_block -1):
            chunk = sound[start:end]
        else:
            chunk = sound[start:]
        
        # Khi nao muon doi model chi can sua ham process_chunk
        output = predict_chunk(model, chunk, gpu )

        if x == 0:
            frames = frames + output[0:-frame_get]
        elif x == n_block - 1:
            frames = frames + output[frame_get:]
        else:
            frames = frames + output[frame_get:-frame_get]
    
    # print('Frames output len: ', len(frames))
    return frames
    # output = net(audio_input.cuda())

def vad_collector(silence_mask, frame_duration_s, window_size, sr):
    # Ham nay generate voice ( start-end cho cac doan voice trong mask silence dua vao silence duration)
    # Ket qua tra ve la 1 mang chua start va end cua cac doan voice
    sample_per_frame = int(sr*frame_duration_s)

    # We use a deque for our sliding window/ring buffer.
    ring_buffer = collections.deque(maxlen=window_size)
    # We have two states: TRIGGERED and NOTTRIGGERED. We start in the
    # NOTTRIGGERED state.
    # triggered = False
    triggered = True
    
    speech_timestamp = []
    start_time = 0
    for i, frame in enumerate(silence_mask):
        is_speech = frame 
        if triggered:
            ring_buffer.append((i, is_speech))
            num_voiced = len([f for f, speech in ring_buffer if speech])

            if num_voiced >= 0.9 * ring_buffer.maxlen:
                triggered = False
                if ring_buffer[0][0] != 0:
                    start_time = (ring_buffer[0][0]-1) * sample_per_frame
                else:
                    start_time = ring_buffer[0][0] * sample_per_frame
                ring_buffer.clear()
        elif i == len(silence_mask) - 1:
            end_time = (len(silence_mask) -1) * sample_per_frame
            speech_timestamp.append({
                    'start': start_time,
                    'end': end_time
                })
            ring_buffer.clear()
        else:
            ring_buffer.append((i, is_speech))
            num_unvoiced = len([f for f, speech in ring_buffer if not speech])
            
            if num_unvoiced == ring_buffer.maxlen:
                # end_time = i * sample_per_frame
                end_time = (ring_buffer[0][0] + 1) * sample_per_frame
                triggered = True
                speech_timestamp.append({
                    'start': start_time,
                    'end': end_time
                })
                ring_buffer.clear()
                # start_time = end_time + 1
            
    return speech_timestamp

def process_single_file(file_path, model, base_name, save_path, gpu):
    frames = predict_single_audio_file(model, file_path, gpu=gpu)
    speech_timestamps = vad_collector(frames, FRAME_SIZE, 10, DATA_REQUIRED_SR)

    file = open(save_path, 'w')
    
    file.writelines('name,start,end\n')
    
    for i, seg in enumerate(speech_timestamps):
        # Chuyen doi sang giay
        start = seg['start'] / DATA_REQUIRED_SR
        end = seg['end'] / DATA_REQUIRED_SR

        file.writelines('{},{},{}\n'.format(base_name + str(i), str(start), str(end)))
    file.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio_dir','-a',type=str,default='',required=True, help="Audio directory to process")
    parser.add_argument('--ckpt', type=str, default='latest', required=False, help="desired checkpoint to restore")
    # parser.add_argument('--save_path', type=str, help="outputs path to write results silence mask file")
    parser.add_argument('--save_dir','-s',type=str,default='',required=True, help="Directory to save silence csv result")

    parser.add_argument("--gpu", type=str2bool, nargs='?',
                        const=True, help="GPU for prediction")

    args = parser.parse_args()

    net = get_network()

    if args.ckpt == 'latest' or args.ckpt == 'best_acc':
        name = args.ckpt    
    else:
        name = "ckpt_epoch{}".format(args.ckpt)

    load_path = os.path.join('/home3/huydd/huydd/model_output/huydd/','model',"{}.pth".format(name))

    print('Load saved model: {}'.format(load_path))
    if args.gpu:
        net.load_state_dict(torch.load(load_path)['model_state_dict'])
    else:
        net.load_state_dict(torch.load(load_path,map_location=torch.device('cpu'))['model_state_dict'])
        
    
    if torch.cuda.device_count() >= 1 and args.gpu:
        print('For single-GPU')
        net = net.cuda()    # For single-GPU
    else:
        net = net
    
    net.eval()

    # filepath = '/home3/huydd/cut_audio_by_silence/Speech-Denoise/result/csty2.wav'

    cannot_do = []
    if not os.path.isdir(args.save_dir):
        os.mkdir(args.save_dir)

    for sub1 in os.listdir(args.audio_dir):
        sub1_path = os.path.join(args.audio_dir, sub1) 

        for sub2 in os.listdir(sub1_path):
            if sub2.endswith('.wav'):
                sub2_path = os.path.join(sub1_path, sub2)
                print(sub2_path)

                new_save_dir = os.path.join(args.save_dir,sub1) + '_re'
                print(new_save_dir)
                if not os.path.isdir(new_save_dir):
                    os.mkdir(new_save_dir)
                save_path = os.path.join(new_save_dir, sub2.replace('.wav', '.csv'))

                try:
                    process_single_file(file_path=sub2_path, model=net, base_name='chunk_', save_path=save_path, gpu=args.gpu)
                except:
                    cannot_do.append(sub2_path)  
    print(cannot_do)

    # process_single_file(file_path=filepath, model=net, base_name='chunk_', save_path=args.save_path, gpu=args.gpu)

  

    # print(frames)
             
if __name__ == '__main__':
    main()
            
            


