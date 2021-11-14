import numpy
import scipy.io.wavfile
from scipy.fftpack import dct

import matplotlib.pyplot as plt
import numpy as np
import librosa
import librosa.display
import torch 

SAMPLE_RATE = 16000

def power_law(data,power=0.3):
    # assume input has negative value
    mask = np.zeros(data.shape)
    mask[data>=0] = 1
    mask[data<0] = -1
    data = np.power(np.abs(data),power)
    data = data*mask
    return data

def real_imag_expand(c_data,dim='new'):
    # dim = 'new' or 'same'
    # expand the complex data to 2X data with true real and image number
    if dim == 'new':
        D = np.zeros((c_data.shape[0],c_data.shape[1],2))
        D[:,:,0] = np.real(c_data)
        D[:,:,1] = np.imag(c_data)
        return D
    if dim =='same':
        D = np.zeros((c_data.shape[0],c_data.shape[1]*2))
        D[:,::2] = np.real(c_data)
        D[:,1::2] = np.imag(c_data)
        return D

def real_imag_test(c_data,dim='new'):
    # dim = 'new' or 'same'
    # expand the complex data to 2X data with true real and image number
    if dim == 'new':
        D = np.zeros((c_data.shape[0],c_data.shape[1]))
        D = np.real(c_data)
        return D
    
def fast_stft(data, power=False, n_fft=78, hop_length=160, win_length=60):
    # directly transform the wav to the input
    # power law = A**0.3 , to prevent loud audio from overwhelming soft audio
    if power:
        data = power_law(data)
    return real_imag_test(librosa.stft(data, n_fft, hop_length, win_length))
    # return real_imag_expand(librosa.stft(data, n_fft, hop_length, win_length))

def audio_normalize(snd):
    """Normalize librosa audio array"""
    max_abs = max(abs(snd))
    if max_abs > 1:
        mult_var = 1.0 / max_abs
        return snd * mult_var
    else:
        return snd

def filter_bank_with_path(path, nfilt=60):
    # Calculate log mel filterbank
    # sample_rate, signal = scipy.io.wavfile.read(path)  # File assumed to be in the same directory
    signal, sample_rate = librosa.load(path, sr=SAMPLE_RATE)
    print(signal.shape)
    # print(signal.shape[0])
    # Append 0 signal to current signal to reach 12 s
    # print(type(signal))
    # padding = numpy.zeros((12*sample_rate - signal.shape[0],))
    # signal = numpy.concatenate((signal, padding), axis=0)

    # Pre-Emphasis
    #---------------------------------
    pre_emphasis = 0.97
    emphasized_signal = numpy.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])

    # Framing
    #---------------------------------
    # frame_size = 0.025
    frame_stride = 0.01
    frame_size = 0.00625

    frame_length, frame_step = frame_size * sample_rate, frame_stride * sample_rate  # Convert from seconds to samples
    signal_length = len(emphasized_signal)
    frame_length = int(round(frame_length))
    frame_step = int(round(frame_step))

    # print("frames step:", frame_step)
    # print("signal length:", signal_length)

    num_frames = int(numpy.ceil(float(numpy.abs(signal_length - frame_length)) / frame_step))  # Make sure that we have at least 1 frame
    # num_frames = int(numpy.ceil(float(numpy.abs(signal_length - 100)) / frame_step))  # Make sure that we have at least 1 frame

    # print("Number of frames:", num_frames)
    # print("frames length:",frame_length)

    pad_signal_length = num_frames * frame_step + frame_length
    z = numpy.zeros((pad_signal_length - signal_length))
    pad_signal = numpy.append(emphasized_signal, z) # Pad Signal to make sure that all frames have equal number of samples without truncating any samples from the original signal

    indices = numpy.tile(numpy.arange(0, frame_length), (num_frames, 1)) + numpy.tile(numpy.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    frames = pad_signal[indices.astype(numpy.int32, copy=False)]
    # print("Frames shape:",frames.shape)

    # Window
    #---------------------------------
    frames *= numpy.hamming(frame_length)
    # frames *= 0.54 - 0.46 * numpy.cos((2 * numpy.pi * n) / (frame_length - 1))  # Explicit Implementation **

    #Fourier-Transform and Power Spectrum
    #---------------------------------
    NFFT = 512
    # NFFT = 700
    mag_frames = numpy.absolute(numpy.fft.rfft(frames, NFFT))  # Magnitude of the FFT
    # print("Mag_frames shape:",mag_frames.shape)

    pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))  # Power Spectrum
    # print("Pow_frames shape:",pow_frames.shape)
    # Mel filterbank
    nfilt = nfilt 

    low_freq_mel = 0
    high_freq_mel = (2595 * numpy.log10(1 + (sample_rate / 2) / 700))  # Convert Hz to Mel
    mel_points = numpy.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale
    hz_points = (700 * (10**(mel_points / 2595) - 1))  # Convert Mel to Hz
    bin = numpy.floor((NFFT + 1) * hz_points / sample_rate)

    fbank = numpy.zeros((nfilt, int(numpy.floor(NFFT / 2 + 1))))
    for m in range(1, nfilt + 1):
        f_m_minus = int(bin[m - 1])   # left
        f_m = int(bin[m])             # center
        f_m_plus = int(bin[m + 1])    # right

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
    filter_banks = numpy.dot(pow_frames, fbank.T)
    filter_banks = numpy.where(filter_banks == 0, numpy.finfo(float).eps, filter_banks)  # Numerical Stability
    filter_banks = 20 * numpy.log10(filter_banks)  # dB

    # print("shape of filterbank:", filter_banks.shape)

    return filter_banks

def filter_bank_with_audio(signal, sample_rate, nfilt=60):
    # Calculate log mel filterbank
    # sample_rate, signal = scipy.io.wavfile.read(path)  # File assumed to be in the same directory
    # print(signal.shape[0])
    # Append 0 signal to current signal to reach 12 s
    # print(type(signal))
    # padding = numpy.zeros((12*sample_rate - signal.shape[0],))
    # signal = numpy.concatenate((signal, padding), axis=0)

    # Pre-Emphasis
    #---------------------------------
    pre_emphasis = 0.97
    emphasized_signal = numpy.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])

    # Framing
    #---------------------------------
    # frame_size = 0.025
    frame_stride = 0.01
    frame_size = 0.00625

    frame_length, frame_step = frame_size * sample_rate, frame_stride * sample_rate  # Convert from seconds to samples
    signal_length = len(emphasized_signal)
    frame_length = int(round(frame_length))
    frame_step = int(round(frame_step))

    # print("frames step:", frame_step)
    # print("signal length:", signal_length)

    num_frames = int(numpy.ceil(float(numpy.abs(signal_length - frame_length)) / frame_step))  # Make sure that we have at least 1 frame
    # num_frames = int(numpy.ceil(float(numpy.abs(signal_length - 100)) / frame_step))  # Make sure that we have at least 1 frame

    # print("Number of frames:", num_frames)
    # print("frames length:",frame_length)

    pad_signal_length = num_frames * frame_step + frame_length
    z = numpy.zeros((pad_signal_length - signal_length))
    pad_signal = numpy.append(emphasized_signal, z) # Pad Signal to make sure that all frames have equal number of samples without truncating any samples from the original signal

    indices = numpy.tile(numpy.arange(0, frame_length), (num_frames, 1)) + numpy.tile(numpy.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    frames = pad_signal[indices.astype(numpy.int32, copy=False)]
    # print("Frames shape:",frames.shape)

    # Window
    #---------------------------------
    frames *= numpy.hamming(frame_length)
    # frames *= 0.54 - 0.46 * numpy.cos((2 * numpy.pi * n) / (frame_length - 1))  # Explicit Implementation **

    #Fourier-Transform and Power Spectrum
    #---------------------------------
    NFFT = 512
    # NFFT = 700
    mag_frames = numpy.absolute(numpy.fft.rfft(frames, NFFT))  # Magnitude of the FFT
    # print("Mag_frames shape:",mag_frames.shape)

    pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))  # Power Spectrum
    # print("Pow_frames shape:",pow_frames.shape)
    # Mel filterbank
    nfilt = nfilt 

    low_freq_mel = 0
    high_freq_mel = (2595 * numpy.log10(1 + (sample_rate / 2) / 700))  # Convert Hz to Mel
    mel_points = numpy.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale
    hz_points = (700 * (10**(mel_points / 2595) - 1))  # Convert Mel to Hz
    bin = numpy.floor((NFFT + 1) * hz_points / sample_rate)

    fbank = numpy.zeros((nfilt, int(numpy.floor(NFFT / 2 + 1))))
    for m in range(1, nfilt + 1):
        f_m_minus = int(bin[m - 1])   # left
        f_m = int(bin[m])             # center
        f_m_plus = int(bin[m + 1])    # right

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
    filter_banks = numpy.dot(pow_frames, fbank.T)
    filter_banks = numpy.where(filter_banks == 0, numpy.finfo(float).eps, filter_banks)  # Numerical Stability
    filter_banks = 20 * numpy.log10(filter_banks)  # dB

    # print("shape of filterbank:", filter_banks.shape)

    return filter_banks

if __name__ == '__main__':
    file = '/Users/huydd/Hedspi-5/code/cut_audio_by_silence/Speech-Denoise/result/csty1.wav'
    # fi_bank = filter_bank_with_path(file)
    fi_bank_audio, sr = librosa.load(file, sr=SAMPLE_RATE)
    fi_bank = filter_bank_with_audio(fi_bank_audio, sr)
    filter_bank = torch.tensor(fi_bank.transpose((1, 0)), dtype=torch.float32)
    filter_bank = filter_bank.unsqueeze(0)

    print(fi_bank.shape)
    print(filter_bank.shape)