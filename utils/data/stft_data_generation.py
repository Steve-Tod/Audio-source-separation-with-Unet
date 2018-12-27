import os
import numpy as np
import scipy
import librosa
import itertools
import random
import argparse
import pickle

SAMPLE_RATE = 11025
WINDOW_SIZE = 1022
HOP_LENGTH = 256
SEG_LENGTH = 262000

parser = argparse.ArgumentParser(description='Generate mixed stft data and label.')
parser.add_argument('train_num', metavar='Ntrain', type=int, default=100, nargs=1,
                    help='The number of train data per combination')
parser.add_argument('test_num', metavar='Ntest', type=int, default=10, nargs=1,
                    help='The number of test data per combination')
parser.add_argument('--input_dir', type=str, default='/data/data_Z2o77zHr/std/trainset/audios/solo/',
                    help='The path of input dir')
parser.add_argument('--output_dir', type=str, required=True,
                    help='The path of output dir')
args = parser.parse_args()

# single stft with log resample
def stft_log(sig, window_size, hop_length, sample_index):
    stft_res = librosa.core.stft(sig, n_fft=window_size, hop_length=hop_length)
    assert stft_res.shape == (512, 256)
    return stft_res[sample_index, :]

# split long audio sequence and generate stft
def generate_stft(sig, window_size, hop_length, sample_index, seg_length):
    length = sig.shape[0]
    seg_num = int(np.floor(length/seg_length))
    spectrogram = np.zeros((seg_num, 2, 256, 256))
    for i in range(seg_num):
        stft_res = stft_log(sig[i*seg_length:(i+1)*seg_length], window_size, hop_length, sample_index)
        spectrogram[i, 0, :, :] = np.real(stft_res)
        spectrogram[i, 1, :, :] = np.imag(stft_res)
    return spectrogram

# mix two audios, downsample and generate stft
def mix_and_generate_spectrogram(wav1, wav2, window_size, hop_length, sample_index, seg_length):
    length = min(wav1.shape[0], wav2.shape[0])
    down_sampled_sig = scipy.signal.decimate(wav1[:length]+wav2[:length], 4)
    spectrogram = generate_stft(down_sampled_sig, window_size, hop_length, sample_index, int(seg_length/4))
    return spectrogram

if __name__ == "__main__":
    audio_prefix = args.input_dir
    save_dir = args.output_dir
    num_per_comb_train = args.train_num[0]
    num_per_comb_test = args.test_num[0]
    #print(num_per_comb_train, num_per_comb_test)
    
    # prepare for log resample
    frequencies = np.linspace(SAMPLE_RATE/2/512, SAMPLE_RATE/2, 512)
    log_freq = np.log10(frequencies)
    sample_freq = np.linspace(log_freq[0], log_freq[-1], 256)
    sample_index = [np.abs(log_freq-x).argmin() for x in sample_freq]

    instruments = tuple(os.listdir(audio_prefix))
    combinations = list(itertools.combinations(instruments, 2))
    
    train_list = []
    test_list = []
    
    for comb in combinations:
        instrument1, instrument2 = comb
        instrument1_seg = list(os.listdir(os.path.join(audio_prefix, instrument1)))
        instrument2_seg = list(os.listdir(os.path.join(audio_prefix, instrument2)))
        seg_product = list(itertools.product(instrument1_seg, instrument2_seg))
        selected_pairs = random.choices(seg_product, k=num_per_comb_train+num_per_comb_test)
        for i, pair in enumerate(selected_pairs):
            wav1_path = os.path.join(audio_prefix, instrument1, pair[0])
            wav2_path = os.path.join(audio_prefix, instrument2, pair[1])
            wav1, _ = librosa.core.load(wav1_path, sr=44100)
            wav2, _ = librosa.core.load(wav2_path, sr=44100)
            mixed_spectrogram = mix_and_generate_spectrogram(wav1, wav2, WINDOW_SIZE, 
                                                             HOP_LENGTH, sample_index, SEG_LENGTH)
            seg_num = mixed_spectrogram.shape[0]
            wav1_gt = np.reshape(wav1[:seg_num*SEG_LENGTH], (seg_num, SEG_LENGTH))
            wav2_gt = np.reshape(wav2[:seg_num*SEG_LENGTH], (seg_num, SEG_LENGTH))
            if mixed_spectrogram.shape[0] == 0:
                break
            if i < num_per_comb_train:
                for index in range(mixed_spectrogram.shape[0]):
                    train_list.append({
                        "wav1_gt": wav1_gt[index, :],
                        "wav2_gt": wav2_gt[index, :],
                        "wav1_instrument": instrument1,
                        "wav2_instrument": instrument2,
                        "mixed_spectrogram": mixed_spectrogram[index, :, :, :]
                })
            else:
                for index in range(mixed_spectrogram.shape[0]):
                    test_list.append({
                        "wav1_gt": wav1_gt[index, :],
                        "wav2_gt": wav2_gt[index, :],
                        "wav1_instrument": instrument1,
                        "wav2_instrument": instrument2,
                        "mixed_spectrogram": mixed_spectrogram[index, :, :, :]
                    })
                
    with open(os.path.join(save_dir, "train.pkl"), 'wb') as f:
        pickle.dump(train_list, f)
    with open(os.path.join(save_dir, "test.pkl"), 'wb') as f:
        pickle.dump(test_list, f)
                