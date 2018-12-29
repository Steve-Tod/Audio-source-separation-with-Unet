import torch.utils.data as data
from torchvision import transforms
import torch
import os
import json
import random
import librosa
import numpy as np

SEG_LENGTH = 130816 # about 3 seconds
WINDOW_SIZE = 1022
HOP_LENGTH = 256
instrument_dict = {'flute': 5, 
                   'acoustic_guitar': 1, 
                   'accordion': 6, 
                   'xylophone': 3, 
                   'saxophone': 2, 
                   'cello': 7, 
                   'violin': 4, 
                   'trumpet': 0}


class TrainTransformerDb(object):
    """Convert stft to absolute value tensor"""

    def __call__(self, sample):
        stft, instrument = sample['stft'], sample['instrument']

        return {'stft': (torch.from_numpy(librosa.amplitude_to_db(np.abs(stft[0]))), 
                         torch.from_numpy(librosa.amplitude_to_db(np.abs(stft[1]))),
                         torch.from_numpy(librosa.amplitude_to_db(np.abs(stft[2])))),
                'stft_angle': (np.angle(stft[0]), np.angle(stft[1]), np.angle(stft[2])),
                'instrument': instrument}
    
    
class TestTransformerDb(object):
    def __call__(self, sample):
        stft, instrument = sample['stft'], sample['instrument']

        return {'stft': (np.abs(stft[0]), np.abs(stft[1]),
                         torch.from_numpy(librosa.amplitude_to_db(np.abs(stft[2])))),
                'stft_angle': (np.angle(stft[0]), np.angle(stft[1]), np.angle(stft[2])),
                'instrument': instrument}
    
class TrainTransformer(object):
    """Convert stft to absolute value tensor"""

    def __call__(self, sample):
        stft, instrument = sample['stft'], sample['instrument']

        return {'stft': (torch.from_numpy(np.abs(stft[0])), 
                         torch.from_numpy(np.abs(stft[1])),
                         torch.from_numpy(np.abs(stft[2]))),
                'stft_angle': (np.angle(stft[0]), np.angle(stft[1]), np.angle(stft[2])),
                'instrument': instrument}
    
    
class TestTransformer(object):
    def __call__(self, sample):
        stft, instrument = sample['stft'], sample['instrument']

        return {'stft': (np.abs(stft[0]), np.abs(stft[1]),
                         torch.from_numpy(np.abs(stft[2]))),
                'stft_angle': (np.angle(stft[0]), np.angle(stft[1]), np.angle(stft[2])),
                'instrument': instrument}
    
def default_audio_loader(wav1_path, wav2_path):
    wav1, _ = librosa.core.load(wav1_path, sr=44100)
    wav2, _ = librosa.core.load(wav2_path, sr=44100)
    length = min(wav1.shape[0], wav2.shape[0])
    scale_factor = max(abs(wav1[:length] + wav2[:length]))
    wav1 /= scale_factor
    wav2 /= scale_factor
    if length < SEG_LENGTH:
        return None
    start_index = random.randint(0, length - SEG_LENGTH)
    segment1 = wav1[start_index: start_index+SEG_LENGTH]
    segment2 = wav2[start_index: start_index+SEG_LENGTH]
    stft1 = librosa.core.stft(segment1, n_fft=WINDOW_SIZE, hop_length=HOP_LENGTH)
    stft2 = librosa.core.stft(segment2, n_fft=WINDOW_SIZE, hop_length=HOP_LENGTH)

    
    stft1 = stft1[np.newaxis, :, :]
    stft2 = stft2[np.newaxis, :, :]
    stft_mix = stft1 + stft2
    # 1 X 512(freq) X 512(time)
    return stft1, stft2, stft_mix


class STFTData(data.Dataset):
    def __init__(self, pair_dict_path, root_path="/data/data_61WwmOjr/std/trainset/audios/solo/", 
                 loader=default_audio_loader, transform=TrainTransformerDb()):
        self.root_path = root_path
        with open(pair_dict_path, 'r') as f:
            self.pair_dict = json.load(f)
        self.pair_info = []
        for k, v in self.pair_dict.items():
            # k :"instrument1-instrument2"
            # v :[{"wav1_path": wav1_path, "wav2_path": wav2_path}, ...]
            instrument1 = k.split('-')[0]
            instrument2 = k.split('-')[1]
            for item in v:
                self.pair_info.append({
                    'instrument1': instrument_dict[instrument1], 
                    'instrument2': instrument_dict[instrument2], 
                    'wav1_path': os.path.join(root_path, item['wav1_path']), 
                    'wav2_path': os.path.join(root_path, item['wav2_path']) 
                })
        self.transform = transform
        self.loader = loader
        
    def __getitem__(self, index):
        info = self.pair_info[index]
        stft1, stft2, stft_mix = self.loader(info['wav1_path'], info['wav2_path'])
        instrument1 = info["instrument1"]
        instrument2 = info["instrument2"]
        assert stft1.shape == (1, 512, 512)
        assert stft2.shape == (1, 512, 512)
        assert stft_mix.shape == (1, 512, 512)
        
        sample = {
            "stft": (stft1, stft2, stft_mix),
            "instrument": (instrument1, instrument2)
        }
        if self.transform:
            sample = self.transform(sample)
            
        return sample
            
    def __len__(self):
        return len(self.pair_info)