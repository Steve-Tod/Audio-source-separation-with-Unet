from __future__ import print_function

import torch
import torch.utils.data as data

import os, json, time

import numpy as np
import librosa
from dataHelper import nussl

from models.unet.unet_model import UNet
from utils.signal import Polar2Complex

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
instrument_array = [
    'trumpet',
    'acoustic_guitar',
    'saxophone',
    'xylophone',
    'violin',
    'flute',
    'accordion',
    'cello'
]

def check_classifier(name, res):
    print(name)
    print(res)

def generate_info_for_audio(video_root, wav_root, classifier, output='./test_info.json'):
    video_name = []
    audio_name = []
    for v in os.listdir(video_root):
        video_name.append(v.rstrip('mp4'))
    for w in os.listdir(wav_root):
        audio_name.append(w.rstrip('wav'))
    # check
    video_name.sort()
    audio_name.sort()
    assert len(audio_name) == len(video_name)
    for i in range(len(video_name)):
        assert video_name[i] == audio_name[i]

    result_info = []
    for v in video_name:
        instrument1, instrument2 = classifier(os.path.join(video_root, v + 'mp4'))
        check_classifier(v, (instrument1, instrument2))
        wav_path = os.path.join(wav_root, v + 'wav')
        assert os.path.exists(wav_path)
        result_info.append({
            'instrument1': instrument1,
            'instrument2': instrument2,
            'mixed_path': wav_path
        })
        
    with open(output, 'w') as f:
        json.dump(result_info, f)
        
# loader for test dataset
def default_test_loader(wav_path):
    wav, _ = librosa.core.load(wav_path, sr=44100)
    length = wav.shape[0]
    seg_num = length // SEG_LENGTH + 1
    pad_num = seg_num * SEG_LENGTH - length
    if pad_num != 0:
        wav = np.pad(wav, (0, pad_num), 'constant')
    assert wav.shape[0] == seg_num * SEG_LENGTH
    stft = np.zeros((seg_num, 1, 512, 512), dtype=complex)
    for s in range(seg_num):
        wav_seg = wav[s*SEG_LENGTH: (s+1)*SEG_LENGTH]
        stft[s, 0, :, :] = librosa.core.stft(wav_seg, n_fft=WINDOW_SIZE, hop_length=HOP_LENGTH)
    # seg_num X 1 X 512(freq) X 512(time)
    return stft, length

class TestSet(data.Dataset):
    def __init__(self, file_list='./test_info.json',
                 loader = default_test_loader, db=True):
        with open(file_list, 'r') as f:
            self.file_list = json.load(f)
        self.loader = loader
        self.db = db
    
    def __getitem__(self, index):
        info = self.file_list[index]
        wav_path = os.path.join(info['mixed_path'])
        stft, length = self.loader(wav_path)
        instrument = (instrument_dict[info['instrument1']], instrument_dict[info['instrument2']])
        if self.db:
            res = {
                'stft': torch.from_numpy(librosa.amplitude_to_db(np.abs(stft))),
                'stft_angle': np.angle(stft),
                'instrument':instrument,
                'length': length
            }
        else:
            res = {
                'stft': torch.from_numpy(np.abs(stft)),
                'stft_angle': np.angle(stft),
                'instrument':instrument,
                'length': length
            }
        return res
    
    def __len__(self):
        return len(self.file_list)
    
def source_separation_once(model, sample, db):
    with torch.no_grad():
        input_stft = sample["stft"].float()
        mask_list = torch.zeros(1, input_stft.shape[1], 8, 512, 512)
        for b in range(input_stft.shape[1]):
            mask_list[:, b] = model(input_stft[:, b].cuda()).cpu()
        mask1 = mask_list[:, :, sample['instrument'][0]]
        mask2 = mask_list[:, :, sample['instrument'][1]]
        output1 = torch.squeeze(mask1 * input_stft)
        output2 = torch.squeeze(mask2 * input_stft)

        angle = torch.squeeze(sample["stft_angle"])
        if db:
            out_stft1 = Polar2Complex(
                librosa.db_to_amplitude(output1.cpu().numpy()),
                angle.numpy())
            out_stft2 = Polar2Complex(
                librosa.db_to_amplitude(output2.cpu().numpy()),
                angle.numpy())
        else:
            out_stft1 = Polar2Complex(
                output1.cpu().numpy(),
                angle.numpy())
            out_stft2 = Polar2Complex(
                output2.cpu().numpy(),
                angle.numpy())

        res1 = np.zeros(angle.shape[0] * SEG_LENGTH)
        res2 = np.zeros(angle.shape[0] * SEG_LENGTH)
        for b in range(out_stft1.shape[0]):
            res1[b * SEG_LENGTH:(b + 1) * SEG_LENGTH] = librosa.core.istft(
                out_stft1[b], hop_length=HOP_LENGTH, win_length=WINDOW_SIZE)
            res2[b * SEG_LENGTH:(b + 1) * SEG_LENGTH] = librosa.core.istft(
                out_stft2[b], hop_length=HOP_LENGTH, win_length=WINDOW_SIZE)
        length = sample['length'].item()
    return res1[:length], res2[:length]

def source_separation(pretrained_path, dataset, result_wav_path, result_json_path):
    assert os.path.isdir(result_wav_path)
    db = dataset.db
    dataloader = data.DataLoader(
        dataset, batch_size=1, num_workers=6, shuffle=False)
    model = UNet(1, 8).float()
    pretrained = torch.load(pretrained_path)
    model.load_state_dict(pretrained)
    model = model.eval().cuda()

    result_json = {}
    print("Start working!")
    start = time.clock()
    for i, sample in enumerate(dataloader):
        res1, res2 = source_separation_once(model, sample, db)
        res_path = dataset.file_list[i]['mixed_path'].split('/')[-1]  # a.wav
        res_path = res_path[:-4]
        res1_path = res_path + '_seg1.wav'
        res1_path = os.path.join(result_wav_path, res1_path)
        res2_path = res_path + '_seg2.wav'
        res2_path = os.path.join(result_wav_path, res2_path)
        # write wav
        wav1 = nussl.AudioSignal(audio_data_array=res1, sample_rate=44100)
        wav1.write_audio_to_file(res1_path)
        wav2 = nussl.AudioSignal(audio_data_array=res2, sample_rate=44100)
        wav2.write_audio_to_file(res2_path)
        #write json
        result_json[res_path + '.mp4'] = [{
            'position':
            0,
            'audio': res1_path.split('/')[-1]
        }, {
            'position':
            1,
            'audio':res2_path.split('/')[-1]
        }]
        print("Finished solving %s!" % (res_path))
    end = time.clock()
    print('Finished %d audios! Spent %d seconds' % (i+1, end-start))
        
    with open(result_json_path, 'w') as f:
        json.dump(result_json, f)
