import torch
import torch.utils.data as data
import torch.optim as optim
import torch.nn as nn

import os
import numpy as np
import librosa
from utils import separation
from utils.signal import Polar2Complex, any_source_silent, compute_sdr
import random

SEG_LENGTH = 130816
WINDOW_SIZE = 1022
HOP_LENGTH = 256
instrument_dict = {
    'flute': 5,
    'acoustic_guitar': 1,
    'accordion': 6,
    'xylophone': 3,
    'saxophone': 2,
    'cello': 7,
    'violin': 4,
    'trumpet': 0
}


def evaluate(model, testloader, instrument_dict, bs, test_number=-1):
    # model eval() and cuda()
    model.eval()
    model.cuda()
    sdr_list = [[] for x in range(len(instrument_dict))]
    if test_number == -1:
        test_number = len(testloader)

    with torch.no_grad():
        for i, sample in enumerate(testloader, 0):
            input_stft = sample["stft"][2].cuda()
            mask_list = model(input_stft)
            ref = torch.arange(0, bs)
            mask1 = mask_list[ref, sample['instrument'][0]]
            mask2 = mask_list[ref, sample['instrument'][1]]
            mask1 = torch.unsqueeze(mask1, 1)
            mask2 = torch.unsqueeze(mask2, 1)
            output1 = mask1 * input_stft
            output2 = mask2 * input_stft

            out_stft1 = Polar2Complex(
                librosa.db_to_amplitude(output1.cpu().numpy()),
                sample["stft_angle"][2].numpy())
            out_stft2 = Polar2Complex(
                librosa.db_to_amplitude(output2.cpu().numpy()),
                sample["stft_angle"][2].numpy())
            gt_stft1 = Polar2Complex(sample['stft'][0].numpy(),
                                     sample["stft_angle"][0].numpy())
            gt_stft2 = Polar2Complex(sample['stft'][1].numpy(),
                                     sample["stft_angle"][1].numpy())
            for b in range(out_stft1.shape[0]):
                sdr_1 = compute_sdr(out_stft1[b, 0, :, :],
                                    gt_stft1[b, 0, :, :])
                sdr_2 = compute_sdr(out_stft2[b, 0, :, :],
                                    gt_stft2[b, 0, :, :])
                sdr_list[sample['instrument'][0][b]].append(sdr_1)
                sdr_list[sample['instrument'][1][b]].append(sdr_2)
            if i >= test_number:
                break

    return sdr_list


def evaluate_nodb(model, testloader, instrument_dict, bs, test_number=-1):
    # model eval() and cuda()
    model.eval()
    model.cuda()
    sdr_list = [[] for x in range(len(instrument_dict))]
    if test_number == -1:
        test_number = len(testloader)

    with torch.no_grad():
        for i, sample in enumerate(testloader, 0):
            input_stft = sample["stft"][2].cuda()
            mask_list = model(input_stft)
            ref = torch.arange(0, bs)
            mask1 = mask_list[ref, sample['instrument'][0]]
            mask2 = mask_list[ref, sample['instrument'][1]]
            mask1 = torch.unsqueeze(mask1, 1)
            mask2 = torch.unsqueeze(mask2, 1)
            output1 = mask1 * input_stft
            output2 = mask2 * input_stft

            out_stft1 = Polar2Complex(output1.cpu().numpy(),
                                      sample["stft_angle"][2].numpy())
            out_stft2 = Polar2Complex(output2.cpu().numpy(),
                                      sample["stft_angle"][2].numpy())
            gt_stft1 = Polar2Complex(sample['stft'][0].numpy(),
                                     sample["stft_angle"][0].numpy())
            gt_stft2 = Polar2Complex(sample['stft'][1].numpy(),
                                     sample["stft_angle"][1].numpy())
            for b in range(out_stft1.shape[0]):
                sdr_1 = compute_sdr(out_stft1[b, 0, :, :],
                                    gt_stft1[b, 0, :, :])
                sdr_2 = compute_sdr(out_stft2[b, 0, :, :],
                                    gt_stft2[b, 0, :, :])
                sdr_list[sample['instrument'][0][b]].append(sdr_1)
                sdr_list[sample['instrument'][1][b]].append(sdr_2)
            if i >= test_number:
                break

    return sdr_list


def eval_train(output1, output2, sample, sdr_list):
    out_stft1 = Polar2Complex(
        librosa.db_to_amplitude(output1.detach().cpu().numpy()),
        sample["stft_angle"][2].numpy())
    out_stft2 = Polar2Complex(
        librosa.db_to_amplitude(output2.detach().cpu().numpy()),
        sample["stft_angle"][2].numpy())
    gt_stft1 = Polar2Complex(
        librosa.db_to_amplitude(sample['stft'][0].numpy()),
        sample["stft_angle"][0].numpy())
    gt_stft2 = Polar2Complex(
        librosa.db_to_amplitude(sample['stft'][1].numpy()),
        sample["stft_angle"][1].numpy())
    for b in range(out_stft1.shape[0]):
        sdr_1 = compute_sdr(out_stft1[b, 0, :, :], gt_stft1[b, 0, :, :])
        sdr_2 = compute_sdr(out_stft2[b, 0, :, :], gt_stft2[b, 0, :, :])
        sdr_list[sample['instrument'][0][b]].append(sdr_1)
        sdr_list[sample['instrument'][1][b]].append(sdr_2)
