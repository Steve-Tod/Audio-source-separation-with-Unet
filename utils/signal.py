import numpy as np
import librosa
import separation

SEG_LENGTH = 130816 # about 3 seconds
WINDOW_SIZE = 1022
HOP_LENGTH = 256

def Polar2Complex(abs_val, angle):
    return abs_val * np.exp(1j * angle)

def any_source_silent(sources):
    """Returns true if the parameter sources has any silent first dimensions"""
    return np.any(np.all(np.sum(
        sources, axis=tuple(range(2, sources.ndim))) == 0, axis=1))

def compute_sdr(stft_out, stft_gt):
    signal_out = librosa.core.istft(
        stft_out, hop_length=HOP_LENGTH, win_length=WINDOW_SIZE)
    signal_gt = librosa.core.istft(
        stft_gt, hop_length=HOP_LENGTH, win_length=WINDOW_SIZE)
    if any_source_silent(signal_gt[np.newaxis, :]):
        signal_gt += 1e-10
    rvalue = separation.bss_eval_sources(
        signal_gt, signal_out, compute_permutation=False)
    return rvalue[0][0]