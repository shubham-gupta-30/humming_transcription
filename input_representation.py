import numpy as np
import torch
import torch.nn as nn
from nnAudio import Spectrogram, features

from constants import (
    ANNOTATIONS_N_SEMITONES,
    MAX_N_SEMITONES,
)


class GetCQT(nn.Module):
    def __init__(self, sr, hop_length, fmin, n_harmonics, bins_per_semitone):
        super(GetCQT, self).__init__()
        self.sr = sr
        self.hop_length = hop_length
        self.fmin = fmin
        self.bins_per_semitone = bins_per_semitone

        # Calculate the number of semitones for the CQT
        n_semitones = np.min([
            int(np.ceil(12.0 * np.log2(n_harmonics)) + ANNOTATIONS_N_SEMITONES),
            MAX_N_SEMITONES
        ])

        # Number of bins for the CQT
        self.n_bins = n_semitones * self.bins_per_semitone
        self.bins_per_octave = 12 * self.bins_per_semitone

        # Define the CQT layer
        self.cqt_layer = Spectrogram.CQT1992v2(
            sr=self.sr, hop_length=self.hop_length, fmin=self.fmin,
            n_bins=self.n_bins, bins_per_octave=self.bins_per_octave,
            pad_mode="constant")

    def forward(self, x, wav_lengths):
        cqt = self.cqt_layer(x)

        # Apply log normalization
        cqt = torch.log1p(cqt)

        B, F, T = cqt.shape

        cqt_lengths = -torch.div(-wav_lengths, self.hop_length, rounding_mode='floor')

        cqt_mask = torch.arange(T).repeat(B, 1).to(cqt_lengths.device) < cqt_lengths[:, None]
        cqt_mask = cqt_mask.int()

        return cqt, cqt_mask # (B, F, T) and (B, T)


class GetChroma(nn.Module):
    def __init__(self, sr, hop_length, n_fft):
        super(GetChroma, self).__init__()
        self.sr = sr
        self.hop_length = hop_length
        self.n_fft = n_fft

        # Define the chroma layer
        self.chroma_layer = torch.tensor(
            features.chroma(sr=self.sr, n_fft=self.n_fft))
        

    def forward(self, x, wav_lengths):
        stft = torch.stft(x, n_fft=self.n_fft, return_complex=True, hop_length=self.hop_length)
        magnitude = torch.abs(stft)
        chroma = torch.bmm(self.chroma_layer[None].repeat(x.size(0), 1, 1).to(x.device), magnitude)
        
        B, _, T = chroma.shape
        chroma_lengths = -torch.div(-wav_lengths, self.hop_length, rounding_mode='floor')
        chroma_mask = torch.arange(T).repeat(B, 1).to(chroma_lengths.device) < chroma_lengths[:, None]
        chroma_mask = chroma_mask.int()
        return chroma, chroma_mask
