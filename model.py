import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

from constants import (
    AUDIO_SAMPLE_RATE,
    CONTOURS_BINS_PER_SEMITONE,
    FFT_HOP,
    N_HARMONICS,
    N_FFT
)
from input_representation import GetCQT, GetChroma


class HarmonicStacking(nn.Module):
    def __init__(self, bins_per_semitone=3, num_harmonics=7, num_subharmonics=1):
        super(HarmonicStacking, self).__init__()
        self.bins_per_semitone = bins_per_semitone
        self.num_harmonics = num_harmonics
        self.num_subharmonics = num_subharmonics

    def forward(self, cqt):
        # Calculate shifts for harmonics and sub-harmonics
        sub_harmonics = [1 / (x + 1) for x in range(1, self.num_subharmonics + 1)]
        harmonics = list(range(1, self.num_harmonics + 1))
        shifts = [int(round(
            12 * self.bins_per_semitone * np.log2(h))) for h in sub_harmonics + harmonics]

        # Stack shifted CQTs
        hcqt = []
        for shift in shifts:
            if shift < 0:  # Handling sub-harmonics
                shifted_cqt = F.pad(cqt[:, :shift, :], (0, 0, -shift, 0))
            else:  # Handling harmonics
                shifted_cqt = F.pad(cqt[:, shift:, :], (0, 0, 0, shift))

            hcqt.append(shifted_cqt)

        output = torch.stack(hcqt, dim=1)

        return  output # Shape: (B, num_harmonics + num_subharmonics, F, T)


def add_noise(audio, noise_level=0.005):
    noise = torch.randn_like(audio) * noise_level
    return audio + noise


def apply_low_pass_filter(audio, sample_rate, cutoff_freq=3000):
    return  torchaudio.functional.lowpass_biquad(audio, sample_rate, cutoff_freq)


def add_reverb(audio, sample_rate, reverberance=30):
    effects = [
        ["reverb", str(reverberance)]
    ]
    augmented_audio, _ = torchaudio.sox_effects.apply_effects_tensor(audio, sample_rate, effects)
    return augmented_audio


def add_transforms(wav_data, p=0.5):
    if torch.rand(1).item() < p:
        wav_data = add_noise(wav_data)
    if torch.rand(1).item() < p:
        wav_data = apply_low_pass_filter(wav_data, AUDIO_SAMPLE_RATE)
    # if torch.rand(1).item() < p:
    #     print("Reverb")
    #     wav_data = add_reverb(wav_data, AUDIO_SAMPLE_RATE)
    return wav_data


class HarmonicCQTConv2D(nn.Module):
    def __init__(self, num_classes=13, in_features=8, do_transforms=False):
        super(HarmonicCQTConv2D, self).__init__()
        self.get_audio_rep = GetCQT(sr=AUDIO_SAMPLE_RATE,
                 hop_length=FFT_HOP,
                 fmin=27.5,  # Lowest frequency (A0)
                 n_harmonics=N_HARMONICS,
                 bins_per_semitone=CONTOURS_BINS_PER_SEMITONE)
        self.model = nn.Sequential(
            HarmonicStacking(),
            nn.Conv2d(in_features, 16, kernel_size=(5, 5), padding="same"),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv2d(16, 8, kernel_size=(3, 3 * 13), padding="same"),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv2d(8, 32, kernel_size=(7, 7),  padding="same"),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv2d(32, 1, kernel_size=(7, 3), padding="same"))
        self.final_layer = nn.Linear(345, num_classes)
        self.do_transforms = do_transforms


    def forward(self, wav_data, wav_lengths):
        if self.do_transforms and self.train:
          wav_data = add_transforms(wav_data)
        x, cqt_mask = self.get_audio_rep(wav_data.float(), wav_lengths)
        x = self.model(x)
        x = x.squeeze(1).transpose(-1, -2)
        x = self.final_layer(x)
        return x, cqt_mask


class CQTConv1D(nn.Module):
    def __init__(self, num_classes=13, n_features=12, do_transforms=False):
        super(CQTConv1D, self).__init__()
        self.get_audio_rep = GetChroma(AUDIO_SAMPLE_RATE, FFT_HOP, N_FFT)
        self.model = nn.Sequential(
            nn.Conv1d(n_features, 64, kernel_size=9, padding=4),
            nn.BatchNorm1d(64, False),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv1d(64, 32, kernel_size=9, padding=4),
            nn.BatchNorm1d(32, False),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv1d(32, num_classes, kernel_size=1))
        self.do_transforms = do_transforms

    def forward(self, wav_data, wav_lengths):
        if self.do_transforms and self.train:
          wav_data = add_transforms(wav_data)
        x, mask = self.get_audio_rep(wav_data.float(), wav_lengths)
        x = self.model(x).transpose(1, 2)
        return x, mask