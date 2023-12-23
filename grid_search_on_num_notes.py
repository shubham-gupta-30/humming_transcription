import itertools
from multiprocessing import Pool
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from nnAudio import features
from tqdm import tqdm

from loader import HumTransDataset


def plot_waveforms(original_signal, modified_signal, threshold, above_threshold, onsets, offsets, fname):
    T = len(modified_signal)
    modified_signal = modified_signal[:T]
    plt.figure(figsize=(40, 4))
    time = np.arange(len(modified_signal))
    plt.plot(time, original_signal, label='Original Signal', alpha=0.7)
    plt.plot(time, modified_signal, label='Modified Signal', color='red', alpha=0.7)
    plt.plot(time, above_threshold, label='AboveThrehsold', color='blue', alpha=0.2)
    onsets_np = np.zeros(modified_signal.shape[0])
    onsets_np[onsets] = 1
    offsets_np = np.zeros(modified_signal.shape[0])
    offsets_np[offsets] = 1
    plt.plot(time, onsets_np, label='AboveThrehsold', color='green', alpha=0.7)
    plt.plot(time, offsets_np, label='AboveThrehsold', color='black', alpha=0.7)
    
    plt.axhline(threshold, color='green', label='Threshold', alpha=0.7)
    plt.title("Waveform Comparison")
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.savefig(fname)

def adjust_onsets_offsets(onsets, offsets, num_time_frames, min_silence_length=1000, min_note_length=3000):
    # Convert onsets and offsets to numpy if they are tensors
    onsets = onsets.numpy() if isinstance(onsets, torch.Tensor) else onsets
    offsets = offsets.numpy() if isinstance(offsets, torch.Tensor) else offsets

    # Initial preprocessing
    if onsets.shape[0] == 0:
        onsets = np.concatenate([np.array([0]), onsets], axis=0)
    if offsets.shape[0] == 0:
        offsets = np.concatenate([offsets, np.array([num_time_frames - 1])], axis=0)
    if onsets[0] > offsets[0]:
        onsets = np.concatenate([np.array([0]), onsets], axis=0)
    if onsets[-1] > offsets[-1]:
        offsets = np.concatenate([offsets, np.array([num_time_frames - 1])], axis=0)

    # Adjusting onsets and offsets
    adjusted_onsets = [onsets[0]] if len(onsets) > 0 else []
    adjusted_offsets = []

    for i in range(1, len(onsets)):
        # Merge intervals if the silence between notes is less than the minimum silence length
        if onsets[i] - offsets[i - 1] < min_silence_length:
            continue
        else:
            # Check if the current note meets the minimum note length requirement
            if offsets[i - 1] - adjusted_onsets[-1] >= min_note_length:
                adjusted_offsets.append(offsets[i - 1])
                adjusted_onsets.append(onsets[i])

    # Closing the last interval
    if len(adjusted_onsets) > len(adjusted_offsets) and len(adjusted_onsets) > 0:
        if offsets[-1] - adjusted_onsets[-1] >= min_note_length:
            adjusted_offsets.append(offsets[-1])
        else:
            adjusted_onsets = adjusted_onsets[:-1]

    return np.array(adjusted_onsets), np.array(adjusted_offsets)


def get_waveform_envelope(signal):

    # Padding for the signal
    padding = 100
    padded_signal = torch.nn.functional.pad(signal, (padding, padding), mode='constant', value=0)

    # Unfold to get sliding windows
    windows_before = padded_signal[:-padding ].unfold(0, padding, 1)
    windows_after = padded_signal[padding:].unfold(0, padding, 1)

    # Compute maximums
    max_before = windows_before.max(dim=1).values
    max_after = windows_after.max(dim=1).values

    # Compute minimum of the two maximums
    modified_signal = torch.min(max_before, max_after)

    return modified_signal




# Function to process a single dataset item
def process(i):    
    signal =  torch.abs(torch.tensor(dataset[i]["wav_data"]).float())
    envelope = get_waveform_envelope(signal)[None]
    envelope = get_waveform_envelope(envelope)[None]
    mw_min = torch.min(envelope)
    mw_max = torch.max(envelope)
    thresholds_for_i = mw_min + thresholds * (mw_max - mw_min)
    above_threshold = envelope > thresholds_for_i

    num_notes_known = dataset[i]["midi_notes"].shape[0]


    for t_idx in range(len(threshold_values)):
        current_above_threshold = above_threshold[t_idx]
        onsets = (current_above_threshold[:-1] < current_above_threshold[1:]).nonzero(as_tuple=True)[0]
        offsets = (current_above_threshold[:-1] > current_above_threshold[1:]).nonzero(as_tuple=True)[0] + 1

        onsets, offsets = adjust_onsets_offsets(onsets, offsets, envelope.shape[0])
        num_notes_discovered = len(offsets)
        if num_notes_discovered == num_notes_known:
                file_path = filtered_folder / f"{dataset[i]['file_name']}_onsets_offsets.txt"
                with file_path.open('w') as f:
                    for onset, offset in zip(onsets, offsets):
                        f.write(f"{onset} {offset}\n")
                plot_waveforms(
                    signal.numpy(),
                    envelope.numpy(),
                    thresholds_for_i[0].item(),
                    current_above_threshold.numpy(), onsets, offsets, f"{i}_{dataset[i]['file_name']}.png")
                return True

    # print(f"Done {i}")
    return False


if __name__ == "__main__":
    
    dataset = HumTransDataset("/tmp/HumTrans", split="valid", filtered_file=None)
    print(len(dataset))

    # Define the ranges for your hyperparameters
    # threshold_values = np.linspace(0.01, 0.2, 5)  # Define your range and number of thresholds
    # min_silence_lengths = np.linspace(500, 3000, 5)  # Define your range and number of note lengths

    threshold_values = np.linspace(1.0, 0.01, 10)
    min_silence_lengths = np.array([1125.0])

    thresholds = torch.from_numpy(threshold_values).unsqueeze(1)


    print(threshold_values)
    print(min_silence_lengths)

    # Initialize variables to track the best hyperparameters and their performance
    best_threshold = None
    best_min_silence_length = None
    min_failed = len(dataset)  # Start with the worst case, where all cases fail

    drawn = 0

    # Initialize a matrix to track the failure count for each combination of hyperparameters
    failures_matrix = np.zeros((len(threshold_values), len(min_silence_lengths)), dtype=int)

    # We want a high time resolution here
    hop_length = 128
    window_length = 512


    filtered_folder = Path('/tmp/HumTrans/filtered_valid')
    filtered_folder.mkdir(parents=True, exist_ok=True)

    # Define the number of workers for the Pool
    num_workers = 10  # Adjust this based on your system's capabilities

    # Initialize the failures matrix
    failures_matrix = np.zeros((len(threshold_values), len(min_silence_lengths)), dtype=int)


    # Use Pool to parallelize the task
    with Pool(num_workers) as pool:
        # imap gives us an iterator that allows tracking progress with tqdm
        T = len(dataset)
        # T = 100
        results = list(tqdm(pool.imap(process, list(range(T))), total=T))

    # results = [process(i) for i in range(1000)]

    indices = []

    success = 0
    # Accumulate the results
    for i, res in enumerate(results):
        success += res
        if not res :
            indices.append(i)

    print(indices)

    print(success, success/T)
    import shutil

    # Create the zip file
    shutil.make_archive(filtered_folder, 'zip', filtered_folder)

    # Move the zip file to the destination folder
    dst_folder = Path("/home/shubhamg/projects/def-ravanelm/datasets/")
    dst_folder.mkdir(parents=True, exist_ok=True)  # Create the destination folder if it doesn't exist
    shutil.move(str(filtered_folder) + ".zip", str(dst_folder))


