import json
import os
import random
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pretty_midi
import soundfile as sf
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


def peak_normalize(signal):
    # Assuming signal is a 1-D numpy array
    signal_max = np.max(np.abs(signal))

    # Avoid dividing by zero
    if signal_max > 0:
        signal = signal / signal_max
    return signal


class HumTransDataset(Dataset):
  def __init__(
          self,
          root_path,
          split="train", 
          synthesize: bool = False, 
          preload: bool = False,
          filtered_file: str = "filtered"):
        """
        possible values for split are train|test|valid
        """

        self.root_path = Path(root_path)
        self.preload = preload

        self.midi_root = self.root_path / "all_midi_unzipped"
        self.midi_path = self.midi_root / "midi_data"
        self.midi_zip = self.root_path / "all_midi.zip"

        self.wav_root = self.root_path/ "all_wav_unzipped"
        self.wav_path = self.wav_root / "wav_data_sync_with_midi"
        self.wav_zip = self.root_path / "all_wav.zip"


        self.json_path = self.root_path / "train_valid_test_keys.json"
        if filtered_file is not None:
            self.filtered_zip = self.root_path / f"{filtered_file}.zip"
        else:
            self.filtered_zip = None
        self.filtered_root = self.root_path / f"{filtered_file}"



        self.unpack_if_needed(self.midi_root, self.midi_zip)
        self.unpack_if_needed(self.wav_root, self.wav_zip)
        if self.filtered_zip is not None:
            self.unpack_if_needed(self.filtered_root, self.filtered_zip)

        # Load the JSON file
        with open(self.json_path, 'r') as file:
            data = json.load(file)

        # Extract file names for the specified split
        self.file_names = data[split.upper()]
        if self.filtered_zip is not None:
            self.file_names = self.filter_file_names(self.file_names)

        self.synthesize = synthesize

        self.midi_data = {}
        self.wav_data = {}
        self.onset_offset_data = {}

        if self.preload:
            self.preload_data()

  def filter_file_names(self, file_names):
        filtered_file_names = []
        for file_name in file_names:
            if (self.filtered_root / f"{file_name}_onsets_offsets.txt").exists():
                filtered_file_names.append(file_name)
        print(f"Kept {len(filtered_file_names)} from {len(file_names)} files.")
        return filtered_file_names

  def preload_data(self):
        print("Preloading data...")

        with ThreadPoolExecutor() as executor:
            futures = []
            for file_name in tqdm(self.file_names, desc="Preloading files"):
                futures.append(executor.submit(self.load_file, file_name))

            for future in tqdm(futures, desc="Processing files"):
                result = future.result()
                self.midi_data[result['file_name']] = result['midi_data']
                self.wav_data[result['file_name']] = result['wav_data']
                self.onset_offset_data[result['file_name']] = result['onset_offset_data']
                del result

  def read_onset_offset_file(self, file_path):
        onset_offset_data = []
        with open(file_path, 'r') as file:
            for line in file:
                onset, offset = map(int, line.strip().split())
                onset_offset_data.append((onset, offset))
        return onset_offset_data

  def load_file(self, file_name):
        midi_file = os.path.join(self.midi_path, file_name + ".mid")
        wav_file = os.path.join(self.wav_path, file_name + ".wav")

        midi_data, midi_notes = self.extract_notes(midi_file)
        wav_data, sr = sf.read(wav_file)
        wav_data = peak_normalize(wav_data)

        midi_audio_data = midi_data.synthesize() if self.synthesize else None

        # Read onset and offset data from the corresponding file in the filtered folder
        if self.filtered_zip is not None:
            onset_offset_file = self.filtered_root / (f"{file_name}_onsets_offsets.txt")
            onset_offset_data = self.read_onset_offset_file(onset_offset_file)
        else:
            onset_offset_data = [(x[0], x[1]) for x in midi_notes]
            
        return {"file_name": file_name, "midi_data": (midi_data, midi_notes), "wav_data": (wav_data, sr),
                "midi_audio_data": midi_audio_data, "onset_offset_data": onset_offset_data}


  def unpack_if_needed(self, unpack_path, zip_path):
      if not unpack_path.exists():
        print(f"{unpack_path} does not exist, looking for zipped file")
        if not zip_path.exists():
            raise FileNotFoundError(f"{zip_path} also does not exist")
        else:
            print(f"Extracting {zip_path} to {unpack_path}")
            shutil.unpack_archive(zip_path, unpack_path)

  def __len__(self):
        return len(self.file_names)

  def __getitem__(self, idx):
        file_name = self.file_names[idx]

        if self.preload:
            midi_data, midi_notes = self.midi_data[file_name]
            wav_data, sr = self.wav_data[file_name]
            onset_offset_data = self.onset_offset_data[file_name]
        else:
            result = self.load_file(file_name)
            midi_data, midi_notes = result['midi_data']
            wav_data, sr = result['wav_data']
            onset_offset_data = result['onset_offset_data']

        wav_data = peak_normalize(wav_data)

        midi_audio_data = midi_data.synthesize() if self.synthesize else None

        return_data = {
            "midi_notes": np.array([x[2] for x in midi_notes]),
            "midi_durations": np.array([x[1] - x[0] for x in midi_notes]),
            "given_onsets": np.array([x[0] for x in midi_notes]),
            "given_offsets": np.array([x[1] for x in midi_notes]),
            "extracted_onsets": np.array([x[0] for x in onset_offset_data]),
            "extracted_offsets": np.array([x[1] for x in onset_offset_data]),
            "wav_data": wav_data,
            "sample_rate": sr,
            "file_name": file_name,
            "wav_length": len(wav_data),
            "midi_notes_whole": midi_notes,
        }

        if midi_audio_data is not None:
            return_data["midi_audio_data"] = midi_audio_data

        return return_data

  @staticmethod
  def extract_notes(midi_file):
      """
      Extracts notes and their onsets and offsets from a midi file. 
      Inspired by: https://github.com/shansongliu/HumTrans/blob/main/calc_transcription_eval_metric.py#L35
      """
      midi_data = pretty_midi.PrettyMIDI(midi_file)
      notes = []
      for instrument in midi_data.instruments:
          if instrument.is_drum:
              continue
          for note in instrument.notes:
              notes.append((note.start, note.end, note.pitch))
      return midi_data, notes


class CustomCollate:
    def __init__(self, is_validation, load_whole_audio, octave_invariant=True):
        self.is_validation = is_validation
        self.load_whole_audio = load_whole_audio
        self.octave_invariant = octave_invariant

    def __call__(self, batch):
        batched_wav_data = []
        batched_midi_notes = []
        batched_num_notes = []
        batched_wav_lengths = []
        batched_midi_durations = []
        batched_sample_rates = []
        batched_extracted_onsets = []
        batched_extracted_offsets = []
        batched_midi_labels = []

        max_wav_segment_len = 0
        max_midi_notes_len = 0
        max_midi_notes_durations_len = 0

        if self.is_validation:
          num_notes_for_batch = 5
        elif not self.load_whole_audio:
          num_notes_for_batch = random.randint(5, 10)
        else:
          note_lengths = [data["given_onsets"].shape[0] for data in batch]
          num_notes_for_batch = np.max(note_lengths)

        padding = 0 if self.load_whole_audio else random.randint(100, 1000)

        for data in batch:
            # Pad the wav data
            wav_len = data['wav_length'] + 2 * padding
            note_starts = (data["extracted_onsets"] + padding) / data["sample_rate"]
            note_ends = (data["extracted_offsets"] + padding) / data["sample_rate"]
            if padding > 0:
              data["wav_data"] = np.pad(data["wav_data"], (padding, padding), 'constant')

            # Select a random start index for 5 contiguous MIDI notes, ensuring we don't go out of bounds
            if len(note_starts) >= num_notes_for_batch:
                start_idx = random.randint(0, len(note_starts) - num_notes_for_batch)
            else:
                start_idx = 0
            end_idx = start_idx + num_notes_for_batch if start_idx + num_notes_for_batch <= len(note_ends) else len(note_ends)
            # Get the start and end time in the WAV file
            wav_start = int(note_starts[start_idx] * data["sample_rate"]) - padding
            wav_end = int(note_ends[end_idx-1] * data["sample_rate"]  + padding if end_idx > 0 else wav_len)
            # wav_end = min(wav_len, wav_end)

            # Update max length if this segment is longer
            max_wav_segment_len = max(max_wav_segment_len, wav_end - wav_start)

            # Get midi notes segment for this chunk
            midi_notes = data["midi_notes"][start_idx: end_idx]
            if self.octave_invariant:
              midi_notes = [x%12 for x in midi_notes]
              midi_notes = np.insert(midi_notes, np.arange(len(midi_notes) + 1), 12)
            else:
              midi_notes = np.insert(midi_notes, np.arange(len(midi_notes) + 1), 89)

            max_midi_notes_len = max(max_midi_notes_len, len(midi_notes))

            midi_durations = data["midi_durations"][start_idx: end_idx]
            max_midi_notes_durations_len = max(max_midi_notes_durations_len, len(midi_durations))

            data['wav_data'] = peak_normalize(data["wav_data"][wav_start: wav_end])
            data['midi_notes'] = midi_notes
            data['midi_durations'] = midi_durations
            data['wav_length'] = wav_end - wav_start
            data['num_notes'] = midi_notes.shape[0]
            time_base = (note_starts[start_idx]) - padding / data["sample_rate"]
            data['extracted_onsets'] =  note_starts[start_idx: end_idx] - time_base
            data['extracted_offsets'] =  note_ends[start_idx: end_idx] - time_base

        for data in batch:
            if data['wav_length'] < max_wav_segment_len:
                batched_wav_data.append(np.pad(data['wav_data'], (0, max_wav_segment_len - data['wav_data'].shape[0]), 'constant'))
            else:
                batched_wav_data.append(data['wav_data'])

            if data['num_notes'] < max_midi_notes_len:
                data['midi_notes'] = np.pad(data['midi_notes'],
                                            (0, max_midi_notes_len - data['midi_notes'].shape[0]), 'constant')
                data['extracted_onsets'] = np.pad(data['extracted_onsets'],
                                             (0, max_midi_notes_durations_len - data['extracted_onsets'].shape[0]), 'constant')
                data['extracted_offsets'] = np.pad(data['extracted_offsets'],
                                           (0, max_midi_notes_durations_len - data['extracted_offsets'].shape[0]), 'constant')
                batched_midi_durations.append(np.pad(data['midi_durations'], (0, max_midi_notes_durations_len - data['midi_durations'].shape[0]), 'constant'))
            else:
                batched_midi_durations.append(data['midi_durations'])

            # Create a one-hot encoded vector for MIDI notes
            if self.octave_invariant:
              midi_labels = np.zeros(13)
            else:
              midi_labels = np.zeros(90)

            midi_labels[data['midi_notes']] = 1
            batched_midi_labels.append(midi_labels)
            batched_midi_notes.append(data['midi_notes'])
            batched_num_notes.append(data["num_notes"])
            batched_wav_lengths.append(data["wav_length"])
            batched_sample_rates.append(data["sample_rate"])
            batched_extracted_onsets.append(data["extracted_onsets"])
            batched_extracted_offsets.append(data["extracted_offsets"])

        # Convert lists to PyTorch tensors
        batched_wav_data = np.stack(batched_wav_data)
        batched_wav_data = torch.FloatTensor(batched_wav_data)
        batched_midi_labels = torch.LongTensor(np.stack(batched_midi_labels))
        batched_midi_notes = torch.LongTensor(np.stack(batched_midi_notes))
        batched_midi_durations = torch.FloatTensor(np.stack(batched_midi_durations))
        batched_sample_rates = torch.FloatTensor(np.stack(batched_sample_rates))
        batched_wav_lengths = torch.FloatTensor(np.stack(batched_wav_lengths))
        batched_num_notes = torch.FloatTensor(np.stack(batched_num_notes))
        batched_extracted_onsets = torch.FloatTensor(np.stack(batched_extracted_onsets))
        batched_extracted_offsets = torch.FloatTensor(np.stack(batched_extracted_offsets))


        # modify note starts and note ends to reflect the dummy notes:
        # Calculate the number of notes
        num_notes = batched_extracted_onsets.shape[1]
        num_dummy_notes = 2 * num_notes + 1
        modified_note_starts = torch.zeros((batched_extracted_onsets.shape[0], num_dummy_notes))
        modified_note_ends = torch.zeros((batched_extracted_onsets.shape[0], num_dummy_notes))

        modified_note_starts[:, 1::2] = batched_extracted_onsets
        modified_note_ends[:, 1::2] = batched_extracted_offsets

        # Fill in the dummy note starts using torch operations
        modified_note_starts[:, 2::2] = batched_extracted_offsets

        # Fill in the dummy note ends using torch operations
        modified_note_ends[:, 0:2*num_notes:2] = batched_extracted_onsets[:, :]
        modified_note_ends[:, -1] = batched_wav_data.shape[1] / batched_sample_rates[0]

        return {
            'wav_data': batched_wav_data,
            'midi_notes': batched_midi_notes,
            "midi_labels": batched_midi_labels,
            'midi_durations': batched_midi_durations,
            'sample_rate': batched_sample_rates,
            'wav_lengths': batched_wav_lengths,
            'num_notes': batched_num_notes,
            'note_starts': modified_note_starts,
            'note_ends': modified_note_ends,
            'file_names': [data['file_name'] for data in batch],
            'dum_starts': batched_extracted_onsets,
            'dum_ends': batched_extracted_offsets
        }