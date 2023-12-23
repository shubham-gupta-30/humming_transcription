

import argparse
import os
import shutil
from pathlib import Path

import tensorflow as tf
from basic_pitch import ICASSP_2022_MODEL_PATH
from basic_pitch.inference import predict
from tqdm import tqdm

from loader import HumTransDataset

# import basic_pitch

def get_dataset(dataset_path, split):
    return HumTransDataset(
        dataset_path,
        split=split, 
        preload=False, 
        filtered_file=None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_path", type=Path)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--midi_save_folder", type=str, default="basic_pitch")

    args = parser.parse_args()

    print("Loading dataset")
    dataset = get_dataset(args.dataset_path, args.split)
    print(len(dataset))

    midi_save_root = args.dataset_path / args.midi_save_folder 
    midi_save_path = midi_save_root / args.midi_save_folder / args.split
    midi_save_path.mkdir(exist_ok=True, parents=True)

    print(midi_save_path)
    print("Loading model")
    model = tf.saved_model.load(str(ICASSP_2022_MODEL_PATH))

    print("Model loaded")

    for fname in tqdm(dataset.file_names):
        file_path = os.path.join(dataset.wav_path, fname + ".wav")
        midi_path = os.path.join(midi_save_path, fname + ".mid") 
        _, midi_data, _ = predict(file_path, model)
        midi_data.write(str(midi_path))

    shutil.make_archive(midi_save_root, 'zip', midi_save_root)
    dst_folder = Path("/home/shubhamg/projects/def-ravanelm/datasets/HumTrans")
    shutil.move(str(midi_save_root) + ".zip", str(dst_folder))
        
