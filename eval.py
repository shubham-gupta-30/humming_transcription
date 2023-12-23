

import argparse
from pathlib import Path

import torch
import numpy as np

from model import HarmonicCQTConv2D
from loader import HumTransDataset
from postprocess import (
    build_log_path_prob_matrix_with_path,
    convert_extracted_notes_to_eval_format,
    get_argmax_path )

from mir_eval.transcription import precision_recall_f1_overlap
from constants import EVAL_TOLERANCE, OCTAVE_INVARIANT_RADIUS
from tqdm import tqdm
import pretty_midi
import shutil
import os


def unpack_if_needed(unpack_path, zip_path):
    if not unpack_path.exists():
        print(f"{unpack_path} does not exist, looking for zipped file")
        if not zip_path.exists():
            raise FileNotFoundError(f"{zip_path} also does not exist")
        else:
            print(f"Extracting {zip_path} to {unpack_path}")
            shutil.unpack_archive(zip_path, unpack_path)


def get_model(model_path):
    model = HarmonicCQTConv2D(num_classes=90)
    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()
    return model


def get_dataset(dataset_path, use_filtered_data):
    filtered_data_file  = "filtered_test" if use_filtered_data else None
    return HumTransDataset(
        dataset_path,
        split="test", 
        preload=False, 
        filtered_file=filtered_data_file)


def midi_to_pitch(m):
    return 440.0 * np.power(2, (float(m) - 69) / 12)


def extract_intervals_and_pitches_from_model(data, model, device, dummy_note):
    wav_data = torch.tensor(data["wav_data"])[None].float().to(device)
    wav_length = torch.tensor([len(data["wav_data"])]).to(device)

    outputs, _ = model(wav_data, wav_length)
    log_probs = torch.log_softmax(outputs, dim=-1).detach().cpu().numpy()[0].T
    L, T = log_probs.shape

    _, path = build_log_path_prob_matrix_with_path(log_probs)
    # path = get_argmax_path(log_probs)
    
    notes = [x[0] for x in path]
    # print(notes)

    intervals, notes = convert_extracted_notes_to_eval_format(notes, dummy_note)
    pitches = np.array([midi_to_pitch(m) for m in notes])
    return intervals, pitches, notes



def read_midi(midi_file):
    midi_data = pretty_midi.PrettyMIDI(midi_file)
    intervals = []
    notes = []
    for instrument in midi_data.instruments:
        if instrument.is_drum:
            continue
        for note in instrument.notes:
            intervals.append((note.start, note.end))
            notes.append(note.pitch)
    return intervals, notes


def extract_intervals_and_pitches_from_midi(file_name: str, midi_folder: Path):
    midi_file = os.path.join(midi_folder, file_name + ".mid")
    assert (midi_folder / (file_name + ".mid")).exists(), midi_file
    intervals, notes = read_midi(midi_file)
    intervals = np.array(intervals)
    pitches = np.array([midi_to_pitch(m) for m in notes])
    return intervals, pitches, notes


def extract_known_intervals_and_pitches(data, type):
    assert type in ["given", "extracted"], f"{type} not in ['given', 'extracted']"
    if type == "given":
        onsets = data["given_onsets"]
        offsets = data["given_offsets"]
    else:
        onsets = [x/44100.0 for x in data["extracted_onsets"]]
        offsets = [x/44100.0 for x in data["extracted_offsets"]]
    intervals = np.array(list(zip(onsets, offsets)))
    pitches = np.array([midi_to_pitch(m) for m in data["midi_notes"]])
    return intervals, pitches, data["midi_notes"]



def evaluate_octave_invariant(ref_intervals, ref_notes, est_intervals, est_pitches, onset_tolerance):
    octaves = list(range(-OCTAVE_INVARIANT_RADIUS, OCTAVE_INVARIANT_RADIUS + 1))
    ps = []
    rs = []
    f1s = []
    overlap_rs = []
    for o in octaves:
        ref_notes_shifted = (o * 12) + ref_notes
        ref_pitches = np.array([midi_to_pitch(x) for x in ref_notes_shifted])
        p, r, f1, overlap_r = precision_recall_f1_overlap(
            ref_intervals,
            ref_pitches,
            est_intervals,
            est_pitches,
            onset_tolerance = onset_tolerance,
            pitch_tolerance = 1.0,
            offset_ratio = None,
        )
        ps.append(p)
        rs.append(r)
        f1s.append(f1)
        overlap_rs.append(overlap_r)
    best_octave_idx = np.argmax(f1s)
    return (
        ps[best_octave_idx],
        rs[best_octave_idx],
        f1s[best_octave_idx],
        overlap_rs[best_octave_idx]
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=Path)
    parser.add_argument("dataset_path", type=Path)
    parser.add_argument("--dummy_note", type=int, default=89)
    parser.add_argument("--model_out_midi_zip", type=str, default=None)
    parser.add_argument("--onset_tolerance", type=float, default=EVAL_TOLERANCE)
    parser.add_argument("--ground_truth_type", type=str, default="extracted")
    parser.add_argument("--use_filtered_data", action='store_true')
    parser.add_argument("--octave_invariant", action='store_true')


    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)


    dataset = get_dataset(args.dataset_path, args.use_filtered_data)
    print(len(dataset))

    use_model = True
    if args.model_out_midi_zip is not None:
        print("Extracting the model_out midi folder")
        unpack_path = args.dataset_path / args.model_out_midi_zip
        zip_path = args.dataset_path / (args.model_out_midi_zip + ".zip")
        unpack_if_needed(unpack_path, zip_path)
        use_model = False

    if use_model:
        model = get_model(args.model_path)

    print(f"Using model: {use_model}")

    result = []
    for i, data in tqdm(enumerate(dataset)):
        if use_model:
            est_intervals, est_pitches, est_notes = extract_intervals_and_pitches_from_model(
                data, model, device, args.dummy_note)
        else:
            est_intervals, est_pitches, est_notes = extract_intervals_and_pitches_from_midi(
                data["file_name"], unpack_path / args.model_out_midi_zip / "test")
        ref_intervals, ref_pitches, ref_notes = extract_known_intervals_and_pitches(
            data, type=args.ground_truth_type)
        
        if args.octave_invariant:
            res = evaluate_octave_invariant(
                ref_intervals, ref_notes, est_intervals, est_pitches, onset_tolerance = args.onset_tolerance,)
        else:
            res = precision_recall_f1_overlap(
                ref_intervals, 
                ref_pitches, 
                est_intervals, 
                est_pitches, 
                onset_tolerance = args.onset_tolerance,
                pitch_tolerance = 1.0,
                offset_ratio = None,)
        result.append(res)
    
    result = np.array(result)
    print(np.mean(result, axis=0))

