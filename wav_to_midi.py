import argparse
from pathlib import Path

import librosa
import pretty_midi
import torch

from constants import AUDIO_SAMPLE_RATE
from eval import extract_intervals_and_pitches_from_model, get_model


def create_midi_file(intervals, pitches, file_name):
    """
    Creates a MIDI file from given intervals and pitches.

    This function generates a MIDI file using the pretty_midi library. It creates a MIDI track using a piano instrument
    and adds notes to this track based on the provided intervals and pitches. Each note is added with a start time,
    end time, and pitch derived from the intervals and pitches. The resulting MIDI file is saved to the specified file path.

    Parameters
    ----------
    intervals : list of tuples or ndarray
        A list or array of tuples where each tuple represents the start and end times (in seconds) of a note.
        Example: [(start_time1, end_time1), (start_time2, end_time2), ...]

    pitches : list or ndarray
        A list or array of pitch values corresponding to each interval. The pitch values should be MIDI note numbers.
        Example: [pitch1, pitch2, ...]

    file_name : str
        The path (including the file name) where the MIDI file will be saved. The file name should end with '.mid'.

    Examples
    --------
    >>> intervals = [(0.5, 1.0), (1.5, 2.0)]
    >>> pitches = [60, 62]
    >>> create_midi_file(intervals, pitches, 'example.mid')

    Notes
    -----
    The function assumes that the time units for the intervals are in seconds and that the pitches are in the MIDI note number format.
    """
    
    # Create a PrettyMIDI object
    midi_file = pretty_midi.PrettyMIDI()

    # Create an Instrument instance for a piano (or choose another instrument)
    piano_program = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
    piano = pretty_midi.Instrument(program=piano_program)

    # Add notes
    for interval, pitch in zip(intervals, pitches):
        # Create a Note instance for each note, specifying start time, end time, and pitch
        start_time = interval[0]  # Start time in seconds
        end_time = interval[1]    # End time in seconds
        note = pretty_midi.Note(
            velocity=100,          
            pitch=pitch,           
            start=start_time, 
            end=end_time
        )
        piano.notes.append(note)

    # Add the instrument to the PrettyMIDI object
    midi_file.instruments.append(piano)

    # Write out the MIDI data to a file
    midi_file.write(file_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=Path)
    parser.add_argument("wav_path", type=Path)
    parser.add_argument("midi_out_path", type=Path)
    parser.add_argument("--dummy_note", type=int, default=89)


    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    model = get_model(args.model_path)

    wav_data, _ = librosa.load(args.wav_path, sr=AUDIO_SAMPLE_RATE)
    data = {'wav_data': wav_data}
    intervals, pitches, notes = extract_intervals_and_pitches_from_model(data, model, device, args.dummy_note)
    create_midi_file(intervals, notes, args.midi_out_path)

