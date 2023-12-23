import numpy as np

from constants import FFT_HOP, AUDIO_SAMPLE_RATE

def get_argmax_path(log_affinity):
    """
    Computes a simple path by selecting the note with the maximum probability at each time frame from the log affinity matrix.

    Parameters
    ----------
    log_affinity : ndarray
        A 2D numpy array of shape (L, T) where T is the number of time frames and L is the number of prediction classes.
        Each element represents the log affinity score for a class at a given time frame.

    Returns
    -------
    path : list of tuples
        A list representing the path, where each tuple (i, t) denotes the note i with the highest probability at time frame t.
    """
    # Find the note with the maximum probability at each time frame
    max_indices = np.argmax(log_affinity, axis=0)
    path = [(note, t) for t, note in enumerate(max_indices)]

    return path


def build_log_path_prob_matrix_with_path(log_affinity):
    """
    Compute the most probable path in a given log affinity matrix using dynamic programming. 
    The path represents the sequence of notes from the first to the last timestamp, with constraints 
    on the transitions between notes.

    The constraints are:
    - The first and the last timestamps should have the same note assigned, which is the dummy note (L-1).
    - If the note at time t is k, then for time t+1:
        - Any note is allowed if k == L-1.
        - Only the same note k or the dummy note L-1 are allowed if k != L-1.

    Parameters
    ----------
    log_affinity : ndarray
        A 2D numpy array of shape (L, T) where T is the number of time frames and L is the number of prediction classes.
        Each element represents the log affinity score for a class at a given time frame.

    Returns
    -------
    prob_table : ndarray
        A 2D numpy array of shape (L, T) representing the probability table used in dynamic programming. Each element 
        at (i, j) represents the log probability of the most probable path ending with note i at time j.

    path : list of tuples
        A list representing the most probable path, where each tuple (i, j) denotes the note i at time frame j. The path
        is returned in chronological order, starting from the first timestamp and ending at the last.

    Notes
    -----
    - The function assumes that the last class (L-1) is the dummy note.
    - The path is initially calculated in reverse (from last to first timestamp) and then reversed to present it 
      from first to last timestamp.
    """
    L, T = log_affinity.shape
    prob_table = np.full((L, T), float('-inf'))
    path_matrix = np.zeros((L, T, 2), dtype=int)

    # Base case
    prob_table[L-1, 0] = 0

    # Iterating over columns
    for j in range(1, T):
        # Vectorized computation for non-last rows
        non_last_rows = np.arange(L-1)
        max_vals = np.maximum(prob_table[non_last_rows, j-1], prob_table[L-1, j-1])
        prob_table[non_last_rows, j] = max_vals + log_affinity[non_last_rows, j]
        path_matrix[non_last_rows, j] = np.vstack([non_last_rows, np.full(L-1, j-1)]).T
        path_matrix[non_last_rows, j, 0] = np.where(prob_table[non_last_rows, j-1] == max_vals, non_last_rows, L-1)

        # Computation for last row, still iterative
        max_index = np.argmax(prob_table[:, j-1])
        prob_table[L-1, j] = prob_table[max_index, j-1] + log_affinity[L-1, j]
        path_matrix[L-1, j] = (max_index, j-1)

    # Retrace path
    path = []
    current_pos = (L-1, T-1)
    while current_pos[1] != 0:
        path.append(current_pos)
        current_pos = tuple(path_matrix[current_pos])

    path.append(current_pos)
    path = path[::-1]
    path = clean_path(path, log_affinity, L-1)
    return prob_table, path



def frame_to_sec(n):
    return float(n * FFT_HOP)/float(AUDIO_SAMPLE_RATE)



def clean_path(path, log_affinity, dummy_note, min_dummy_duration=3):
    """
        Cleans a given path by removing spurious occurrences of a specified dummy note. This function ensures that each
        instance of the dummy note in the path, except at the beginning and end, persists for at least a minimum number
        of frames. If a dummy note sequence is shorter than this minimum and not at the start or end, it is considered
        spurious and replaced with the most probable note according to a log affinity matrix.

        The replacement note is chosen based on the highest cumulative log affinity score in the segment where the 
        spurious dummy note is detected.

        Parameters
        ----------
        path : list of tuples
            A list representing the path, where each tuple (note, frame) denotes the note at a given time frame.
            The path is assumed to start and end with the dummy note.

        log_affinity : ndarray
            A 2D numpy array of shape (L, T) where L is the number of notes (including the dummy note) and T is the 
            number of time frames. Each element represents the log affinity score for a note at a given time frame.

        dummy_note : int
            The note value that is used as a dummy note.

        min_dummy_duration : int, optional
            The minimum number of frames for a dummy note sequence to be considered valid (default is 3).

        Returns
        -------
        cleaned_path : list of tuples
            A list representing the cleaned path with spurious dummy notes replaced. Each tuple (note, frame) denotes 
            the note at a given time frame.

        Notes
        -----
        - The function preserves the initial and final dummy notes, regardless of their duration.
        - Spurious dummy notes are identified as sequences of the dummy note that are shorter than `min_dummy_duration`
        and are not at the start or end of the path.
        - The replacement for spurious dummy notes is determined based on the log affinity scores in the corresponding 
        segment of the path.
    """
    cleaned_path = []
    segment_start = 0
    i = 0

    # Preserve initial dummy notes
    while i < len(path) and path[i][0] == dummy_note:
        cleaned_path.append(path[i])
        i += 1
    segment_start = i  # Start of the first non-dummy note segment

    spurious_dummy_note_detected = False

    while i < len(path):
        if path[i][0] == dummy_note:
            # Process a dummy note segment
            start = i
            while i < len(path) and path[i][0] == dummy_note:
                i += 1
            end = i

            if end - start < min_dummy_duration and end < len(path):
                # Spurious dummy note detected
                spurious_dummy_note_detected = True
                continue
            else:
                # Non-spurious dummy note sequence or end of sequence
                if spurious_dummy_note_detected:
                    # Replace segment with best note if spurious dummy note was detected
                    segment = range(segment_start, start)
                    best_note = np.argmax(log_affinity[:, segment].sum(axis=1))
                    for j in segment:
                        cleaned_path.append((best_note, j))
                else:
                    # Append original notes if no spurious dummy note was detected
                    cleaned_path.extend(path[segment_start:start])
                
                spurious_dummy_note_detected = False

                # Append the dummy note sequence as it is
                cleaned_path.extend(path[start:end])
                segment_start = end
        else:
            i += 1

    return cleaned_path



def convert_extracted_notes_to_eval_format(extracted_notes, dummy_note_value):
    """
    Convert a list of detected notes for each time frame into intervals and corresponding notes, excluding a specified dummy note.

    Parameters
    ----------
    extracted_notes : ndarray
        An array of shape (T,) where T is the number of time frames. Each element in the array represents
        a detected note at the corresponding time frame.

    hop_length : int
        The hop length used for the creation of the input representation, used for conversion to time frames.

    dummy_note_value : int
        The value used to represent the dummy note, which indicates the beginning, end, and spaces between notes.

    Returns
    -------
    intervals : ndarray
        An array of shape (N, 2), where N is the number of notes (excluding the dummy note). Each row in the array contains 
        [start time, end time] of a note in seconds.

    notes : ndarray
        An array of shape (N,) containing the detected notes (excluding the dummy note).
    """
    notes = []
    intervals = []
    start_frame = None

    for i, note in enumerate(extracted_notes):
        if note != dummy_note_value and start_frame is None:
            # Note onset
            start_frame = i
        elif (note == dummy_note_value or i == len(extracted_notes) - 1) and start_frame is not None:
            # Note offset
            end_frame = i
            intervals.append([frame_to_sec(start_frame), frame_to_sec(end_frame)])
            notes.append(extracted_notes[start_frame])
            start_frame = None

    # Convert lists to numpy arrays
    intervals = np.array(intervals)
    notes = np.array(notes)

    return intervals, notes
