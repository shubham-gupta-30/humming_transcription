# Introduction

This is repo containing code for our course project for [Machine Learning for Signal Processing](https://ycemsubakan.github.io/mlsp.html). We explore various methods for humming transcription - classical chroma feature based methods, HMM based methods and finally CNN based neural networks.

Our presentation and report are contained in the `documents/` folder.

# Dataset
The original HumTrans dataset that we use for all our methods can be downloaded from this link: https://huggingface.co/datasets/dadinghh2/HumTrans

# Corrected ground truth through semi-supervised learning
We realized early on that the dataset we are working with has incorrect onsets and offsets as ground truth. For classical methods, we correct the provided onsets in the original dataset using supervision from librosa's onset detection. These corrected onsets are in the folder `corrected_onsets_using_librosa/`. For our CNN based methods, we needed both precise onsets and offsets and so we came up with a semi supervised onset and offset correction mechanism, that provides more precise-ish onsets and offsets, at the expense of not being able to do so for around `45%` of the dataset, and thus forcing us to work  with a filteredc `55%` of the train, test and validation set. These filtered onsets and offsets so obtained are contained in the folder `heurestic_filtered_onsets_offsets/`.

# Chroma based and HMM based methods
The notebooks contained in the `notebook/` folder contain all the explorations related to the chroma features and HMM based offset, onset and note detection for the HumTrans dataset.

# CNN based method
## Model checkpoint
The trained model checkpoint we present in our report is at `checkpoints/harmonic_cqt.pt`. This is trained for `3` epochs, as we stopped seeing any further improvements in tracked metrics and the network seems to converge. We also experiemnted with a smaller model, which we do not include in our report, and the checkpoint for that is located at `checkpoints/cqt_1d.pt`.

## Data Loader
The loader for the humtrans  dataset is located in `loader.py`. To only work with the filtered dataset with correct onsets, just transfer the zip files from the `heurestic_filtered_onsets_offsets` folder in  the main `HumTrans` dataset folder and depending on the split you desire, you can provide the appropriate file name  while instantiating the `HumTransDataset` class. This dataset class and the accompanying collate function `CustomCollate` for creating a DataLoader has support for a lot of useful functionality:
- Its possible to preload the audio files for faster training (if there is enough CPU memory)
- Its possible to provide different flags for specifying a train mode (only small random number of notes are loaded), validation mode (randomization is turned off, and the first 5 notes are returned pr sample), or load whole audio instead of few notes for inference purposes.

## Training 
The simplest way to train the model is:

```
python train.py /path/to/dataset /path/to/model/checkpoint/
```

## Evaluation
To obtain the eval results in our report for our CNN based model, we use two major flags. The flag `octave_invariant` when set, does octave invariant analysis but when absent, does an octave aware analysis. The default value of the `onset_tolerance` flag  analyses both notes + onsets accuracy. To do a notes only accuracy analysis, set this to a large value, we set it to `5` for the results in our report. Hence, 

So for example, to obtain octave invariant notes only analysis for a model checkpoint use:
```
python eval.py /path/to/model/checkpoint /path/to/dataset --onset_tolerance 5 --octave_invariant --use_filtered_data
```
All other results in our report table can be obtained by plating around with the 2 flags we mentioned above. the flag `use_filtered_data` just means that we are only evaluating over the filtered test set we created. We evaluated our model over the entrie test set, and not just the filtered test set and compared it with other methods, and did not see significant change in performance. Due to space constraints in the report, we oomit that discussion there. 


## Contact
Please feel free to contact the authors for any further clarifications.
