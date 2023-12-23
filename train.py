import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from constants import AUDIO_SAMPLE_RATE, FFT_HOP
from loader import CustomCollate, HumTransDataset
from model import HarmonicCQTConv2D, CQTConv1D
from postprocess import build_log_path_prob_matrix_with_path

import time

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Define a function to convert onsets/offsets to frame numbers
def convert_to_frames(values, hop_length=512):
    return (values * AUDIO_SAMPLE_RATE) // FFT_HOP


def expand_labels_to_frames(labels, frame_onsets, frame_offsets, cqt_mask):
  B, T = cqt_mask.shape

  # Create indices for selecting the relevant frames
  frame_indices = torch.arange(T).unsqueeze(0).to(device)

  # Expand labels to match the shape of frame_indices
  expanded_labels = labels.unsqueeze(2).expand(-1, -1, T)

  # Create a mask for selecting the frames within the onset and offset ranges
  mask = (frame_indices >= frame_onsets[:, :, None]) & (frame_indices < frame_offsets[:, :, None])

  # Use advanced indexing to assign labels to the selected frames
  expanded_labels = torch.where(mask, expanded_labels, torch.zeros_like(expanded_labels)).sum(-2)

  return expanded_labels


def run_model_for_batch(batch, criterion):
    wav_data = batch['wav_data'].to(device)
    wav_lengths = batch['wav_lengths'].to(device)
    onsets = batch['note_starts'].to(device)
    offsets = batch['note_ends'].to(device)

    frame_onsets = convert_to_frames(onsets)
    frame_offsets = convert_to_frames(offsets)

    outputs, cqt_mask = model(wav_data.float(), wav_lengths)

    raw_labels = batch['midi_notes'].to(device)
    labels = expand_labels_to_frames(raw_labels, frame_onsets, frame_offsets, cqt_mask).long()
    loss =  criterion(outputs.transpose(1, 2), labels)
    masked_loss = loss * cqt_mask
    mean_loss = masked_loss.sum() / cqt_mask.sum()
    return outputs, labels, mean_loss


# Define a validation loop
def validation_loop(model, dataloader, criterion, num_validation_steps=None):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0.0
    correct = 0
    total = 0

    num_elems = 0
    with torch.no_grad():
        for i, batch in tqdm(enumerate(dataloader)):
            outputs, labels, loss = run_model_for_batch(batch, criterion)
            total_loss += loss.item() * labels.size(0)

            _, predicted = torch.max(outputs, dim=-1)
            total += labels.size(0) * labels.size(1)
            correct += (predicted == labels).sum().item()

            num_elems += labels.size(0)

            if num_validation_steps is not None and i == num_validation_steps:
              break

    val_accuracy = 100 * correct / total
    val_loss = total_loss / num_elems

    wandb.log({"val": {
                "acc": val_accuracy,
                "loss": val_loss,
                "epoch": epoch}})

    model.train()
    return val_loss, val_accuracy



def visualize_validation_samples(
        model, 
        dataloader, 
        criterion, 
        save_dir, 
        num_samples):
    
    model.eval()  # Set the model to evaluation mode

    save_dir.mkdir(exist_ok=True, parents=True)

    with torch.no_grad():
        for i in range(num_samples):
            # Get a random sample from the dataloader
            batch = next(iter(dataloader))

            print(batch)

            # Convert the batch to numpy for visualization
            wav_data = batch['wav_data'].to(device)
            wav_lengths = batch['wav_lengths'].to(device)

            # print(batch["midi_notes"])
            outputs, labels, _ = run_model_for_batch(batch, criterion)
            # print("Model output: ", labels.shape, labels)

            spectrogram = torch.log1p(model.get_audio_rep(wav_data, wav_lengths)[0])[0].detach().cpu().numpy()


            _, predicted = torch.max(outputs, dim=-1)
            predicted = predicted[0].cpu().numpy()

            # Create a figure for visualization
            fig, ax1 = plt.subplots(figsize=(12, 6))
            ax2 = ax1.twinx()  # Create a secondary y-axis for predicted and labels

            ax1.imshow(spectrogram, cmap='viridis', aspect='auto', origin='lower',
                       extent=[0, spectrogram.shape[1], 0, spectrogram.shape[0]], alpha=0.8)
            ax1.set_ylabel('Chroma Features')
            ax1.set_xlabel('Frame Index')

            ax2.plot(predicted, label='Predicted', marker='o', color='red')
            ax2.plot(labels[0].cpu().numpy(), label='Ground Truth', linestyle='--', marker='x', color='blue')
            ax2.set_ylabel('Class')
            ax2.legend(loc='upper left')

            plt.title(f'Spectrogram with Predicted and Ground Truth Labels (Sample {i+1})')
            plt.savefig(save_dir/f"{i}_spec.png")
            plt.close()

            # Create a figure for visualization
            fig, ax1 = plt.subplots(figsize=(12, 6))
            # ax2 = ax1.twinx()  # Create a secondary y-axis for predicted and labels

            print("outputs: ", outputs.shape)
            ax1.imshow(torch.softmax(outputs, dim=-1).detach().cpu().numpy()[0].T, cmap='viridis', aspect='auto', origin='lower',
                       extent=[0, outputs.shape[1], 0, outputs.shape[2]], alpha=0.8)
            ax1.set_ylabel('Softmax of output')
            ax1.set_xlabel('Frame Index')

            ax1.plot(predicted, label='Predicted', marker='o', color='red', markersize=0.2)
            ax1.plot(labels[0].cpu().numpy(), label='Ground Truth', linestyle='--', marker='x', color='blue', markersize=0.2)
            ax1.set_ylabel('Class')
            ax1.legend(loc='upper left')

            plt.title(f'Spectrogram with Predicted and Ground Truth Labels (Sample {i+1})')
            plt.savefig(save_dir/f"{i}_out.png")
            plt.close()

            log_probs = torch.log_softmax(outputs, dim=-1).detach().cpu().numpy()[0].T
            _, path = build_log_path_prob_matrix_with_path(log_probs)

            # Convert path to a format suitable for plotting
            path_x, path_y = zip(*path)
            # fig, ax1 = plt.subplots(figsize=(12, 6))

            plt.imshow(log_probs, cmap='viridis', aspect='auto')
            plt.plot(path_y, path_x, color="red", marker='o',  markersize=0.2)  # Path coordinates are inverted for plotting (y, x)
            plt.plot(labels[0].cpu().numpy(), label='Ground Truth', linestyle='--', marker='x', color='blue', markersize=0.2)

            plt.colorbar(label='Log Affinity')
            plt.xlabel('Time Index')
            plt.ylabel('L Index')
            plt.title('Path Superimposed on Log Affinity Matrix')
            plt.savefig(save_dir/f"{i}_path.png")
            plt.close()
    model.train()

def get_model(model_class):
    if model_class == "HarmonicCQTConv2D":
        return HarmonicCQTConv2D(
            num_classes=args.num_classes,
            do_transforms=args.input_transforms_in_train)
    elif model_class == "CQTConv1D":
        return CQTConv1D(
            num_classes=args.num_classes, 
            do_transforms=args.input_transforms_in_train)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_path", type=Path)
    parser.add_argument("model_save_dir", type=Path)
    parser.add_argument("--model_load_path", type=Path)
    parser.add_argument("--viz_save_path", type=Path)
    parser.add_argument("--preload_train_dataset", action='store_true')
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--valid_batch_size", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_classes", type=int, default=90)
    parser.add_argument("--valid_step_interval", type=int, default=50)
    parser.add_argument("--num_validation_steps", type=int, default=10)
    parser.add_argument("--num_viz_samples", type=int, default=5)
    parser.add_argument("--input_transforms_in_train", action='store_true')
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--experiment_name", type=str, default="exp")
    parser.add_argument("--model_class", type=str, default="HarmonicCQTConv2D")

    args = parser.parse_args()

    # Initailize weights and biases
    print("Initializing weights and biases")
    wandb.init(
        project="piano_transcription",
        name=f"exp_{args.experiment_name}",
        settings=wandb.Settings(start_method="fork"))
    wandb.config.update(args)

    # Set seed values for reproducability:
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # get train and validation datasets and dataloaders
    train_dataset = HumTransDataset(
        args.dataset_path,
        split="train",
        preload=args.preload_train_dataset,
        filtered_file="filtered")
    
    valid_dataset = HumTransDataset(
        args.dataset_path, 
        split="valid",
        filtered_file="filtered_valid")
    
    print(f"Length of train_dataset: {len(train_dataset)}")
    print(f"Length of valid_dataset: {len(valid_dataset)}")

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=CustomCollate(is_validation=False, load_whole_audio=False, octave_invariant=False))
    
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=args.valid_batch_size,
        shuffle=True,
        collate_fn=CustomCollate(is_validation=True, load_whole_audio=False, octave_invariant=False))

    # Initialize the model
    model = get_model(args.model_class)
    print(model)
    model = model.to(device)

    print(f"Number of parameters in model: {count_parameters(model)}")


    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss(reduction='none', label_smoothing=args.label_smoothing)
    optimizer = optim.Adam(model.parameters(), lr=0.001)


    # Training loop
    for epoch in range(args.epochs):
        model.train()
        for i, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            outputs, labels, loss = run_model_for_batch(batch, criterion)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs, dim=-1)
            total = labels.size(0) * labels.size(1)
            correct = (predicted == labels).sum().item()
            print(f"Epoch: {epoch}, i: {i}, loss: {loss.item(): .4f}, Accuracy: {correct/total: .4f}")

            wandb.log({"train": {
                "acc": correct/total,
                "loss": loss.item(),
                "epoch": epoch}})

            if i % args.valid_step_interval == 0:
                loss, accuracy = validation_loop(
                    model, 
                    valid_dataloader, 
                    criterion, 
                    num_validation_steps=args.num_validation_steps)
                print(f"VALLLLLL Epoch: {epoch}, i: {i}, loss: {loss}, accuracy: {accuracy}")
                visualize_validation_samples(
                    model,
                    DataLoader(
                        valid_dataset,
                        batch_size=1,
                        collate_fn=CustomCollate(is_validation=True, load_whole_audio=False, octave_invariant=False)),
                    criterion,
                    args.viz_save_path,
                    num_samples=args.num_viz_samples)
                
                print("Saving model")
                checkpoint = {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'step': i,
                }

                args.model_save_dir.mkdir(exist_ok=True, parents=True)
                torch.save(checkpoint, args.model_save_dir / f"{args.experiment_name}.pt")

    print('Finished Training')






        

