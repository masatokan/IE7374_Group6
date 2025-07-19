# The purpose of this script is to load the dataset for training/finetuning the model.

# Import necessary libraries

import os
import torch
import random
import numpy as np
from torchvision import transforms
from collections import defaultdict
from glob import glob

# ------------------- Configuration Parameters -------------------

def set_seed(seed=102):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    print(f"Using seed: {seed}")

# ------------------- Data Preprocessing & Loading -------------------

# def get_image_transforms(resolution):
#     return transforms.Compose([
#         transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
#         transforms.CenterCrop(resolution),
#         transforms.RandomHorizontalFlip(p=0.5),
#         transforms.RandomRotation(degrees=10),
#         transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
#         transforms.ToTensor(),
#         transforms.Normalize([0.5], [0.5]),
#     ])

def extract_artist_from_filename(filename):
    base_name = os.path.splitext(os.path.basename(filename))[0]
    parts = base_name.split("_")
    return " ".join(parts[:-1])  # e.g., "Vincent van Gogh"

def load_artist_dataset(dataset_dir):
    artist_images = defaultdict(list)
    for image_path in glob(f"{dataset_dir}/*.jpg") + glob(f"{dataset_dir}/*.jpeg"):
        artist = extract_artist_from_filename(image_path)
        artist_images[artist].append(image_path)
    return artist_images

def load_dataset(dataset_path):
    dataset = []
    artist_counts = defaultdict(int)

    for filename in os.listdir(dataset_path):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            file_path = os.path.join(dataset_path, filename)
            base_name = os.path.splitext(filename)[0]
            parts = base_name.split("_")
            artist_name = " ".join(parts[:-1])
            ## Excluded these lines. No prompts are being added to the dataset.
            # prompt = f"A painting in the style of {artist_name}, an apple"
            # dataset.append((file_path, prompt, artist_name))
            dataset.append((file_path, artist_name))
            artist_counts[artist_name] += 1

    return dataset, artist_counts


def split_dataset(dataset, validation_split=0.1):
    random.shuffle(dataset)
    val_size = int(len(dataset) * validation_split)
    return dataset[val_size:], dataset[:val_size]

def data_loader():
    dataset_dir = '../data/processed'
    artist_datasets = load_artist_dataset(dataset_dir)
    print(f"Found {len(artist_datasets)} artists.")
    for artist, images in artist_datasets.items():
        print(f"{artist}: {len(images)} images")
    
    dataset, artist_counts = load_dataset(dataset_dir)
    train_dataset, validation_dataset = split_dataset(dataset)

    return train_dataset, validation_dataset