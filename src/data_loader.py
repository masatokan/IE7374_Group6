import os
from glob import glob
from collections import defaultdict
import random
from torchvision import transforms
from PIL import Image
import logging

logger = logging.getLogger(__name__)

def get_image_transforms(resolution):
    """Define image transformation pipeline."""
    return transforms.Compose([
        transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(resolution),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

def extract_artist_from_filename(filename):
    """Extract artist name from filename."""
    base_name = os.path.splitext(os.path.basename(filename))[0]
    parts = base_name.split("_")
    return " ".join(parts[:-1])

def load_artist_dataset(dataset_dir="data/processed"):
    """Load images organized by artist."""
    try:
        artist_images = defaultdict(list)
        for image_path in glob(f"{dataset_dir}/*.jpg") + glob(f"{dataset_dir}/*.jpeg"):
            artist = extract_artist_from_filename(image_path)
            artist_images[artist].append(image_path)
        logger.info(f"Loaded {len(artist_images)} artists from {dataset_dir}")
        return artist_images
    except Exception as e:
        logger.error(f"Error loading artist dataset: {e}")
        raise

def load_dataset(dataset_dir="data/processed"):
    """Load dataset with prompts."""
    try:
        dataset = []
        artist_counts = defaultdict(int)
        for filename in os.listdir(dataset_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                file_path = os.path.join(dataset_dir, filename)
                artist_name = extract_artist_from_filename(filename)
                prompt = f"A man with a hat in the style of {artist_name}"
                dataset.append((file_path, prompt, artist_name))
                artist_counts[artist_name] += 1
        logger.info(f"Loaded dataset from {dataset_dir} with {len(dataset)} images")
        return dataset, artist_counts
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise

def split_dataset(dataset, validation_split=0.1):
    """Split dataset into training and validation sets."""
    try:
        random.shuffle(dataset)
        val_size = int(len(dataset) * validation_split)
        logger.info(f"Split dataset: {len(dataset) - val_size} training, {val_size} validation")
        return dataset[val_size:], dataset[:val_size]
    except Exception as e:
        logger.error(f"Error splitting dataset: {e}")
        raise