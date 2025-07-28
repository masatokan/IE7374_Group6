import os
import torch
import random
from PIL import Image
from tqdm.auto import tqdm
from accelerate import Accelerator
from diffusers import StableDiffusionPipeline
from peft import LoraConfig, get_peft_model
from utils.helper import setup_device, load_pipeline, apply_lora
from src.data_loader import get_image_transforms, load_artist_dataset
import logging

logger = logging.getLogger(__name__)

def train_lora_for_artist(artist_name, image_paths, base_pipeline, output_dir, config, sample_size=None):
    """Fine-tune LoRA for a specific artist, optionally using a sample of images."""
    try:
        device = setup_device()
        unet = get_peft_model(base_pipeline.unet, LoraConfig(
            r=config["lora_rank"],
            lora_alpha=config["lora_rank"],
            target_modules=["to_q", "to_k", "to_v", "to_out.0"],
            lora_dropout=0.05,
            bias="none",
            use_dora=False
        ))
        unet.train()
        preprocess = get_image_transforms(config["resolution"])
        accelerator = Accelerator()
        optimizer = torch.optim.AdamW(unet.parameters(), lr=config["learning_rate"])
        prompt_template = f"A painting in the style of {artist_name}, a man with a hat"

        # Sample images if specified
        if sample_size is not None and len(image_paths) > sample_size:
            image_paths = random.sample(image_paths, sample_size)
            logger.info(f"Sampled {sample_size} images for {artist_name}")

        for epoch in range(config["num_epochs"]):
            random.shuffle(image_paths)
            for img_path in tqdm(image_paths, desc=f"[{artist_name}] Epoch {epoch+1}"):
                image = Image.open(img_path).convert("RGB")
                img_tensor = preprocess(image).unsqueeze(0).to(device, dtype=torch.float16)
                with torch.no_grad():
                    latents = base_pipeline.vae.encode(img_tensor).latent_dist.sample() * base_pipeline.vae.config.scaling_factor
                timesteps = torch.randint(0, base_pipeline.scheduler.config.num_train_timesteps, (1,), device=latents.device).long()
                noise = torch.randn_like(latents)
                noisy_latents = base_pipeline.scheduler.add_noise(latents, noise, timesteps)
                with torch.no_grad():
                    text_inputs = base_pipeline.tokenizer(prompt_template, return_tensors="pt").input_ids.to(device)
                    text_embeddings = base_pipeline.text_encoder(text_inputs)[0]
                noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states=text_embeddings).sample
                loss = torch.nn.functional.mse_loss(noise_pred, noise)
                optimizer.zero_grad()
                accelerator.backward(loss)
                optimizer.step()
                if device == "mps":
                    torch.mps.empty_cache()

        save_path = os.path.join(output_dir, artist_name.replace(" ", "_"))
        os.makedirs(save_path, exist_ok=True)
        unet.save_pretrained(save_path, safe_serialization=False)
        logger.info(f"[{artist_name}] LoRA saved to {save_path}")
        return unet
    except Exception as e:
        logger.error(f"Error training LoRA for {artist_name}: {e}")
        raise

def run_all_lora_training(dataset_dir="data/processed", output_dir="outputs/lora_weights", sample_size=None):
    """Run LoRA training for all artists, optionally using a sample of images."""
    try:
        os.makedirs(output_dir, exist_ok=True)
        artist_datasets = load_artist_dataset(dataset_dir)
        logger.info(f"Found {len(artist_datasets)} artists")
        for artist, images in artist_datasets.items():
            logger.info(f"{artist}: {len(images)} images")

        config = {
            "learning_rate": 1e-4,
            "num_epochs": 10,
            "lora_rank": 4,
            "resolution": 256,
        }
        base_model = "stabilityai/sd-turbo"
        device = setup_device()
        pipe = load_pipeline(base_model, device)

        for artist_name, image_paths in artist_datasets.items():
            train_lora_for_artist(artist_name, image_paths, pipe, output_dir, config, sample_size=sample_size)

        logger.info("All artists fine-tuned and saved")
        logger.info("----- Training Summary -----")
        logger.info(f"Total Artists: {len(artist_datasets)}")
        for artist, images in artist_datasets.items():
            logger.info(f"{artist}: {len(images)} images → LoRA saved in '{output_dir}/{artist.replace(' ', '_')}'")
    except Exception as e:
        logger.error(f"Error in training orchestration: {e}")
        raise