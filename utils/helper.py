import torch
import random
import numpy as np
import logging
from diffusers import StableDiffusionPipeline, DDIMScheduler
from peft import LoraConfig, get_peft_model

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def set_seed(seed=102):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    logger.info(f"Using seed: {seed}")

def setup_device():
    """Set up the computation device."""
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    return device

def load_pipeline(pretrained_model_path, device):
    """Load Stable Diffusion pipeline."""
    try:
        pipe = StableDiffusionPipeline.from_pretrained(
            pretrained_model_path, torch_dtype=torch.float16
        )
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe.to(device)
        pipe.vae.requires_grad_(False)
        pipe.text_encoder.requires_grad_(False)
        logger.info(f"Loaded pipeline from {pretrained_model_path}")
        return pipe
    except Exception as e:
        logger.error(f"Error loading pipeline: {e}")
        raise

def apply_lora(pipe, lora_rank):
    """Apply LoRA to the UNet."""
    try:
        config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_rank,
            target_modules=["to_q", "to_k", "to_v", "to_out.0"],
            lora_dropout=0.05,
            bias="none",
            use_dora=False
        )
        pipe.unet = get_peft_model(pipe.unet, config)
        logger.info(f"Applied LoRA with rank {lora_rank}")
        return pipe
    except Exception as e:
        logger.error(f"Error applying LoRA: {e}")
        raise