import os
import torch
import random
import yaml
from diffusers import StableDiffusionPipeline
from peft import PeftModel
from transformers import CLIPModel, CLIPProcessor
from tqdm.auto import tqdm
from src.data_loader import load_dataset
from utils.helper import setup_device, load_pipeline
import logging

logger = logging.getLogger(__name__)

def merge_lora_adapters(base_unet, adapter_paths, weights):
    """Merge multiple LoRA adapters into the UNet."""
    try:
        assert len(adapter_paths) == len(weights), "Mismatch in number of adapters and weights"
        merged = None
        for path, w in zip(adapter_paths, weights):
            if not (os.path.exists(path) and os.path.exists(os.path.join(path, 'adapter_config.json'))):
                raise ValueError(f"Adapter path '{path}' does not exist or is missing 'adapter_config.json'")
            temp = PeftModel.from_pretrained(base_unet, path)
            temp_state = temp.state_dict()
            if merged is None:
                merged = {k: w * v.clone() for k, v in temp_state.items()}
            else:
                for k in merged:
                    merged[k] += w * temp_state[k]
        base_unet.load_state_dict(merged, strict=False)
        logger.info(f"Merged LoRA adapters: {adapter_paths}")
        return base_unet
    except Exception as e:
        logger.error(f"Error merging LoRA adapters: {e}")
        raise

def load_fused_lora_pipeline(base_model="stabilityai/sd-turbo", adapter_paths=None, weights=None):
    """Load a pipeline with fused LoRA adapters."""
    try:
        if not adapter_paths or not weights:
            raise ValueError("You must provide adapter_paths and weights for fusion")
        device = setup_device()
        base_pipe = load_pipeline(base_model, device)
        base_pipe.unet = merge_lora_adapters(base_pipe.unet, adapter_paths, weights)
        base_pipe.unet.eval()
        logger.info(f"Loaded fused pipeline with adapters: {adapter_paths}")
        return base_pipe
    except Exception as e:
        logger.error(f"Error loading fused pipeline: {e}")
        raise

@torch.no_grad()
def generate_images(pipeline, prompts, output_dir, label="fusion"):
    """Generate images using the pipeline."""
    try:
        os.makedirs(output_dir, exist_ok=True)
        generated = {}
        for i, prompt_tuple in enumerate(tqdm(prompts, desc=f"Generating with {label}")):
            prompt = prompt_tuple[0] if isinstance(prompt_tuple, tuple) else prompt_tuple
            if not isinstance(prompt, str):
                raise ValueError(f"Prompt at index {i} is not a string: {prompt}")
            image = pipeline(prompt, num_inference_steps=25, guidance_scale=7.5).images[0]
            filename = os.path.join(output_dir, f"{label}_{i}.png")
            image.save(filename)
            generated[prompt] = image
        logger.info(f"Generated images saved to {output_dir}")
        return generated
    except Exception as e:
        logger.error(f"Error generating images: {e}")
        raise

def load_clip_model():
    """Load CLIP model and processor."""
    try:
        device = setup_device()
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        logger.info("Loaded CLIP model and processor")
        return model, processor
    except Exception as e:
        logger.error(f"Error loading CLIP model: {e}")
        raise

@torch.no_grad()
def compute_clip_score(clip_model, clip_processor, image, text):
    """Compute CLIP similarity score."""
    try:
        inputs = clip_processor(text=[text], images=image, return_tensors="pt", padding=True).to(clip_model.device)
        outputs = clip_model(**inputs)
        return torch.cosine_similarity(outputs.image_embeds, outputs.text_embeds).item()
    except Exception as e:
        logger.error(f"Error computing CLIP score: {e}")
        raise

@torch.no_grad()
def score_clip_fusion(clip_model, clip_processor, images_dict, style_descriptions):
    """Score fused images against individual style prompts."""
    try:
        fusion_scores = {}
        for prompt, image in images_dict.items():
            _, style_prompts = next((p for p in style_descriptions if p[0] == prompt), (None, []))
            if not style_prompts:  # Skip if no style prompts
                logger.debug(f"No style prompts for {prompt}, skipping fusion scoring")
                fusion_scores[prompt] = {"style_scores": {}, "fusion_score": 0.0}
                continue
            scores = [compute_clip_score(clip_model, clip_processor, image, s) for s in style_prompts]
            if not scores or any(s <= 0 for s in scores):
                logger.warning(f"Invalid scores for {prompt}: {scores}, setting fusion score to 0.0")
                fusion_scores[prompt] = {"style_scores": dict(zip(style_prompts, scores)), "fusion_score": 0.0}
            else:
                harmonic = len(scores) / sum(1.0 / s for s in scores)
                fusion_scores[prompt] = {
                    "style_scores": dict(zip(style_prompts, scores)),
                    "fusion_score": harmonic
                }
        logger.info("Computed CLIP fusion scores")
        return fusion_scores
    except Exception as e:
        logger.error(f"Error scoring CLIP fusion: {e}")
        raise

@torch.no_grad()
def score_clip_similarity(clip_model, clip_processor, images_dict):
    """Score images against their prompts."""
    try:
        scores = {}
        for prompt, image in images_dict.items():
            score = compute_clip_score(clip_model, clip_processor, image, prompt)
            scores[prompt] = score
        logger.info("Computed CLIP similarity scores")
        return scores
    except Exception as e:
        logger.error(f"Error scoring CLIP similarity: {e}")
        raise

def evaluate_pipeline(adapter_paths, weights, base_model="stabilityai/sd-turbo", prompts=None):
    """Evaluate base and fused models."""
    try:
        if prompts is None:
            prompts = [
                ("A man with a hat in the fusion style of Dali and Monet", ["A painting in the style of Dali", "A painting in the style of Monet"]),
                ("A man with a hat in the style of Picasso and Van Gogh", ["A painting in the style of Picasso", "A painting in the style of Van Gogh"])
            ]
        fusion_name = "+".join([os.path.basename(p) for p in adapter_paths])
        logger.info(f"Evaluating fusion: {fusion_name}")

        base_output_dir = "outputs/images/base_outputs"
        fusion_output_dir = os.path.join("outputs/images/fusion_outputs", fusion_name)

        base_pipe = load_pipeline(base_model, setup_device())
        base_images = generate_images(base_pipe, prompts, base_output_dir, label="base")

        lora_pipe = load_fused_lora_pipeline(base_model, adapter_paths, weights)
        fusion_images = generate_images(lora_pipe, prompts, fusion_output_dir, label=f"fusion_{fusion_name}")

        clip_model, clip_processor = load_clip_model()
        base_scores = score_clip_similarity(clip_model, clip_processor, base_images)
        fusion_scores = score_clip_similarity(clip_model, clip_processor, fusion_images)
        fusion_detailed_scores = score_clip_fusion(clip_model, clip_processor, fusion_images, prompts)

        logger.info("\nCLIP Prompt Alignment Comparison:")
        for full_prompt, _ in prompts:
            base = base_scores[full_prompt]
            fusion = fusion_scores[full_prompt]
            logger.info(f"Prompt: {full_prompt}")
            logger.info(f"  Base Score: {base:.4f}")
            logger.info(f"  Fusion Score: {fusion:.4f} {'Better' if fusion > base else 'Worse'}")
            logger.info(f"  Detailed Fusion Scores: {fusion_detailed_scores[full_prompt]['style_scores']}")
            logger.info(f"  Harmonic Fusion Score: {fusion_detailed_scores[full_prompt]['fusion_score']:.4f}")

        avg_base = sum(base_scores.values()) / len(base_scores)
        avg_fusion = sum(fusion_scores.values()) / len(fusion_scores)
        logger.info(f"\nAverage CLIP Score → Base: {avg_base:.4f} | Fusion ({fusion_name}): {avg_fusion:.4f}")
    except Exception as e:
        logger.error(f"Error in evaluation pipeline: {e}")
        raise

def main():
    """Main function to run inference on 5–10 samples."""
    try:
        # Load configuration
        with open("configs/model_config.yaml", "r") as f:
            config = yaml.safe_load(f)
        
        base_model = config["model"]["base_model"]
        adapter_paths = config["fusion"]["adapter_paths"]
        weights = config["fusion"]["weights"]
        fusion_name = "+".join([os.path.basename(p) for p in adapter_paths])
        output_dir = os.path.join("outputs/images/fusion_outputs", fusion_name)

        # Load dataset and select prompts
        dataset, _ = load_dataset("data/processed")
        dataset_prompts = [prompt for _, prompt, _ in dataset]
        random.shuffle(dataset_prompts)
        num_samples = min(10, max(5, len(dataset_prompts)))  # 5–10 samples
        selected_dataset_prompts = dataset_prompts[:num_samples // 2]
        
        # Combine with predefined prompts
        predefined_prompts = [
            ("A man with a hat in the fusion style of Dali and Monet", ["A painting in the style of Dali", "A painting in the style of Monet"]),
            ("A man with a hat in the style of Picasso and Van Gogh", ["A painting in the style of Picasso", "A painting in the style of Van Gogh"])
        ]
        all_prompts = [(p, []) for p in selected_dataset_prompts] + predefined_prompts
        all_prompts = all_prompts[:10]  # Ensure 5–10 prompts

        # Load model and generate images
        pipeline = load_fused_lora_pipeline(base_model, adapter_paths, weights)
        generate_images(pipeline, all_prompts, output_dir, label=f"fusion_{fusion_name}")

        # Optional: Run evaluation
        logger.info("Running evaluation...")
        evaluate_pipeline(adapter_paths, weights, base_model, all_prompts)
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise

if __name__ == "__main__":
    main()