# IE7374_Group6

# Artist Style Fusion with Stable Diffusion and LoRA

Our team’s goal is to explore how generative AI can be used to create novel artistic styles by blending influences from multiple iconic artists. In this project, we are using fine-tunes Stable Diffusion with LoRA to adapt to artists' styles, merges styles, and evaluates the results using CLIP.  We want to understand how different text inputs influence the visual characteristics of the output, and this serves as a first step toward our broader goal of generating entirely new, blended painting styles.


## Repository Structure
```
project-root/
├── src/
│   ├── __init__.py        # Make Directory a proper python package
│   ├── data_loader.py     # Data loading and preprocessing
│   ├── train.py           # LoRA fine-tuning
│   ├── model_runner.py    # Inference and evaluation
├── utils/
│   ├── __init__.py        # Make Directory a proper python package
│   ├── helpers.py         # Utility functions
├── configs/
│   ├── model_config.yaml  # Hyperparameters
├── outputs/
│   ├── lora_weights/      # LoRA adapter weights
│   ├── images/            # Generated images
├── data/
│   ├── processed/         # Training images (e.g., Pablo_Picasso_001.jpg)
├── Dockerfile             # Container definition
├── requirements.txt       # Dependencies
├── README.md              # This file
├── __init__.py            # Make Directory a proper python package
```

## Setup
1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
2. **Prepare Data**:
   - Place artist images in `data/processed/` with filenames like `Artist_Name_001.jpg`.
3. **Configure**:
   - Update `configs/model_config.yaml` with desired hyperparameters or adapter paths.

## Usage
1. **Train LoRA Adapters**:
   ```bash
   python src/train.py
   ```
   - Fine-tunes LoRA for each artist, saving weights to `outputs/lora_weights/`.
2. **Run Inference and Evaluation**:
   ```bash
   python src/model_runner.py
   ```
   - Generates base and fused images, evaluates CLIP scores, and saves images to `outputs/images/`.
3. **Docker**:
   ```bash
   docker build -t style-fusion .
   docker run -v $(pwd)/data:/app/data -v $(pwd)/outputs:/app/outputs style-fusion
   ```

## Requirements
- Python 3.10+
- GPU (CUDA or MPS) recommended for faster training/inference.
- Internet access for downloading pretrained models.

## Notes
- Ensure `data/processed/` contains images with consistent filename formats.
- Update `adapter_paths` in `model_config.yaml` for different artist combinations.
- Monitor memory usage on MPS devices; the code includes `torch.mps.empty_cache()` for optimization.