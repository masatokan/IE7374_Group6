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

### Running the Web App
- Start the Flask web app with WebSocket support for real-time progress updates:
  ```bash
  python app.py
  ```
- The app will automatically open `http://localhost:5001` in your default browser.
- Select two artists, enter a prompt, and click "Generate Image" to create fused-style images.
- A progress bar and status messages will display during training and generation.
- Generated images are saved in `outputs/images/web_outputs/` and displayed in the browser.

### Running the Pipeline via Notebook
- Run the demo notebook for an interactive walkthrough:
  ```bash
  cd notebooks
  jupyter notebook demo_pipeline.ipynb
  ```

### Outputs
- **LoRA Weights**: Saved in `outputs/lora_weights/<artist_name>/` (e.g., `outputs/lora_weights/Pablo_Picasso/`).
- **Generated Images**: Saved in `outputs/images/fusion_outputs/<fusion_name>/` (notebook) or `outputs/images/web_outputs/<fusion_name>/` (web app).
- **Logs**: Evaluation results (CLIP scores) are logged to the console or `outputs/samples.txt` if redirected.

### Docker
Build and run the web app in a Docker container:
```bash
docker build -t style-fusion .
docker run -p 5001:5001 -v $(pwd)/data:/IE7374_Group6/data -v $(pwd)/outputs:/IE7374_Group6/outputs style-fusion
```
For the notebook, modify the `Dockerfile` to include Jupyter and run:
```bash
docker run -p 8888:8888 -v $(pwd)/data:/IE7374_Group6/data -v $(pwd)/outputs:/IE7374_Group6/outputs style-fusion jupyter notebook --ip=0.0.0.0 --allow-root --no-browser
```

## Requirements
- Python 3.10+
- GPU (CUDA or MPS) recommended for faster training/inference.
- Internet access for downloading pretrained models.

## Notes
- Ensure `data/processed/` contains images with consistent filename formats.
- Update `adapter_paths` in `model_config.yaml` for different artist combinations.
- Monitor memory usage on MPS devices; the code includes `torch.mps.empty_cache()` for optimization.