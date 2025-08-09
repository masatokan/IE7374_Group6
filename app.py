import os
import logging
import webbrowser
from flask import Flask, render_template, request, send_file
from flask_socketio import SocketIO, emit
from src.data_loader import load_artist_dataset
from src.train import train_lora_for_artist
from src.model_runner import load_fused_lora_pipeline, generate_images
from utils.helper import setup_device, load_pipeline
import yaml

app = Flask(__name__, template_folder='templates', static_folder='static')
socketio = SocketIO(app)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Custom logging handler to emit logs to WebSocket
class WebSocketHandler(logging.Handler):
    def emit(self, record):
        socketio.emit('progress', {'message': self.format(record)})

logger.addHandler(WebSocketHandler())

project_root = os.path.abspath('.')
dataset_dir = os.path.join(project_root, 'data/processed')
output_dir = os.path.join(project_root, 'outputs/images/web_outputs')
os.makedirs(output_dir, exist_ok=True)

@app.route('/', methods=['GET'])
def index():
    """Render the main page with artist selection."""
    artist_datasets = load_artist_dataset(dataset_dir)
    artists = list(artist_datasets.keys())
    logger.info(f"Found artists: {artists}")
    return render_template('index.html', artists=artists)

@app.route('/image/<path:image_path>')
def serve_image(image_path):
    """Serve images from outputs/images/web_outputs/."""
    image_path = os.path.join(project_root, 'outputs/images/web_outputs', image_path)
    if not os.path.exists(image_path):
        return "Image not found", 404
    return send_file(image_path, mimetype='image/png')

@app.route('/generate', methods=['POST'])
def generate():
    """Handle form submission and generate images."""
    try:
        # Get form data
        artist1 = request.form.get('artist1')
        artist2 = request.form.get('artist2')
        prompt = request.form.get('prompt', 'A landscape')
        logger.info(f"Received request: artist1={artist1}, artist2={artist2}, prompt={prompt}")

        # Validate inputs
        if not artist1 or not artist2:
            logger.error("Missing artist selection")
            return render_template('index.html', artists=load_artist_dataset(dataset_dir).keys(), error="Please select two artists.")

        # Load config
        with open(os.path.join(project_root, 'configs/model_config.yaml'), 'r') as f:
            config = yaml.safe_load(f)
        base_model = config['model']['base_model']
        adapter_paths = [
            os.path.join(project_root, 'outputs/lora_weights', artist1.replace(' ', '_')),
            os.path.join(project_root, 'outputs/lora_weights', artist2.replace(' ', '_'))
        ]
        weights = config['fusion']['weights']

        # Check and train LoRA adapters if missing
        device = setup_device()
        base_pipeline = load_pipeline(base_model, device)
        artist_datasets = load_artist_dataset(dataset_dir)
        for artist, path in zip([artist1, artist2], adapter_paths):
            if not (os.path.exists(path) and os.path.exists(os.path.join(path, 'adapter_config.json'))):
                logger.info(f"Training LoRA for {artist}")
                socketio.emit('progress', {'message': f'Training LoRA for {artist}...', 'progress': 10})
                train_config = {'learning_rate': 1e-4, 'num_epochs': 3, 'lora_rank': 4, 'resolution': 256}
                train_lora_for_artist(artist, artist_datasets[artist], base_pipeline, os.path.join(project_root, 'outputs/lora_weights'), train_config, sample_size=5)
                socketio.emit('progress', {'message': f'Finished training LoRA for {artist}', 'progress': 50})

        # Generate images
        logger.info("Loading fused model")
        socketio.emit('progress', {'message': 'Loading fused model...', 'progress': 60})
        pipeline = load_fused_lora_pipeline(base_model, adapter_paths, weights)
        fusion_name = '+'.join([artist1.replace(' ', '_'), artist2.replace(' ', '_')])
        prompts = [(f"{prompt} in the fusion style of {artist1} and {artist2}", [])]
        logger.info("Generating images")
        socketio.emit('progress', {'message': 'Generating image...', 'progress': 80})
        generated_images = generate_images(pipeline, prompts, output_dir, label=fusion_name)
        socketio.emit('progress', {'message': 'Image generation complete', 'progress': 100})

        # Prepare image paths for display
        image_paths = [f"{fusion_name}_{i}.png" for i in range(len(generated_images))]
        return render_template('results.html', images=image_paths, prompt=prompt, fusion_name=fusion_name)
    except Exception as e:
        logger.error(f"Error generating images: {e}")
        socketio.emit('progress', {'message': f'Error: {str(e)}', 'progress': 0})
        return render_template('index.html', artists=load_artist_dataset(dataset_dir).keys(), error=str(e))

if __name__ == '__main__':
    webbrowser.open('http://localhost:5001')
    socketio.run(app, debug=True, host='0.0.0.0', port=5001)