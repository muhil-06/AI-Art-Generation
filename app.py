from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import os
import torch
import uuid
from models import Generator
from utils import text_to_latent, save_image, generate_procedural_art

app = Flask(__name__)
CORS(app)

# Configuration
LATENT_DIM = 100
IMG_SHAPE = (3, 64, 64)  # Example shape: RGB 64x64
OUTPUT_DIR = os.path.join('static', 'generated')

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

# Initialize Generator
generator = Generator(LATENT_DIM, IMG_SHAPE)
# Note: In a real scenario, we would load pre-trained weights here
# generator.load_state_dict(torch.load('gan_generator.pth'))
generator.eval()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    try:
        data = request.json
        prompt = data.get('prompt', 'A beautiful landscape')
        style = data.get('style', 'neon')
        color = data.get('color', 'vibrant')
        resolution = data.get('resolution', '512')
        
        print(f"Generating image for prompt: {prompt}, style: {style}, color: {color}")
        
        # Generate image using procedural artistic logic
        generated_img = generate_procedural_art(prompt, style, color, resolution)
        
        # Save image
        filename = f"{uuid.uuid4()}.png"
        filepath = os.path.join(OUTPUT_DIR, filename)
        save_image(generated_img, filepath)
        
        print(f"Image saved to: {filepath}")
        
        return jsonify({
            'status': 'success',
            'image_url': f'/static/generated/{filename}',
            'prompt': prompt
        })
    except Exception as e:
        print(f"Error during generation: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
