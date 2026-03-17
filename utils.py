import torch
import numpy as np
import hashlib
from PIL import Image, ImageDraw, ImageFilter
import os
import random

def text_to_latent(text, latent_dim):
    # Deterministic mapping from text to latent vector using hash
    hash_object = hashlib.sha256(text.encode())
    hash_hex = hash_object.hexdigest()
    
    # Use the hash to seed numpy random
    seed = int(hash_hex[:8], 16)
    np.random.seed(seed)
    
    latent_vector = np.random.normal(0, 1, (1, latent_dim))
    return torch.FloatTensor(latent_vector)

def get_color_palette(color_theme):
    palettes = {
        'vibrant': [(0, 210, 255), (157, 80, 187), (255, 0, 200), (0, 255, 128)],
        'monochrome': [(20, 20, 20), (80, 80, 80), (150, 150, 150), (220, 220, 220)],
        'pastel': [(255, 179, 186), (255, 223, 186), (255, 255, 186), (186, 255, 201)],
        'dark': [(5, 5, 5), (30, 0, 50), (0, 20, 40), (20, 20, 20)]
    }
    return palettes.get(color_theme, palettes['vibrant'])

def generate_procedural_art(prompt, style, color_theme, resolution):
    try:
        res = int(resolution)
    except:
        res = 512
        
    # Create base image as RGBA to support transparency during drawing
    img = Image.new('RGBA', (res, res), color=(10, 10, 10, 255))
    draw = ImageDraw.Draw(img)
    
    # Seed based on prompt
    seed = int(hashlib.sha256(prompt.encode()).hexdigest()[:8], 16)
    random.seed(seed)
    
    palette = get_color_palette(color_theme)
    
    # Draw some artistic layers
    if style == 'cyberpunk' or style == 'neon':
        # Draw neon lines and glows
        for _ in range(30):
            c = random.choice(palette)
            x1, y1 = random.randint(0, res), random.randint(0, res)
            x2, y2 = random.randint(0, res), random.randint(0, res)
            # Main line
            draw.line([x1, y1, x2, y2], fill=(*c, 255), width=random.randint(1, 3))
            # Glow effect
            draw.ellipse([x1-15, y1-15, x1+15, y1+15], fill=(*c, 80))
            
    elif style == 'oil-painting':
        # Draw blobby brush strokes
        for _ in range(150):
            c = random.choice(palette)
            x, y = random.randint(0, res), random.randint(0, res)
            r = random.randint(30, 120)
            draw.ellipse([x-r, y-r, x+r, y+r], fill=(*c, 180))
            
    elif style == 'vaporwave':
        # Draw geometric shapes and gradients
        for i in range(5):
            c = palette[i % len(palette)]
            y = (i * res) // 5
            draw.rectangle([0, y, res, y + res // 5], fill=(*c, 255))
        for _ in range(15):
            c = (255, 255, 255, 150)
            x = random.randint(0, res)
            y = random.randint(0, res)
            # Draw a circle instead if polygon fails
            draw.ellipse([x-20, y-20, x+20, y+20], fill=c)

    else: # Abstract/Surreal
        for _ in range(60):
            c = random.choice(palette)
            x, y = random.randint(0, res), random.randint(0, res)
            w, h = random.randint(50, 300), random.randint(50, 300)
            draw.chord([x, y, x+w, y+h], random.randint(0, 360), random.randint(0, 360), fill=(*c, 120))

    # Apply artistic filters
    # Convert back to RGB for filters that might not support RGBA or for final save
    img = img.convert('RGB')
    img = img.filter(ImageFilter.GaussianBlur(radius=random.randint(1, 4)))
    img = img.filter(ImageFilter.SHARPEN)
    
    return img

def save_image(img, path):
    if isinstance(img, torch.Tensor):
        # Fallback for GAN tensor
        image = img.detach().cpu().numpy()[0]
        image = (image + 1) / 2.0
        image = (image * 255).astype(np.uint8)
        if image.shape[0] == 3:
            image = image.transpose(1, 2, 0)
        img = Image.fromarray(image)
        img = img.resize((512, 512), Image.LANCZOS)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.save(path)
