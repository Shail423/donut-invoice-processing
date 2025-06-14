import json
import os
import argparse
from PIL import Image
import torch
import requests
from io import BytesIO
import base64
from transformers import DonutProcessor
import gc

def clear_memory():
    """Clear memory to prevent out-of-memory errors"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    if hasattr(os, 'sync'):
        os.sync()

def predict_with_api(image_path, api_key=None):
    """Use Hugging Face Inference API to run prediction"""
    # Load and prepare image
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    
    # Encode image to base64
    encoded_image = base64.b64encode(image_bytes).decode("utf-8")
    
    # API endpoint (free tier of Hugging Face Inference API)
    API_URL = "https://api-inference.huggingface.co/models/naver-clova-ix/donut-base-finetuned-cord-v2"
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    
    # Make API request
    response = requests.post(
        API_URL,
        headers=headers,
        json={"inputs": {"image": encoded_image}}
    )
    
    # Parse response
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": f"API request failed with status code {response.status_code}: {response.text}"}

def predict_locally(image_path, model_dir):
    """Process image locally with saved processor"""
    try:
        # Load processor
        processor = DonutProcessor.from_pretrained(model_dir, use_fast=False)
        if isinstance(processor, tuple):
            processor = processor[0]
        
        # Load and process image
        image = Image.open(image_path).convert("RGB")
        pixel_values = processor(images=image, return_tensors="pt").pixel_values
        
        # Since we can't run the full model on CPU, just return the processed image info
        return {
            "filename": os.path.basename(image_path),
            "image_shape": [image.width, image.height],
            "pixel_values_shape": [int(x) for x in pixel_values.shape],
            "note": "Full prediction requires GPU. Only image processing completed."
        }
    except Exception as e:
        return {"error": str(e)}
    finally:
        clear_memory()

def main():
    parser = argparse.ArgumentParser(description="Predict with Donut model")
    parser.add_argument("--image_path", required=True, help="Path to the input image")
    parser.add_argument("--model_dir", default="./donut-rvlcdip-invoice-model", 
                        help="Path to the saved model directory")
    parser.add_argument("--api_key", help="Hugging Face API key for inference API")
    parser.add_argument("--use_api", action="store_true", 
                        help="Use Hugging Face Inference API instead of local processing")
    args = parser.parse_args()
    
    # Check if image exists
    if not os.path.exists(args.image_path):
        print(f"Error: Image file not found: {args.image_path}")
        return
    
    print(f"Processing image: {args.image_path}")
    
    # Run prediction
    if args.use_api:
        print("Using Hugging Face Inference API for prediction...")
        result = predict_with_api(args.image_path, args.api_key)
    else:
        print("Using local processor for image processing...")
        result = predict_locally(args.image_path, args.model_dir)
    
    # Save result
    output_file = os.path.splitext(args.image_path)[0] + "_prediction.json"
    with open(output_file, "w") as f:
        json.dump(result, f, indent=2)
    
    print(f"Prediction saved to {output_file}")

if __name__ == "__main__":
    main()