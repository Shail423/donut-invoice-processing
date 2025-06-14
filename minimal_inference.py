import os
import json
import argparse
from PIL import Image
import torch
from transformers import DonutProcessor

def main():
    parser = argparse.ArgumentParser(description="Process invoice with Donut")
    parser.add_argument("--image_path", required=True, help="Path to the input image")
    args = parser.parse_args()
    
    # Use the pre-trained model
    model_name = "naver-clova-ix/donut-base-finetuned-rvlcdip"
    print(f"Loading processor from {model_name}")
    
    # Load only the processor (not the full model)
    processor = DonutProcessor.from_pretrained(model_name)
    
    # Process the image
    print(f"Processing image: {args.image_path}")
    image = Image.open(args.image_path).convert("RGB")
    
    # Process image
    pixel_values = processor(images=image, return_tensors="pt").pixel_values
    
    print(f"Image processed successfully! Shape: {pixel_values.shape}")
    print("For full model inference, use a cloud service with GPU access.")
    
    # Extract filename without extension
    base_name = os.path.splitext(os.path.basename(args.image_path))[0]
    
    # Create a simple JSON output with image information
    output = {
        "filename": os.path.basename(args.image_path),
        "image_size": [image.width, image.height],
        "document_type": "invoice",
        "processed": True
    }
    
    # Save the output
    output_file = f"output_{base_name}.json"
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"Output saved to {output_file}")

if __name__ == "__main__":
    main()