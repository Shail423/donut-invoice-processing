#!/usr/bin/env python
"""
Script to download pre-trained models for the Donut invoice processor.
"""
import os
import sys
import argparse
from transformers import DonutProcessor, VisionEncoderDecoderModel

def download_model(model_name, output_dir):
    """Download a model from Hugging Face and save it locally."""
    print(f"Downloading {model_name} to {output_dir}...")
    
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Download processor and model
        processor = DonutProcessor.from_pretrained(model_name)
        model = VisionEncoderDecoderModel.from_pretrained(model_name)
        
        # Save to disk
        processor.save_pretrained(output_dir)
        model.save_pretrained(output_dir)
        
        print(f"✅ Successfully downloaded {model_name} to {output_dir}")
        return True
    except Exception as e:
        print(f"❌ Error downloading model: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Download pre-trained models for Donut invoice processor")
    parser.add_argument("--model", default="naver-clova-ix/donut-base-finetuned-cord-v2", 
                        help="Model name on Hugging Face (default: naver-clova-ix/donut-base-finetuned-cord-v2)")
    parser.add_argument("--output", default="donut-invoice-model", 
                        help="Output directory (default: donut-invoice-model)")
    
    args = parser.parse_args()
    
    print("Donut Model Downloader")
    print("======================")
    
    success = download_model(args.model, args.output)
    
    if success:
        print("\nNext steps:")
        print("1. Use the model with: python predict.py --model_dir", args.output)
    else:
        print("\nDownload failed. Please check your internet connection and try again.")
        sys.exit(1)

if __name__ == "__main__":
    main()