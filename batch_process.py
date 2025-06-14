import os
import glob
import argparse
import subprocess
from concurrent.futures import ThreadPoolExecutor

def process_invoice(image_path, model_dir):
    """Process a single invoice image"""
    print(f"Processing: {os.path.basename(image_path)}")
    subprocess.run([
        "python", "predict_with_finetuned.py",
        "--image_path", image_path,
        "--model_dir", model_dir
    ])
    return image_path

def main():
    parser = argparse.ArgumentParser(description="Batch process invoice images")
    parser.add_argument("--image_dir", default="data/images", help="Directory containing invoice images")
    parser.add_argument("--model_dir", default="./donut-invoice-model", help="Path to fine-tuned model folder")
    parser.add_argument("--pattern", default="FACTU*.jpg", help="File pattern to match")
    parser.add_argument("--max_workers", type=int, default=1, help="Maximum number of parallel workers")
    args = parser.parse_args()
    
    # Find all matching images
    image_pattern = os.path.join(args.image_dir, args.pattern)
    image_files = glob.glob(image_pattern)
    print(f"Found {len(image_files)} invoice images to process")
    
    # Process images in parallel
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [executor.submit(process_invoice, img, args.model_dir) for img in image_files]
        for future in futures:
            future.result()
    
    print("All invoices processed!")

if __name__ == "__main__":
    main()