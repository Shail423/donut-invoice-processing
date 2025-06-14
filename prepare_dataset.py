import os
import json
import argparse
from pathlib import Path
import random
import shutil

def create_dataset_entry(image_path, invoice_data):
    """
    Create a dataset entry in the format required by Donut
    
    Args:
        image_path: Path to the invoice image
        invoice_data: Dictionary containing invoice data fields
    
    Returns:
        Dictionary with image_path and text fields
    """
    # Convert the invoice data to a JSON string
    json_str = json.dumps(invoice_data)
    
    # Wrap in the required format for Donut
    text = f"<s_invoice>{json_str}</s_invoice>"
    
    return {
        "image_path": str(image_path),
        "text": text
    }

def prepare_dataset(image_dir, output_file, schema_file, train_ratio=0.8, seed=42):
    """
    Prepare a dataset from a directory of invoice images
    
    Args:
        image_dir: Directory containing invoice images
        output_file: Base name for output files (will create train and val files)
        schema_file: Path to the JSON schema file
        train_ratio: Ratio of images to use for training
        seed: Random seed for reproducibility
    """
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Load the schema
    with open(schema_file, 'r') as f:
        schema = json.load(f)
    
    # Get all image files
    image_dir = Path(image_dir)
    image_files = []
    for ext in ['jpg', 'jpeg', 'png', 'tif', 'tiff']:
        image_files.extend(list(image_dir.glob(f"*.{ext}")))
        image_files.extend(list(image_dir.glob(f"*.{ext.upper()}")))
    
    print(f"Found {len(image_files)} images in {image_dir}")
    
    # Shuffle the images
    random.shuffle(image_files)
    
    # Split into train and validation sets
    split_idx = int(len(image_files) * train_ratio)
    train_images = image_files[:split_idx]
    val_images = image_files[split_idx:]
    
    print(f"Creating {len(train_images)} training samples and {len(val_images)} validation samples")
    
    # Create output directory if it doesn't exist
    output_dir = Path(output_file).parent
    os.makedirs(output_dir, exist_ok=True)
    
    # Create the training dataset
    train_file = f"{output_file.rstrip('.jsonl')}_train.jsonl"
    val_file = f"{output_file.rstrip('.jsonl')}_val.jsonl"
    
    # For this example, we'll create placeholder data
    # In a real scenario, you would extract this data from existing annotations or ask users to provide it
    
    # Create training data
    with open(train_file, 'w') as f:
        for img_path in train_images:
            # Create a sample invoice data (in real use, this would come from annotations)
            invoice_data = schema.copy()
            invoice_data["invoice_number"] = f"INV-{random.randint(1000, 9999)}"
            invoice_data["date"] = f"2025-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}"
            invoice_data["product_amount"] = f"{random.randint(100, 5000)}"
            invoice_data["total_amount"] = f"{random.randint(100, 5000)}"
            
            # Create the dataset entry
            entry = create_dataset_entry(img_path, invoice_data)
            f.write(json.dumps(entry) + "\n")
    
    # Create validation data
    with open(val_file, 'w') as f:
        for img_path in val_images:
            # Create a sample invoice data (in real use, this would come from annotations)
            invoice_data = schema.copy()
            invoice_data["invoice_number"] = f"INV-{random.randint(1000, 9999)}"
            invoice_data["date"] = f"2025-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}"
            invoice_data["product_amount"] = f"{random.randint(100, 5000)}"
            invoice_data["total_amount"] = f"{random.randint(100, 5000)}"
            
            # Create the dataset entry
            entry = create_dataset_entry(img_path, invoice_data)
            f.write(json.dumps(entry) + "\n")
    
    print(f"Created training dataset: {train_file}")
    print(f"Created validation dataset: {val_file}")
    
    return train_file, val_file

def main():
    parser = argparse.ArgumentParser(description="Prepare invoice dataset for Donut model")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory containing invoice images")
    parser.add_argument("--output_file", type=str, default="./data/invoices.jsonl", help="Base name for output files")
    parser.add_argument("--schema_file", type=str, default="./data/invoice_schema.json", help="Path to invoice schema file")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Ratio of images to use for training")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    prepare_dataset(args.image_dir, args.output_file, args.schema_file, args.train_ratio, args.seed)

if __name__ == "__main__":
    main()