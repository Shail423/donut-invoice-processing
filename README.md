<<<<<<< HEAD
# Donut Invoice Processing

This project uses the Donut (Document Understanding Transformer) model to extract invoice numbers from invoice images.

## Project Structure

- `train.py`: Script for training the Donut model (CPU-optimized)
- `predict_with_finetuned.py`: Script for running predictions with the fine-tuned model
- `batch_process.py`: Script for batch processing multiple invoice images
- `analyze_results.py`: Script for analyzing extraction results
- `export_to_excel.py`: Script for exporting results to Excel

## Setup

1. Install dependencies:
   ```
   pip install transformers torch pillow pandas openpyxl
   ```

2. Prepare your data:
   - Place invoice images in `data/images/`
   - Ensure the model is in `donut-invoice-model/`

## Usage

### Single Invoice Prediction

```
python predict_with_finetuned.py --image_path data/images/FACTU2020040046.jpg --model_dir donut-invoice-model
```

### Batch Processing

```
python batch_process.py --image_dir data/images --pattern FACTU*.jpg
```

### Analyze Results

```
python analyze_results.py --results_dir data/images
```

### Export to Excel

```
python export_to_excel.py --results_dir data/images
```

## Batch Files

- `run_extract_invoice.bat`: Process a single invoice
- `run_batch_process.bat`: Process all invoices and analyze results
- `run_export_excel.bat`: Export results to Excel

## Model Details

- Base model: Donut (Document Understanding Transformer)
- Fine-tuned on: Invoice images with invoice numbers
- Input: Invoice images
- Output: Extracted invoice numbers (format: FA##/####/######)

## Performance

The model's performance can be analyzed using the `analyze_results.py` script, which provides:
- Success rate
- Sample of extracted invoice numbers
- Common error types
=======
# donut-invoice-processing
>>>>>>> b3349b2a7ada12df6b4a6fd66e2e890629d5f014
