# Setting Up Donut Invoice Processor on a New Machine

This guide will help you quickly set up the Donut Invoice Processor project on a new machine.

## Prerequisites

- Python 3.8+ installed
- Git installed
- Pip installed

## Step 1: Clone the Repository

```bash
git clone https://github.com/your-username/donut-invoice-processor.git
cd donut-invoice-processor
```

## Step 2: Set Up Development Environment

Run the setup script to create a virtual environment and install dependencies:

```bash
python setup_dev_env.py
```

## Step 3: Activate the Virtual Environment

On Windows:
```bash
.\donut-env\Scripts\activate
```

On macOS/Linux:
```bash
source ./donut-env/bin/activate
```

## Step 4: Download Pre-trained Models

Download the required pre-trained models:

```bash
python download_models.py
```

This will download the default model to the `donut-invoice-model` directory.

## Step 5: Prepare Your Data

Place your invoice images in the `data/images` directory.

## Step 6: Run the Application

```bash
python app.py
```

## Using Make Commands (Optional)

If you have Make installed, you can use the following commands:

```bash
make setup           # Set up the development environment
make download-model  # Download pre-trained models
make run             # Run the application
make batch-process   # Process all invoices in the data directory
make export-excel    # Export results to Excel
make test            # Run tests
make clean           # Clean up temporary files
```

## Troubleshooting

If you encounter any issues:

1. Make sure all dependencies are installed correctly
2. Check that the model files are downloaded properly
3. Verify that your invoice images are in the correct format (JPG, PNG)
4. Check the logs directory for error messages