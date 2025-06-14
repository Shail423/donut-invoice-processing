.PHONY: setup download-model run test clean

# Default model path
MODEL_DIR = donut-invoice-model
DATA_DIR = data/images

setup:
	python setup_dev_env.py

download-model:
	python download_models.py --output $(MODEL_DIR)

run:
	python app.py

batch-process:
	python batch_process.py --image_dir $(DATA_DIR) --model_dir $(MODEL_DIR)

export-excel:
	python export_to_excel.py --results_dir data/output_json

test:
	pytest

clean:
	rm -rf __pycache__
	rm -rf */__pycache__
	rm -rf *.egg-info
	rm -rf dist
	rm -rf build