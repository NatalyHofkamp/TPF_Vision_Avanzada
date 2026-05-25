install:
	pip install -r requirements.txt

validate:
	python scripts/validate_setup.py

download-dataset:
	python scripts/download_klifs_dataset.py

explore-dataset:
	python scripts/explore_dataset.py

train:
	python training/finetune.py

evaluate:
	python evaluation/evaluate.py

visualize:
	python evaluation/visualize.py

preprocess:
	python scripts/preprocess_dataset.py

# Pipeline completo
pipeline: validate download-dataset explore-dataset preprocess

.PHONY: install validate download-dataset explore-dataset train evaluate visualize preprocess pipeline


