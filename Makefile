install:
	pip3 install -r requirements.txt

validate:
	python3 scripts/validate_setup.py

download-dataset:
	python3 scripts/download_klifs_dataset.py

explore-dataset:
	python3 scripts/explore_dataset.py

preprocess:
	python3 scripts/preprocess_dataset.py

train:
	python3 training/finetune.py

evaluate:
	python3 evaluation/evaluate.py

visualize:
	python3 evaluation/visualize.py

# Pipeline completo
pipeline: validate download-dataset explore-dataset preprocess

.PHONY: install validate download-dataset explore-dataset preprocess train evaluate visualize pipeline