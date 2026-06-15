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

esm-loko:
	python3 scripts/build_esm_loko.py

install-foldflow:
	./scripts/install_foldflow.sh

download-foldflow:
	./scripts/download_foldflow.sh sfm

validate-foldflow:
	GEOMSTATS_BACKEND=pytorch python3 scripts/validate_foldflow_integration.py

train:
	python3 training/finetune.py

train-fold1:
	GEOMSTATS_BACKEND=pytorch python3 train_single_fold.py

visualize-fold1:
	python3 visualize_fold1_prediction.py

evaluate:
	python3 evaluation/evaluate.py

visualize:
	python3 evaluation/visualize.py

# Pipeline completo
pipeline: validate download-dataset explore-dataset preprocess

.PHONY: install validate download-dataset explore-dataset preprocess esm-loko install-foldflow download-foldflow validate-foldflow train train-fold1 visualize-fold1 evaluate visualize pipeline
