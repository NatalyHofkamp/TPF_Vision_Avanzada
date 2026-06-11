# Flow Matching + CFG comparison

## Notes
- Downstream probe validation ROC-AUC: 0.6828; test ROC-AUC: 0.1873.

## Final comparison table

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | RMSD | Diversity |
| ------ | -------- | --------- | ------ | -- | ------- | ------ | ---- | --------- |
| Existing baseline | 0.0938 | 0.0000 | 0.0000 | 0.0000 | 0.0316 | 0.7825 | 34.8567 | 24.8175 |
| Flow Matching no CFG | 0.0938 | 0.0000 | 0.0000 | 0.0000 | 0.0287 | 0.7714 | 34.7935 | 24.7492 |
| Flow Matching + CFG 1 | 0.0938 | 0.0000 | 0.0000 | 0.0000 | 0.0431 | 0.7832 | 34.8085 | 24.7223 |
| Flow Matching + CFG 2 | 0.0938 | 0.0000 | 0.0000 | 0.0000 | 0.0374 | 0.7863 | 35.0408 | 25.1498 |
| Flow Matching + CFG 3 | 0.0938 | 0.0000 | 0.0000 | 0.0000 | 0.0489 | 0.7976 | 35.0360 | 25.4833 |
| Flow Matching + CFG 5 | 0.0938 | 0.0000 | 0.0000 | 0.0000 | 0.0259 | 0.7702 | 35.3136 | 26.3002 |

## Limitations

- The current workspace may not contain the processed tensors or split files required for end-to-end training.
- Metrics for generative comparison rely on a downstream probe trained only on the training split.
- The notebook is designed to be rerun on the full dataset without changing splits or labels.