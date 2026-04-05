# Reproducing experiments

## Prerequisites

- Install packages per the root `README.md`.
- Use the same **preprocessed** multimodal windows as in your reference run: `data/preprocessed_data/train_windows` and `test_windows`.
- Fix randomness via `random_seed` in `config/default.json`; use `eval_seeds` as a JSON list for multi-seed runs.

## Optional determinism

For stabler numbers across runs (often slower):

```python
import torch
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

## Run training

```bash
python train.py
```

Inspect `outputs/test_results.json` for `accuracy`, `macro_f1`, etc.

## Evaluate without retraining

```bash
python evaluate.py --checkpoint outputs/best_ReFusion.pth
```

## Output schema

See `test_results.example.json` in this folder.
