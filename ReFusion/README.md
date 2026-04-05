# ReFusion

![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)

This repository provides **ReFusion**, a multimodal reliability fusion network for **bearing / rotating machinery fault diagnosis** from windowed vibration, current, and RPM signals. It combines dual-domain encoders (time + frequency), per-modality heads, an **uncertainty-aware reliability gate**, hybrid logits–feature fusion, and an optional **Refiner**.

No raw dataset is shipped: you place **preprocessed** CSV windows under `data/preprocessed_data/` (see **Data layout**).

---

## Requirements

- Python **3.9+**
- PyTorch **2.x** (CUDA build optional; install from the [official guide](https://pytorch.org/))

Install dependencies:

```bash
cd ReFusion
python -m pip install -r requirements.txt
```

---

## Quick start

**1. Data** — Under `data/preprocessed_data/`, provide:

- `train_windows/*.csv`
- `test_windows/*.csv`

Each file must include `window_idx` and columns `*_t0` … `*_t1023` (default window length 1024). Filenames must encode fault type (see `_filename_to_canonical_fault` in `data/data_loader.py`).

**2. Train**

```bash
python train.py
```

Hyperparameters: `config/default.json` (or `config.json` in the repo root; first match wins). Best weights → `outputs/best_ReFusion.pth`, metrics → `outputs/test_results.json`.

**3. Evaluate a checkpoint**

```bash
python evaluate.py --checkpoint outputs/best_ReFusion.pth
```

---

## Key files

| Path | Role |
|------|------|
| `refusion/network.py` | **ReFusion** (`nn.Module`), gates, fusion, Refiner |
| `refusion/layers.py` | Per-modality linear classification head |
| `data/data_loader.py` | CSV loaders, stratified split, augmentation |
| `train.py` | Training loop, validation selection, test metrics |
| `evaluate.py` | Load `ReFusion_state_dict` and run test evaluation |
| `config/default.json` | Default hyperparameters and paths |
| `reproduce/` | Reproduction notes and example result schema |

Checkpoint format: `{"ReFusion_state_dict": ..., "fault_mapping": ...}`. Older files with `model_state_dict` are still accepted in `evaluate.py`.

---

## Transfer learning (optional)

Set `pretrained_path` in `config/default.json`. Weights are aligned by modality order; legacy checkpoints containing `attention.*` keys are remapped when shapes match.

---

## Citation

If you use this code in research, please cite the associated work (add your paper entry here) or acknowledge the repository.

```bibtex
% @article{...}
```

---

## License

Use and redistribution are subject to your project’s license file (add `LICENSE` if needed).
