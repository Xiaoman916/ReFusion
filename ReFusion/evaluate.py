"""
Load a checkpoint and evaluate on the test set.
Set DATA_DIR or use config/default.json for data_dir (must match training).
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from data.data_loader import create_dataloaders
from refusion.network import ReFusion

import train as train_mod


def load_config() -> dict:
    cfg = {}
    for name in ("config/default.json", "config.json"):
        p = _ROOT / name
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            break
    return cfg


def main():
    ap = argparse.ArgumentParser(description="Load weights and evaluate on the test set")
    ap.add_argument("--checkpoint", type=str, default="outputs/best_ReFusion.pth")
    ap.add_argument("--data-dir", type=str, default=os.environ.get("DATA_DIR", ""))
    ap.add_argument("--seed", type=int, default=43)
    args = ap.parse_args()

    cfg = load_config()
    if args.data_dir:
        cfg["data_dir"] = args.data_dir
    data_dir = cfg.get("data_dir", str(_ROOT / "data" / "preprocessed_data"))
    if not Path(data_dir).is_absolute():
        data_dir = str((_ROOT / data_dir).resolve())
    cfg["data_dir"] = data_dir

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = _ROOT / ckpt_path
    if not ckpt_path.exists():
        raise SystemExit(f"Checkpoint not found: {ckpt_path}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, test_loader, fault_mapping = create_dataloaders(
        data_dir=data_dir,
        batch_size=cfg.get("batch_size", 32),
        window_size=cfg.get("window_size", 1024),
        num_workers=cfg.get("num_workers", 0),
        random_seed=args.seed,
        augment_train=False,
        use_all_data=cfg.get("use_all_data", True),
        train_ratio=cfg.get("train_ratio", 0.7),
        val_ratio=cfg.get("val_ratio", 0.0),
        augment_repeat=1,
        test_repeat=cfg.get("test_repeat", 1),
        use_raw_modalities=cfg.get("use_raw_modalities", True),
    )

    num_classes = fault_mapping["num_classes"]
    modalities = fault_mapping["modalities"]

    model = ReFusion(
        modalities=modalities,
        input_size=cfg.get("window_size", 1024),
        latent_dim=cfg.get("latent_dim", 256),
        num_classes=num_classes,
        dropout=cfg.get("dropout", 0.0),
        lstm_hidden=cfg.get("lstm_hidden", 192),
        lstm_layers=cfg.get("lstm_layers", 1),
        use_hybrid_fusion=cfg.get("use_hybrid_fusion", True),
        use_refiner=cfg.get("use_refiner", True),
        prob_weight=cfg.get("prob_weight", 0.6),
        feat_weight=cfg.get("feat_weight", 0.4),
        temperature=cfg.get("temperature", 0.7),
        learnable_temperature=True,
        use_bad_modality=False,
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("ReFusion_state_dict", ckpt.get("model_state_dict", ckpt))
    model.load_state_dict(state, strict=True)

    import torch.nn as nn
    criterion = nn.CrossEntropyLoss()
    metrics = train_mod.evaluate(model, test_loader, criterion, device, fault_mapping)
    print(json.dumps({k: v for k, v in metrics.items() if k != "confusion_matrix"}, ensure_ascii=False, indent=2))
    print("accuracy:", metrics["accuracy"])


if __name__ == "__main__":
    main()
