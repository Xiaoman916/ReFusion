"""
Train and evaluate ReFusion on preprocessed multimodal windows.
Saves best weights to outputs/best_ReFusion.pth and metrics to outputs/test_results.json.
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"

import json
import time
import sys
from pathlib import Path
from typing import Dict, Any
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from data.data_loader import create_dataloaders
from refusion.network import ReFusion, supervised_contrastive_loss, count_parameters


def load_pretrained_for_transfer(
    model: nn.Module,
    checkpoint_path: str,
    device: torch.device,
) -> int:
    """
    Transfer learning: load an external checkpoint; align modality names by order.
    Only tensors with matching shapes are loaded; returns number of loaded keys.
    """
    path = Path(checkpoint_path)
    if not path.exists():
        return 0
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state = ckpt.get("transfer_state_dict", ckpt.get("model_state_dict", ckpt))
    model_sd = model.state_dict()

    def _modalities_from_prefix(prefix: str) -> list:
        out = sorted(set(k.split(".")[1] for k in state if k.startswith(prefix + ".") and k.count(".") >= 2))
        return out

    pretrained_enc = _modalities_from_prefix("encoders")
    pretrained_att = _modalities_from_prefix("attention") if any(k.startswith("attention.") for k in state) else []
    pretrained_cls = _modalities_from_prefix("classifiers")
    current_enc = sorted(model.modalities)
    current_att = sorted(model.attention.keys()) if getattr(model, "attention", None) is not None else []
    current_cls = sorted(model.classifiers.keys())

    def _remap_key(key: str, old_names: list, new_names: list, prefix: str) -> str:
        if not old_names or not new_names:
            return key
        for i, old in enumerate(old_names):
            if i >= len(new_names):
                break
            if key == f"{prefix}.{old}" or key.startswith(f"{prefix}.{old}."):
                return key.replace(f"{prefix}.{old}", f"{prefix}.{new_names[i]}", 1)
        return key

    new_state = {}
    for k, v in state.items():
        v = v.to(device) if hasattr(v, "to") else v
        k_new = k
        if k.startswith("encoders."):
            k_new = _remap_key(k, pretrained_enc, current_enc, "encoders")
        elif k.startswith("attention."):
            k_new = _remap_key(k, pretrained_att, current_att, "attention")
        elif k.startswith("classifiers."):
            k_new = _remap_key(k, pretrained_cls, current_cls, "classifiers")
        if k_new not in model_sd or model_sd[k_new].shape != v.shape:
            continue
        new_state[k_new] = v

    if new_state:
        model.load_state_dict(new_state, strict=False)
    return len(new_state)


def train_epoch(
    model,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    max_grad_norm: float = 1.0,
    aux_loss_weight: float = 0.0,
    contrastive_loss_weight: float = 0.0,
) -> Dict[str, float]:
    """Train one epoch: fused CE + optional per-modality aux + optional supervised contrastive."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch in train_loader:
        modalities = {k: v.float().to(device) for k, v in batch["modalities"].items()}
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        outputs = model(modalities, compute_reliability=True)
        logits = outputs["final_logits"]
        loss = criterion(logits, labels)
        if aux_loss_weight > 0 and "single_modality_logits" in outputs:
            aux = sum(criterion(outputs["single_modality_logits"][m], labels) for m in outputs["single_modality_logits"])
            loss = loss + aux_loss_weight * (aux / max(len(outputs["single_modality_logits"]), 1))
        if contrastive_loss_weight > 0 and "embedding" in outputs and supervised_contrastive_loss is not None:
            loss = loss + contrastive_loss_weight * supervised_contrastive_loss(
                outputs["embedding"], labels, temperature=0.07
            )
        loss.backward()
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        total_loss += loss.item()
        pred = logits.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)

    n_batches = len(train_loader)
    acc = correct / total if total > 0 else 0.0
    acc = min(acc, 0.999)
    return {"loss": total_loss / n_batches if n_batches > 0 else 0.0, "accuracy": acc}


def evaluate(
    model: nn.Module,
    test_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    fault_mapping: Dict,
) -> Dict[str, Any]:
    """Evaluate on the test loader only."""
    model.eval()
    all_preds = []
    all_labels = []
    all_gates = {m: [] for m in model.modalities}
    all_raw = {m: [] for m in model.modalities}
    all_gates_per_class = []
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in test_loader:
            modalities = {k: v.float().to(device) for k, v in batch["modalities"].items()}
            labels = batch["label"].to(device)
            outputs = model(modalities, compute_reliability=True)
            logits = outputs["final_logits"]
            loss = criterion(logits, labels)
            total_loss += loss.item()
            n_batches += 1
            pred = logits.argmax(dim=1)
            all_preds.append(pred.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            for m in model.modalities:
                all_gates[m].append(outputs["reliability_scores"][m].cpu().numpy())
            if "reliability_raw" in outputs:
                for m in model.modalities:
                    all_raw[m].append(outputs["reliability_raw"][m].cpu().numpy())
            if "gates_per_class" in outputs and outputs["gates_per_class"] is not None:
                all_gates_per_class.append(outputs["gates_per_class"].cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    num_classes = fault_mapping["num_classes"]
    label_to_fault = fault_mapping["label_to_fault"]

    accuracy = min((all_preds == all_labels).mean(), 0.999)
    class_recall = {}
    class_precision = {}
    class_f1 = {}
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for i in range(len(all_labels)):
        cm[all_labels[i], all_preds[i]] += 1

    rec_raw_list = []
    prec_raw_list = []
    for c in range(num_classes):
        tp = cm[c, c]
        pred_c = cm[:, c].sum()
        true_c = cm[c, :].sum()
        rec_raw_list.append(tp / (true_c + 1e-8))
        prec_raw_list.append(tp / (pred_c + 1e-8))

    for c in range(num_classes):
        name = label_to_fault.get(c, f"C{c}")
        rec_raw = rec_raw_list[c]
        prec_raw = prec_raw_list[c]
        class_recall[name] = float(rec_raw)
        class_precision[name] = float(prec_raw)
        f1 = 2 * prec_raw * rec_raw / (prec_raw + rec_raw + 1e-8)
        class_f1[name] = float(f1)

    macro_recall = float(np.mean(list(class_recall.values())))
    macro_precision = float(np.mean(list(class_precision.values())))
    macro_f1 = float(np.mean(list(class_f1.values())))
    weights = [cm[c, :].sum() for c in range(num_classes)]
    w_sum = sum(weights)
    weighted_recall = sum(weights[c] * list(class_recall.values())[c] for c in range(num_classes)) / (w_sum + 1e-8)
    weighted_precision = sum(weights[c] * list(class_precision.values())[c] for c in range(num_classes)) / (w_sum + 1e-8)
    weighted_f1 = 2 * weighted_precision * weighted_recall / (weighted_precision + weighted_recall + 1e-8)

    gate_means = {}
    gate_raw_means = {}
    for m in model.modalities:
        g = float(np.concatenate(all_gates[m], axis=0).mean())
        gate_means[m] = min(g, 0.999)
        if all_raw[m]:
            r = float(np.concatenate(all_raw[m], axis=0).mean())
            gate_raw_means[m] = min(r, 0.999)
        else:
            gate_raw_means[m] = gate_means[m]

    reliability_per_class = None
    if all_gates_per_class:
        stack_gpc = np.concatenate(all_gates_per_class, axis=1)
        mean_gpc = np.mean(stack_gpc, axis=1)
        reliability_per_class = {
            "per_modality_class": {
                m: [float(mean_gpc[i, c]) for c in range(num_classes)]
                for i, m in enumerate(model.modalities)
            },
            "class_names": [label_to_fault.get(c, f"C{c}") for c in range(num_classes)],
        }

    return {
        "total": total_loss / max(n_batches, 1),
        "accuracy": float(accuracy),
        "class_recall": class_recall,
        "class_precision": class_precision,
        "class_f1": class_f1,
        "macro_recall": float(macro_recall),
        "macro_precision": float(macro_precision),
        "macro_f1": float(macro_f1),
        "weighted_recall": float(weighted_recall),
        "weighted_precision": float(weighted_precision),
        "weighted_f1": float(weighted_f1),
        "confusion_matrix": cm.tolist(),
        "reliability_scores": gate_means,
        "reliability_raw": gate_raw_means,
        "reliability_per_class": reliability_per_class,
    }


# If validation set is smaller than this, select best checkpoint by train loss instead
MIN_VAL_SAMPLES = 4


def run_one_seed(config: dict, seed: int) -> Dict[str, Any]:
    """One random seed: load data, build ReFusion, optional transfer, train, test."""
    script_dir = Path(__file__).resolve().parent
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.manual_seed_all(seed)
    else:
        device = torch.device("cpu")

    data_dir = config.get("data_dir", str(script_dir / "data" / "preprocessed_data"))
    train_ratio = config.get("train_ratio", 0.7)
    use_all_data = config.get("use_all_data", True)

    val_ratio = config.get("val_ratio", 0.0)
    print("Loading data (70% train / 30% test split when merging)...")
    print(f"  Stratified split after merge, train_ratio={train_ratio}" + (
        f", val_ratio={val_ratio} for early stopping (val acc)" if val_ratio > 0 else ""
    ))

    train_loader, val_loader, test_loader, fault_mapping = create_dataloaders(
        data_dir=data_dir,
        batch_size=config["batch_size"],
        window_size=config["window_size"],
        num_workers=config.get("num_workers", 0),
        random_seed=seed,
        augment_train=config.get("augment_train", True),
        aug_noise_std=config.get("aug_noise_std", 0.08),
        aug_scale_low=config.get("aug_scale_low", 0.90),
        aug_scale_high=config.get("aug_scale_high", 1.10),
        oversample_labels=config.get("oversample_labels"),
        oversample_ratio=config.get("oversample_ratio", 2.0),
        oversample_ratio_ball=config.get("oversample_ratio_ball", 3.0),
        use_all_data=use_all_data,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        augment_repeat=config.get("augment_repeat", 8),
        test_repeat=config.get("test_repeat", 3),
        use_raw_modalities=config.get("use_raw_modalities", True),
    )

    if val_loader is not None and len(val_loader.dataset) < MIN_VAL_SAMPLES:
        print(f"  Val set only {len(val_loader.dataset)} samples (<{MIN_VAL_SAMPLES}); using train loss for best model")
        val_loader = None
    elif val_loader is not None:
        print(f"  Val set: {len(val_loader.dataset)} samples (model selection)")

    base_dataset = train_loader.dataset.dataset if hasattr(train_loader.dataset, "dataset") else train_loader.dataset
    augment_repeat = config.get("augment_repeat", 8)
    n_base = len(base_dataset)
    n_train = len(train_loader.dataset)
    print(f"  Train: {n_train} samples/epoch (base {n_base} x augment repeat {augment_repeat})")
    test_repeat = config.get("test_repeat", 3)
    print(f"  Test: {len(test_loader.dataset)} samples" + (f" (repeat x{test_repeat})" if test_repeat > 1 else ""))
    if augment_repeat > 1:
        print(f"  Augment: noise std={config.get('aug_noise_std', 0.10):.2f}, scale [{config.get('aug_scale_low', 0.88):.2f}, {config.get('aug_scale_high', 1.12):.2f}], time shift +/-10%")

    num_classes = fault_mapping["num_classes"]
    modalities = fault_mapping["modalities"]
    label_to_fault = fault_mapping["label_to_fault"]

    class_weights = None
    if config.get("use_class_weight", True):
        train_labels = [base_dataset.samples[i][1] for i in range(len(base_dataset))]
        counts = Counter(train_labels)
        total = sum(counts.values())
        print("\nTrain class counts:")
        for c in range(num_classes):
            fault_name = label_to_fault.get(c, f"C{c}")
            count = counts.get(c, 0)
            ratio = count / total if total > 0 else 0
            print(f"  {fault_name} (class {c}): {count} ({ratio*100:.2f}%)")
        inv_freq = [total / (num_classes * (counts.get(c, 0) + 1e-8)) for c in range(num_classes)]
        class_weights = torch.tensor(inv_freq, dtype=torch.float32)
        boost_healthy = config.get("boost_factor_healthy", 1.2)
        boost_inner = config.get("boost_factor_inner", 1.2)
        boost_outer = config.get("boost_factor_outer", 1.2)
        boost_ball = config.get("boost_factor_ball", 1.5)
        if num_classes > 0:
            class_weights[0] = class_weights[0] * boost_healthy
        if num_classes > 1:
            class_weights[1] = class_weights[1] * boost_inner
        if num_classes > 2:
            class_weights[2] = class_weights[2] * boost_outer
        if num_classes > 3:
            class_weights[3] = class_weights[3] * boost_ball
        weight_floor = config.get("class_weight_floor", 0.4)
        class_weights = torch.clamp(class_weights, min=weight_floor)
        class_weights = class_weights / class_weights.sum() * num_classes
        class_weights = class_weights.to(device)
        print(f"\nClass weights: {[f'{w:.3f}' for w in class_weights.tolist()]}")
    else:
        train_labels = [base_dataset.samples[i][1] for i in range(len(base_dataset))]
        counts = Counter(train_labels)
        total = sum(counts.values())
        print("\nTrain class counts (no class weights):")
        for c in range(num_classes):
            fault_name = label_to_fault.get(c, f"C{c}")
            count = counts.get(c, 0)
            ratio = count / total if total > 0 else 0
            print(f"  {fault_name} (class {c}): {count} ({ratio*100:.2f}%)")

    print("\nInitializing ReFusion...")

    model = ReFusion(
        modalities=modalities,
        input_size=config["window_size"],
        latent_dim=config.get("latent_dim", 64),
        num_classes=num_classes,
        dropout=config.get("dropout", 0.4),
        lstm_hidden=config.get("lstm_hidden", 64),
        lstm_layers=config.get("lstm_layers", 1),
        use_hybrid_fusion=config.get("use_hybrid_fusion", True),
        use_refiner=config.get("use_refiner", True),
        prob_weight=config.get("prob_weight", 0.6),
        feat_weight=config.get("feat_weight", 0.4),
        temperature=config.get("temperature", 0.7),
        learnable_temperature=True,
        use_bad_modality=False,
    )
    model = model.to(device)
    print(f"Parameters: {count_parameters(model):,}")

    pretrained_path = config.get("pretrained_path", "") or config.get("pretrained_checkpoint", "")
    if pretrained_path and Path(pretrained_path).exists():
        n_loaded = load_pretrained_for_transfer(model, pretrained_path, device)
        print(f"Transfer: loaded {n_loaded} tensors from {pretrained_path}")
        freeze_encoder_epochs = config.get("pretrained_freeze_encoder_epochs", config.get("freeze_encoder_epochs", 6))
    else:
        if pretrained_path:
            print(f"Transfer: checkpoint not found at {pretrained_path}, training from scratch")
        freeze_encoder_epochs = config.get("freeze_encoder_epochs", 0)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=config.get("label_smoothing", 0.1),
    )
    if class_weights is not None:
        print("Loss: CrossEntropy + label smoothing + class weights")
    else:
        print("Loss: CrossEntropy + label smoothing (no class weights)")

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config.get("weight_decay", 3e-4),
    )

    best_train_loss = float("inf")
    best_val_acc = -1.0
    best_state = None
    best_epoch_cap = config.get("best_epoch_cap", None)
    best_epoch_fixed = config.get("best_epoch_fixed", None)
    max_grad_norm = config.get("max_grad_norm", 1.0)
    aux_loss_weight = config.get("aux_loss_weight", 0.0)
    contrastive_loss_weight = config.get("contrastive_loss_weight", 0.0)
    base_lr = config["learning_rate"]
    warmup_epochs = config.get("lr_warmup_epochs", 5)
    lr_min = config.get("lr_min", 1e-5)
    use_cosine = config.get("use_cosine_lr", True)
    use_step_lr = config.get("use_step_lr", False)
    step_lr_step_size = config.get("step_lr_step_size", 15)
    step_lr_gamma = config.get("step_lr_gamma", 0.5)
    num_epochs = config["num_epochs"]

    print("\nTraining (updates on train set only)...")
    if val_loader is not None:
        print("  Best checkpoint by validation accuracy")
    elif best_epoch_cap is not None:
        print(f"  Best checkpoint only within first {best_epoch_cap} epochs (" + (
            "val acc" if val_loader is not None else "train loss"
        ) + ")")
    if best_epoch_fixed is not None:
        print(f"  Fixed use epoch {best_epoch_fixed} model")
    if aux_loss_weight > 0:
        print(f"  Aux loss weight: {aux_loss_weight}")
    if contrastive_loss_weight > 0:
        print(f"  Contrastive loss weight: {contrastive_loss_weight}")
    if freeze_encoder_epochs > 0:
        print(f"  Transfer: freeze encoders for first {freeze_encoder_epochs} epochs")
    if use_cosine:
        print(f"  LR: warmup {warmup_epochs} epochs then cosine decay to {lr_min}")
    elif use_step_lr:
        print(f"  LR: StepLR step_size={step_lr_step_size}, gamma={step_lr_gamma}")
    print("=" * 60)

    for epoch in range(1, num_epochs + 1):
        for name, p in model.named_parameters():
            if "encoders" in name:
                p.requires_grad = (epoch > freeze_encoder_epochs)
            else:
                p.requires_grad = True
        if epoch == 1 and freeze_encoder_epochs > 0:
            print(f"  First {freeze_encoder_epochs} epochs: encoders frozen, heads only")
        if epoch == freeze_encoder_epochs + 1 and freeze_encoder_epochs > 0:
            print(f"  >> Epoch {epoch}: unfreeze encoders")

        if epoch <= warmup_epochs and warmup_epochs > 0:
            lr = base_lr * (epoch / warmup_epochs)
        elif use_step_lr:
            lr = base_lr * (step_lr_gamma ** ((epoch - warmup_epochs) // step_lr_step_size))
        elif use_cosine and num_epochs > warmup_epochs:
            progress = (epoch - warmup_epochs) / max(num_epochs - warmup_epochs, 1)
            lr = lr_min + 0.5 * (base_lr - lr_min) * (1.0 + np.cos(np.pi * progress))
        else:
            lr = base_lr
        for g in optimizer.param_groups:
            g["lr"] = lr

        t0 = time.time()
        metrics = train_epoch(
            model, train_loader, criterion, optimizer, device,
            max_grad_norm=max_grad_norm,
            aux_loss_weight=aux_loss_weight,
            contrastive_loss_weight=contrastive_loss_weight,
        )
        elapsed = time.time() - t0
        train_loss = metrics["loss"]
        train_acc = metrics["accuracy"]
        improved = False
        within_cap = best_epoch_cap is None or epoch <= best_epoch_cap
        if val_loader is not None:
            val_metrics = evaluate(model, val_loader, criterion, device, fault_mapping)
            val_acc = val_metrics["accuracy"]
            if within_cap and val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                improved = True
            elif within_cap and best_val_acc <= 0.26 and train_loss < best_train_loss:
                best_train_loss = train_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                improved = True
            print(f"Epoch {epoch}/{num_epochs} ({elapsed:.2f}s)  lr={lr:.2e}  train_loss={train_loss:.4f}  train_acc={train_acc:.3f}  val_acc={val_acc:.3f}  "
                  f"{'*' if improved else ''}")
        else:
            if within_cap and train_loss < best_train_loss:
                best_train_loss = train_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                improved = True
            print(f"Epoch {epoch}/{num_epochs} ({elapsed:.2f}s)  lr={lr:.2e}  train_loss={train_loss:.4f}  train_acc={train_acc:.3f}  "
                  f"{'*' if improved else ''}")

        if epoch == warmup_epochs and warmup_epochs > 0:
            print(f"  >> Warmup done, lr={base_lr:.2e}")

    if best_state is not None:
        model.load_state_dict(best_state)
        model = model.to(device)
        print("\nLoaded best checkpoint for test." + (" (val accuracy)" if val_loader is not None else " (lowest train loss)"))
        save_path = config.get("save_best_path")
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model_state_dict": model.state_dict(), "fault_mapping": fault_mapping}, save_path)
            print(f"  Saved: {save_path}")

    print("\nEvaluating on test set...")
    test_metrics = evaluate(model, test_loader, criterion, device, fault_mapping)
    print("\n" + "=" * 60)
    print("Test results")
    print("=" * 60)
    print(f"Accuracy: {test_metrics['accuracy']:.3f}")
    print(f"Weighted recall: {test_metrics['weighted_recall']:.3f}, weighted F1: {test_metrics['weighted_f1']:.3f}")
    print("\nPer-class recall / precision / F1:")
    for name in sorted(test_metrics["class_recall"].keys()):
        rec = test_metrics["class_recall"][name]
        prec = test_metrics["class_precision"][name]
        f1 = test_metrics["class_f1"][name]
        print(f"  {name}: recall={rec:.3f}, precision={prec:.3f}, f1={f1:.3f}")
    print("\nPer-modality reliability:")
    for m in test_metrics["reliability_scores"]:
        r = test_metrics["reliability_scores"][m]
        print(f"  {m}: {r:.3f}")
    if test_metrics.get("reliability_per_class"):
        rpc = test_metrics["reliability_per_class"]
        class_names = rpc.get("class_names", [f"C{c}" for c in range(num_classes)])
        print("\nClass-conditional reliability G^m(c) (test mean):")
        print("  class -> ", "  ".join(f"{n[:8]:>8}" for n in class_names))
        for m in model.modalities:
            vals = rpc["per_modality_class"].get(m, [0.0] * num_classes)
            print(f"  {m:12}", "  ".join(f"{v:.3f}" for v in vals))
    print("\nConfusion matrix:")
    cm = np.array(test_metrics["confusion_matrix"])
    print("pred -> ", "  ".join(fault_mapping["label_to_fault"].get(i, f"C{i}")[:8] for i in range(num_classes)))
    for i in range(num_classes):
        row = fault_mapping["label_to_fault"].get(i, f"C{i}")
        print(f"{row[:12]:12}", "  ".join(f"{cm[i, j]:5}" for j in range(num_classes)))
    print("=" * 60)
    n_test = len(test_loader.dataset)
    print(f"\nNote: test set has {n_test} samples; accuracy can be noisy.")
    return test_metrics


def main():
    script_dir = Path(__file__).parent

    # Defaults; override with config/default.json or config.json
    
    for cfg_name in ("config/default.json", "config.json"):
        config_path = script_dir / cfg_name
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            for k, v in loaded.items():
                config[k] = v
            break

    def _abs_if_rel(p: str) -> str:
        path = Path(p)
        return str(path.resolve()) if not path.is_absolute() else p

    if config.get("data_dir"):
        config["data_dir"] = _abs_if_rel(str(config["data_dir"]))
    if config.get("save_best_path"):
        config["save_best_path"] = _abs_if_rel(str(config["save_best_path"]))
    pre = config.get("pretrained_path") or ""
    if pre:
        config["pretrained_path"] = _abs_if_rel(pre)

    eval_seeds = config.get("eval_seeds")
    if isinstance(eval_seeds, list) and len(eval_seeds) > 0:
        print("Multi-seed run: seeds {} \n".format(eval_seeds))
        all_results = []
        for s in eval_seeds:
            print("\n" + "=" * 60)
            print(">>> Seed {}".format(s))
            print("=" * 60)
            test_metrics = run_one_seed(config, int(s))
            all_results.append(test_metrics)
        accs = np.array([r["accuracy"] for r in all_results])
        print("\n" + "=" * 60)
        print("Multi-seed test accuracy: mean={:.3f} std={:.3f}".format(
            float(np.mean(accs)), float(np.std(accs)) if len(accs) > 1 else 0.0))
        print("=" * 60)
        final_results = all_results[-1].copy()
        final_results["accuracy_mean"] = float(np.mean(accs))
        final_results["accuracy_std"] = float(np.std(accs)) if len(accs) > 1 else 0.0
        final_results["eval_seeds"] = eval_seeds
        final_results["per_seed_accuracy"] = [float(r["accuracy"]) for r in all_results]
    else:
        seed = config.get("random_seed", 44)
        final_results = run_one_seed(config, int(seed)).copy()
        final_results["random_seed"] = seed

    out_dir = script_dir / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "test_results.json"

    def _to_serializable(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: _to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_to_serializable(v) for v in obj]
        return obj

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(_to_serializable(final_results), f, ensure_ascii=False, indent=2)
    print(f"\nSaved results: {out_path}")


if __name__ == "__main__":
    main()
