"""
Multimodal fault diagnosis data loader.
Four classes: Healthy, Inner_Race_Faults, Outer_Race_Faults, Ball_Faults.
Preprocessed CSV columns: window_idx, modality_t0, ..., modality_t1023.
Filename pattern: {idx}_{CanonicalFault}_{fault}_{run}.csv (second token encodes fault type).
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

# Four-class label space
CANONICAL_FAULT_TYPES = [
    "Healthy",
    "Inner_Race_Faults",
    "Outer_Race_Faults",
    "Ball_Faults",
]

# Map raw channels to three modalities: vibration / current / rpm
MODALITY_GROUPS = {
    "vibration": ["bearingA_x", "bearingA_y", "bearingB_x", "bearingB_y"],
    "current": ["current_R", "current_S", "current_T"],
    "rpm": ["rpm"],
}


def _filename_to_canonical_fault(csv_path: Path) -> Optional[str]:
    """
    Parse fault type from CSV filename.
    Format: {idx}_{CanonicalFault}_{fault}_{run}.csv
    e.g. 1_Healthy_normal_0.csv -> Healthy, 2_Inner_inner_0.csv -> Inner_Race_Faults
    """
    stem = csv_path.stem
    parts = stem.split("_")
    if len(parts) < 2:
        return None
    first = parts[1]
    if first == "Healthy":
        return "Healthy"
    if first == "Inner":
        return "Inner_Race_Faults"
    if first == "Outer":
        return "Outer_Race_Faults"
    if first == "Ball":
        return "Ball_Faults"
    return None


class MultimodalFaultDataset(Dataset):
    """One sample = one time window with multimodal sequences and a class label."""

    def __init__(
        self,
        data_dir: str,
        split: str,
        window_size: int = 1024,
        modalities: Optional[List[str]] = None,
        augment: bool = False,
        aug_noise_std: float = 0.01,
        aug_scale_low: float = 0.95,
        aug_scale_high: float = 1.05,
        file_list: Optional[List[Tuple[Path, int]]] = None,
        use_modality_groups: bool = True,
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.window_size = window_size
        self.augment = augment
        self.aug_noise_std = aug_noise_std
        self.aug_scale_low = aug_scale_low
        self.aug_scale_high = aug_scale_high
        self.use_modality_groups = use_modality_groups
        fault_to_label = {f: i for i, f in enumerate(CANONICAL_FAULT_TYPES)}
        label_to_fault = {i: f for f, i in fault_to_label.items()}
        self.samples: List[Tuple[Dict[str, np.ndarray], int]] = []

        if file_list is not None:
            csv_files_for_mod = [p for p, _ in file_list if p.exists()]
            if not csv_files_for_mod:
                raise ValueError("file_list has no valid CSV paths")
            if modalities is None:
                sample_df = pd.read_csv(csv_files_for_mod[0], nrows=0)
                mods = []
                for col in sample_df.columns:
                    if col == "window_idx":
                        continue
                    m = col.rsplit("_t", 1)[0]
                    if m not in mods:
                        mods.append(m)
                self.modalities = sorted(mods) if not use_modality_groups else list(MODALITY_GROUPS.keys())
            else:
                self.modalities = modalities
            for csv_file, label in file_list:
                if not csv_file.exists():
                    continue
                try:
                    df = pd.read_csv(csv_file)
                except Exception:
                    continue
                for row_idx in range(len(df)):
                    features_dict = self._extract_window(df, row_idx)
                    if features_dict is not None:
                        self.samples.append((features_dict, label))
            self.fault_to_label = fault_to_label
            self.label_to_fault = label_to_fault
            self.num_classes = len(CANONICAL_FAULT_TYPES)
            return

        self.augment = self.augment and (split == "train")
        split_dir = self.data_dir / f"{split}_windows"
        if not split_dir.exists():
            raise FileNotFoundError(f"Directory does not exist: {split_dir}")
        csv_files = list(split_dir.glob("*.csv"))
        if not csv_files:
            raise ValueError(f"No CSV files in: {split_dir}")

        if modalities is None:
            sample_df = pd.read_csv(csv_files[0], nrows=0)
            mods = []
            for col in sample_df.columns:
                if col == "window_idx":
                    continue
                m = col.rsplit("_t", 1)[0]
                if m not in mods:
                    mods.append(m)
            self.modalities = sorted(mods) if not use_modality_groups else list(MODALITY_GROUPS.keys())
        else:
            self.modalities = modalities

        for csv_file in csv_files:
            canonical = _filename_to_canonical_fault(csv_file)
            if canonical is None:
                continue
            label = fault_to_label[canonical]
            try:
                df = pd.read_csv(csv_file)
            except Exception:
                continue
            for row_idx in range(len(df)):
                features_dict = self._extract_window(df, row_idx)
                if features_dict is not None:
                    self.samples.append((features_dict, label))

        self.fault_to_label = fault_to_label
        self.label_to_fault = label_to_fault
        self.num_classes = len(CANONICAL_FAULT_TYPES)

    def _extract_window(self, df: pd.DataFrame, row_idx: int) -> Optional[Dict[str, np.ndarray]]:
        out = {}
        if self.use_modality_groups:
            for mod_name, channel_names in MODALITY_GROUPS.items():
                parts = []
                for ch in channel_names:
                    cols = [c for c in df.columns if c.startswith(ch + "_t")]
                    if not cols:
                        return None
                    cols.sort(key=lambda x: int(x.split("_t")[-1]))
                    data = df.loc[row_idx, cols].values.astype(np.float32)
                    if len(data) != self.window_size or np.any(np.isnan(data)):
                        return None
                    parts.append(data)
                out[mod_name] = np.mean(parts, axis=0).astype(np.float32)
        else:
            for mod_name in self.modalities:
                cols = [c for c in df.columns if c.startswith(mod_name + "_t")]
                if not cols:
                    return None
                cols.sort(key=lambda x: int(x.split("_t")[-1]))
                data = df.loc[row_idx, cols].values.astype(np.float32)
                if len(data) != self.window_size or np.any(np.isnan(data)):
                    return None
                out[mod_name] = data
        return out

    def __len__(self) -> int:
        return len(self.samples)

    def _augment(self, data: np.ndarray) -> np.ndarray:
        out = data.astype(np.float32).copy()
        if self.aug_noise_std > 0:
            out += np.random.randn(*out.shape).astype(np.float32) * self.aug_noise_std
        scale = np.random.uniform(self.aug_scale_low, self.aug_scale_high)
        out *= scale
        # Circular time shift for diversity
        shift = np.random.randint(-int(len(out) * 0.1), int(len(out) * 0.1) + 1)
        if shift != 0:
            out = np.roll(out, shift)
        return out

    def __getitem__(self, idx: int) -> Dict:
        features_dict, label = self.samples[idx]
        if self.augment:
            modality_tensors = {
                mod: torch.from_numpy(self._augment(features_dict[mod]))
                for mod in self.modalities
            }
        else:
            modality_tensors = {
                mod: torch.from_numpy(features_dict[mod].copy()) for mod in self.modalities
            }
        return {
            "modalities": modality_tensors,
            "label": torch.tensor(label, dtype=torch.long),
        }


class RepeatedAugmentDataset(Dataset):
    """Repeat indices so each base sample is seen N times per epoch with fresh augmentation."""

    def __init__(self, dataset: MultimodalFaultDataset, repeat: int):
        self.dataset = dataset
        self.repeat = max(1, repeat)

    def __len__(self) -> int:
        return len(self.dataset) * self.repeat

    def __getitem__(self, idx: int) -> Dict:
        return self.dataset[idx % len(self.dataset)]


def create_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    window_size: int = 1024,
    num_workers: int = 4,
    random_seed: int = 42,
    augment_train: bool = True,
    aug_noise_std: float = 0.01,
    aug_scale_low: float = 0.95,
    aug_scale_high: float = 1.05,
    oversample_labels: Optional[List[int]] = None,
    oversample_ratio: float = 2.5,
    oversample_ratio_ball: float = 3.0,
    use_all_data: bool = False,
    train_ratio: float = 0.7,
    val_ratio: float = 0.0,
    augment_repeat: int = 1,
    test_repeat: int = 1,
    use_raw_modalities: bool = False,
) -> Tuple[DataLoader, Optional[DataLoader], DataLoader, Dict]:
    """
    Build train / optional val / test DataLoaders.
    use_all_data=False: use pre-split train_windows and test_windows.
    use_all_data=True: merge files then stratify by train_ratio; if val_ratio>0, split val from train.
    """
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    data_dir = Path(data_dir)
    train_dir = data_dir / "train_windows"
    test_dir = data_dir / "test_windows"
    if not train_dir.exists() or not test_dir.exists():
        raise FileNotFoundError(f"Required: {train_dir} and {test_dir}")

    fault_to_label = {f: i for i, f in enumerate(CANONICAL_FAULT_TYPES)}
    label_to_fault = {i: f for f, i in fault_to_label.items()}
    num_classes = len(CANONICAL_FAULT_TYPES)

    val_dataset = None
    if use_all_data:
        all_files: List[Tuple[Path, int]] = []
        for d in [train_dir, test_dir]:
            for csv_file in d.glob("*.csv"):
                canonical = _filename_to_canonical_fault(csv_file)
                if canonical is not None:
                    all_files.append((csv_file, fault_to_label[canonical]))
        if not all_files:
            raise ValueError("No valid CSV files found")
        by_label = defaultdict(list)
        for path, label in all_files:
            by_label[label].append((path, label))
        train_files, test_files = [], []
        for c in range(num_classes):
            lst = by_label.get(c, [])
            np.random.shuffle(lst)
            n = len(lst)
            if n >= 2 and train_ratio < 1.0:
                cut = max(1, int(n * train_ratio))
                train_files.extend(lst[:cut])
                test_files.extend(lst[cut:])
            else:
                train_files.extend(lst)
                if train_ratio >= 1.0:
                    test_files.extend(lst)
        # Optional stratified val split from train
        val_files: List[Tuple[Path, int]] = []
        if val_ratio > 0 and train_files:
            by_label_train = defaultdict(list)
            for path, label in train_files:
                by_label_train[label].append((path, label))
            train_files_new = []
            for c in range(num_classes):
                lst = by_label_train.get(c, [])
                np.random.shuffle(lst)
                n = len(lst)
                if n >= 2:
                    val_cut = max(1, int(n * val_ratio))
                    train_files_new.extend(lst[val_cut:])
                    val_files.extend(lst[:val_cut])
                else:
                    train_files_new.extend(lst)
            train_files = train_files_new
        modalities = None
        train_dataset = MultimodalFaultDataset(
            str(data_dir), "train", window_size, modalities=modalities,
            augment=augment_train, aug_noise_std=aug_noise_std,
            aug_scale_low=aug_scale_low, aug_scale_high=aug_scale_high,
            file_list=train_files,
            use_modality_groups=not use_raw_modalities,
        )
        modalities = train_dataset.modalities
        test_dataset = MultimodalFaultDataset(
            str(data_dir), "test", window_size, modalities=modalities,
            augment=False, file_list=test_files,
            use_modality_groups=not use_raw_modalities,
        )
        val_dataset = None
        if val_files:
            val_dataset = MultimodalFaultDataset(
                str(data_dir), "val", window_size, modalities=modalities,
                augment=False, file_list=val_files,
                use_modality_groups=not use_raw_modalities,
            )
    else:
        test_dataset = MultimodalFaultDataset(
            data_dir, "test", window_size, use_modality_groups=not use_raw_modalities
        )
        modalities = test_dataset.modalities
        train_dataset = MultimodalFaultDataset(
            data_dir, "train", window_size, modalities=modalities,
            augment=augment_train, aug_noise_std=aug_noise_std,
            aug_scale_low=aug_scale_low, aug_scale_high=aug_scale_high,
            use_modality_groups=not use_raw_modalities,
        )
        print(f"Data: using pre-split train_windows / test_windows (70/30)")

    fault_mapping = {
        "num_classes": num_classes,
        "label_to_fault": label_to_fault,
        "fault_to_label": fault_to_label,
        "modalities": list(modalities),
    }

    if augment_repeat > 1 and augment_train:
        train_dataset = RepeatedAugmentDataset(train_dataset, augment_repeat)
        print(f"Augment repeat: {augment_repeat}x -> effective train size {len(train_dataset)}/epoch")

    if test_repeat > 1:
        test_dataset = RepeatedAugmentDataset(test_dataset, test_repeat)
        print(f"Test repeat: {test_repeat}x -> test tensor count {len(test_dataset)} (metrics unchanged)")

    sampler = None
    base_ds = train_dataset.dataset if hasattr(train_dataset, "dataset") else train_dataset
    if oversample_labels and len(base_ds.samples) > 0:
        labels_arr = np.array([base_ds.samples[i][1] for i in range(len(base_ds))])
        weights = np.ones(len(labels_arr), dtype=np.float64)
        for i in range(len(labels_arr)):
            c = labels_arr[i]
            if c in oversample_labels:
                weights[i] = oversample_ratio_ball if c == 3 else oversample_ratio
        weights = weights / weights.sum()
        sampler = WeightedRandomSampler(
            torch.from_numpy(weights), num_samples=len(train_dataset), replacement=True
        )
        ball_note = f", Ball(3)x{oversample_ratio_ball}" if 3 in oversample_labels else " (Ball not oversampled)"
        print(f"Oversampling: labels {oversample_labels} weight x{oversample_ratio}{ball_note}")

    # Avoid empty train_loader when len(train) < batch_size
    drop_last = len(train_dataset) > batch_size
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader: Optional[DataLoader] = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    # Per-class test counts (sanity check for all classes present)
    test_base = test_dataset.dataset if hasattr(test_dataset, "dataset") else test_dataset
    if hasattr(test_base, "samples") and test_base.samples:
        from collections import Counter
        test_counts = Counter(s[1] for s in test_base.samples)
        label_to_fault = fault_mapping["label_to_fault"]
        test_dist = [f"{label_to_fault.get(c, c)}:{test_counts.get(c, 0)}" for c in range(num_classes)]
        print(f"Test set per-class counts: {test_dist}")
    val_note = f", val: {len(val_dataset)} samples" if val_dataset is not None else " (no val set)"
    print(f"Train: {len(train_dataset)} samples, test: {len(test_dataset)} samples{val_note}")
    print(f"Fault types: {list(fault_mapping['label_to_fault'].values())}")
    print(f"Modalities: {fault_mapping['modalities']}")
    return train_loader, val_loader, test_loader, fault_mapping


if __name__ == "__main__":
    data_dir = "./data/preprocessed_data"
    train_loader, val_loader, test_loader, fault_mapping = create_dataloaders(
        data_dir, batch_size=16, window_size=1024, use_all_data=True, train_ratio=0.7, val_ratio=0.15
    )
    batch = next(iter(train_loader))
    print("batch keys:", batch.keys())
    print("modalities:", list(batch["modalities"].keys()))
    print("label shape:", batch["label"].shape)
