"""Dataloaders for the foreshock-aftershock dataset."""

from typing import Any, Dict, Optional

import numpy as np
import torch

from dataloaders import foreshock_aftershock_dataset as dataset


def prepare_foreshock_aftershock_dataloaders(
  *,
  data_dir: str,
  num_classes: int,
  batch_size: int,
  event_split_method: str,
  component_order: str,
  seed: int = 42,
  remove_class_overlapping_dates: bool = False,
  train_frac: float = 0.70,
  val_frac: float = 0.10,
  test_frac: float = 0.20,
  demean_axis: Optional[int] = -1,
  amp_norm_axis: Optional[int] = -1,
  amp_norm_type: str = "peak",
  num_workers: int = 8,
  dimension_order: str = "NCW",
  collator: Any = None,
) -> Dict[str, torch.utils.data.DataLoader]:
  """Create dataloaders for the foreshock-aftershock dataset."""

  datasets = dataset.create_foreshock_aftershock_datasets(
    data_dir=data_dir,
    num_classes=num_classes,
    event_split_method=event_split_method,
    component_order=component_order,
    dimension_order=dimension_order,
    seed=seed,
    remove_class_overlapping_dates=remove_class_overlapping_dates,
    train_frac=train_frac,
    val_frac=val_frac,
    test_frac=test_frac,
  )

  # Use full sequence length as prepared upstream (SeisLM-style); avoid hardcoded cropping
  X_train, y_train = datasets["train"]["X"], datasets["train"]["y"]
  X_val, y_val = datasets["val"]["X"], datasets["val"]["y"]
  X_test, y_test = datasets["test"]["X"], datasets["test"]["y"]

  # Ensure chunked mLSTM backend compatibility while minimizing padding.
  # With ConvDown×4 and pools [4,4], using chunk_size=96 allows padded_len multiples of 2048:
  # stage lengths become 1536, 384, 96 (all divisible by 96). So we pad to the nearest multiple of 2048.
  def pad_to_multiple_of(x: np.ndarray, base: int = 2048) -> np.ndarray:
    if x.ndim != 3:
      return x
    # Handle both NCW (N, C, W) and NWC (N, W, C) formats
    if dimension_order == "NCW":
      L = x.shape[2]  # Time axis is shape[2] for NCW
      rem = L % base
      if rem == 0:
        return x
      pad = base - rem
      pad_width = ((0, 0), (0, 0), (0, pad))  # Pad time axis (axis 2)
    else:  # NWC
      L = x.shape[1]  # Time axis is shape[1] for NWC
      rem = L % base
      if rem == 0:
        return x
      pad = base - rem
      pad_width = ((0, 0), (0, pad), (0, 0))  # Pad time axis (axis 1)
    return np.pad(x, pad_width=pad_width, mode='constant')

  X_train = pad_to_multiple_of(X_train, base=2048)
  X_val = pad_to_multiple_of(X_val, base=2048)
  X_test = pad_to_multiple_of(X_test, base=2048)

  def normalize(X: np.ndarray) -> np.ndarray:
    if demean_axis is not None:
      X = X - np.mean(X, axis=demean_axis, keepdims=True)

    if amp_norm_axis is not None:
      if amp_norm_type == "std":
        X = X / (np.std(X, axis=amp_norm_axis, keepdims=True) + 1e-10)
      elif amp_norm_type == "peak":
        X = X / (np.max(np.abs(X), axis=amp_norm_axis, keepdims=True) + 1e-10)
      else:
        raise ValueError(f"Normalization type {amp_norm_type} not supported")
    return X

  X_train = normalize(X_train)
  X_val = normalize(X_val)
  X_test = normalize(X_test)

  X_train, y_train = torch.from_numpy(X_train), torch.from_numpy(y_train)
  X_val, y_val = torch.from_numpy(X_val), torch.from_numpy(y_val)
  X_test, y_test = torch.from_numpy(X_test), torch.from_numpy(y_test)

  loaders = {
    "train": torch.utils.data.DataLoader(
      torch.utils.data.TensorDataset(X_train, y_train),
      batch_size=batch_size,
      shuffle=True,
      pin_memory=True,
      num_workers=num_workers,
      collate_fn=collator,
    ),
    "val": torch.utils.data.DataLoader(
      torch.utils.data.TensorDataset(X_val, y_val),
      batch_size=batch_size,
      shuffle=False,
      pin_memory=True,
      num_workers=num_workers,
      collate_fn=collator,
    ),
    "test": torch.utils.data.DataLoader(
      torch.utils.data.TensorDataset(X_test, y_test),
      batch_size=batch_size,
      shuffle=False,
      pin_memory=True,
      num_workers=num_workers,
      collate_fn=collator,
    ),
  }

  return loaders
