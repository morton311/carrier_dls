from __future__ import annotations

from typing import Optional, Tuple, Union
import numpy as np
import torch
from torch.utils.data import DataLoader


def make_Sequence(
    time_lag: int,
    data: np.ndarray,
    pred_length: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate time-delay sequence data.

    Args:
        time_lag: Time lag for the sequence
        data: Numpy array of shape [Ntime, Nmode]
        pred_length: Prediction length for the sequence

    Returns:
        X: Input array
        Y: Output array
    """
    from tqdm import tqdm

    nSamples = data.shape[0] - time_lag - pred_length
    X = np.zeros([nSamples, time_lag, data.shape[-1]], dtype=np.float32)
    Y = np.zeros([nSamples, pred_length, data.shape[-1]], dtype=np.float32)
    for j in tqdm(np.arange(nSamples)):
        X[j] = data[j: j + time_lag, :]
        Y[j] = data[j + time_lag: j + time_lag + pred_length, :]
    return X, Y


def sample_series(
    series: np.ndarray,
    n_samples: int,
    time_lag: int = 64,
    train_ahead: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Randomly sample n_samples windows from a series."""
    if series.shape[0] - time_lag - train_ahead < n_samples:
        raise ValueError("n_samples must be less than or equal to the length of the series")
    rng = np.random.default_rng(42)
    if n_samples != 0:
        indices = rng.choice(series.shape[0] - time_lag - train_ahead, n_samples,
                             replace=True, shuffle=False)
    else:
        indices = np.arange(series.shape[0] - time_lag - train_ahead)
    indices = np.sort(indices)

    if len(series.shape) == 1:
        series = series[:, np.newaxis]

    X = np.zeros((n_samples, time_lag) + series.shape[1:])
    Y = np.zeros((n_samples, train_ahead) + series.shape[1:])
    for i, idx in enumerate(indices):
        X[i] = series[idx:idx + time_lag]
        Y[i] = series[idx + time_lag:idx + time_lag + train_ahead]
    return X, Y


def sample_series_indices(
    series_length: int,
    n_samples: int,
    time_lag: int = 64,
    train_ahead: int = 5,
    seed: int = 42,
) -> np.ndarray:
    """Return starting indices for (X, Y) windows sampled from a series of given length."""
    if series_length - time_lag - train_ahead < n_samples and n_samples != 0:
        raise ValueError("n_samples must be less than or equal to the length of the series")
    rng = np.random.default_rng(seed)
    if n_samples != 0:
        indices = rng.choice(series_length - time_lag - train_ahead, n_samples,
                             replace=True, shuffle=False)
    else:
        indices = np.arange(series_length - time_lag - train_ahead)
    return np.sort(indices)


def make_dataloader(
    X: torch.Tensor,
    Y: torch.Tensor,
    batch_size: int = 32,
    shuffle: bool = True,
    distributed: bool = False,
) -> Union[DataLoader, Tuple[DataLoader, object]]:
    """Create a DataLoader for (X, Y) tensor pairs.

    When distributed=True, returns (dataloader, sampler).
    """
    from torch.utils.data import TensorDataset

    dataset = TensorDataset(X, Y)
    if distributed:
        import os
        from torch.utils.data.distributed import DistributedSampler
        world_rank = int(os.environ["RANK"])
        sampler = DistributedSampler(dataset, seed=world_rank)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, sampler=sampler)
        return dataloader, sampler
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


class LocalGlobalDataset(torch.utils.data.Dataset):
    """Snapshot-grouped dataset for the global-local encoder-decoder model.

    One item = all GFEM elements of one time window starting at t0:
        X   (E, time_lag, F)            local input windows
        Y   (E, train_ahead, F)         local targets
        tok (E, context_window * F)     encoder tokens at the context time
        t0  scalar window start index
    where F = num_components * dof_elem. Nodal DOF arrays are stored once and
    gathered through lltogl_mat per item, so no dense per-sample copies are made.
    """

    def __init__(
        self,
        dof_comps: list,
        lltogl_mat: torch.Tensor,
        t0_list: np.ndarray,
        time_lag: int,
        train_ahead: int,
        mean: torch.Tensor,
        std: torch.Tensor,
        context_window: int = 1,
        context_time: str = 'window_start',
    ):
        if context_window > time_lag:
            raise ValueError("context_window must be <= time_lag")
        if context_time not in ('window_start', 'window_end'):
            raise ValueError(f"context_time should be 'window_start' or 'window_end', not {context_time}")
        self.dof_comps = [torch.as_tensor(c, dtype=torch.float32) for c in dof_comps]
        self.lltogl_mat = torch.as_tensor(lltogl_mat, dtype=torch.long)
        self.t0_list = np.asarray(t0_list)
        self.time_lag = time_lag
        self.train_ahead = train_ahead
        self.mean = mean
        self.std = std
        self.context_window = context_window
        self.context_time = context_time

    def __len__(self) -> int:
        return len(self.t0_list)

    def __getitem__(self, i: int):
        t0 = int(self.t0_list[i])
        tl, ta, cw = self.time_lag, self.train_ahead, self.context_window
        # (tl+ta, E, F): gather element-local slots from each component and concatenate
        block = torch.cat(
            [c[t0:t0 + tl + ta][:, self.lltogl_mat] for c in self.dof_comps], dim=2
        )
        block = (block - self.mean) / self.std
        X = block[:tl].permute(1, 0, 2)
        Y = block[tl:tl + ta].permute(1, 0, 2)
        c0 = 0 if self.context_time == 'window_start' else tl - cw
        tok = block[c0:c0 + cw].permute(1, 0, 2).reshape(block.shape[1], -1)
        return X, Y, tok, t0


def make_group_dataloader(
    dataset: torch.utils.data.Dataset,
    batch_size: int = 8,
    shuffle: bool = True,
    distributed: bool = False,
) -> Union[DataLoader, Tuple[DataLoader, object]]:
    """Create a DataLoader over a snapshot-grouped Dataset.

    When distributed=True, returns (dataloader, sampler).
    """
    if distributed:
        import os
        from torch.utils.data.distributed import DistributedSampler
        world_rank = int(os.environ["RANK"])
        sampler = DistributedSampler(dataset, seed=world_rank)  # Ensure different shuffling for each process
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, sampler=sampler, pin_memory=True, num_workers=4, persistent_workers=True)
        return dataloader, sampler
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def normalize_data(
    data: np.ndarray,
    mean: Union[float, np.ndarray],
    std: Union[float, np.ndarray],
) -> np.ndarray:
    return (data - mean) / std


def denormalize_data(
    data: np.ndarray,
    mean: Union[float, np.ndarray],
    std: Union[float, np.ndarray],
) -> np.ndarray:
    return (data * std) + mean
