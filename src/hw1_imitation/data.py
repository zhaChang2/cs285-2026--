"""Dataset utilities for Push-T."""

from __future__ import annotations

import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import zarr
from torch.utils.data import Dataset

PUSHT_URL = "https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip"
ZARR_RELATIVE_PATH = Path("pusht") / "pusht_cchi_v7_replay.zarr"


@dataclass(frozen=True)
class Normalizer:
    """Feature-wise normalizer for states and actions."""
    # 对 state / action 做按维度标准化:
    # x_norm = (x - mean) / std

    state_mean: np.ndarray
    state_std: np.ndarray
    action_mean: np.ndarray
    action_std: np.ndarray

    @staticmethod
    def _safe_std(std: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        return np.maximum(std, eps)

    @classmethod
    def from_data(cls, states: np.ndarray, actions: np.ndarray) -> "Normalizer":
        # states: [N, state_dim], actions: [N, action_dim]
        state_mean = states.mean(axis=0)
        state_std = cls._safe_std(states.std(axis=0))
        action_mean = actions.mean(axis=0)
        action_std = cls._safe_std(actions.std(axis=0))
        return cls(state_mean, state_std, action_mean, action_std)

    def normalize_state(self, state: np.ndarray) -> np.ndarray:
        return (state - self.state_mean) / self.state_std

    def normalize_action(self, action: np.ndarray) -> np.ndarray:
        return (action - self.action_mean) / self.action_std

    def denormalize_action(self, action: np.ndarray) -> np.ndarray:
        return action * self.action_std + self.action_mean


def download_pusht(dataset_dir: Path) -> Path:#->函数返回值是一个Path对象，表示解压后的Zarr数据集的路径
    """Download and extract the Push-T dataset if needed.

    Returns the path to the extracted Zarr dataset.
    """

    dataset_dir.mkdir(parents=True, exist_ok=True)
    zarr_path = dataset_dir / ZARR_RELATIVE_PATH
    if zarr_path.exists():
        return zarr_path

    zip_path = dataset_dir / "pusht.zip"
    if not zip_path.exists():
        urllib.request.urlretrieve(PUSHT_URL, zip_path)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(dataset_dir)

    return zarr_path


def load_pusht_zarr(zarr_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # zarr 中保存的是把所有 episode 拼接在一起的时间序列
    root = zarr.open(zarr_path, mode="r")
    # states: [N, state_dim]，Push-T 通常 state_dim=5
    states = np.asarray(root["data"]["state"][:], dtype=np.float32)
    # actions: [N, action_dim]，Push-T 通常 action_dim=2
    actions = np.asarray(root["data"]["action"][:], dtype=np.float32)
    # episode_ends: [num_episodes]，每个值是该 episode 结束后的全局下标
    episode_ends = np.asarray(root["meta"]["episode_ends"][:], dtype=np.int64)
    return states, actions, episode_ends


def build_valid_indices(episode_ends: np.ndarray, chunk_size: int) -> np.ndarray:
    # 计算所有可作为“chunk 起点”的时间步 t
    # 要求 [t, t+chunk_size) 全部落在同一个 episode 内，不能跨 episode 边界
    starts = np.concatenate(([0], episode_ends[:-1]))
    indices: list[int] = []
    for start, end in zip(starts, episode_ends, strict=True):
        last_start = end - chunk_size
        if last_start < start:
            continue
        indices.extend(range(start, last_start + 1))
    return np.asarray(indices, dtype=np.int64)


class PushtChunkDataset(Dataset):
    """Dataset of (state, action_chunk) pairs using a sliding window."""

    def __init__(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        episode_ends: np.ndarray,
        chunk_size: int,
        normalizer: Normalizer | None = None,
    ) -> None:
        self.states = states
        self.actions = actions
        self.chunk_size = chunk_size
        self.normalizer = normalizer
        # indices 里存的是所有合法起点 t
        self.indices = build_valid_indices(episode_ends, chunk_size)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # idx 是第几个样本；先映射到全局时间步 t
        t = int(self.indices[idx])
        # obs: 单个时刻的状态，形状 [state_dim]
        state = self.states[t]
        # chunk: 从 t 开始长度为 chunk_size 的动作序列，形状 [chunk_size, action_dim]
        action_chunk = self.actions[t : t + self.chunk_size]

        if self.normalizer is not None:
            state = self.normalizer.normalize_state(state)
            action_chunk = self.normalizer.normalize_action(action_chunk)

        # 最终返回:
        # obs -> FloatTensor[state_dim]
        # chunk -> FloatTensor[chunk_size, action_dim]
        return (
            torch.from_numpy(state).float(),
            torch.from_numpy(action_chunk).float(),
        )
