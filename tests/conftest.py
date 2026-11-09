"""pytest 公共 fixture：内存构造小样本，不依赖外部数据集。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from diting.common import PROJECT_ROOT, load_config
from diting.data.schema import FRAME_COLUMNS


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def project_root() -> Path:
    return PROJECT_ROOT


def _make_frames(n_segments: int = 24, frames_per_segment: int = 200, seed: int = 7) -> pd.DataFrame:
    """构造与统一 schema 一致的内存帧表（正常段与注入段交替）。"""
    rng = np.random.default_rng(seed)
    rows = []
    t = 0.0
    for s in range(n_segments):
        is_attack = s % 2 == 1
        for i in range(frames_per_segment):
            t += float(rng.uniform(0.0003, 0.0008)) if is_attack else float(rng.uniform(0.004, 0.012))
            injected = is_attack and (i % 7 != 6)
            can_id = 0x000 if (is_attack and injected) else int(rng.integers(0x100, 0x500))
            dlc = 8
            data = {f"data_{j}": int(rng.integers(0, 256)) for j in range(8)}
            rows.append(
                {
                    "frame_index": len(rows),
                    "timestamp": t,
                    "can_id": can_id,
                    "dlc": dlc,
                    **data,
                    "flag": "T" if injected else "R",
                    "source_tag": "simulated",
                    "label": 1 if injected else 0,
                    "session_id": "unit-session",
                    "segment_id": f"unit-session__seg{s:04d}",
                    "attack_kind": "DoS" if is_attack else "Normal",
                }
            )
        t += 0.05
    return pd.DataFrame(rows)[FRAME_COLUMNS]


@pytest.fixture(scope="session")
def sample_frames() -> pd.DataFrame:
    return _make_frames()


@pytest.fixture(scope="session")
def sample_windows(sample_frames, cfg):
    from diting.data.labels import build_windows

    return build_windows(
        sample_frames,
        window_size=int(cfg["window"]["size_frames"]),
        stride=int(cfg["window"]["stride_frames"]),
        theta=float(cfg["window"]["attack_ratio_theta"]),
    )


@pytest.fixture(scope="session")
def sample_split(sample_windows, cfg):
    from diting.split.make_split import build_split_manifest

    return build_split_manifest(sample_windows, cfg)
