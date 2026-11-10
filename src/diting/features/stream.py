"""D4 · 流式特征计算（真实数据量级下的分批实现）。

W1 的 `compute_window_features(frames, windows, id_freq)` 要求一次性把全量帧表
读入内存；真实数据集约 500 万帧，目标环境 8 GB 内存不适合。本模块按 segment 分批：

1. 顺序扫描 frames.parquet，每次只在内存里保留一个 segment 的帧；
2. 把该 segment 的 frame_index 重映射为 0..n-1，同时把该段窗口的
   frame_start/frame_end 减去同一 offset，保持与 W1 完全一致的索引语义；
3. 复用 W1 唯一的 `compute_window_features` 实现算出 G1–G6 特征，
   再把 frame_start/frame_end 还原为全局值。

等价性说明：W1 中帧级中间量按 `segment_id` 分组计算（`_dt`、`_same_id_dt`、
`same_id_prev`），窗口不跨 segment；因此「按段独立计算」与「全表一次计算」
逐值等价（段首帧的 `valid_prev` 为 False，不受跨段前驱影响）。

ID 频次表（只在 train 折 fit）也按段累加 value_counts，避免全表载入。
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pandas as pd

from ..data.local_raw import iter_segments
from .build import FEATURE_COLUMNS, compute_window_features


def iter_segment_batches(
    interim_dir: Path, windows: pd.DataFrame, chunk_size: int
) -> Iterator[tuple[str, int, pd.DataFrame, pd.DataFrame]]:
    """yield (segment_id, frame_index_offset, 局部帧表, 局部窗口表)。"""
    by_seg = {
        str(seg): grp.reset_index(drop=True)
        for seg, grp in windows.groupby("segment_id", sort=False)
    }
    seen: set[str] = set()
    for seg_id, seg_frames in iter_segments(interim_dir, chunk_size):
        seen.add(seg_id)
        w = by_seg.get(seg_id)
        if w is None or w.empty:
            continue
        offset = int(seg_frames["frame_index"].min())
        local_frames = seg_frames.copy()
        local_frames["frame_index"] = local_frames["frame_index"].to_numpy() - offset
        local_windows = w.copy()
        local_windows["frame_start"] = local_windows["frame_start"].to_numpy() - offset
        local_windows["frame_end"] = local_windows["frame_end"].to_numpy() - offset
        yield seg_id, offset, local_frames, local_windows
    missing = sorted(set(by_seg) - seen)
    if missing:
        raise ValueError(f"窗口引用的 segment 在帧表中不存在（前 5 个）：{missing[:5]}")


def fit_id_frequency_streaming(
    interim_dir: Path, train_segments: set[str], chunk_size: int
) -> dict[str, float]:
    """只在 train 折帧上统计 CAN ID 频次编码（流式累加，等价于全表 value_counts）。"""
    counts: dict[int, int] = {}
    total = 0
    for seg_id, seg_frames in iter_segments(interim_dir, chunk_size):
        if seg_id not in train_segments:
            continue
        vc = seg_frames["can_id"].value_counts()
        for k, v in vc.items():
            key = int(k)
            counts[key] = counts.get(key, 0) + int(v)
        total += len(seg_frames)
    if total == 0:
        raise ValueError("train 折无帧，无法 fit ID 频次表")
    return {str(k): float(v / total) for k, v in counts.items()}


def compute_window_features_streaming(
    interim_dir: Path, windows: pd.DataFrame, id_freq: dict[str, float], chunk_size: int
) -> pd.DataFrame:
    """按 segment 分批计算 G1–G6 窗口特征，返回与 W1 同结构的特征表。"""
    parts: list[pd.DataFrame] = []
    for seg_id, offset, local_frames, local_windows in iter_segment_batches(
        interim_dir, windows, chunk_size
    ):
        part = compute_window_features(local_frames, local_windows, id_freq)
        # 还原全局 frame_index（划分清单与报告使用全局值）
        part["frame_start"] = part["frame_start"].to_numpy() + offset
        part["frame_end"] = part["frame_end"].to_numpy() + offset
        parts.append(part)
    if not parts:
        raise ValueError("未计算出任何窗口特征")
    out = pd.concat(parts, ignore_index=True)
    missing_cols = [c for c in FEATURE_COLUMNS if c not in out.columns]
    if missing_cols:
        raise ValueError(f"特征列缺失：{missing_cols}")
    return out


def load_segment_frames(interim_dir: Path, segment_id: str) -> pd.DataFrame:
    """只读取指定 segment 的帧（供 golden vector 与测试使用，避免全表载入）。"""
    import pyarrow.parquet as pq

    table = pq.read_table(
        interim_dir / "frames.parquet", filters=[("segment_id", "=", segment_id)]
    )
    return table.to_pandas()
