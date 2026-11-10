"""D2 · 标签生成（帧级 / 窗口级），标签与特征分目录存放。

- 帧级标签：Flag(T→1, R→0)，由 D1 归一化时映射，此处单独导出到 labels/。
- 窗口级标签：窗口内注入帧占比 >= theta 标为攻击，否则 Normal。
- 窗口按固定帧数 N、步长 N/2 重叠划分，且**窗口不跨 segment 边界**。
- 标签目录 data/processed/labels/ 与特征目录 data/processed/features/ 物理隔离。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def build_windows(
    frames: pd.DataFrame,
    window_size: int,
    stride: int,
    theta: float,
) -> pd.DataFrame:
    """按 segment 内部滑窗，产出窗口索引清单与窗口级标签。"""
    if window_size < 2 or stride < 1 or stride > window_size:
        raise ValueError("窗口参数非法：需 window_size>=2 且 1<=stride<=window_size")

    records: list[dict[str, Any]] = []
    for segment_id, seg in frames.groupby("segment_id", sort=False):
        idx = seg["frame_index"].to_numpy()
        labels = seg["label"].to_numpy()
        ts = seg["timestamp"].to_numpy()
        if len(idx) < window_size:
            continue
        for start in range(0, len(idx) - window_size + 1, stride):
            end = start + window_size
            w_labels = labels[start:end]
            ratio = float(w_labels.mean())
            records.append(
                {
                    "window_id": f"{segment_id}__w{start:06d}",
                    "segment_id": segment_id,
                    "session_id": str(seg["session_id"].iloc[0]),
                    "source_tag": str(seg["source_tag"].iloc[0]),
                    "frame_start": int(idx[start]),
                    "frame_end": int(idx[end - 1]),
                    "t_start": float(ts[start]),
                    "t_end": float(ts[end - 1]),
                    "n_frames": int(end - start),
                    "n_injected": int(w_labels.sum()),
                    "attack_ratio": ratio,
                    "label": int(ratio >= theta),
                    "dominant_class": (
                        str(seg["attack_kind"].mode().iloc[0])
                        if "attack_kind" in seg and ratio >= theta
                        else "Normal"
                    ),
                }
            )
    win = pd.DataFrame.from_records(records)
    if win.empty:
        raise ValueError("未生成任何窗口：请减小 window.size_frames 或增大样本量")
    return win


def write_labels(
    frames: pd.DataFrame,
    windows: pd.DataFrame,
    labels_dir: Path,
    theta: float,
) -> dict[str, Any]:
    """写出帧级 / 窗口级标签，并返回统计。"""
    labels_dir.mkdir(parents=True, exist_ok=True)
    frame_labels = frames[["frame_index", "label", "flag", "segment_id", "source_tag"]].copy()
    frame_labels.to_parquet(labels_dir / "frame_labels.parquet", index=False, engine="pyarrow")
    window_labels = windows[
        [
            "window_id",
            "segment_id",
            "session_id",
            "source_tag",
            "frame_start",
            "frame_end",
            "t_start",
            "t_end",
            "n_frames",
            "n_injected",
            "attack_ratio",
            "label",
            "dominant_class",
        ]
    ].copy()
    window_labels.to_parquet(labels_dir / "window_labels.parquet", index=False, engine="pyarrow")

    return {
        "theta": theta,
        "frame_label_rows": len(frame_labels),
        "frame_positive": int(frame_labels["label"].sum()),
        "window_label_rows": len(window_labels),
        "window_positive": int(window_labels["label"].sum()),
        "window_negative": int((1 - window_labels["label"]).sum()),
        "files": ["frame_labels.parquet", "window_labels.parquet"],
        "storage_note": "标签存于 data/processed/labels/，特征存于 data/processed/features/，物理隔离",
    }


def attach_attack_kind(frames: pd.DataFrame, sim_meta: dict[str, Any] | None) -> pd.DataFrame:
    """把 simulated 生成时的攻击类型回填到帧表（仅标签元数据，不进特征）。"""
    if not sim_meta:
        return frames
    lookup: dict[tuple[str, int], str] = {}
    for f in sim_meta.get("files", []):
        session = f["session_id"]
        for seg in f["segments"]:
            lookup[(session, seg["segment_no"])] = seg["attack_kind"]
    if not lookup:
        return frames
    # segment_no 在 parse 时由 Flag 连续段重排，这里按 session + 段序号近似对齐
    kinds = []
    for session_id, seg in frames.groupby("session_id", sort=False):
        ordered = sorted(seg["segment_id"].unique())
        for i, sid in enumerate(ordered):
            kinds.append((sid, lookup.get((session_id, i), "unknown")))
    mapping = dict(kinds)
    out = frames.copy()
    out["attack_kind"] = out["segment_id"].map(mapping).fillna("unknown")
    return out
