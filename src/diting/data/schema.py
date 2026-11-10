"""D1 · 统一 schema 归一（官方 §1.1 数据属性 → Parquet）。

统一 schema 口径：
    timestamp, can_id, dlc, data_0..data_7, flag, source_tag
另附工程列（不进入特征矩阵）：frame_index, label, session_id, segment_id, attack_kind。

解析器兼容官方 CSV 的两种常见排版：
- 带表头逗号分隔：Timestamp,CAN ID,DLC,DATA[0..7],Flag
- 无表头、空格/逗号混合分隔（含 DATA[0] 缺位留空）
CAN ID 与 DATA 字节按十六进制解析；DLC<8 时缺失字节填 -1。

segment_id 按「时间间隔 > gap_threshold」切分（而非 Flag 连续段）：
官方数据每次注入持续 3–5 秒且注入段内会混入正常帧，用 Flag 连续段会把一个
注入片段切碎；用时间间隙切分才对应「一次注入片段 / 一段连续会话」的分组语义。
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# 统一 schema 的列顺序（帧表）
FRAME_COLUMNS = [
    "frame_index",
    "timestamp",
    "can_id",
    "dlc",
    *[f"data_{i}" for i in range(8)],
    "flag",
    "source_tag",
    "label",
    "session_id",
    "segment_id",
    "attack_kind",
]

DATA_COLUMNS = [f"data_{i}" for i in range(8)]
# 只由统一 schema 计算得到的「合法原始输入列」，供特征层做来源校验
ALLOWED_RAW_COLUMNS = ["timestamp", "can_id", "dlc", *DATA_COLUMNS]

# 从 session 名推断攻击类别（仅标签元数据，不进特征）
_KIND_HINTS = {
    "dos": "DoS",
    "fuzzy": "Fuzzy",
    "gear": "Spoofing_the_drive_gear",
    "rpm": "Spoofing_the_RPM_gauge",
}


def _infer_kind(session_id: str) -> str:
    low = session_id.lower()
    for key, kind in _KIND_HINTS.items():
        if key in low:
            return kind
    return "Normal" if "normal" in low else "Attack"


def _split_line(line: str) -> list[str]:
    """按逗号或空白切分一行，容忍多余空格。"""
    if "," in line:
        return [tok.strip() for tok in line.split(",")]
    return [tok.strip() for tok in re.split(r"\s+", line.strip()) if tok]


def _is_header(tokens: list[str]) -> bool:
    return bool(tokens) and tokens[0].lower().startswith("timestamp")


def _parse_hex(token: str) -> int:
    """解析十六进制字节/CAN ID；支持 0x 前缀与空值。"""
    t = token.strip().lower()
    if t in {"", "none", "nan", "na", "null", "-"}:
        return -1
    t = t.removeprefix("0x")
    try:
        return int(t, 16)
    except ValueError:
        # 少数镜像把 DATA 写成十进制，兜底按十进制解析
        return int(t, 10)


def parse_raw_file(
    path: str | Path,
    source_tag: str,
    session_id: str,
    gap_threshold: float = 0.02,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """解析单个原始 CSV，返回统一 schema 帧表与统计。"""
    path = Path(path)
    rows: list[list[str]] = []
    skipped = 0
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            tokens = _split_line(line)
            if _is_header(tokens):
                continue
            if len(tokens) < 4:
                skipped += 1
                continue
            rows.append(tokens)

    records: list[dict[str, Any]] = []
    for tokens in rows:
        try:
            timestamp = float(tokens[0])
        except ValueError:
            skipped += 1
            continue
        can_id = _parse_hex(tokens[1])
        try:
            dlc = int(tokens[2])
        except ValueError:
            skipped += 1
            continue
        # Flag 在最后一列；某些镜像在 DLC<8 时会压缩空 DATA 列，导致列数不足 12。
        # 先剥离 Flag，再取最多 8 个 DATA 列并按 -1 补齐，避免 Flag 被误当字节解析。
        tail = tokens[-1].strip().upper()
        if tail in {"T", "R"}:
            flag = tail
            payload_tokens = tokens[3:-1]
        else:
            flag = "R"
            payload_tokens = tokens[3:]
        data_tokens = payload_tokens[:8]
        data = [_parse_hex(t) for t in data_tokens] + [-1] * (8 - len(data_tokens))
        records.append(
            {
                "timestamp": timestamp,
                "can_id": can_id,
                "dlc": dlc,
                **{f"data_{i}": data[i] for i in range(8)},
                "flag": flag,
                "source_tag": source_tag,
                "label": 1 if flag == "T" else 0,
                "session_id": session_id,
                "attack_kind": _infer_kind(session_id),
            }
        )

    if not records:
        raise ValueError(f"原始文件无有效数据行：{path}")

    df = pd.DataFrame.from_records(records)
    df = df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    df["frame_index"] = np.arange(len(df), dtype=np.int64)
    # 时间间隔超过阈视为新片段（注入片段 / 会话连续段）
    dt = df["timestamp"].diff().fillna(0.0)
    new_seg = (dt > gap_threshold).cumsum()
    df["segment_id"] = (
        df["session_id"].astype(str) + "__seg" + new_seg.astype(int).astype(str).str.zfill(4)
    )
    df = df[FRAME_COLUMNS]

    stats = {
        "path": str(path),
        "session_id": session_id,
        "source_tag": source_tag,
        "raw_data_lines": len(rows),
        "parsed_rows": len(df),
        "skipped_rows": skipped,
        "n_injected": int(df["label"].sum()),
        "n_normal": int((1 - df["label"]).sum()),
        "n_segments": int(df["segment_id"].nunique()),
        "unique_can_ids": int(df["can_id"].nunique()),
        "t_start": float(df["timestamp"].min()),
        "t_end": float(df["timestamp"].max()),
        "row_count_match": len(df) + skipped == len(rows),
    }
    return df, stats


def normalize_all(
    raw_dir: Path,
    interim_dir: Path,
    source_tag: str,
    gap_threshold: float = 0.02,
    exclude_dirs: tuple[str, ...] = (),
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """归一化目录下所有原始 CSV，拼为一份帧表并写 Parquet。

    exclude_dirs：跳过指定子目录（W2 起用于把 `car-hacking/` 本地真实数据目录
    与 W1 的 simulated 测试路径隔离，避免两套数据混进同一份帧表）。
    """
    files = sorted(
        [
            p for p in raw_dir.rglob("*.csv")
            if p.is_file() and not any(part in exclude_dirs for part in p.parts)
        ]
    )
    if not files:
        raise FileNotFoundError(f"未找到原始 CSV：{raw_dir}")

    frames: list[pd.DataFrame] = []
    per_file: list[dict[str, Any]] = []
    for p in files:
        is_sim = "SIMULATED" in p.name or "simulated" in p.parts
        tag = "simulated" if is_sim else source_tag
        session_id = p.stem.replace("SIMULATED_NOT_REAL_DATA__", "")
        df, stats = parse_raw_file(p, tag, session_id, gap_threshold=gap_threshold)
        stats["is_real_data"] = not is_sim
        frames.append(df)
        per_file.append(stats)

    full = pd.concat(frames, ignore_index=True)
    # 按全局时间排序后重编号，保证 frame_index 与 timestamp 同序
    full = full.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    full["frame_index"] = np.arange(len(full), dtype=np.int64)

    interim_dir.mkdir(parents=True, exist_ok=True)
    out_path = interim_dir / "frames.parquet"
    full.to_parquet(out_path, index=False, engine="pyarrow")

    total_raw = sum(f["raw_data_lines"] for f in per_file)
    total_parsed = sum(f["parsed_rows"] for f in per_file)
    stats_all: dict[str, Any] = {
        "schema": FRAME_COLUMNS,
        "parquet": str(out_path),
        "n_files": len(per_file),
        "raw_data_lines_total": total_raw,
        "parquet_rows_total": len(full),
        "per_file_parsed_rows_total": total_parsed,
        "row_count_match": len(full) == total_parsed == total_raw,
        "n_injected": int(full["label"].sum()),
        "n_normal": int((1 - full["label"]).sum()),
        "n_segments": int(full["segment_id"].nunique()),
        "n_sessions": int(full["session_id"].nunique()),
        "source_tags": sorted(full["source_tag"].unique().tolist()),
        "is_real_data": bool((full["source_tag"] != "simulated").any()),
        "gap_threshold_s": gap_threshold,
        "per_file": per_file,
    }
    return full, stats_all


def load_frames(interim_dir: Path) -> pd.DataFrame:
    return pd.read_parquet(interim_dir / "frames.parquet", engine="pyarrow")
