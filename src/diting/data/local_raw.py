"""本地真实数据接入（HCRL Car-Hacking / Kaggle 镜像），流式分批实现。

本模块是「本地原始文件接入」路径，与 `download.py` + `simulate.py`
（simulated 替代路径，仍保留用于测试）并存。

真实文件为公开数据集 zip 的解压产物，放在 `data/raw/car-hacking/`，共 5 个：

- 攻击集 CSV（4 个）：`Timestamp,CAN ID(HEX),DLC,DATA[0..DLC-1](HEX),Flag(T/R)`，
  **无表头**，行尾 CRLF，DLC<8 时 DATA 列数随 DLC 变化，Flag 恒为最后一个字段。
- `normal_run_data.txt`：candump 文本，
  `Timestamp: <秒> ID: <hex> 000 DLC: <n> <hex bytes...>`，**无 Flag 列**，
  按设计约定，Attack-free 全部视为正常帧（label=0），仅作正常流量来源。

流式策略（内存约束：不得一次性读入内存）：

- 逐文件、逐 chunk 解析，用 `pyarrow.parquet.ParquetWriter` 追加写 frames.parquet；
- 以「时间间隙 > gap_threshold」或「单段帧数达 segment_max_frames」切 segment，
  使每个 segment 的帧在 parquet 中连续、帧数有上界，便于后续按段流式算窗口与特征；
- 全局 frame_index 按文件顺序单调递增，不做逐帧全局时间重排 —— 5 个子集是
  不同时段采集的独立会话，全局重排反而会打散同一会话的帧块、破坏防泄漏分组。

本次不访问任何外部网络（数据已本地化）。
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ..common import PROJECT_ROOT, sha256_file, utc_now, write_json
from .schema import FRAME_COLUMNS, _parse_hex

# 真实数据统一来源标记（设计约束）
SOURCE_TAG = "public:car-hacking"

# 5 个真实文件的字段级规格；official_count 取自官方页面 §1.2（configs/default.yaml）
RAW_FILE_SPECS: tuple[dict[str, Any], ...] = (
    {
        "file": "DoS_dataset.csv", "session_id": "DoS", "fmt": "attack_csv",
        "attack_kind": "DoS", "official_key": "DoS", "has_flag": True,
    },
    {
        "file": "Fuzzy_dataset.csv", "session_id": "Fuzzy", "fmt": "attack_csv",
        "attack_kind": "Fuzzy", "official_key": "Fuzzy", "has_flag": True,
    },
    {
        "file": "RPM_dataset.csv", "session_id": "RPM", "fmt": "attack_csv",
        "attack_kind": "Spoofing_the_RPM_gauge", "official_key": "RPM", "has_flag": True,
    },
    {
        "file": "gear_dataset.csv", "session_id": "Gear", "fmt": "attack_csv",
        "attack_kind": "Spoofing_the_drive_gear", "official_key": "Gear", "has_flag": True,
    },
    {
        "file": "normal_run_data.txt", "session_id": "Attack-free", "fmt": "candump",
        "attack_kind": "Normal", "official_key": "Attack-free", "has_flag": False,
    },
)

# 与 schema.FRAME_COLUMNS 顺序完全一致的 Arrow schema，保证各 chunk 类型稳定。
PA_SCHEMA = pa.schema(
    [
        ("frame_index", pa.int64()),
        ("timestamp", pa.float64()),
        ("can_id", pa.int64()),
        ("dlc", pa.int64()),
        *[(f"data_{i}", pa.int64()) for i in range(8)],
        ("flag", pa.string()),
        ("source_tag", pa.string()),
        ("label", pa.int64()),
        ("session_id", pa.string()),
        ("segment_id", pa.string()),
        ("attack_kind", pa.string()),
    ]
)


def count_lines_and_hash(path: Path) -> dict[str, Any]:
    """二进制流式统计行数与 SHA-256（不把文件读入内存）。"""
    h = hashlib.sha256()
    newlines = 0
    tail = b""
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
            newlines += block.count(b"\n")
            tail = block[-1:]
    return {
        "size_bytes": path.stat().st_size,
        "sha256": h.hexdigest(),
        "physical_newlines": newlines,
        # 末行无换行符时仍算一个逻辑行（本数据集 normal 文件末尾恰有一行残缺）
        "logical_lines": newlines + (0 if tail == b"\n" else 1),
    }


def inventory_raw_files(cfg: dict[str, Any], raw_dir: Path) -> dict[str, Any]:
    """逐文件核对校验和、字节数、行数与官方计数（±5% 容差，验收标准 #1）。"""
    spec_cfg = cfg["local_data"]
    official = cfg["dataset"]["official_counts"]
    records: list[dict[str, Any]] = []
    for spec in RAW_FILE_SPECS:
        p = raw_dir / spec["file"]
        if not p.is_file():
            raise FileNotFoundError(f"真实数据文件缺失：{p}")
        base = count_lines_and_hash(p)
        oc = int(official[spec["official_key"]])
        line_count = int(base["physical_newlines"])
        dev = (line_count - oc) / oc
        n_used = min(int(spec_cfg["max_rows_per_file"]), line_count)
        records.append(
            {
                "file": spec["file"],
                "session_id": spec["session_id"],
                "fmt": spec["fmt"],
                "attack_kind": spec["attack_kind"],
                "has_flag": spec["has_flag"],
                "size_bytes": base["size_bytes"],
                "sha256": base["sha256"],
                "physical_newlines": line_count,
                "logical_lines": base["logical_lines"],
                "official_count": oc,
                "deviation_pct": round(dev * 100, 6),
                "within_5pct": abs(dev) <= 0.05,
                "rows_requested": int(spec_cfg["max_rows_per_file"]),
                "rows_used": n_used,
                "is_subset": n_used < line_count,
            }
        )
    return {
        "checked_at": utc_now(),
        "source_tag": SOURCE_TAG,
        "files": records,
        "n_files": len(records),
        "all_within_5pct": all(r["within_5pct"] for r in records),
        "raw_dir": str(raw_dir),
    }


def verify_zip(cfg: dict[str, Any]) -> dict[str, Any]:
    """校验输入 zip 的 SHA-256 / 字节数（只读，不解压、不改写输入）。"""
    z = cfg["local_data"]["zip"]
    p = Path(z["path"])
    if not p.is_absolute():
        p = (PROJECT_ROOT / p).resolve()
    rec: dict[str, Any] = {
        "path": str(p),
        "exists": p.is_file(),
        "expected_sha256": z["sha256"],
        "expected_size_bytes": int(z["size_bytes"]),
    }
    if not p.is_file():
        rec["verified"] = False
        rec["error"] = "zip 不存在"
        return rec
    actual = sha256_file(p)
    size = p.stat().st_size
    rec.update(
        {
            "actual_sha256": actual,
            "actual_size_bytes": size,
            "sha256_match": actual == z["sha256"],
            "size_match": size == int(z["size_bytes"]),
        }
    )
    rec["verified"] = bool(rec["sha256_match"] and rec["size_match"])
    return rec


def _parse_attack_csv_line(line: str) -> dict[str, Any] | None:
    """解析无表头攻击集 CSV 行；DLC<8 时先剥离末尾 Flag 再取 DATA。"""
    tokens = [t.strip() for t in line.strip().split(",")]
    if len(tokens) < 4:
        return None
    try:
        timestamp = float(tokens[0])
        dlc = int(tokens[2])
    except ValueError:
        return None
    can_id = _parse_hex(tokens[1])
    tail = tokens[-1].upper()
    if tail in {"T", "R"}:
        flag = tail
        payload_tokens = tokens[3:-1]
    else:
        # 容错：个别行缺 Flag，按正常帧处理并记录（不伪造标签）
        flag = "R"
        payload_tokens = tokens[3:]
    data_tokens = payload_tokens[:8]
    data = [_parse_hex(t) for t in data_tokens]
    data += [-1] * (8 - len(data))
    return {
        "timestamp": timestamp,
        "can_id": can_id,
        "dlc": dlc,
        "data": data,
        "flag": flag,
    }


def _parse_candump_line(line: str) -> dict[str, Any] | None:
    """解析 candump 文本行；末尾残缺行（字段不足）直接跳过。"""
    tokens = line.split()
    if len(tokens) < 8 or tokens[0] != "Timestamp:" or tokens[2] != "ID:":
        return None
    if tokens[5] != "DLC:":
        return None
    try:
        timestamp = float(tokens[1])
        dlc = int(tokens[6])
    except ValueError:
        return None
    can_id = _parse_hex(tokens[3])
    data_tokens = tokens[7 : 7 + dlc][:8]
    data = [_parse_hex(t) for t in data_tokens]
    data += [-1] * (8 - len(data))
    return {
        "timestamp": timestamp,
        "can_id": can_id,
        "dlc": dlc,
        "data": data,
        "flag": "R",  # Attack-free：无 Flag 列，全部正常帧
    }


def _record_from_parsed(
    parsed: dict[str, Any], spec: dict[str, Any], frame_index: int, segment_id: str
) -> dict[str, Any]:
    data = parsed["data"]
    flag = parsed["flag"]
    return {
        "frame_index": frame_index,
        "timestamp": parsed["timestamp"],
        "can_id": parsed["can_id"],
        "dlc": parsed["dlc"],
        **{f"data_{i}": data[i] for i in range(8)},
        "flag": flag,
        "source_tag": SOURCE_TAG,
        "label": 1 if flag == "T" else 0,
        "session_id": spec["session_id"],
        "segment_id": segment_id,
        "attack_kind": spec["attack_kind"],
    }


def write_frames_streaming(
    cfg: dict[str, Any], raw_dir: Path, interim_dir: Path, inventory: dict[str, Any]
) -> dict[str, Any]:
    """逐文件逐 chunk 解析并追加写 frames.parquet（内存只占一个 chunk）。"""
    lc = cfg["local_data"]
    chunk_size = int(lc["chunk_size"])
    gap = float(lc["gap_threshold"])
    seg_max = int(lc["segment_max_frames"])
    interim_dir.mkdir(parents=True, exist_ok=True)
    out_path = interim_dir / "frames.parquet"

    per_file: list[dict[str, Any]] = []
    segment_index: list[dict[str, Any]] = []
    frame_index = 0
    total_injected = 0
    total_normal = 0
    raw_lines_total = 0

    writer = pq.ParquetWriter(out_path, PA_SCHEMA, compression="snappy")
    try:
        for spec, inv in zip(RAW_FILE_SPECS, inventory["files"]):
            path = raw_dir / spec["file"]
            parser = _parse_attack_csv_line if spec["fmt"] == "attack_csv" else _parse_candump_line
            max_rows = int(inv["rows_used"])
            n_parsed = 0
            n_skipped = 0
            n_injected = 0
            prev_ts: float | None = None
            seg_no = 0
            seg_frames = 0
            seg_start_frame = frame_index
            seg_t_start: float | None = None
            seg_t_end: float | None = None
            seg_injected = 0
            buf: list[dict[str, Any]] = []

            def flush() -> None:
                nonlocal buf
                if not buf:
                    return
                df = pd.DataFrame.from_records(buf, columns=FRAME_COLUMNS)
                table = pa.Table.from_pandas(df, schema=PA_SCHEMA, preserve_index=False)
                writer.write_table(table, row_group_size=chunk_size)
                buf = []

            with path.open("r", encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    if n_parsed >= max_rows:
                        break
                    raw_lines_total += 1
                    parsed = parser(line)
                    if parsed is None:
                        n_skipped += 1
                        continue
                    ts = parsed["timestamp"]
                    dt = 0.0 if prev_ts is None else ts - prev_ts
                    # 同一会话内：时间间隙过大或单段帧数达上限 → 切新段（流式分块）
                    if prev_ts is None or dt > gap or dt < 0 or seg_frames >= seg_max:
                        if seg_frames > 0:
                            segment_index.append(
                                {
                                    "segment_id": f"{spec['session_id']}__seg{seg_no:04d}",
                                    "session_id": spec["session_id"],
                                    "source_tag": SOURCE_TAG,
                                    "frame_start": seg_start_frame,
                                    "frame_end": frame_index - 1,
                                    "n_frames": seg_frames,
                                    "n_injected": seg_injected,
                                    "t_start": seg_t_start,
                                    "t_end": seg_t_end,
                                }
                            )
                        seg_no += 1
                        seg_frames = 0
                        seg_injected = 0
                        seg_start_frame = frame_index
                        seg_t_start = ts
                    seg_id = f"{spec['session_id']}__seg{seg_no:04d}"
                    rec = _record_from_parsed(parsed, spec, frame_index, seg_id)
                    buf.append(rec)
                    n_parsed += 1
                    n_injected += rec["label"]
                    seg_injected += rec["label"]
                    seg_frames += 1
                    seg_t_end = ts
                    prev_ts = ts
                    frame_index += 1
                    if len(buf) >= chunk_size:
                        flush()
            flush()
            if seg_frames > 0:
                segment_index.append(
                    {
                        "segment_id": f"{spec['session_id']}__seg{seg_no:04d}",
                        "session_id": spec["session_id"],
                        "source_tag": SOURCE_TAG,
                        "frame_start": seg_start_frame,
                        "frame_end": frame_index - 1,
                        "n_frames": seg_frames,
                        "n_injected": seg_injected,
                        "t_start": seg_t_start,
                        "t_end": seg_t_end,
                    }
                )
            per_file.append(
                {
                    "file": spec["file"],
                    "session_id": spec["session_id"],
                    "fmt": spec["fmt"],
                    "rows_used": n_parsed,
                    "skipped_rows": n_skipped,
                    "n_injected": n_injected,
                    "n_normal": n_parsed - n_injected,
                    "n_segments": seg_no,
                    "source_tag": SOURCE_TAG,
                    "is_real_data": True,
                }
            )
            total_injected += n_injected
            total_normal += n_parsed - n_injected
    finally:
        writer.close()

    total_rows = frame_index
    seg_path = interim_dir / "segment_index.json"
    write_json(
        seg_path,
        {
            "generated_at": utc_now(),
            "segment_strategy": "时间间隙 > gap_threshold 或单段帧数达 segment_max_frames（流式分块）",
            "gap_threshold_s": gap,
            "segment_max_frames": seg_max,
            "n_segments": len(segment_index),
            "segments": segment_index,
        },
    )
    return {
        "schema": FRAME_COLUMNS,
        "parquet": str(out_path),
        "segment_index_path": str(seg_path),
        "n_files": len(per_file),
        "raw_data_lines_total": raw_lines_total,
        "parquet_rows_total": total_rows,
        "row_count_match": total_rows == sum(f["rows_used"] for f in per_file),
        "n_injected": total_injected,
        "n_normal": total_normal,
        "n_segments": len(segment_index),
        "n_sessions": len(per_file),
        "source_tags": [SOURCE_TAG],
        "is_real_data": True,
        "chunk_size": chunk_size,
        "gap_threshold_s": gap,
        "segment_max_frames": seg_max,
        "per_file": per_file,
        "subset_scope": {
            "max_rows_per_file": int(lc["max_rows_per_file"]),
            "note": "每个攻击集取前 N 行；Attack-free 文件不足 N 行则全量使用。详见 inventory.files[].rows_used",
            "files": [
                {"file": f["file"], "rows_used": f["rows_used"], "is_subset": f["is_subset"]}
                for f in inventory["files"]
            ],
        },
    }


def iter_segments(interim_dir: Path, batch_size: int = 200_000) -> Iterator[tuple[str, pd.DataFrame]]:
    """顺序扫描 frames.parquet，按 segment 边界 yield（内存只占一个 batch + 一个 segment）。"""
    pf = pq.ParquetFile(interim_dir / "frames.parquet")
    pending_id: str | None = None
    pending: list[pd.DataFrame] = []
    for batch in pf.iter_batches(batch_size=batch_size):
        df = batch.to_pandas()
        for seg_id, grp in df.groupby("segment_id", sort=False):
            sid = str(seg_id)
            if pending_id is None:
                pending_id, pending = sid, [grp]
            elif pending_id == sid:
                pending.append(grp)
            else:
                yield pending_id, pd.concat(pending, ignore_index=True)
                pending_id, pending = sid, [grp]
    if pending_id is not None and pending:
        yield pending_id, pd.concat(pending, ignore_index=True)


def build_windows_and_labels_streaming(
    cfg: dict[str, Any], interim_dir: Path, processed_dir: Path
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """按段流式生成窗口级/帧级标签，避免把全量帧表读入内存。"""
    from .labels import build_windows

    labels_dir = processed_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    wsize = int(cfg["window"]["size_frames"])
    stride = int(cfg["window"]["stride_frames"])
    theta = float(cfg["window"]["attack_ratio_theta"])

    frame_label_schema = pa.schema(
        [
            ("frame_index", pa.int64()), ("label", pa.int64()), ("flag", pa.string()),
            ("segment_id", pa.string()), ("source_tag", pa.string()),
        ]
    )
    frame_writer = pq.ParquetWriter(
        labels_dir / "frame_labels.parquet", frame_label_schema, compression="snappy"
    )
    window_chunks: list[pd.DataFrame] = []
    n_frame_labels = 0
    n_frame_positive = 0
    try:
        for seg_id, seg_frames in iter_segments(interim_dir, int(cfg["local_data"]["chunk_size"])):
            fl = seg_frames[
                ["frame_index", "label", "flag", "segment_id", "source_tag"]
            ].copy()
            frame_writer.write_table(
                pa.Table.from_pandas(fl, schema=frame_label_schema, preserve_index=False),
                row_group_size=100_000,
            )
            n_frame_labels += len(fl)
            n_frame_positive += int(fl["label"].sum())
            win = build_windows(seg_frames, wsize, stride, theta)
            window_chunks.append(win)
    finally:
        frame_writer.close()

    windows = pd.concat(window_chunks, ignore_index=True)
    window_labels = windows[
        [
            "window_id", "segment_id", "session_id", "source_tag", "frame_start", "frame_end",
            "t_start", "t_end", "n_frames", "n_injected", "attack_ratio", "label", "dominant_class",
        ]
    ].copy()
    window_labels.to_parquet(
        labels_dir / "window_labels.parquet", index=False, engine="pyarrow"
    )
    windows.to_parquet(labels_dir / "windows_full.parquet", index=False, engine="pyarrow")
    stats = {
        "theta": theta,
        "frame_label_rows": n_frame_labels,
        "frame_positive": n_frame_positive,
        "window_label_rows": len(window_labels),
        "window_positive": int(window_labels["label"].sum()),
        "window_negative": int((1 - window_labels["label"]).sum()),
        "n_segments_with_windows": len(window_chunks),
        "files": ["frame_labels.parquet", "window_labels.parquet", "windows_full.parquet"],
        "storage_note": "标签存于 data/processed/labels/，特征存于 data/processed/features/，物理隔离",
    }
    return windows, stats


def write_real_provenance(
    cfg: dict[str, Any], artifacts_dir: Path, inventory: dict[str, Any], zip_check: dict[str, Any]
) -> dict[str, Any]:
    """真实数据来源记录：官方出处 + Kaggle 镜像 + 校验和 + 逐文件行数（不访问网络）。"""
    ds = cfg["dataset"]
    kaggle = cfg["local_data"]["kaggle"]
    official_download = [
        "https://drive.google.com", "https://www.dropbox.com",
    ]
    prov = {
        "provenance_version": "2.0",
        "checked_at": utc_now(),
        "dataset": {
            "name": ds["name"],
            "official_page": ds["official_page"],
            "paper_page": ds["paper_page"],
            "citation_primary": ds["citation_primary"],
            "citation_secondary": ds["citation_secondary"],
            "license_note": ds["license_note"],
            "official_schema_fields": ds["schema_fields"],
            "official_counts": ds["official_counts"],
        },
        "source_of_this_copy": {
            "statement": "本次实际文件来源为 Kaggle 公开镜像，非官方直发（not official direct download）。",
            "is_official_direct_download": False,
            "kaggle_mirror": kaggle["url"],
            "kaggle_license": kaggle["license"],
            "acquired_by": "本地数据文件，路径见 input_zip.path",
            "official_download_hosts": official_download,
            "official_download_note": (
                "官方页面列出的下载入口指向 Google Drive / Dropbox；在本网络不可达且不在白名单内，"
                "W1 曾如实记录该阻塞。本次未发起任何网络访问，数据完全由本地 zip 提供。"
            ),
            "network_accessed_this_run": False,
        },
        "input_zip": zip_check,
        "file_inventory": inventory,
        "field_level_verification": {
            "attack_csv": "Timestamp,CAN ID(HEX),DLC,DATA[0..7](HEX),Flag(T/R)；无表头；DLC<8 时 DATA 列数随 DLC 变化",
            "normal_candump": "Timestamp: <秒> ID: <hex> 000 DLC: <n> <hex bytes...>；无 Flag 列，全部正常帧",
            "flag_mapping": "T -> label=1（注入帧），R -> label=0（正常帧）",
            "attack_free_policy": "Attack-free 数据集不参与攻击类标签，仅作正常流量来源（设计约定）",
            "evidence": "逐文件 SHA-256、字节数、行数（physical_newlines）与官方 §1.2 计数逐一比对，见 file_inventory.files",
        },
        "subset_scope": {
            "max_rows_per_file": int(cfg["local_data"]["max_rows_per_file"]),
            "note": "目标环境内存上限 8 GB，全量约 1755 万帧不可一次性处理；按设计约束使用配置化子集",
            "files": [
                {"file": f["file"], "rows_used": f["rows_used"], "is_subset": f["is_subset"]}
                for f in inventory["files"]
            ],
        },
        "real_data_available": True,
        "status": "available",
        "source_tag": SOURCE_TAG,
        "conclusion": (
            "真实 HCRL Car-Hacking 数据集已由 Kaggle 公开镜像（MIT）本地化并接入，"
            "zip 与 5 个文件的 SHA-256 / 行数均已核对留痕；W1 的 simulated 数值仅用于管线验证，"
            "已被本次真实数据评估结果取代。"
        ),
        "whitelist": list(cfg["local_data"].get("whitelist", [])),
        "network_scope_note": "本阶段数据已本地化，未访问外部网络；原始数据不提交 git、不重新分发。",
    }
    write_json(artifacts_dir / "dataset_provenance.json", prov)
    return prov
