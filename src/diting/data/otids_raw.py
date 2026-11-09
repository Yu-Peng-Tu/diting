"""谛听 W4 · OTIDS（Kia Soul）数据接入、窗口化与冻结特征管线复用。

本模块是 TASK-20261109-0827 新增的独立路径，与 HCRL 侧 ``local_raw.py`` 并存，
**不修改、不覆盖、不重命名任何既有文件**。

数据口径（全部写进 manifest，禁止静默丢弃数据）：
- 输入 zip 只读流式解析，主口径只用 4 个逐帧带标签文件；
  ``CAN_HCRL_OTIDS_B.csv`` / ``CAN_HCRL_OTIDS_UB.csv`` 是第三方二次加工
  （SMOTE 平衡/合并、十进制化），来源口径不明，**仅作规模参考，禁止作为评估口径**。
- 列结构 ``TS,ID1,ID0,LEN,DLC0..DLC7,target``：
  - ``TS`` -> timestamp(float)；``LEN`` -> dlc(int)；``DLC0..7`` -> data_0..7(int, 空=-1)
  - ``ID1`` 为 4 位十六进制字符串 -> ``can_id = int(ID1, 16)``。
    交叉印证：B 文件的 ``ID1`` 为十进制整数（如 339=0x153、1349=0x545），
    取值范围上界 1680 恰等于 4 个主口径文件 ``int(ID1,16)`` 的上界 0x690=1680；
    且 4 个文件的 ``ID0`` 只取 {000,100} 两个值，非 ID 低位。
    故 ``ID0`` 作为帧属性标记 ``id0_token`` 保留在帧表，**不进特征矩阵**。
  - ``target`` 是**文件级标签**（Attack-Free=0 / DoS=1 / Fuzzy=2 / Impersonation=3），
    攻击文件内正常 ID 的帧同样被标为攻击（实测 dataset1.csv 中 ID=0x0080 的帧 target=1）。
- 标签双口径（窗口级，均复用 HCRL 同一个 ``build_windows`` 与 θ=0.20）：
  1. **主口径 primary**：帧 label = (target > 0)，窗口 label = attack_ratio>=0.20。
     因文件级标签在段内恒定，attack_ratio 退化为 0/1，窗口标签等价于文件标签；
     θ 仍按同一函数与同一参数执行，口径代码路径与 HCRL 完全一致（如实声明）。
  2. **官方描述敏感性口径 official**：DoS 以 ID=0x000 为注入帧；
     Fuzzy / Impersonation 以「250 秒后为攻击段」为注入帧；Attack-Free 全正常。
     再按同样的窗口参数与 θ=0.20 生成窗口标签，仅作敏感性对照，不参与主结论。

窗口与特征：
- 窗口参数（100 帧 / 步长 50 / θ=0.20 / gap 0.02 / 单段上限 20000）与 HCRL 完全一致；
- 特征复用 ``features.build.compute_window_features``（G1–G6 33 维，唯一实现），
  经 ``features.stream.compute_window_features_streaming`` 按 segment 流式分批；
- **归一化参数与 ID 频次表全部冻结复用 HCRL train 折产物**，OTIDS 侧零拟合 ——
  这是「模型与阈值完全冻结、零样本应用」的必要条件；
- ``assert_feature_matrix_clean`` / ``assert_no_forbidden_fields`` 防泄漏断言全部沿用。
"""

from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ..common import PROJECT_ROOT, assert_no_forbidden_fields, sha256_file, utc_now, write_json
from ..data.labels import build_windows
from ..data.local_raw import iter_segments
from ..data.schema import FRAME_COLUMNS
from ..features.build import FEATURE_COLUMNS, apply_normalization, assert_feature_matrix_clean
from ..features.stream import compute_window_features_streaming

OTIDS_SOURCE_TAG = "public:otids:kia-soul"
OTIDS_EXTRA_COLUMNS = ("id0_token", "file_label", "official_label")
OTIDS_FRAME_COLUMNS = [*FRAME_COLUMNS, *OTIDS_EXTRA_COLUMNS]

OTIDS_PA_SCHEMA = pa.schema(
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
        ("id0_token", pa.string()),
        ("file_label", pa.int64()),
        ("official_label", pa.int64()),
    ]
)

# 主口径 4 个文件；B / UB 明确排除
PRIMARY_MEMBERS = ("dataset.csv", "dataset1.csv", "dataset2.csv", "dataset3.csv")
EXCLUDED_MEMBERS = ("CAN_HCRL_OTIDS_B.csv", "CAN_HCRL_OTIDS_UB.csv")


def load_xdomain_config(path: str | Path | None = None) -> dict[str, Any]:
    """读取 configs/xdomain.yaml，返回 xdomain 配置段并注入根目录与校验和。"""
    from ..common import load_config

    p = Path(path) if path else PROJECT_ROOT / "configs" / "xdomain.yaml"
    raw = load_config(p)
    if "xdomain" not in raw:
        raise ValueError("configs/xdomain.yaml 缺少 xdomain 配置段")
    cfg = raw["xdomain"]
    cfg["_config_path"] = raw["_config_path"]
    cfg["_config_sha256"] = raw["_config_sha256"]
    cfg["_root"] = raw["_root"]
    return cfg


def _resolve(cfg: dict[str, Any], relative: str) -> Path:
    p = Path(relative)
    return p if p.is_absolute() else PROJECT_ROOT / p


def verify_input_zip(cfg: dict[str, Any]) -> dict[str, Any]:
    """只读校验输入 zip 的 SHA-256 / 字节数 / 成员清单（不解压、不改写输入）。"""
    zcfg = cfg["input_zip"]
    p = _resolve(cfg, zcfg["path"])
    rec: dict[str, Any] = {
        "path": str(p),
        "exists": p.is_file(),
        "expected_sha256": zcfg["sha256"],
        "expected_size_bytes": int(zcfg["size_bytes"]),
    }
    if not p.is_file():
        rec["verified"] = False
        rec["error"] = "输入 zip 不存在"
        return rec
    actual = sha256_file(p)
    size = p.stat().st_size
    with zipfile.ZipFile(p) as z:
        members = z.namelist()
    expected = list(zcfg["expected_members"])
    rec.update(
        {
            "actual_sha256": actual,
            "actual_size_bytes": size,
            "sha256_match": actual == zcfg["sha256"],
            "size_match": size == int(zcfg["size_bytes"]),
            "members": members,
            "expected_members": expected,
            "all_expected_members_present": all(m in members for m in expected),
        }
    )
    rec["verified"] = bool(
        rec["sha256_match"] and rec["size_match"] and rec["all_expected_members_present"]
    )
    if not rec["verified"]:
        raise AssertionError(f"输入 zip 校验失败：{rec}")
    return rec


def _parse_hex_byte(token: str) -> int:
    """解析十六进制数据字节；空值/占位 -> -1（与 HCRL 的 _parse_hex 语义一致）。"""
    t = token.strip().lower()
    if t in {"", "none", "nan", "na", "null", "-"}:
        return -1
    return int(t, 16)


def _scan_file_time_range(z: zipfile.ZipFile, name: str) -> dict[str, Any]:
    """只读流式预扫描：取该文件时间戳上下界，供官方描述口径的 250 秒判据使用。"""
    ts_min, ts_max, n_rows = None, None, 0
    with z.open(name) as fh:
        fh.readline()  # 表头
        for raw in fh:
            line = raw.decode("utf-8", "replace").strip()
            if not line:
                continue
            head = line.split(",", 1)[0].strip()
            try:
                ts = float(head)
            except ValueError:
                continue
            ts_min = ts if ts_min is None else min(ts_min, ts)
            ts_max = ts if ts_max is None else max(ts_max, ts)
            n_rows += 1
    return {"ts_min": float(ts_min), "ts_max": float(ts_max), "n_scanned_rows": n_rows}


def parse_otids_line(
    line: str,
    spec: dict[str, Any],
    t_start: float,
    official_start_s: float,
) -> dict[str, Any] | None:
    """解析单行 OTIDS CSV，返回统一 schema 记录（不含 frame_index / segment_id）。

    返回 None 表示该行无法解析（列数异常或数值非法），调用方必须计数并如实记录，
    禁止静默丢弃。
    """
    tokens = [t.strip() for t in line.strip().split(",")]
    if len(tokens) != 13:
        return None
    try:
        timestamp = float(tokens[0])
        can_id = int(tokens[1], 16)
        id0_token = tokens[2]
        dlc = int(tokens[3])
        target = int(tokens[12])
        data = [_parse_hex_byte(t) for t in tokens[4:12]]
    except (ValueError, IndexError):
        return None
    if len(data) != 8:
        return None
    if target != int(spec["file_label"]):
        return None

    # 主口径：文件级标签下放到帧
    label = 1 if target > 0 else 0
    # 官方描述敏感性口径
    file_label = int(spec["file_label"])
    if file_label == 1:  # DoS：ID=0x000 即注入
        official_label = 1 if can_id == 0 else 0
    elif file_label in (2, 3):  # Fuzzy / Impersonation：250 秒后为攻击段
        official_label = 1 if (timestamp - t_start) >= official_start_s else 0
    else:  # Attack-Free
        official_label = 0

    return {
        "timestamp": timestamp,
        "can_id": can_id,
        "dlc": dlc,
        **{f"data_{i}": data[i] for i in range(8)},
        "flag": "NA",  # OTIDS 无逐帧 T/R Flag；仅作 schema 占位，不参与任何计算
        "source_tag": OTIDS_SOURCE_TAG,
        "label": label,
        "session_id": spec["session_id"],
        "attack_kind": spec["attack_kind"],
        "id0_token": id0_token,
        "file_label": file_label,
        "official_label": official_label,
    }


def write_otids_frames_streaming(cfg: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    """逐文件逐 chunk 解析并追加写 frames.parquet（内存只占一个 chunk）。

    segment 切分策略与 HCRL ``write_frames_streaming`` 完全一致：
    时间间隙 > gap_threshold，或单段帧数达 segment_max_frames。
    frame_index 全局单调递增，segment_id 带 session 前缀（同段不跨文件）。
    """
    zpath = _resolve(cfg, cfg["input_zip"]["path"])
    win = cfg["window"]
    gap = float(win["gap_threshold"])
    seg_max = int(win["segment_max_frames"])
    chunk_size = int(win["chunk_size"])
    official_start_s = float(cfg["label_policy"]["official_attack_start_s"])
    specs = list(cfg["files"])
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_path = out_dir / "frames.parquet"

    per_file: list[dict[str, Any]] = []
    segment_index: list[dict[str, Any]] = []
    frame_index = 0
    skipped_total = 0

    with zipfile.ZipFile(zpath) as z:
        writer = pq.ParquetWriter(frames_path, OTIDS_PA_SCHEMA, compression="zstd")
        try:
            for spec in specs:
                name = str(spec["name"])
                scan = _scan_file_time_range(z, name)
                t_start = scan["ts_min"]
                n_parsed = n_skipped = 0
                n_injected = n_official_injected = 0
                prev_ts: float | None = None
                seg_no = 0
                seg_frames = 0
                seg_start_frame = frame_index
                seg_t_start: float | None = None
                seg_t_end: float | None = None
                buf: list[dict[str, Any]] = []

                def flush() -> None:
                    nonlocal buf
                    if not buf:
                        return
                    df = pd.DataFrame.from_records(buf, columns=OTIDS_FRAME_COLUMNS)
                    table = pa.Table.from_pandas(df, schema=OTIDS_PA_SCHEMA, preserve_index=False)
                    writer.write_table(table, row_group_size=chunk_size)
                    buf = []

                with z.open(name) as fh:
                    fh.readline()  # 表头
                    for raw in fh:
                        line = raw.decode("utf-8", "replace").strip()
                        if not line:
                            continue
                        rec = parse_otids_line(line, spec, t_start, official_start_s)
                        if rec is None:
                            n_skipped += 1
                            continue
                        ts = rec["timestamp"]
                        dt = 0.0 if prev_ts is None else ts - prev_ts
                        if prev_ts is None or dt > gap or dt < 0 or seg_frames >= seg_max:
                            if seg_frames > 0:
                                segment_index.append(
                                    {
                                        "segment_id": f"{spec['session_id']}__seg{seg_no:04d}",
                                        "session_id": spec["session_id"],
                                        "frame_start": seg_start_frame,
                                        "frame_end": frame_index - 1,
                                        "n_frames": seg_frames,
                                        "t_start": seg_t_start,
                                        "t_end": seg_t_end,
                                    }
                                )
                            seg_no += 1
                            seg_frames = 0
                            seg_start_frame = frame_index
                            seg_t_start = ts
                        seg_id = f"{spec['session_id']}__seg{seg_no:04d}"
                        rec["frame_index"] = frame_index
                        rec["segment_id"] = seg_id
                        buf.append(rec)
                        n_parsed += 1
                        n_injected += rec["label"]
                        n_official_injected += rec["official_label"]
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
                            "frame_start": seg_start_frame,
                            "frame_end": frame_index - 1,
                            "n_frames": seg_frames,
                            "t_start": seg_t_start,
                            "t_end": seg_t_end,
                        }
                    )
                per_file.append(
                    {
                        "file": name,
                        "session_id": spec["session_id"],
                        "file_label": int(spec["file_label"]),
                        "attack_kind": spec["attack_kind"],
                        "zip_member_bytes": z.getinfo(name).file_size,
                        "ts_min": scan["ts_min"],
                        "ts_max": scan["ts_max"],
                        "span_seconds": round(scan["ts_max"] - scan["ts_min"], 6),
                        "rows_scanned": scan["n_scanned_rows"],
                        "rows_parsed": n_parsed,
                        "rows_skipped": n_skipped,
                        "row_count_match": (n_parsed + n_skipped) == scan["n_scanned_rows"],
                        "n_primary_injected": n_injected,
                        "n_official_injected": n_official_injected,
                        "n_segments": seg_no,
                    }
                )
                skipped_total += n_skipped
        finally:
            writer.close()

    write_json(
        out_dir / "segment_index.json",
        {
            "generated_at": utc_now(),
            "segment_strategy": "时间间隙 > gap_threshold 或单段帧数达 segment_max_frames（与 HCRL 一致）",
            "gap_threshold_s": gap,
            "segment_max_frames": seg_max,
            "n_segments": len(segment_index),
            "segments": segment_index,
        },
    )
    stats = {
        "schema": OTIDS_FRAME_COLUMNS,
        "parquet": str(frames_path),
        "n_files": len(per_file),
        "parquet_rows_total": frame_index,
        "skipped_rows_total": skipped_total,
        "all_row_counts_match": all(f["row_count_match"] for f in per_file),
        "n_segments": len(segment_index),
        "n_sessions": len(per_file),
        "source_tags": [OTIDS_SOURCE_TAG],
        "excluded_zip_members": list(EXCLUDED_MEMBERS),
        "primary_zip_members": list(PRIMARY_MEMBERS),
        "per_file": per_file,
    }
    write_json(out_dir / "parse_stats.json", stats)
    return stats


def build_otids_windows(cfg: dict[str, Any], out_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """按 segment 流式生成窗口表，含主口径与官方描述敏感性双口径标签。

    复用既有 ``iter_segments`` 与 ``build_windows``，窗口参数与 HCRL 完全一致。
    官方口径通过「把帧 label 换成 official_label 后重跑同一 build_windows」得到，
    再按 window_id 对齐回主窗口表，保证两次窗口划分逐窗一致。
    """
    win = cfg["window"]
    wsize = int(win["size_frames"])
    stride = int(win["stride_frames"])
    theta = float(win["attack_ratio_theta"])
    chunk_size = int(win["chunk_size"])

    chunks: list[pd.DataFrame] = []
    for _seg_id, seg_frames in iter_segments(out_dir, chunk_size):
        primary = build_windows(seg_frames, wsize, stride, theta)
        alt_frames = seg_frames.copy()
        alt_frames["label"] = alt_frames["official_label"].to_numpy()
        official = build_windows(alt_frames, wsize, stride, theta)
        official = official[["window_id", "label", "attack_ratio"]].rename(
            columns={"label": "official_label", "attack_ratio": "official_attack_ratio"}
        )
        merged = primary.merge(official, on="window_id", how="inner", validate="one_to_one")
        if len(merged) != len(primary):
            raise AssertionError("双口径窗口按 window_id 对齐后行数变化")
        chunks.append(merged)

    if not chunks:
        raise ValueError("OTIDS 未生成任何窗口")
    windows = pd.concat(chunks, ignore_index=True)
    windows["split"] = "otids"
    if not windows["window_id"].is_unique:
        raise AssertionError("OTIDS 存在重复 window_id")
    # 窗口级防泄漏：窗口不跨 segment（build_windows 已保证，此处双保险）
    if windows["segment_id"].isna().any():
        raise AssertionError("窗口缺 segment_id")

    labels_dir = out_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    windows.to_parquet(out_dir / "windows.parquet", index=False, engine="pyarrow")
    windows.to_parquet(labels_dir / "window_labels.parquet", index=False, engine="pyarrow")

    by_kind: dict[str, Any] = {}
    for kind, grp in windows.groupby("dominant_class", sort=True):
        by_kind[str(kind)] = {
            "n_windows": len(grp),
            "primary_positive": int(grp["label"].sum()),
            "official_positive": int(grp["official_label"].sum()),
        }
    stats = {
        "theta": theta,
        "window_size_frames": wsize,
        "stride_frames": stride,
        "n_windows": len(windows),
        "window_positive_primary": int(windows["label"].sum()),
        "window_negative_primary": int((1 - windows["label"]).sum()),
        "window_positive_official": int(windows["official_label"].sum()),
        "window_negative_official": int((1 - windows["official_label"]).sum()),
        "by_dominant_class": by_kind,
        "files": ["windows.parquet", "labels/window_labels.parquet"],
        "label_policy": {
            "primary": cfg["label_policy"]["primary"],
            "official_sensitivity": cfg["label_policy"]["official_sensitivity"],
            "note": (
                "主口径因文件级标签在段内恒定，attack_ratio 退化为 0/1，窗口标签等价于文件标签；"
                "θ=0.20 仍由与 HCRL 完全相同的 build_windows 执行，口径代码路径一致。"
            ),
        },
    }
    write_json(out_dir / "label_policy.json", stats)
    return windows, stats


def build_otids_features(cfg: dict[str, Any], out_dir: Path, windows: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """用冻结的 ID 频次表 + 归一化参数生成 OTIDS 33 维特征（OTIDS 侧零拟合）。"""
    from ..common import read_json

    id_freq_path = _resolve(cfg, cfg["frozen"]["id_frequency"])
    params_path = _resolve(cfg, cfg["frozen"]["feature_params"])
    id_freq = read_json(id_freq_path)["id_frequency"]
    params = read_json(params_path)
    chunk_size = int(cfg["window"]["chunk_size"])

    # 防泄漏断言：模型特征矩阵恰为 33 维 G1–G6，不含标签等价字段
    assert_no_forbidden_fields(FEATURE_COLUMNS, context="OTIDS 窗口特征矩阵")
    if len(FEATURE_COLUMNS) != 33:
        raise AssertionError("期望 33 维窗口特征")

    features_raw = compute_window_features_streaming(out_dir, windows, id_freq, chunk_size)
    clean = assert_feature_matrix_clean(features_raw)
    features_norm = apply_normalization(features_raw, params)

    features_raw.to_parquet(out_dir / "window_features_raw.parquet", index=False, engine="pyarrow")
    features_norm.to_parquet(out_dir / "window_features.parquet", index=False, engine="pyarrow")
    meta = {
        "n_windows": len(features_norm),
        "n_features": len(FEATURE_COLUMNS),
        "feature_columns": list(FEATURE_COLUMNS),
        "feature_function": "diting.features.build.compute_window_features (G1-G6, 唯一实现)",
        "streaming_wrapper": "diting.features.stream.compute_window_features_streaming",
        "id_frequency_source": str(id_freq_path),
        "id_frequency_sha256": sha256_file(id_freq_path),
        "id_frequency_fit_on": "HCRL train（冻结复用，OTIDS 侧零拟合）",
        "normalization_params_source": str(params_path),
        "normalization_params_sha256": sha256_file(params_path),
        "normalization_fit_on": params.get("fit_on"),
        "normalization_clip": params.get("clip"),
        "leakage_assertions": clean,
        "files": ["window_features_raw.parquet", "window_features.parquet"],
    }
    write_json(out_dir / "feature_meta.json", meta)
    return features_norm, meta


def write_otids_manifest(
    cfg: dict[str, Any],
    out_dir: Path,
    zip_check: dict[str, Any],
    parse_stats: dict[str, Any],
    window_stats: dict[str, Any],
    feature_meta: dict[str, Any],
) -> dict[str, Any]:
    """OTIDS 产物目录 manifest：zip SHA-256 溯源、口径声明、冻结资产校验和。"""
    from ..common import git_revision, read_json

    frozen_paths = {k: _resolve(cfg, v) for k, v in cfg["frozen"].items() if isinstance(v, str) and v.endswith((".json", ".joblib", ".parquet"))}
    frozen_records = {
        k: {"path": str(p), "sha256": sha256_file(p) if p.is_file() else None, "exists": p.is_file()}
        for k, p in frozen_paths.items()
    }
    thr = read_json(_resolve(cfg, cfg["frozen"]["threshold_json"]))
    manifest = {
        "manifest_version": "1.0",
        "stage": "otids_xdomain_ingest",
        "version": cfg["version"],
        "task": cfg["task"],
        "created_at": utc_now(),
        "random_seed": int(cfg["seed"]),
        "config_sha256": cfg["_config_sha256"],
        "git_revision_at_ingest": git_revision(),
        "dataset": cfg["dataset"],
        "source_tag": OTIDS_SOURCE_TAG,
        "data_scope": cfg["data_scope"],
        "level": cfg["level"],
        "input_zip": zip_check,
        "primary_members": list(PRIMARY_MEMBERS),
        "excluded_members": list(EXCLUDED_MEMBERS),
        "excluded_reason": (
            "B / UB 为第三方二次加工（SMOTE 平衡/合并、十进制化、列结构不同），"
            "来源口径不明，仅作规模参考，禁止作为评估口径；本任务未读取其数据行"
        ),
        "field_level_rules": {
            "TS": "-> timestamp (float)",
            "ID1": "-> can_id = int(ID1, 16)，4 位十六进制；B 文件十进制 ID1 取值上界 1680=0x690 交叉印证",
            "ID0": "-> id0_token 帧属性标记（仅取 {000,100}），保留在帧表，不进特征矩阵",
            "LEN": "-> dlc (int)",
            "DLC0..DLC7": "-> data_0..7 (int, 空值 -1)",
            "target": "-> file_label 文件级标签 0/1/2/3",
            "flag": "OTIDS 无逐帧 T/R Flag，统一占位 'NA'，不参与任何标签或特征计算",
        },
        "window_policy": {
            "size_frames": int(cfg["window"]["size_frames"]),
            "stride_frames": int(cfg["window"]["stride_frames"]),
            "attack_ratio_theta": float(cfg["window"]["attack_ratio_theta"]),
            "gap_threshold_s": float(cfg["window"]["gap_threshold"]),
            "segment_max_frames": int(cfg["window"]["segment_max_frames"]),
            "same_as_hcrl": True,
        },
        "label_policy": cfg["label_policy"],
        "frozen_assets": frozen_records,
        "frozen_threshold": float(thr["selected_threshold"]),
        "frozen_threshold_version": thr["version"],
        "parse_stats": parse_stats,
        "window_stats": window_stats,
        "feature_meta": feature_meta,
        "hcrl_mixing_forbidden": (
            "OTIDS 帧表、窗口、特征全部写入 data/processed/otids/ 独立目录；"
            "未写入任何 HCRL 训练/划分/特征产物；HCRL test 折全程封存、零读取"
        ),
        "outputs": [
            "frames.parquet", "segment_index.json", "parse_stats.json",
            "windows.parquet", "labels/window_labels.parquet", "label_policy.json",
            "window_features_raw.parquet", "window_features.parquet",
            "feature_meta.json", "manifest.json",
        ],
    }
    write_json(out_dir / "manifest.json", manifest)
    return manifest
