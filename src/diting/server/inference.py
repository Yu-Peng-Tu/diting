"""W5 工作台 · 冻结模型实时推理与演示样本库。

三条硬纪律：
1. **真实推理**：``/api/detect`` 直接加载 ``artifacts/models/l1_v3/model.joblib``
   里的二分类 / 五分类 HistGradientBoosting 模型做 predict_proba，不是 mock，
   不重新训练、不改动任何模型权重。
2. **test 折零载入**：演示样本只从 ``split-v3 val`` 与 ``OTIDS`` 构建；
   split-v3 划分表断言 split 集合恰为 {train, val}；特征表用 pyarrow
   ``window_id in-list`` 下推，只读取选中样本行，test 特征行根本不进内存。
3. **特征口径一致**：粘贴 candump 的路径复用 W1 唯一的
   ``compute_window_features`` / ``apply_normalization``，ID 频次表与归一化
   参数都是 HCRL train 拟合的冻结产物，不重新 fit。

特征贡献是「单特征置零（归一化均值）扰动法」的局部归因，不是 SHAP、不是全局
特征重要性；该口径在响应里显式声明，避免与 W3 的 permutation importance 混用。
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from ..common import PROJECT_ROOT, read_json
from ..data.schema import DATA_COLUMNS
from ..features.build import (
    FEATURE_COLUMNS,
    FEATURE_GROUPS,
    apply_normalization,
    compute_window_features,
)

MODEL_PATH = PROJECT_ROOT / "artifacts" / "models" / "l1_v3" / "model.joblib"
THRESHOLD_PATH = PROJECT_ROOT / "artifacts" / "models" / "l1_v3" / "threshold.json"
FEATURE_PARAMS_PATH = PROJECT_ROOT / "artifacts" / "features" / "feature_params.json"
ID_FREQ_PATH = PROJECT_ROOT / "artifacts" / "features" / "id_frequency_train.json"

V3_WINDOWS = PROJECT_ROOT / "data" / "processed" / "split" / "split-v3" / "windows.parquet"
HCRL_FEATURES = PROJECT_ROOT / "data" / "processed" / "features" / "window_features.parquet"
OTIDS_LABELS = PROJECT_ROOT / "data" / "processed" / "otids" / "labels" / "window_labels.parquet"
OTIDS_FEATURES = PROJECT_ROOT / "data" / "processed" / "otids" / "window_features.parquet"

# 每类挑 2 个确定性样本（按 window_id 排序后取首尾附近，避免随机性）
SAMPLES_PER_CLASS = 2
HCRL_CLASSES = ("Normal", "DoS", "Fuzzy", "Gear", "RPM")
OTIDS_CLASSES = ("Normal", "DoS", "Fuzzy", "Impersonation")

FEATURE_GROUP_OF: dict[str, str] = {
    col: group for group, cols in FEATURE_GROUPS.items() for col in cols
}

_CANDUMP_LINE = re.compile(
    r"^\s*(?:\(\s*(?P<ts>[0-9]+(?:\.[0-9]+)?)\s*\)\s*)?"
    r"(?:[A-Za-z0-9_.-]+\s+)?"
    r"(?P<can_id>[0-9A-Fa-f]{3,8})\s*#\s*(?P<data>[0-9A-Fa-f]*)"
)


class DetectInputError(ValueError):
    """演示输入不合法（缺帧、解析失败、样本 id 未在白名单登记）。"""


@lru_cache(maxsize=1)
def model_bundle() -> dict[str, Any]:
    """加载冻结模型并做结构断言（进程内只加载一次）。"""
    bundle = joblib.load(MODEL_PATH)
    required = ("binary_model", "multiclass_model", "binary_feature_columns", "multiclass_feature_columns")
    missing = [k for k in required if k not in bundle]
    if missing:
        raise AssertionError(f"冻结模型缺少键：{missing}")
    if list(bundle.get("binary_classes", [])) != [0, 1]:
        raise AssertionError("冻结二分类模型的类别不是 [0, 1]")
    for k in ("binary_feature_columns", "multiclass_feature_columns"):
        if list(bundle[k]) != list(FEATURE_COLUMNS):
            raise AssertionError(f"冻结模型 {k} 与 33 维 G1–G6 特征列不一致")
    return bundle


@lru_cache(maxsize=1)
def frozen_threshold() -> float:
    return float(read_json(THRESHOLD_PATH)["selected_threshold"])


@lru_cache(maxsize=1)
def _feature_params() -> dict[str, Any]:
    return read_json(FEATURE_PARAMS_PATH)


@lru_cache(maxsize=1)
def _id_frequency() -> dict[str, float]:
    return dict(read_json(ID_FREQ_PATH)["id_frequency"])


@lru_cache(maxsize=1)
def sample_catalog() -> dict[str, Any]:
    """构建只读演示样本库，并返回 test 折封存的逐条断言留痕。"""
    v3 = pq.read_table(
        V3_WINDOWS,
        columns=["window_id", "segment_id", "split", "dominant_class", "label", "attack_ratio"],
    ).to_pandas()
    splits = sorted(set(v3["split"].astype(str)))
    if splits != ["train", "val"]:
        raise AssertionError(f"split-v3 划分表 split 集合异常：{splits}（含 test 即违反封存纪律）")
    if "test" in splits:
        raise AssertionError("split-v3 划分表含 test 行，违反封存纪律")

    val = v3[v3["split"] == "val"].copy()
    ot = pq.read_table(
        OTIDS_LABELS,
        columns=["window_id", "segment_id", "dominant_class", "label", "attack_ratio"],
    ).to_pandas()

    def pick(df: pd.DataFrame, classes: tuple[str, ...], source: str) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for cls in classes:
            sub = df[df["dominant_class"].astype(str) == cls].sort_values("window_id")
            if sub.empty:
                continue
            positions = np.linspace(0, len(sub) - 1, min(SAMPLES_PER_CLASS, len(sub)), dtype=int)
            for pos in positions:
                r = sub.iloc[int(pos)]
                rows.append(
                    {
                        "sample_id": str(r["window_id"]),
                        "window_id": str(r["window_id"]),
                        "segment_id": str(r["segment_id"]),
                        "source": source,
                        "split": "val" if source == "hcrl_val" else "otids",
                        "dominant_class": str(r["dominant_class"]),
                        "label": int(r["label"]),
                        "attack_ratio": float(r["attack_ratio"]),
                    }
                )
        return rows

    samples = pick(val, HCRL_CLASSES, "hcrl_val") + pick(ot, OTIDS_CLASSES, "otids")
    ids = [s["sample_id"] for s in samples]
    if len(ids) != len(set(ids)):
        raise AssertionError("演示样本库出现重复 window_id")
    allowed_val_ids = set(val["window_id"].astype(str))
    allowed_otids_ids = set(ot["window_id"].astype(str))
    for s in samples:
        pool = allowed_val_ids if s["source"] == "hcrl_val" else allowed_otids_ids
        if s["sample_id"] not in pool:
            raise AssertionError(f"样本 {s['sample_id']} 不在允许的 val/OTIDS 窗口池内")

    return {
        "samples": samples,
        "sealing": {
            "split_v3_rows": len(v3),
            "split_v3_splits": splits,
            "split_v3_val_rows": len(val),
            "otids_rows": len(ot),
            "test_rows_loaded": 0,
            "test_set_status": "sealed — HCRL test 折零载入",
            "assertions": [
                "A1: split-v3/windows.parquet 的 split 集合 == {train, val}（无 test 行）",
                "A2: HCRL 演示样本全部来自 split-v3 val 窗口，train 与 test 均不进入样本库",
                "A3: OTIDS 演示样本全部来自 OTIDS 窗口表（其 split 列恒为 otids）",
                "A4: 特征表用 pyarrow filters [('window_id','in', 选中 id)] 下推，test 特征行零载入",
                "A5: /api/detect 只接受样本库中登记过的 sample_id，拒绝任意 window_id",
            ],
        },
    }


def find_sample(sample_id: str) -> dict[str, Any]:
    """按白名单查样本；未登记 id（含任何 test 窗口）一律拒绝。"""
    for s in sample_catalog()["samples"]:
        if s["sample_id"] == sample_id:
            return s
    raise DetectInputError(f"sample_id 未在演示样本库登记：{sample_id}（test 折与 train 折不可用于演示）")


def feature_row_for(sample_id: str) -> dict[str, float]:
    """用 in-list 下推读取选中窗口的 33 维已归一化特征（只读 1 行）。"""
    entry = find_sample(sample_id)
    path = HCRL_FEATURES if entry["source"] == "hcrl_val" else OTIDS_FEATURES
    table = pq.read_table(
        path,
        columns=["window_id", *FEATURE_COLUMNS],
        filters=[("window_id", "in", [sample_id])],
    )
    df = table.to_pandas()
    if len(df) != 1:
        raise DetectInputError(f"样本 {sample_id} 在特征表中命中 {len(df)} 行（期望 1 行）")
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise AssertionError(f"特征表缺少声明列：{missing}")
    return {c: float(df.loc[df.index[0], c]) for c in FEATURE_COLUMNS}


def _parse_hex_pairs(token: str) -> list[int]:
    token = token.strip()
    if len(token) % 2 != 0:
        raise DetectInputError("candump 数据段十六进制字符数必须为偶数")
    try:
        return [int(token[i : i + 2], 16) for i in range(0, len(token), 2)]
    except ValueError as exc:
        raise DetectInputError(f"candump 数据段不是合法十六进制：{token[:40]}") from exc


def parse_candump(text: str) -> tuple[dict[str, float], dict[str, Any]]:
    """把粘贴的 candump / 空格分隔 / CSV 文本解析成 33 维归一化特征向量。

    支持三种行格式：
    - ``(1479121434.850202) can0 316#052100090021006F``（标准 candump）
    - ``1479121434.850202 316 8 05 21 00 09 00 21 00 6F``（空格分隔）
    - ``1479121434.850202,0316,8,05,21,00,09,00,21,00,6F``（CSV）
    至少 8 帧；超过 100 帧时取最后 100 帧（与训练窗口一致）。
    返回 (特征向量, 输入口径说明)。
    """
    rows: list[tuple[float | None, int, list[int]]] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith(("#", "//")):
            continue
        if line.lower().startswith("timestamp"):
            continue
        m = _CANDUMP_LINE.match(line)
        if m:
            ts = float(m.group("ts")) if m.group("ts") else None
            can_id = int(m.group("can_id"), 16)
            data = _parse_hex_pairs(m.group("data")) if m.group("data") else []
        else:
            parts = [p.strip() for p in (line.split(",") if "," in line else line.split()) if p.strip()]
            if len(parts) < 3:
                raise DetectInputError(f"无法解析行：{raw[:80]}")
            try:
                ts = float(parts[0])
            except ValueError:
                ts = None
            try:
                can_id = int(parts[1], 16)
            except ValueError as exc:
                raise DetectInputError(f"CAN ID 不是合法十六进制：{parts[1]}") from exc
            maybe_dlc = (
                len(parts[2]) == 1
                and parts[2].isdigit()
                and 0 <= int(parts[2]) <= 8
                and len(parts) - 3 == int(parts[2])
            )
            body = parts[3:] if maybe_dlc else parts[2:]
            try:
                data = [int(p, 16) for p in body[:8]]
            except ValueError as exc:
                raise DetectInputError(f"数据字节不是合法十六进制：{raw[:80]}") from exc
        rows.append((ts, can_id, data))

    if len(rows) < 8:
        raise DetectInputError(f"至少需要 8 帧 CAN 报文，实际解析到 {len(rows)} 帧")
    rows = rows[-100:]

    # 缺少时间戳时按固定 0.5ms 步长补足（仅用于计算帧间隔特征，响应里会标注）
    have_ts = all(r[0] is not None for r in rows)
    if have_ts:
        timestamps = [float(r[0]) for r in rows]
    else:
        timestamps = [i * 0.0005 for i in range(len(rows))]

    n = len(rows)
    frame: dict[str, Any] = {
        "frame_index": np.arange(n, dtype=np.int64),
        "timestamp": np.asarray(timestamps, dtype=float),
        "can_id": np.asarray([r[1] for r in rows], dtype=np.int64),
        "dlc": np.asarray([len(r[2]) for r in rows], dtype=float),
        "segment_id": "candump__seg0001",
        "session_id": "candump",
        "source_tag": "user:candump",
        "flag": "R",
        "label": 0,
        "attack_kind": "unknown",
    }
    for j, col in enumerate(DATA_COLUMNS):
        # 与 W1 帧表一致：不足 8 字节用 -1 填充，add_frame_level_columns 用 >=0 掩码处理
        frame[col] = np.asarray([r[2][j] if j < len(r[2]) else -1 for r in rows], dtype=float)
    frames = pd.DataFrame(frame)

    windows = pd.DataFrame(
        [
            {
                "window_id": "CANDUMP_DEMO",
                "segment_id": "candump__seg0001",
                "session_id": "candump",
                "source_tag": "user:candump",
                "split": "val",
                "frame_start": 0,
                "frame_end": n - 1,
                "t_start": float(frames["timestamp"].min()),
                "t_end": float(frames["timestamp"].max()),
                "label": 0,
                "attack_ratio": 0.0,
            }
        ]
    )
    raw_feat = compute_window_features(frames, windows, _id_frequency())
    norm = apply_normalization(raw_feat, _feature_params())
    vector = {c: float(norm.loc[0, c]) for c in FEATURE_COLUMNS}
    return vector, {
        "n_frames": n,
        "timestamp_source": "parsed" if have_ts else "synthetic_0.5ms_step",
        "window_policy": "取最后 100 帧构造单个窗口（与训练窗口 size_frames=100 / stride=50 对齐）",
        "caveat": (
            "粘贴文本的特征由冻结参数即时计算，采样环境/DLC/时间戳基准可能与 HCRL 不同；"
            "结果仅作演示，不进入任何评估统计，不可与 val / OTIDS 指标混用"
        ),
    }


def detect(feature_vector: dict[str, float], threshold: float | None = None) -> dict[str, Any]:
    """用冻结模型做真实二分类 + 五分类推理，并给出局部特征贡献。"""
    bundle = model_bundle()
    thr = frozen_threshold() if threshold is None else float(threshold)

    binary_cols = list(bundle["binary_feature_columns"])
    x = np.asarray([[feature_vector[c] for c in binary_cols]], dtype=np.float64)
    classes_bin = np.asarray(bundle["binary_model"].classes_)
    pos_idx = int(np.where(classes_bin == 1)[0][0])
    base_proba = bundle["binary_model"].predict_proba(x)
    attack_probability = float(base_proba[0, pos_idx])

    multi_cols = list(bundle["multiclass_feature_columns"])
    xm = np.asarray([[feature_vector[c] for c in multi_cols]], dtype=np.float64)
    multi_proba = bundle["multiclass_model"].predict_proba(xm)[0]
    multi_classes = [str(c) for c in bundle["multiclass_model"].classes_]
    probabilities = {c: float(v) for c, v in zip(multi_classes, multi_proba)}
    predicted_class = max(probabilities, key=probabilities.get)

    # 单特征置零扰动：把该维替换为归一化空间的 train 均值（0），观察攻击概率变化
    perturbed = np.repeat(x, len(binary_cols), axis=0)
    for i in range(len(binary_cols)):
        perturbed[i, i] = 0.0
    perturbed_proba = bundle["binary_model"].predict_proba(perturbed)[:, pos_idx]
    contributions = [
        {
            "feature": col,
            "group": FEATURE_GROUP_OF.get(col, "unknown"),
            "value": float(x[0, i]),
            "contribution_to_attack_proba": float(attack_probability - perturbed_proba[i]),
        }
        for i, col in enumerate(binary_cols)
    ]
    top = sorted(contributions, key=lambda d: abs(d["contribution_to_attack_proba"]), reverse=True)[:8]

    return {
        "attack_probability": attack_probability,
        "normal_probability": float(base_proba[0, int(np.where(classes_bin == 0)[0][0])]),
        "threshold": thr,
        "threshold_is_frozen": threshold is None,
        "decision": "attack" if attack_probability >= thr else "normal",
        "multiclass": {
            "predicted_class": predicted_class,
            "probabilities": probabilities,
            "note": "五分类是 HCRL 类目（Normal/DoS/Fuzzy/Gear/RPM）；跨域样本无对应真值类，结果仅作定性映射参考",
        },
        "top_features": top,
        "contribution_method": (
            "单特征置零扰动法：逐维替换为归一化空间 train 均值(0) 后重算冻结模型预测概率，"
            "取 ΔP(attack) 作为局部贡献；非 SHAP、非全局 permutation importance"
        ),
        "model_version": bundle.get("model_version", "l1-gbdt-v3"),
        "split_version": bundle.get("split_version", "split-v3"),
        "n_features": len(binary_cols),
    }
