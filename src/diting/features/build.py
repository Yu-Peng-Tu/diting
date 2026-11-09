"""D4 · 特征工程（G1–G6），训练侧与服务侧共用的唯一实现。

对应策划方案 §4.5 与 §4.6 泄漏防线：
- 特征只从统一 schema 的 timestamp / can_id / dlc / data_0..7 计算。
- flag、label、can_id==0x000 布尔等标签等价信息一律不进特征矩阵（代码断言）。
- 归一化参数与 ID 频次表只在 **train 折** fit，导出带版本号的 feature_params.json。
- golden vector：固定样本输入 → 特征值快照，供训练/推理一致性逐值比对。
- G4 的 min_id_share 是「窗口内数值最小 ID 的占比」这一相对统计量，
  不是帧级 can_id==0x000 布尔；已列入 W2 消融清单（§4.6 L2 / §5.2）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..common import assert_no_forbidden_fields, set_seed, write_json
from ..data.schema import DATA_COLUMNS

# 特征分组（G1–G6），顺序固定，供 golden vector 与参数文件稳定复现
FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "G1": ("g1_dlc_mean", "g1_dlc_std", "g1_dlc_max", "g1_id_freq_mean", "g1_id_freq_std", "g1_id_freq_min"),
    "G2": ("g2_dt_mean", "g2_dt_std", "g2_dt_min", "g2_dt_max", "g2_same_id_dt_mean", "g2_same_id_dt_std", "g2_same_id_dt_cv"),
    "G3": ("g3_byte_mean", "g3_byte_std", "g3_byte_entropy_mean", "g3_hamming_mean", "g3_payload_change_rate"),
    "G4": ("g4_frame_count", "g4_unique_ids", "g4_id_entropy", "g4_top1_id_ratio", "g4_min_id_share"),
    "G5": ("g5_same_id_dt_mean", "g5_same_id_dt_std", "g5_same_id_dt_max", "g5_dt_p50", "g5_dt_p95"),
    "G6": ("g6_payload_change_rate", "g6_byte_entropy_mean", "g6_dlc_entropy", "g6_hamming_p95", "g6_unique_payload_ratio"),
}
FEATURE_COLUMNS: list[str] = [c for cols in FEATURE_GROUPS.values() for c in cols]

# 只承诺这些原始列可作为特征来源
RAW_FEATURE_INPUTS = ("timestamp", "can_id", "dlc", *DATA_COLUMNS)


def _entropy(counts: np.ndarray) -> float:
    total = float(counts.sum())
    if total <= 0:
        return 0.0
    p = counts[counts > 0] / total
    return float(-(p * np.log2(p)).sum())


def _byte_entropy(row: np.ndarray) -> float:
    vals = row[row >= 0]
    if vals.size == 0:
        return 0.0
    counts = np.bincount(vals.astype(int), minlength=256)
    return _entropy(counts)


def fit_id_frequency(train_frames: pd.DataFrame) -> dict[str, float]:
    """只在 train 折帧上统计 CAN ID 频次编码（不使用 0x000 布尔）。"""
    counts = train_frames["can_id"].value_counts(normalize=True)
    return {str(int(k)): float(v) for k, v in counts.items()}


def add_frame_level_columns(frames: pd.DataFrame) -> pd.DataFrame:
    """计算帧级中间量（不产出最终特征列）。"""
    df = frames.copy()
    df["_dt"] = df.groupby("segment_id", sort=False)["timestamp"].diff().fillna(0.0)
    df["_same_id_dt"] = (
        df.groupby(["segment_id", "can_id"], sort=False)["timestamp"].diff().fillna(0.0)
    )
    payload = df[DATA_COLUMNS].to_numpy(dtype=np.float64)
    mask = payload >= 0
    safe = np.where(mask, payload, 0.0)
    cnt = mask.sum(axis=1)
    byte_mean = np.divide(safe.sum(axis=1), np.maximum(cnt, 1))
    sq_mean = np.divide((safe**2).sum(axis=1), np.maximum(cnt, 1))
    byte_std = np.sqrt(np.maximum(sq_mean - byte_mean**2, 0.0))
    byte_mean[cnt == 0] = 0.0
    byte_std[cnt == 0] = 0.0
    df["_byte_mean"] = byte_mean
    df["_byte_std"] = byte_std
    df["_byte_entropy"] = np.array([_byte_entropy(r) for r in payload])
    prev = np.vstack([np.full((1, 8), np.nan), payload[:-1]])
    same_id_prev = df.groupby(["segment_id", "can_id"], sort=False)[DATA_COLUMNS[0]].shift(1)
    valid_prev = same_id_prev.notna().to_numpy()
    changed = ((payload != prev) & (payload >= 0) & (prev >= 0)).sum(axis=1)
    df["_hamming"] = np.where(valid_prev, changed, 0).astype(float)
    df["_payload_changed"] = (df["_hamming"] > 0).astype(float)
    return df


def compute_window_features(
    frames: pd.DataFrame,
    windows: pd.DataFrame,
    id_freq: dict[str, float],
) -> pd.DataFrame:
    """按窗口聚合 G1–G6 特征；特征矩阵不含任何标签字段。"""
    f = add_frame_level_columns(frames)
    f = f.set_index("frame_index", drop=False)
    default_freq = float(np.mean(list(id_freq.values()))) if id_freq else 0.0
    freq_vec = f["can_id"].astype(str).map(id_freq).fillna(default_freq).to_numpy()

    rows: list[dict[str, Any]] = []
    for w in windows.itertuples(index=False):
        sl = f.loc[w.frame_start : w.frame_end]
        if sl.empty:
            raise ValueError(f"窗口 {w.window_id} 找不到对应帧")
        dt = sl["_dt"].to_numpy()
        same_dt = sl["_same_id_dt"].to_numpy()
        same_dt_nz = same_dt[same_dt > 0]
        ids = sl["can_id"].to_numpy()
        uniq, id_counts = np.unique(ids, return_counts=True)
        # frame_index 在归一化后为连续整数，直接按位置取频次编码
        fw = sl.index.to_numpy(dtype=int)
        freq = freq_vec[fw]
        dlc = sl["dlc"].to_numpy(dtype=float)
        bmean = np.nan_to_num(sl["_byte_mean"].to_numpy(), nan=0.0)
        bstd = np.nan_to_num(sl["_byte_std"].to_numpy(), nan=0.0)
        bent = sl["_byte_entropy"].to_numpy()
        ham = sl["_hamming"].to_numpy()
        pchange = sl["_payload_changed"].to_numpy()
        payload = sl[DATA_COLUMNS].to_numpy()
        unique_payloads = len(np.unique(payload, axis=0))

        top1 = float(id_counts.max() / len(ids)) if len(ids) else 0.0
        min_id_share = float((ids == ids.min()).mean()) if len(ids) else 0.0
        _, dlc_counts = np.unique(dlc, return_counts=True)

        rec: dict[str, Any] = {
            "window_id": w.window_id,
            "segment_id": w.segment_id,
            "session_id": w.session_id,
            "source_tag": w.source_tag,
            "split": w.split,
            "frame_start": w.frame_start,
            "frame_end": w.frame_end,
            "t_start": w.t_start,
            "t_end": w.t_end,
            "label": int(w.label),
            "attack_ratio": float(w.attack_ratio),
            # G1 帧级·标识
            "g1_dlc_mean": float(dlc.mean()),
            "g1_dlc_std": float(dlc.std()),
            "g1_dlc_max": float(dlc.max()),
            "g1_id_freq_mean": float(freq.mean()),
            "g1_id_freq_std": float(freq.std()),
            "g1_id_freq_min": float(freq.min()),
            # G2 帧级·时间
            "g2_dt_mean": float(dt.mean()),
            "g2_dt_std": float(dt.std()),
            "g2_dt_min": float(dt.min()),
            "g2_dt_max": float(dt.max()),
            "g2_same_id_dt_mean": float(same_dt_nz.mean()) if same_dt_nz.size else 0.0,
            "g2_same_id_dt_std": float(same_dt_nz.std()) if same_dt_nz.size else 0.0,
            "g2_same_id_dt_cv": float(same_dt_nz.std() / same_dt_nz.mean()) if same_dt_nz.size and same_dt_nz.mean() > 0 else 0.0,
            # G3 帧级·payload
            "g3_byte_mean": float(bmean.mean()),
            "g3_byte_std": float(bstd.mean()),
            "g3_byte_entropy_mean": float(bent.mean()),
            "g3_hamming_mean": float(ham.mean()),
            "g3_payload_change_rate": float(pchange.mean()),
            # G4 窗口·频率
            "g4_frame_count": float(len(ids)),
            "g4_unique_ids": float(len(uniq)),
            "g4_id_entropy": _entropy(id_counts),
            "g4_top1_id_ratio": top1,
            "g4_min_id_share": min_id_share,
            # G5 窗口·周期
            "g5_same_id_dt_mean": float(same_dt_nz.mean()) if same_dt_nz.size else 0.0,
            "g5_same_id_dt_std": float(same_dt_nz.std()) if same_dt_nz.size else 0.0,
            "g5_same_id_dt_max": float(same_dt_nz.max()) if same_dt_nz.size else 0.0,
            "g5_dt_p50": float(np.percentile(dt, 50)),
            "g5_dt_p95": float(np.percentile(dt, 95)),
            # G6 窗口·变化
            "g6_payload_change_rate": float(pchange.mean()),
            "g6_byte_entropy_mean": float(bent.mean()),
            "g6_dlc_entropy": _entropy(dlc_counts),
            "g6_hamming_p95": float(np.percentile(ham, 95)),
            "g6_unique_payload_ratio": float(unique_payloads / len(ids)) if len(ids) else 0.0,
        }
        rows.append(rec)

    out = pd.DataFrame.from_records(rows)
    return out


def assert_feature_matrix_clean(feature_df: pd.DataFrame) -> dict[str, Any]:
    """泄漏断言：flag / label / can_id==0x000 布尔不得出现在特征列中。"""
    feature_cols = [c for c in feature_df.columns if c.startswith("g")]
    assert_no_forbidden_fields(feature_cols, context="窗口特征矩阵")
    # 显式检查：特征列名集合与 FEATURE_COLUMNS 完全一致，杜绝夹带
    assert set(feature_cols) == set(FEATURE_COLUMNS), (
        f"特征列与声明不一致：{set(feature_cols) ^ set(FEATURE_COLUMNS)}"
    )
    # 显式检查：不存在任何 0x000 布尔派生列（帧级/窗口级都不允许）
    zero_bool = [c for c in feature_df.columns if "zero" in c.lower() or "0x000" in c.lower() or "id000" in c.lower()]
    assert not zero_bool, f"存在 can_id==0x000 布尔派生列：{zero_bool}"
    # 显式检查：没有任何特征列恒等于标签（防止误把标签当特征）
    label_alias = [c for c in feature_cols if np.allclose(feature_df[c].to_numpy(), feature_df["label"].to_numpy())]
    assert not label_alias, f"存在与标签完全一致的伪特征：{label_alias}"
    return {
        "passed": True,
        "n_feature_columns": len(feature_cols),
        "feature_groups": {k: list(v) for k, v in FEATURE_GROUPS.items()},
        "forbidden_fields_absent": ["flag", "label", "can_id==0x000 布尔"],
        "note": "g4_min_id_share 为窗口内数值最小 ID 的相对占比，非帧级 0x000 布尔；已列入 W2 消融清单",
    }


def fit_normalization(train_features: pd.DataFrame, clip: float = 6.0) -> dict[str, Any]:
    """只在 train 折上 fit 归一化参数（均值/标准差）。"""
    params: dict[str, Any] = {
        "version": "feat-params-v1",
        "fit_on": "train",
        "clip": clip,
        "columns": {},
    }
    for col in FEATURE_COLUMNS:
        v = train_features[col].to_numpy(dtype=float)
        mean = float(np.mean(v))
        std = float(np.std(v))
        if not np.isfinite(mean):
            mean = 0.0
        if not np.isfinite(std) or std < 1e-9:
            std = 1.0
        params["columns"][col] = {"mean": mean, "std": std}
    return params


def apply_normalization(features: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    """用 train 折参数变换全部分折（含服务侧推理，共用本函数）。"""
    out = features.copy()
    clip = float(params["clip"])
    for col in FEATURE_COLUMNS:
        p = params["columns"][col]
        z = (out[col].to_numpy(dtype=float) - p["mean"]) / p["std"]
        out[col] = np.clip(np.nan_to_num(z, nan=0.0, posinf=clip, neginf=-clip), -clip, clip)
    return out


def build_golden_vector(
    frames: pd.DataFrame,
    windows: pd.DataFrame,
    id_freq: dict[str, float],
    n_frames: int,
) -> dict[str, Any]:
    """固定样本输入 → 特征值快照（未归一化的原始特征值）。"""
    set_seed(0)
    sample_seg = min(frames["segment_id"].unique())
    seg_frames = frames[frames["segment_id"] == sample_seg].head(n_frames).reset_index(drop=True)
    seg_frames = seg_frames.copy()
    seg_frames["frame_index"] = np.arange(len(seg_frames), dtype=np.int64)
    idx = seg_frames["frame_index"].to_numpy()
    win = pd.DataFrame(
        [
            {
                "window_id": "GOLDEN",
                "segment_id": sample_seg,
                "session_id": str(seg_frames["session_id"].iloc[0]),
                "source_tag": str(seg_frames["source_tag"].iloc[0]),
                "split": "train",
                "frame_start": int(idx[0]),
                "frame_end": int(idx[-1]),
                "t_start": float(seg_frames["timestamp"].min()),
                "t_end": float(seg_frames["timestamp"].max()),
                "n_frames": len(idx),
                "n_injected": int(seg_frames["label"].sum()),
                "attack_ratio": float(seg_frames["label"].mean()),
                "label": int(seg_frames["label"].mean() >= 0.2),
                "dominant_class": "unknown",
            }
        ]
    )
    fv = compute_window_features(seg_frames, win, id_freq)
    vector = {c: float(fv.loc[0, c]) for c in FEATURE_COLUMNS}
    return {
        "version": "golden-v1",
        "description": "固定样本输入 → G1–G6 原始特征值快照，供训练/推理特征一致性逐值比对",
        "sample": {
            "segment_id": sample_seg,
            "n_frames": len(seg_frames),
            "frame_start": int(idx[0]),
            "frame_end": int(idx[-1]),
            "t_start": float(seg_frames["timestamp"].min()),
            "t_end": float(seg_frames["timestamp"].max()),
            "sha256_of_input": None,
        },
        "feature_columns": list(FEATURE_COLUMNS),
        "values": vector,
    }


def write_feature_artifacts(
    features_raw: pd.DataFrame,
    features_norm: pd.DataFrame,
    params: dict[str, Any],
    golden: dict[str, Any],
    id_freq: dict[str, float],
    cfg: dict[str, Any],
    processed_dir: Path,
    artifacts_dir: Path,
    clean_report: dict[str, Any],
) -> dict[str, str]:
    feat_dir = processed_dir / "features"
    feat_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    raw_path = feat_dir / "window_features_raw.parquet"
    norm_path = feat_dir / "window_features.parquet"
    params_path = artifacts_dir / "feature_params.json"
    golden_path = artifacts_dir / "golden_vector.json"
    idfreq_path = artifacts_dir / "id_frequency_train.json"
    guard_path = artifacts_dir / "leakage_assertions.json"

    features_raw.to_parquet(raw_path, index=False, engine="pyarrow")
    features_norm.to_parquet(norm_path, index=False, engine="pyarrow")
    params["version"] = cfg["features"]["version"]
    params["random_seed"] = cfg["project"]["seed"]
    params["feature_groups"] = {k: list(v) for k, v in FEATURE_GROUPS.items()}
    write_json(params_path, params)
    write_json(golden_path, golden)
    write_json(idfreq_path, {"fit_on": "train", "id_frequency": id_freq})
    write_json(guard_path, clean_report)

    return {
        "window_features_raw": str(raw_path),
        "window_features": str(norm_path),
        "feature_params": str(params_path),
        "golden_vector": str(golden_path),
        "id_frequency_train": str(idfreq_path),
        "leakage_assertions": str(guard_path),
    }
