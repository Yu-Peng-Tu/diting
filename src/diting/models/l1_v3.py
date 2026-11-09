"""谛听 W3 · split-v3 上的 L1 数据加载（复用既有 L1Dataset 与训练函数）。

与 W2 的 ``models/l1.py`` 的唯一实质差别是 **split 来源**：
- W2 直接信任 ``window_features.parquet`` 里的 ``split`` 列（属于 split-v2）；
- W3 必须从 ``split-v3/windows.parquet`` 读取 window_id → split 映射，
  并用 ``window_id in-list`` 下推特征表与标签表，**覆盖**特征表自带的旧 split 列。

test 封存纪律：
- split-v3/windows.parquet 只含 train/val 行（生成时已断言），故 in-list 中
  绝无 test 的 window_id，test 窗口的特征与标签根本不进内存；
- 载入后断言 split 集合恰为 {train, val}，行数与 v3 清单一致。

复用性：返回的 ``L1Dataset`` 与 W2 完全同构，可直接喂给既有
``fit_one`` / ``predict_proba`` / ``run_ablations`` /
``compute_permutation_importance`` / ``binary_metrics_at_threshold`` /
``calibrate_threshold`` / ``multiclass_metrics``，不改动任何既有代码。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from ..common import PROJECT_ROOT, assert_no_forbidden_fields, load_config, read_json, sha256_file
from ..features.build import FEATURE_COLUMNS, FEATURE_GROUPS
from .l1 import L1Dataset

V3_CONFIG_NAME = "l1_v3.yaml"


def load_l1_v3_config(path: str | Path | None = None) -> dict[str, Any]:
    """读取 configs/l1_v3.yaml，返回 l1 配置段并注入根目录与校验和。"""
    p = Path(path) if path else PROJECT_ROOT / "configs" / V3_CONFIG_NAME
    raw = load_config(p)
    if "l1" not in raw:
        raise ValueError("configs/l1_v3.yaml 缺少 l1 配置段")
    cfg = raw["l1"]
    cfg["_config_path"] = raw["_config_path"]
    cfg["_config_sha256"] = raw["_config_sha256"]
    cfg["_root"] = raw["_root"]
    cfg["_default_config_sha256"] = load_config()["_config_sha256"]
    return cfg


def _resolve(cfg: dict[str, Any], key: str) -> Path:
    p = Path(cfg["inputs"][key])
    return p if p.is_absolute() else PROJECT_ROOT / p


def load_l1_v3_dataset(cfg: dict[str, Any]) -> tuple[L1Dataset, pd.DataFrame]:
    """加载 split-v3 的 train/val 窗口，返回 (L1Dataset, 归因/L0 用元数据表)。

    ``aux`` 含 window_id / segment_id / source_tag / split(v3) / label /
    dominant_class / attack_ratio + 33 维特征，仅供 L0 重算与误报归因；
    其中 label / dominant_class 只从标签表读取，绝不来自特征表。
    模型特征矩阵 ``L1Dataset.features`` 仍严格只有 33 维 G1–G6。
    """
    feat_path = _resolve(cfg, "features")
    lab_path = _resolve(cfg, "labels")
    v3_win_path = _resolve(cfg, "split_v3_windows")
    v3_man_path = _resolve(cfg, "split_manifest")

    v3_windows = pq.read_table(v3_win_path).to_pandas()
    v3_manifest = read_json(v3_man_path)
    if "test" in set(v3_windows["split"].astype(str).unique()):
        raise AssertionError("split-v3 划分产物含 test 行，违反封存纪律")
    if set(v3_windows["split"].astype(str).unique()) != {"train", "val"}:
        raise AssertionError("split-v3 划分产物的 split 集合不是 {train, val}")

    split_map = dict(zip(v3_windows["window_id"].astype(str), v3_windows["split"].astype(str)))
    ids = list(split_map)
    if len(ids) != len(set(ids)):
        raise AssertionError("split-v3 存在重复 window_id")
    expected_n = (
        v3_manifest["split_windows"]["train"] + v3_manifest["split_windows"]["val"]
    )
    if len(ids) != expected_n:
        raise AssertionError(f"split-v3 行数 {len(ids)} 与清单 {expected_n} 不一致")

    # 特征表：window_id in-list 下推，test 行不进内存；不取 split 列（避免误用 v2 划分）
    feat_cols = ["window_id", "segment_id", "session_id", "source_tag", *FEATURE_COLUMNS]
    feat = pq.read_table(
        feat_path, columns=feat_cols, filters=[("window_id", "in", ids)]
    ).to_pandas()
    if len(feat) != len(ids):
        raise AssertionError(f"特征表 in-list 下推后行数 {len(feat)} != v3 窗口数 {len(ids)}")

    # 标签表：同样 in-list 下推；标签只从这里来
    lab_cols = ["window_id", "label", "dominant_class", "attack_ratio"]
    lab = pq.read_table(
        lab_path, columns=lab_cols, filters=[("window_id", "in", ids)]
    ).to_pandas()
    if len(lab) != len(ids):
        raise AssertionError(f"标签表 in-list 下推后行数 {len(lab)} != v3 窗口数 {len(ids)}")

    merged = feat.merge(lab, on="window_id", how="inner", validate="one_to_one")
    if len(merged) != len(ids):
        raise AssertionError("特征与标签按 window_id 内连接后行数变化，存在孤立窗口")

    # 用 split-v3 的映射覆盖划分（特征表自带 split 列未被读取）
    merged["split"] = merged["window_id"].astype(str).map(split_map)
    if merged["split"].isna().any():
        raise AssertionError("存在未能映射到 split-v3 划分的窗口")
    splits = set(merged["split"].astype(str).unique())
    if splits != {"train", "val"} or "test" in splits:
        raise AssertionError(f"载入后 split 集合异常：{splits}")

    # 泄漏断言：模型特征矩阵恰为 33 维 G1–G6，不含标签等价字段
    assert_no_forbidden_fields(FEATURE_COLUMNS, context="L1-v3 训练特征矩阵")
    assert len(FEATURE_COLUMNS) == 33, "L1-v3 期望 33 维窗口特征"
    missing = [c for c in FEATURE_COLUMNS if c not in merged.columns]
    if missing:
        raise AssertionError(f"特征矩阵缺少声明列：{missing}")

    y_binary = merged["label"].to_numpy(dtype=int)
    if not set(np.unique(y_binary)).issubset({0, 1}):
        raise AssertionError("二分类标签出现非 0/1 值")

    mapping = dict(cfg["classes"]["mapping"])
    order = list(cfg["classes"]["order"])
    raw_cls = merged["dominant_class"].astype(str)
    unknown = sorted(set(raw_cls.unique()) - set(mapping))
    if unknown:
        raise ValueError(f"dominant_class 出现未映射类别：{unknown}")
    y_multiclass = raw_cls.map(mapping).to_numpy()
    if not set(np.unique(y_multiclass)).issubset(set(order)):
        raise AssertionError("五分类标签映射后超出固定类别顺序")

    # 每折覆盖统计 + segment 不跨折断言（双保险，生成时已跑一遍）
    coverage: dict[str, Any] = {}
    for s in ("train", "val"):
        mask = (merged["split"] == s).to_numpy()
        coverage[s] = {
            "n_windows": int(mask.sum()),
            "n_attack": int((y_binary[mask] == 1).sum()),
            "n_normal": int((y_binary[mask] == 0).sum()),
            "class_counts": {c: int((y_multiclass[mask] == c).sum()) for c in order},
        }
        absent = [c for c, n in coverage[s]["class_counts"].items() if n <= 0]
        if absent:
            raise AssertionError(f"split-v3 的 {s} 折缺类别 {absent}，分层修复失败")
    cross = merged.groupby("segment_id")["split"].nunique()
    if int((cross > 1).sum()) > 0:
        raise AssertionError("存在跨 train/val 的 segment，违反防泄漏纪律")
    train_classes = {c for c, n in coverage["train"]["class_counts"].items() if n > 0}
    val_classes = {c for c, n in coverage["val"]["class_counts"].items() if n > 0}
    coverage["train_only_classes"] = sorted(train_classes - val_classes)
    coverage["val_only_classes"] = sorted(val_classes - train_classes)
    coverage["shared_classes"] = sorted(train_classes & val_classes)

    meta = {
        "features_sha256": sha256_file(feat_path),
        "labels_sha256": sha256_file(lab_path),
        "split_manifest_sha256": sha256_file(v3_man_path),
        "split_windows_sha256": sha256_file(v3_win_path),
        "split_manifest_version": v3_manifest.get("version"),
        "split_manifest_content_sha256": v3_manifest.get("content_sha256"),
        "n_rows": len(merged),
        "n_features": len(FEATURE_COLUMNS),
        "feature_groups": {k: list(v) for k, v in FEATURE_GROUPS.items()},
        "split_filter": (
            "pyarrow filters [('window_id','in', split-v3 train/val window_ids)] — "
            "test 窗口不在 in-list 中，特征/标签行均未载入内存"
        ),
        "split_source": "data/processed/split/split-v3/windows.parquet（覆盖特征表自带的 split-v2 列）",
        "label_source": "data/processed/labels/window_labels.parquet（既有窗口标签）",
        "test_rows_loaded": 0,
        "test_set_status": cfg["test_set_status"],
        "class_coverage": coverage,
        "data_scope": cfg["data_scope"],
    }
    dataset = L1Dataset(
        features=merged[FEATURE_COLUMNS].reset_index(drop=True),
        window_id=merged["window_id"].to_numpy(),
        split=merged["split"].to_numpy(),
        source_tag=merged["source_tag"].to_numpy(),
        y_binary=y_binary,
        y_multiclass=y_multiclass,
        feature_columns=list(FEATURE_COLUMNS),
        meta=meta,
    )
    aux_cols = [
        "window_id", "segment_id", "session_id", "source_tag", "split",
        "label", "dominant_class", "attack_ratio", *FEATURE_COLUMNS,
    ]
    aux = merged[aux_cols].reset_index(drop=True).copy()
    return dataset, aux
