"""L1 · 手工窗口特征 + GBDT 检测/识别模型。

职责与边界（硬性纪律）：
- 只加载 train/val 行：用 pyarrow 的 filters 下推，**test 行根本不进入内存**；
- 标签一律来自既有 ``window_labels.parquet``（不重算、不从特征表夹带 label/attack_ratio）；
- 特征矩阵复用既有 ``window_features.parquet`` 的 33 维 G1–G6（不重新实现特征计算，
  训练与推理共用同一份特征矩阵文件）；
- 沿用既有泄漏断言：flag / label / can_id==0x000 布尔等标签等价信息禁止入特征；
- 提供二分类（Normal vs Attack）与五分类（Normal/DoS/Fuzzy/Gear/RPM）两种模型，
  超参与类别映射全部由 configs/l1.yaml 驱动。

代码注释用中文，标识符用英文。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.ensemble import HistGradientBoostingClassifier

from ..common import PROJECT_ROOT, assert_no_forbidden_fields, load_config
from ..features.build import FEATURE_COLUMNS, FEATURE_GROUPS


@dataclass
class L1Dataset:
    """一次加载的 train/val 窗口级数据（test 不含在内）。"""

    features: pd.DataFrame            # 仅 33 维特征列，index 已 reset
    window_id: np.ndarray
    split: np.ndarray
    source_tag: np.ndarray
    y_binary: np.ndarray              # 0=Normal, 1=Attack
    y_multiclass: np.ndarray          # 固定 5 类名字符串
    feature_columns: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)


def load_l1_config(path: str | Path | None = None) -> dict[str, Any]:
    """读取 configs/l1.yaml，返回 l1 配置段并注入根目录与校验和。"""
    p = Path(path) if path else PROJECT_ROOT / "configs" / "l1.yaml"
    raw = load_config(p)
    if "l1" not in raw:
        raise ValueError("configs/l1.yaml 缺少 l1 配置段")
    cfg = raw["l1"]
    cfg["_config_path"] = raw["_config_path"]
    cfg["_config_sha256"] = raw["_config_sha256"]
    cfg["_root"] = raw["_root"]
    cfg["_default_config_sha256"] = load_config()["_config_sha256"]
    return cfg


def _resolve(cfg: dict[str, Any], key: str) -> Path:
    """把配置中的相对路径解析为项目根下的绝对路径。"""
    p = Path(cfg["inputs"][key])
    return p if p.is_absolute() else PROJECT_ROOT / p


def validate_ablation_groups(cfg: dict[str, Any]) -> dict[str, Any]:
    """校验三组消融配置合法：组内列均属于 33 维、组间互斥、保留组非空。

    该校验同时被 pytest 复用（验收项「消融配置合法」）。
    """
    groups = cfg["ablation"]["groups"]
    declared = set(FEATURE_COLUMNS)
    seen: set[str] = set()
    report: dict[str, Any] = {"groups": {}, "retained_base_columns": [], "passed": True}
    for name, spec in groups.items():
        cols = list(spec["columns"])
        unknown = [c for c in cols if c not in declared]
        if unknown:
            raise ValueError(f"消融组 {name} 含未声明特征列：{unknown}")
        dup = seen.intersection(cols)
        if dup:
            raise ValueError(f"消融组 {name} 与其它组存在重叠列：{sorted(dup)}")
        seen.update(cols)
        report["groups"][name] = {"label": spec["label"], "n_columns": len(cols), "columns": cols}
    retained = list(cfg["ablation"]["retained_base_columns"])
    if any(c not in declared for c in retained):
        raise ValueError("保留基础组含未声明特征列")
    if seen.intersection(retained):
        raise ValueError("保留基础组与消融组存在重叠列")
    if set(retained) | seen != declared:
        missing = declared - (set(retained) | seen)
        raise ValueError(f"消融组 + 保留组未覆盖全部 33 维，缺：{sorted(missing)}")
    if not retained:
        raise ValueError("保留基础组不能为空（否则消融后无特征可训练）")
    report["retained_base_columns"] = retained
    report["n_total"] = len(declared)
    report["n_ablated"] = len(seen)
    return report


def load_l1_dataset(cfg: dict[str, Any]) -> L1Dataset:
    """加载 train/val 窗口特征与标签，强制 test 不留痕、标签不夹带。

    1) 特征表用 ``filters=[('split','in',['train','val'])]`` 下推读取，
       且 columns 只取 window_id/split/source_tag + 33 维特征，不取 label/attack_ratio；
    2) 标签表无 split 列，用 window_id in-list 下推读取，同样不把 test 行载入内存；
    3) 两者按 window_id 内连接，校验行数一致、标签只取 0/1、类别映射全覆盖。
    """
    feat_path = _resolve(cfg, "features")
    lab_path = _resolve(cfg, "labels")
    columns = ["window_id", "split", "source_tag", *FEATURE_COLUMNS]

    # 读取特征：只保留 train/val；断言 split 集合，杜绝 test 混入
    feat_tbl = pq.read_table(
        feat_path, columns=columns, filters=[("split", "in", ["train", "val"])]
    )
    feat = feat_tbl.to_pandas()
    splits = set(feat["split"].astype(str).unique())
    if splits != {"train", "val"}:
        raise AssertionError(f"特征表下推过滤后仍含非 train/val 集合：{splits}")
    if "test" in splits:
        raise AssertionError("test 集被载入，违反封存纪律")

    # 泄漏断言：特征列必须是且仅是 33 维 G1–G6，且不含标签等价字段
    assert_no_forbidden_fields(FEATURE_COLUMNS, context="L1 训练特征矩阵")
    assert len(FEATURE_COLUMNS) == 33, "L1 期望 33 维窗口特征"
    missing_cols = [c for c in FEATURE_COLUMNS if c not in feat.columns]
    if missing_cols:
        raise AssertionError(f"特征矩阵缺少声明列：{missing_cols}")

    ids = feat["window_id"].tolist()
    # 读取标签：用 window_id in-list 下推，只取标签列（不含特征）
    lab_tbl = pq.read_table(
        lab_path,
        columns=["window_id", "label", "dominant_class"],
        filters=[("window_id", "in", ids)],
    )
    lab = lab_tbl.to_pandas()
    if len(lab) != len(feat):
        raise AssertionError(f"标签与特征行数不一致：{len(lab)} vs {len(feat)}")

    merged = feat.merge(lab[["window_id", "label", "dominant_class"]], on="window_id", how="inner", validate="one_to_one")
    if len(merged) != len(feat):
        raise AssertionError("标签与特征按 window_id 内连接后行数变化，存在孤立窗口")

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

    # 类别覆盖统计（如实暴露 split-v2 的类别错位，不做任何隐藏或重划分）
    coverage: dict[str, Any] = {}
    for s in ("train", "val"):
        mask = (merged["split"] == s).to_numpy()
        coverage[s] = {
            "n_windows": int(mask.sum()),
            "n_attack": int((y_binary[mask] == 1).sum()),
            "n_normal": int((y_binary[mask] == 0).sum()),
            "class_counts": {c: int((y_multiclass[mask] == c).sum()) for c in order},
        }
    train_classes = {c for c, n in coverage["train"]["class_counts"].items() if n > 0}
    val_classes = {c for c, n in coverage["val"]["class_counts"].items() if n > 0}
    coverage["train_only_classes"] = sorted(train_classes - val_classes)
    coverage["val_only_classes"] = sorted(val_classes - train_classes)
    coverage["shared_classes"] = sorted(train_classes & val_classes)

    meta = {
        "features_sha256": _sha(feat_path),
        "labels_sha256": _sha(lab_path),
        "n_rows": len(merged),
        "n_features": len(FEATURE_COLUMNS),
        "feature_groups": {k: list(v) for k, v in FEATURE_GROUPS.items()},
        "split_filter": "pyarrow filters [('split','in',['train','val'])] — test 行未载入内存",
        "label_source": "data/processed/labels/window_labels.parquet（既有窗口标签）",
        "test_rows_loaded": 0,
        "test_set_status": cfg["test_set_status"],
        "class_coverage": coverage,
        "data_scope": cfg["data_scope"],
    }
    return L1Dataset(
        features=merged[FEATURE_COLUMNS].reset_index(drop=True),
        window_id=merged["window_id"].to_numpy(),
        split=merged["split"].to_numpy(),
        source_tag=merged["source_tag"].to_numpy(),
        y_binary=y_binary,
        y_multiclass=y_multiclass,
        feature_columns=list(FEATURE_COLUMNS),
        meta=meta,
    )


def _sha(path: Path) -> str:
    from ..common import sha256_file

    return sha256_file(path)


def build_estimator(cfg: dict[str, Any], task: str = "binary") -> HistGradientBoostingClassifier:
    """按 YAML 超参构造 HistGradientBoostingClassifier（固定随机种子）。"""
    params = dict(cfg["model"]["params"])
    params.update(cfg["model"].get(f"{task}_params", {}) or {})
    params["random_state"] = int(cfg["seed"])
    return HistGradientBoostingClassifier(**params)


def select_columns(cfg: dict[str, Any], drop_group: str | None = None) -> list[str]:
    """返回本次训练使用的特征列；drop_group 非空时剔除对应消融组。"""
    if drop_group is None:
        return list(FEATURE_COLUMNS)
    groups = cfg["ablation"]["groups"]
    if drop_group not in groups:
        raise ValueError(f"未知消融组：{drop_group}")
    drop = set(groups[drop_group]["columns"])
    return [c for c in FEATURE_COLUMNS if c not in drop]


def fit_one(
    cfg: dict[str, Any],
    dataset: L1Dataset,
    task: str = "binary",
    drop_group: str | None = None,
) -> tuple[HistGradientBoostingClassifier, list[str]]:
    """只用 train 折拟合一个模型；返回 (模型, 实际使用的特征列)。"""
    cols = select_columns(cfg, drop_group)
    train_mask = dataset.split == "train"
    X = dataset.features.loc[train_mask, cols].to_numpy(dtype=np.float64)
    y = dataset.y_binary[train_mask] if task == "binary" else dataset.y_multiclass[train_mask]
    est = build_estimator(cfg, task)
    est.fit(X, y)
    return est, cols


def predict_proba(
    model: HistGradientBoostingClassifier,
    dataset: L1Dataset,
    cols: list[str],
    split: str = "val",
) -> tuple[np.ndarray, np.ndarray]:
    """在指定 split（默认 val）上输出概率矩阵与模型类别顺序。"""
    mask = dataset.split == split
    X = dataset.features.loc[mask, cols].to_numpy(dtype=np.float64)
    return model.predict_proba(X), np.asarray(model.classes_)


def positive_column(classes: np.ndarray) -> int:
    """返回二分类正类（label==1）在 predict_proba 中的列下标。"""
    idx = np.where(classes == 1)[0]
    if len(idx) != 1:
        raise AssertionError(f"二分类模型类别异常：{classes.tolist()}")
    return int(idx[0])
