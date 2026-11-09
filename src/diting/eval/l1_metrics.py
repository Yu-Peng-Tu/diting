"""L1 · 二分类 / 五分类评估、阈值标定与 L0 同口径对照。

口径纪律（TASK-20261108-1319）：
- 评估只在 val；阈值标定只用 val；test 全程封存且不载入内存。
- 二分类指标复用既有 ``compute_metrics``（不依赖 sklearn，便于逐项复算核查），
  另补 AP（PR-AUC）与 ROC-AUC 作为阈值无关指标。
- 五分类固定 5 类顺序，报 macro P/R/F1 与 5x5 混淆矩阵；
  当 val 类别覆盖不全时，如实写出该指标的解释边界，不粉饰。
- 所有返回结构都带 ``data_scope`` / ``evaluated_on`` 标注。
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

from ..common import read_json
from .metrics import compute_metrics


def binary_metrics_at_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """给定阈值计算二分类全套指标（复用既有 compute_metrics 保证口径一致）。"""
    y_pred = (np.asarray(y_score) >= float(threshold)).astype(int)
    m = compute_metrics(y_true, y_pred)
    m["threshold"] = round(float(threshold), 6)
    m["average_precision"] = round(float(average_precision_score(y_true, y_score)), 6)
    m["roc_auc"] = round(float(roc_auc_score(y_true, y_score)), 6) if len(np.unique(y_true)) > 1 else None
    m["evaluated_on"] = cfg["evaluated_on"]
    m["data_scope"] = cfg["data_scope"]
    m["level"] = "window"
    return m


def calibrate_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """只在 val 上按 Youden's J = Recall - FPR 最大选定阈值，并留痕完整曲线。

    标定过程不接触 train 标签（train 仅用于拟合模型）、不接触 test。
    """
    g = cfg["threshold"]["grid"]
    grid = np.round(np.arange(float(g["start"]), float(g["stop"]) + 1e-9, float(g["step"])), 4)
    curve: list[dict[str, Any]] = []
    for thr in grid:
        y_pred = (y_score >= thr).astype(int)
        tp = int(((y_true == 1) & (y_pred == 1)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        tn = int(((y_true == 0) & (y_pred == 0)).sum())
        fn = int(((y_true == 1) & (y_pred == 0)).sum())
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        fpr = fp / (fp + tn) if (fp + tn) else 0.0
        curve.append({
            "threshold": round(float(thr), 4),
            "recall": round(recall, 6),
            "fpr": round(fpr, 6),
            "youden_j": round(recall - fpr, 6),
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        })
    best_j = max(c["youden_j"] for c in curve)
    # 平局时取更高阈值（更保守、误报更少），由配置声明
    candidates = [c for c in curve if abs(c["youden_j"] - best_j) < 1e-12]
    best = max(candidates, key=lambda c: c["threshold"])
    return {
        "version": cfg["threshold"]["version"],
        "criterion": cfg["threshold"]["criterion"],
        "criterion_note": "Youden's J = Recall - FPR，在 val 上取最大；平局取更高阈值",
        "calibrated_on": "val",
        "train_used_for": "model fitting only — 阈值未使用 train 标签",
        "test_set_status": cfg["test_set_status"],
        "default_threshold": float(cfg["threshold"]["default"]),
        "selected_threshold": best["threshold"],
        "selected_youden_j": best["youden_j"],
        "n_grid_points": len(curve),
        "grid_range": [float(g["start"]), float(g["stop"]), float(g["step"])],
        "curve": curve,
        "data_scope": cfg["data_scope"],
    }


def multiclass_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    cfg: dict[str, Any],
    model_classes: list[str] | None = None,
) -> dict[str, Any]:
    """五分类 macro P/R/F1 + 5x5 混淆矩阵（固定类别顺序）。"""
    order = list(cfg["classes"]["order"])
    labels = [str(c) for c in order]
    p, r, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    pm, rm, fm, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    per_class = {
        c: {
            "precision": round(float(p[i]), 6),
            "recall": round(float(r[i]), 6),
            "f1": round(float(f1[i]), 6),
            "support": int(support[i]),
        }
        for i, c in enumerate(labels)
    }
    # 定位互混：Gear 与 RPM 的双向错判（任务单点名观察项）
    gear_idx, rpm_idx = labels.index("Gear"), labels.index("RPM")
    gear_as_rpm = int(cm[gear_idx, rpm_idx])
    rpm_as_gear = int(cm[rpm_idx, gear_idx])
    return {
        "macro_precision": round(float(pm), 6),
        "macro_recall": round(float(rm), 6),
        "macro_f1": round(float(fm), 6),
        "per_class": per_class,
        "confusion_matrix": {
            "labels": labels,
            "matrix": cm.astype(int).tolist(),
            "note": "行=真实类别，列=预测类别；固定 5 类顺序",
        },
        "gear_rpm_confusion": {
            "Gear_predicted_as_RPM": gear_as_rpm,
            "RPM_predicted_as_Gear": rpm_as_gear,
            "total_bidirectional": gear_as_rpm + rpm_as_gear,
            "observable_on_val": bool(per_class["Gear"]["support"] > 0 and per_class["RPM"]["support"] > 0),
        },
        "model_classes_seen_in_train": list(model_classes) if model_classes is not None else None,
        "evaluated_on": cfg["evaluated_on"],
        "data_scope": cfg["data_scope"],
        "level": "window",
    }


def l0_comparison(
    l1_default: dict[str, Any],
    l1_calibrated: dict[str, Any],
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """L0 vs L1 同口径（同一 val 集、同一窗口级）对照表数据。"""
    from ..models.l1 import _resolve

    l0 = read_json(_resolve(cfg, "l0_metrics"))
    m = l0["metrics"]
    l0_row = {
        "model": "L0 统计规则基线",
        "recall": m["recall"], "fpr": m["fpr"],
        "precision": m["precision"], "f1": m["f1"],
        "false_alarms_per_10k_normal": m["false_alarms_per_10k_normal"],
        "threshold": "规则投票 >= 1",
    }
    rows = [l0_row]
    for tag, mm in (("L1 @0.5", l1_default), ("L1 @标定阈值", l1_calibrated)):
        rows.append({
            "model": tag,
            "recall": mm["recall"], "fpr": mm["fpr"],
            "precision": mm["precision"], "f1": mm["f1"],
            "false_alarms_per_10k_normal": mm["false_alarms_per_10k_normal"],
            "threshold": mm["threshold"],
        })
    base = l0_row
    deltas = []
    for row in rows[1:]:
        deltas.append({
            "model": row["model"],
            "fpr_absolute_change": round(row["fpr"] - base["fpr"], 6),
            "fpr_relative_change_pct": (
                round(100.0 * (row["fpr"] - base["fpr"]) / base["fpr"], 4)
                if base["fpr"] else None
            ),
            "recall_absolute_change": round(row["recall"] - base["recall"], 6),
            "f1_absolute_change": round(row["f1"] - base["f1"], 6),
            "false_alarms_per_10k_change": round(
                row["false_alarms_per_10k_normal"] - base["false_alarms_per_10k_normal"], 4
            ),
        })
    return {
        "evaluated_on": cfg["evaluated_on"],
        "data_scope": cfg["data_scope"],
        "level": "window",
        "l0_source": "artifacts/baseline/l0_metrics.json（只读既有产物）",
        "l0_split_manifest_version": l0.get("split_manifest_version"),
        "l0_n_samples": m["n_samples"],
        "l1_n_samples": l1_default["n_samples"],
        "same_val_note": (
            "L0 与 L1 均在同一 split-v2 val 集上按窗口级口径评估；"
            "若 n_samples 不一致则对照无效，需停工核查"
        ),
        "rows": rows,
        "deltas_vs_l0": deltas,
    }
