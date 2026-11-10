"""评估指标与报告（L0 留出验证集评估）。

指标口径：类别不平衡下不以 accuracy 为主指标，
报 Recall / FPR / Precision / F1 / 混淆矩阵，并同时给「每万正常窗口误报数」。
帧级 / 窗口级 / 事件级三套口径需分开报，W1 只在**窗口级**评估 L0。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..common import utc_now, write_json


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    """二分类指标 + 2x2 混淆矩阵（不依赖 sklearn，便于核查）。"""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "n_samples": len(y_true),
        "n_positive": int((y_true == 1).sum()),
        "n_negative": int((y_true == 0).sum()),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "recall": round(recall, 6),
        "precision": round(precision, 6),
        "fpr": round(fpr, 6),
        "f1": round(f1, 6),
        "false_alarms_per_10k_normal": round(1e4 * fpr, 4),
        "confusion_matrix": {
            "labels": ["Normal", "Attack"],
            "matrix": [[tn, fp], [fn, tp]],
        },
        "note": "窗口级口径；类别不平衡，不使用 accuracy 作为主指标",
    }


def evaluate_l0(
    predictions: pd.DataFrame,
    split_manifest: dict[str, Any],
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """只在留出验证集 val 上评估；test 集保持封存。"""
    val = predictions[predictions["split"] == "val"]
    if val.empty:
        raise ValueError("val 集合为空，无法评估")
    m = compute_metrics(val["label"].to_numpy(), val["pred"].to_numpy())

    by_source: dict[str, Any] = {}
    for tag, grp in val.groupby("source_tag"):
        by_source[str(tag)] = compute_metrics(grp["label"].to_numpy(), grp["pred"].to_numpy())

    return {
        "version": cfg["eval"]["version"],
        "model": "L0 统计规则 baseline",
        "model_version": cfg["baseline"]["version"],
        "evaluated_on": "val（留出验证集）",
        "test_set_status": "sealed — test 折全程封存，未评估",
        "level": "window",
        "split_manifest_version": split_manifest["version"],
        "evaluated_at": utc_now(),
        "is_real_data": bool((val["source_tag"] != "simulated").any()),
        "data_warning": (
            "评估基于 source_tag=simulated 自生成小样本，仅用于验证管线代码可运行；"
            "不是真实公开数据集结果，不得作为项目性能指标对外引用。"
            if not (val["source_tag"] != "simulated").any()
            else None
        ),
        "metrics": m,
        "metrics_by_source_tag": by_source,
        "rule_vote_distribution": {
            str(k): int(v) for k, v in val["rule_votes"].value_counts().sort_index().items()
        },
    }


def write_eval_artifacts(
    predictions: pd.DataFrame,
    report: dict[str, Any],
    rule: dict[str, Any],
    artifacts_dir: Path,
) -> dict[str, str]:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    pred_path = artifacts_dir / "l0_predictions.parquet"
    metrics_path = artifacts_dir / "l0_metrics.json"
    cm_path = artifacts_dir / "confusion_matrix.json"
    rule_path = artifacts_dir / "l0_rule_thresholds.json"
    report_path = artifacts_dir / "baseline_report.md"

    predictions.to_parquet(pred_path, index=False, engine="pyarrow")
    write_json(metrics_path, report)
    write_json(cm_path, report["metrics"]["confusion_matrix"])
    write_json(rule_path, rule)

    m = report["metrics"]
    lines = [
        "# L0 统计规则基线 · 留出验证集评估报告",
        "",
        f"- 评估口径：窗口级；评估集合：val（test 封存）；模型版本：{report['model_version']}",
        f"- 划分清单版本：{report['split_manifest_version']}；评估时间：{report['evaluated_at']}",
        "",
        "> **数据边界（务必先读）**：" + (report["data_warning"] or "真实公开数据集（HCRL Car-Hacking，Kaggle 镜像）留出验证集评估。"),
        "> 本项目不采集实车数据，不使用任何硬件；指标仅为离线评估结果。",
        "> W1 在 source_tag=simulated 小样本上的 Recall=1.0 / FPR=0.0833 属**仅管线验证，非项目指标**，",
        "> 已被本次真实数据结果取代，不得写入公开报告或对外报告。",
        "",
        "## 指标",
        "",
        "| 指标 | 值 |",
        "|---|---:|",
        f"| Recall（攻击类） | {m['recall']:.4f} |",
        f"| FPR | {m['fpr']:.4f} |",
        f"| Precision | {m['precision']:.4f} |",
        f"| F1 | {m['f1']:.4f} |",
        f"| 每万正常窗口误报数 | {m['false_alarms_per_10k_normal']:.2f} |",
        f"| TP / FP / TN / FN | {m['tp']} / {m['fp']} / {m['tn']} / {m['fn']} |",
        f"| 样本数（正/负） | {m['n_samples']}（{m['n_positive']}/{m['n_negative']}） |",
        "",
        "## 混淆矩阵",
        "",
        "| 真实 \\ 预测 | Normal | Attack |",
        "|---|---:|---:|",
        f"| Normal | {m['tn']} | {m['fp']} |",
        f"| Attack | {m['fn']} | {m['tp']} |",
        "",
        "## 规则与阈值",
        "",
        f"- 阈值拟合范围：`{rule['fit_on']}`（{rule['n_train_normal_windows']} 个训练正常窗口），不接触 val 标签与 test。",
        f"- 规则特征：`{'`, `'.join(rule['rule_features'])}`",
        f"- 排除特征：`{'`, `'.join(rule['excluded_from_rules'])}` — {rule['exclusion_reason']}",
        "",
        "L0 的价值是给后续所有模型一条共同下限并验证评估管线本身没写错；",
        "它不追求高指标，本阶段也不实现 L1 GBDT 及以后模型。",
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")

    return {
        "l0_predictions": str(pred_path),
        "l0_metrics": str(metrics_path),
        "confusion_matrix": str(cm_path),
        "l0_rule_thresholds": str(rule_path),
        "baseline_report": str(report_path),
    }
