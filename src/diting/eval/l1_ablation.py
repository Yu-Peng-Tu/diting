"""L1 · 三组特征消融 + train 折内类别可分性诊断。

三组消融（对应防泄漏叙事）：
  ① 剔除 ID 频率组 → 检验「模型学行为还是背 ID」（任务单点名结论项）；
  ② 剔除周期间隔组；
  ③ 剔除 payload 字节组。
每组都用同样的超参、同样的 train/val 口径重训重评，阈值固定为 0.5 与主模型
标定阈值两套并列，另给阈值无关的 AP（PR-AUC）作为最稳健的崩塌判据。

train 折内诊断：因 split-v2 的 val 只含 Normal/DoS，Gear↔RPM 互混在 val 上
物理不可观察。此诊断**只在 train 折内部**做一次分层留出，单独标注为
diagnostic_only / 非验收指标 / 不参与任何阈值或超参选择，仅用于如实回答
「训练见过的类别里，Gear 与 RPM 是否互混」。不改动划分、不触碰 test。
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split

from ..models.l1 import L1Dataset, fit_one
from .l1_metrics import binary_metrics_at_threshold


def _val_scores(model: Any, dataset: L1Dataset, cols: list[str], cfg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回 val 上的 y_true / y_score / y_pred@argmax（二分类模型）。"""
    from ..models.l1 import positive_column, predict_proba

    proba, classes = predict_proba(model, dataset, cols, split="val")
    y_true = dataset.y_binary[dataset.split == "val"]
    y_score = proba[:, positive_column(classes)]
    return y_true, y_score, classes


def run_ablations(
    cfg: dict[str, Any],
    dataset: L1Dataset,
    baseline_binary: dict[str, Any],
    baseline_threshold: float,
) -> dict[str, Any]:
    """重训重评三组消融，返回指标对比与解释数据。"""
    groups = cfg["ablation"]["groups"]
    results: list[dict[str, Any]] = []

    # 全特征基线（与 baseline_binary 一致，此处复算 AP 便于同表对照）
    full = {
        "variant": "full_33_features",
        "label": "全特征基线（33 维）",
        "dropped_group": None,
        "n_features": 33,
        "metrics_at_0_5": {k: baseline_binary.get(k) for k in ("recall", "fpr", "precision", "f1", "false_alarms_per_10k_normal")},
        "average_precision": baseline_binary["average_precision"],
    }
    results.append(full)

    for name in groups:
        # fit_one 内部按 drop_group 选择特征列并重训；此处不再重复 select_columns
        model, used = fit_one(cfg, dataset, task="binary", drop_group=name)
        y_true, y_score, _ = _val_scores(model, dataset, used, cfg)
        m05 = binary_metrics_at_threshold(y_true, y_score, 0.5, cfg)
        mcal = binary_metrics_at_threshold(y_true, y_score, baseline_threshold, cfg)
        ap = float(average_precision_score(y_true, y_score))
        results.append({
            "variant": f"drop_{name}",
            "label": f"剔除 {groups[name]['label']}（{len(groups[name]['columns'])} 维）",
            "dropped_group": name,
            "n_features": len(used),
            "dropped_columns": list(groups[name]["columns"]),
            "metrics_at_0_5": {k: m05[k] for k in ("recall", "fpr", "precision", "f1", "false_alarms_per_10k_normal")},
            "metrics_at_full_calibrated_threshold": {
                k: mcal[k] for k in ("recall", "fpr", "precision", "f1", "false_alarms_per_10k_normal")
            },
            "average_precision": round(ap, 6),
            "roc_auc": m05["roc_auc"],
        })

    # 与全特征基线的差值
    for r in results[1:]:
        r["delta_vs_full"] = {
            "ap_change": round(r["average_precision"] - full["average_precision"], 6),
            "ap_relative_change_pct": (
                round(100.0 * (r["average_precision"] - full["average_precision"]) / full["average_precision"], 4)
                if full["average_precision"] else None
            ),
            "f1_at_0_5_change": round(r["metrics_at_0_5"]["f1"] - full["metrics_at_0_5"]["f1"], 6),
            "fpr_at_0_5_change": round(r["metrics_at_0_5"]["fpr"] - full["metrics_at_0_5"]["fpr"], 6),
            "recall_at_0_5_change": round(r["metrics_at_0_5"]["recall"] - full["metrics_at_0_5"]["recall"], 6),
        }

    # 「去 ID 组」明确结论：以阈值无关 AP 为主判据，避免阈值选择干扰
    id_row = next(r for r in results if r["dropped_group"] == "id_frequency")
    ap_drop = full["average_precision"] - id_row["average_precision"]
    rel = ap_drop / full["average_precision"] if full["average_precision"] else 0.0
    collapsed = bool(rel >= 0.10)
    if collapsed:
        conclusion = (
            f"去掉 ID 频率组后 AP 从 {full['average_precision']:.6f} 降到 "
            f"{id_row['average_precision']:.6f}（相对下降 {100*rel:.2f}%，≥10%），"
            "指标明显崩塌：模型较依赖 ID 频率类特征，存在「背 ID」倾向，需按 §4.6 复核泄漏与特征设计。"
        )
    else:
        conclusion = (
            f"去掉 ID 频率组后 AP 从 {full['average_precision']:.6f} 到 "
            f"{id_row['average_precision']:.6f}（相对变化 {-100*rel:.2f}%，未达 10% 崩塌线），"
            "指标未崩塌：模型主要依赖行为类（周期/payload）特征，而非单纯记忆具体 ID。"
        )
    return {
        "version": cfg["ablation"]["version"],
        "evaluated_on": cfg["evaluated_on"],
        "data_scope": cfg["data_scope"],
        "threshold_policy": "统一报 0.5 与全特征模型标定阈值两套，并以阈值无关 AP 作为崩塌主判据",
        "baseline_calibrated_threshold": baseline_threshold,
        "collapse_rule": "AP 相对下降 ≥ 10% 判为崩塌（对应规格说明书 §5.4「去掉 ID 特征后 PR-AUC 下降 ≤ 0.10」）",
        "variants": results,
        "id_group_conclusion": {
            "collapsed": collapsed,
            "ap_full": full["average_precision"],
            "ap_without_id": id_row["average_precision"],
            "ap_relative_drop_pct": round(100 * rel, 4),
            "conclusion": conclusion,
        },
    }


def run_train_diagnostic(cfg: dict[str, Any], dataset: L1Dataset) -> dict[str, Any]:
    """train 折内部的分层留出诊断（非评估指标、不参与任何选择）。

    只用于观察训练见过的类别之间（尤其 Gear↔RPM）是否互混。
    """
    d = cfg["train_diagnostic"]
    if not d.get("enabled"):
        return {"enabled": False, "role": "diagnostic_only"}
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import confusion_matrix

    mask = dataset.split == "train"
    X = dataset.features.loc[mask, dataset.feature_columns].to_numpy(dtype=np.float64)
    y = dataset.y_multiclass[mask]
    present = sorted(set(y.tolist()))
    # 分层留出，固定种子；仅用 train，不触碰 val/test
    X_tr, X_ho, y_tr, y_ho = train_test_split(
        X, y, test_size=float(d["holdout_ratio"]),
        random_state=int(d["seed"]), stratify=y,
    )
    params = dict(cfg["model"]["params"])
    params["random_state"] = int(d["seed"])
    clf = HistGradientBoostingClassifier(**params)
    clf.fit(X_tr, y_tr)
    y_hat = clf.predict(X_ho)
    cm = confusion_matrix(y_ho, y_hat, labels=present)
    gear_rpm = None
    if "Gear" in present and "RPM" in present:
        gi, ri = present.index("Gear"), present.index("RPM")
        gear_rpm = {
            "Gear_predicted_as_RPM": int(cm[gi, ri]),
            "RPM_predicted_as_Gear": int(cm[ri, gi]),
            "Gear_support": int((y_ho == "Gear").sum()),
            "RPM_support": int((y_ho == "RPM").sum()),
        }
        tot = gear_rpm["Gear_predicted_as_RPM"] + gear_rpm["RPM_predicted_as_Gear"]
        gear_rpm["bidirectional_total"] = tot
        gear_rpm["note"] = (
            "train 内留出诊断：Gear 与 RPM 双向互混计数；"
            "这不是 val 评估指标，不作为验收结论，不参与阈值/超参选择。"
        )
    return {
        "enabled": True,
        "role": d["role"],
        "is_acceptance_metric": d["is_acceptance_metric"],
        "warning": (
            "本诊断仅在 train 折内部做分层留出，因 split-v2 的 val 只含 Normal/DoS、"
            "不含 Gear/RPM，任务单点名的「Gear↔RPM 互混」在 val 上无法观察。"
            "此结果仅作定性补充证据，严禁写入对外指标或公开报告。"
        ),
        "classes_in_train": present,
        "holdout_ratio": float(d["holdout_ratio"]),
        "seed": int(d["seed"]),
        "n_train_fit": len(X_tr),
        "n_train_holdout": len(X_ho),
        "confusion_matrix": {"labels": present, "matrix": cm.astype(int).tolist()},
        "gear_rpm_confusion": gear_rpm,
        "test_set_status": cfg["test_set_status"],
        "data_scope": cfg["data_scope"],
    }
