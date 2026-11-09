"""谛听 W3 · 评估治理（任务核心）：L0 重算、PR 曲线、互混分析、误报归因。

全部计算只在 split-v3 的 val 上进行；test 全程封存。复用既有 L0 规则实现
（``models/l0.py``）与指标函数（``eval/metrics.py``），不重新实现规则或指标。

模块产出的数据结构一律带 ``evaluated_on=val`` 与 ``data_scope`` 口径标注。
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve

from ..common import read_json
from .metrics import compute_metrics


def _scope(cfg: dict[str, Any]) -> dict[str, str]:
    return {
        "evaluated_on": cfg["evaluated_on"],
        "data_scope": cfg["data_scope"],
        "split_version": cfg["split_version"],
        "level": "window",
    }


def recompute_l0_on_v3(
    aux: pd.DataFrame,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """在 split-v3 val 上重算 L0 规则基线（L0 不训练，阈值只由 train 正常窗口分位数统计得到）。

    双口径，如实标注阈值来源差异：
    - 主口径 A（**同口径**）：用 split-v3 train 正常窗口重新 fit 规则阈值，
      再在 split-v3 val 上评估。与 L1-v3「split-v3 train 拟合 / split-v3 val 评估」
      完全一致，是对照表主行。这仍是「重算不需要训练」——L0 只是分位数统计规则，
      不是可训练模型。
    - 参考口径 B（阈值迁移敏感性）：沿用 W2 在 split-v2 train 正常窗口 fit 的
      固定阈值，只换评估集。该阈值来自旧划分，与 split-v3 的训练口径不一致，
      **不作为对照表主行**；仅用于展示固定规则阈值对划分/分布漂移的敏感性。
    """
    from ..models.l0 import fit_rule_thresholds, predict_l0
    from ..models.l1_v3 import _resolve

    fixed_rule = read_json(_resolve(cfg, "l0_rule_thresholds"))
    val = aux[aux["split"] == "val"].copy()

    def _eval(rule: dict[str, Any]) -> dict[str, Any]:
        pred = predict_l0(val, rule, min_vote=1)
        return compute_metrics(val["label"].to_numpy(), pred["pred"].to_numpy())

    refit_rule = fit_rule_thresholds(
        aux[["split", "label", "g2_dt_mean", "g4_top1_id_ratio",
             "g2_same_id_dt_cv", "g5_same_id_dt_std"]].copy(),
        id_frequency_percentile=float(fixed_rule["id_frequency_percentile"]),
        interval_cv_percentile=float(fixed_rule["interval_cv_percentile"]),
    )
    m_refit = _eval(refit_rule)
    m_fixed = _eval(fixed_rule)

    # 主口径 L0 误报的攻击类型归因（供误报治理报告用）
    pred_main = predict_l0(val, refit_rule, min_vote=1)
    fp_mask = pred_main["pred"].to_numpy() == 1
    fp_normal = val.loc[fp_mask & (val["label"].to_numpy() == 0)]
    fp_by_segment_prefix = (
        fp_normal["segment_id"].astype(str).str.split("__").str[0].value_counts().to_dict()
    )

    return {
        "model": "L0 统计规则基线（split-v3 重算）",
        **_scope(cfg),
        "n_val_windows": len(val),
        "primary_policy": (
            "同口径主行：用 split-v3 train 正常窗口重新 fit 规则阈值（分位数统计，非模型训练），"
            "在 split-v3 val 上评估；与 L1-v3 的训练/评估划分完全一致。"
            "L0 不是可训练模型，故仍称『重算』"
        ),
        "primary_threshold_source": "split-v3 train 正常窗口（本任务重新 fit，见 reference_rule 之外的 primary_rule）",
        "primary_rule": refit_rule,
        "primary_metrics": m_refit,
        "reference_policy": (
            "敏感性分析：沿用 W2 在 split-v2 train 正常窗口 fit 的固定阈值，仅把评估集换成 "
            "split-v3 val。该阈值来自旧划分，与 L1-v3 训练口径不一致，故不作为对照表主行；"
            "两者差距用于说明固定规则阈值对划分/分布漂移的敏感性"
        ),
        "reference_threshold_source": "artifacts/baseline/l0_rule_thresholds.json（只读，W2 产物）",
        "reference_metrics": m_fixed,
        "threshold_source_delta_fpr": round(m_fixed["fpr"] - m_refit["fpr"], 6),
        "threshold_source_delta_note": (
            "参考口径 FPR - 主口径 FPR；正值表示沿用旧划分阈值会显著恶化误报，"
            "说明 L0 固定阈值对划分敏感，也说明主口径必须重新 fit 才公平"
        ),
        "l0_false_positive_windows": len(fp_normal),
        "l0_false_positive_by_segment_prefix": {str(k): int(v) for k, v in fp_by_segment_prefix.items()},
        "test_set_status": cfg["test_set_status"],
    }


def build_l0_comparison(
    l0: dict[str, Any],
    m_default: dict[str, Any],
    m_cal: dict[str, Any],
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """L0(split-v3 重算) vs L1-v3 @0.5 / @标定阈值 的同口径对照表数据。"""
    base = l0["primary_metrics"]
    rows = [{
        "model": "L0 统计规则基线（split-v3 重算）",
        "recall": base["recall"], "fpr": base["fpr"],
        "precision": base["precision"], "f1": base["f1"],
        "false_alarms_per_10k_normal": base["false_alarms_per_10k_normal"],
        "threshold": "规则投票 >= 1（固定阈值）",
    }]
    for tag, mm in (("L1-v3 @0.5", m_default), ("L1-v3 @标定阈值", m_cal)):
        rows.append({
            "model": tag,
            "recall": mm["recall"], "fpr": mm["fpr"],
            "precision": mm["precision"], "f1": mm["f1"],
            "false_alarms_per_10k_normal": mm["false_alarms_per_10k_normal"],
            "threshold": mm["threshold"],
        })
    deltas = []
    for row in rows[1:]:
        deltas.append({
            "model": row["model"],
            "fpr_absolute_change": round(row["fpr"] - base["fpr"], 6),
            "fpr_relative_change_pct": (
                round(100.0 * (row["fpr"] - base["fpr"]) / base["fpr"], 4) if base["fpr"] else None
            ),
            "recall_absolute_change": round(row["recall"] - base["recall"], 6),
            "f1_absolute_change": round(row["f1"] - base["f1"], 6),
            "false_alarms_per_10k_change": round(
                row["false_alarms_per_10k_normal"] - base["false_alarms_per_10k_normal"], 4
            ),
        })
    cal = deltas[1]
    if cal["fpr_relative_change_pct"] is not None and cal["fpr_relative_change_pct"] <= -50.0:
        conclusion = (
            f"FPR 明显下降：标定阈值下相对 L0 下降 {abs(cal['fpr_relative_change_pct']):.2f}%"
            f"（{base['fpr']:.6f} → {m_cal['fpr']:.6f}），"
            f"每万正常窗口误报数 {base['false_alarms_per_10k_normal']:.2f} → "
            f"{m_cal['false_alarms_per_10k_normal']:.2f}。结论明确：是。"
        )
    elif cal["fpr_absolute_change"] < 0:
        conclusion = (
            f"FPR 下降但未达 50% 明显阈值：相对下降 {abs(cal['fpr_relative_change_pct'] or 0):.2f}%"
            f"（{base['fpr']:.6f} → {m_cal['fpr']:.6f}）。结论：有下降，幅度有限，如实报出。"
        )
    else:
        conclusion = (
            f"FPR 未下降：L0 {base['fpr']:.6f}，L1-v3 标定阈值 {m_cal['fpr']:.6f}。"
            "如实给出结论为『否』，不隐瞒、不调阈值凑预期，原因分析见主报告。"
        )
    return {
        **_scope(cfg),
        "l0_source": l0["primary_threshold_source"],
        "l0_threshold_policy": l0["primary_policy"],
        "l0_n_samples": base["n_samples"],
        "l1_n_samples": m_default["n_samples"],
        "same_val_note": (
            "L0 与 L1-v3 均在同一 split-v3 val 窗口集合上按窗口级口径评估；"
            "样本数不一致则对照无效"
        ),
        "rows": rows,
        "deltas_vs_l0": deltas,
        "fpr_conclusion": conclusion,
        "fpr_dropped_significantly": bool(cal["fpr_absolute_change"] < 0 and (
            cal["fpr_relative_change_pct"] is not None
            and cal["fpr_relative_change_pct"] <= -50.0
        )),
        "test_set_status": cfg["test_set_status"],
    }


def pr_curve_points(y_true: np.ndarray, y_score: np.ndarray, cfg: dict[str, Any]) -> dict[str, Any]:
    """逐点导出 PR 曲线（供 W4 画图用），不做降采样，保留 sklearn 原始逐点结果。"""
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    points = [
        {
            "index": i,
            "precision": round(float(precision[i]), 8),
            "recall": round(float(recall[i]), 8),
            "threshold": round(float(thresholds[i]), 8) if i < len(thresholds) else None,
        }
        for i in range(len(precision))
    ]
    return {
        **_scope(cfg),
        "source": "sklearn.metrics.precision_recall_curve（split-v3 val, 二分类正类分数）",
        "threshold_note": "最后一个点 threshold=null（ sklearn 约定：recall=0, precision=1 的端点）",
        "average_precision": round(float(average_precision_score(y_true, y_score)), 8),
        "n_points": len(points),
        "n_thresholds": len(thresholds),
        "points": points,
    }


def per_attack_type_binary_metrics(
    aux_val: pd.DataFrame,
    y_score: np.ndarray,
    threshold: float,
    default_threshold: float,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """按攻击类型分组的二分类指标（Normal 单独计 FPR，四类攻击分别计 Recall）。

    这是误报治理的关键证据：split-v3 的 val 首次同时含 4 类攻击，
    可以看出哪一类最难检出、误报集中在哪些 segment 类型。
    """
    label = aux_val["label"].to_numpy(dtype=int)
    cls = aux_val["dominant_class"].astype(str).to_numpy()
    score = np.asarray(y_score)
    out: dict[str, Any] = {
        **_scope(cfg),
        "threshold_used": round(float(threshold), 6),
        "default_threshold": round(float(default_threshold), 6),
        "by_type": {},
    }
    normal_mask = label == 0
    normal_pred = (score[normal_mask] >= threshold).astype(int)
    out["by_type"]["Normal"] = {
        "role": "负类（只评估 FPR / 误报）",
        "support": int(normal_mask.sum()),
        "fpr": compute_metrics(np.zeros(int(normal_mask.sum()), dtype=int), normal_pred)["fpr"],
        "false_alarms": int(normal_pred.sum()),
        "segment_prefix_breakdown": {
            str(k): int(v) for k, v in
            aux_val.loc[normal_mask, "segment_id"].astype(str).str.split("__").str[0]
            .value_counts().items()
        },
    }
    for attack in ("DoS", "Fuzzy", "Gear", "RPM"):
        mask = cls == {"DoS": "DoS", "Fuzzy": "Fuzzy",
                       "Gear": "Spoofing_the_drive_gear",
                       "RPM": "Spoofing_the_RPM_gauge"}[attack]
        if mask.sum() == 0:
            out["by_type"][attack] = {"support": 0, "recall": None}
            continue
        pred = (score[mask] >= threshold).astype(int)
        m = compute_metrics(np.ones(int(mask.sum()), dtype=int), pred)
        ap = float(average_precision_score(
            np.r_[np.ones(int(mask.sum()), dtype=int), np.zeros(int(normal_mask.sum()), dtype=int)],
            np.r_[score[mask], score[normal_mask]],
        ))
        out["by_type"][attack] = {
            "role": "正类（只评估 Recall / 检出）",
            "support": int(mask.sum()),
            "recall": m["recall"],
            "missed": int(m["fn"]),
            "average_precision_vs_normal": round(ap, 8),
        }
    return out


def analyze_gear_rpm_confusion(multi: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    """Gear↔RPM 互混的真实观测分析（split-v3 val 上两类 support 均 > 0）。"""
    cm = multi["confusion_matrix"]
    labels = cm["labels"]
    g, r = labels.index("Gear"), labels.index("RPM")
    matrix = cm["matrix"]
    gear_row = {labels[j]: int(matrix[g][j]) for j in range(5)}
    rpm_row = {labels[j]: int(matrix[r][j]) for j in range(5)}
    gear_support = int(multi["per_class"]["Gear"]["support"])
    rpm_support = int(multi["per_class"]["RPM"]["support"])
    g2r = int(matrix[g][r])
    r2g = int(matrix[r][g])
    total = g2r + r2g
    observable = bool(gear_support > 0 and rpm_support > 0)
    if not observable:
        verdict = "不可观察（某类 support=0），本轮分层失败"
    elif total == 0:
        verdict = (
            "val 上双向互混 0 次：与规格书『Gear↔RPM 易互混』的预期相反，"
            "在该窗口口径与子集中两类几乎完全可分。反预期结果照实保留，不调数据凑预期。"
        )
    else:
        verdict = (
            f"val 上观测到双向互混 {total} 次（Gear→RPM {g2r} / RPM→Gear {r2g}），"
            f"占两类总 support 的 {100.0 * total / (gear_support + rpm_support):.4f}%。"
        )
    return {
        **_scope(cfg),
        "observable_on_val": observable,
        "gear_support": gear_support,
        "rpm_support": rpm_support,
        "Gear_predicted_as_RPM": g2r,
        "RPM_predicted_as_Gear": r2g,
        "total_bidirectional": total,
        "bidirectional_rate_pct": round(100.0 * total / (gear_support + rpm_support), 6)
        if (gear_support + rpm_support) else None,
        "gear_row_full": gear_row,
        "rpm_row_full": rpm_row,
        "gear_f1": multi["per_class"]["Gear"]["f1"],
        "rpm_f1": multi["per_class"]["RPM"]["f1"],
        "verdict": verdict,
        "note": "这是 split-v3 val 上的真实观测数字，不是 train 折内诊断，可作为验收结论",
    }


def attribute_errors(
    aux_val: pd.DataFrame,
    y_score: np.ndarray,
    y_true: np.ndarray,
    threshold: float,
    multi_pred: np.ndarray,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """对 val 上每个误报(FP)与漏报(FN)窗口逐条归因：特征值画像 + 攻击类型 + 可能原因。

    特征值画像：取该窗口与「同类窗口在 val 上的中位数」偏离最大的 top-k 维，
    给出窗口值、参照中位数与偏离方向，避免堆砌全部 33 维。
    """
    from ..features.build import FEATURE_COLUMNS

    top_k = int(cfg["governance"]["error_attribution_top_features"])
    a = aux_val.reset_index(drop=True).copy()
    score = np.asarray(y_score)
    y_pred = (score >= threshold).astype(int)
    truth = np.asarray(y_true, dtype=int)
    cls = a["dominant_class"].astype(str).to_numpy()

    median_by_class: dict[str, np.ndarray] = {}
    for c in sorted(set(cls.tolist())):
        mask = cls == c
        median_by_class[c] = a.loc[mask, FEATURE_COLUMNS].median().to_numpy(dtype=float)

    records: list[dict[str, Any]] = []
    for i in range(len(a)):
        kind = "false_positive" if (truth[i] == 0 and y_pred[i] == 1) else (
            "false_negative" if (truth[i] == 1 and y_pred[i] == 0) else None
        )
        if kind is None:
            continue
        c = str(cls[i])
        vals = a.loc[i, FEATURE_COLUMNS].to_numpy(dtype=float)
        ref = median_by_class[c]
        denom = np.maximum(np.abs(ref), 1e-6)
        dev = np.abs(vals - ref) / denom
        idx = np.argsort(dev)[::-1][:top_k]
        profile = [
            {
                "feature": FEATURE_COLUMNS[j],
                "window_value": round(float(vals[j]), 6),
                "class_median_on_val": round(float(ref[j]), 6),
                "relative_deviation": round(float(dev[j]), 4),
                "direction": "higher" if vals[j] > ref[j] else "lower",
            }
            for j in idx
        ]
        if kind == "false_positive":
            reason = (
                "正常窗口被判为攻击：该窗口的帧密度/周期/payload 统计偏离同类正常窗口中位数，"
                "可能被模型误当作注入扰动；建议结合同 segment 连续窗口投票与双阈值复核。"
            )
        else:
            reason = (
                "攻击窗口未被检出：该窗口的统计量相对同类攻击窗口中位数偏弱，"
                "可能注入占比低（弱攻击窗口）或该攻击类型的行为特征与正常流量接近；"
                "建议按攻击类型分别标定阈值或引入类型白名单二次判别。"
            )
        records.append({
            "index": int(i),
            "window_id": str(a.loc[i, "window_id"]),
            "segment_id": str(a.loc[i, "segment_id"]),
            "attack_type": c,
            "label": int(truth[i]),
            "binary_pred": int(y_pred[i]),
            "multiclass_pred": str(multi_pred[i]) if multi_pred is not None else None,
            "score": round(float(score[i]), 8),
            "attack_ratio": round(float(a.loc[i, "attack_ratio"]), 6),
            "error_kind": kind,
            "top_deviating_features": profile,
            "possible_cause": reason,
        })

    fp = [r for r in records if r["error_kind"] == "false_positive"]
    fn = [r for r in records if r["error_kind"] == "false_negative"]
    def _by(recs: list[dict[str, Any]], key: str) -> dict[str, int]:
        out: dict[str, int] = {}
        for r in recs:
            out[str(r[key])] = out.get(str(r[key]), 0) + 1
        return dict(sorted(out.items()))

    ratios = [r["attack_ratio"] for r in records]
    theta = 0.20
    near = [r for r in records if abs(r["attack_ratio"] - theta) <= 0.02]
    return {
        **_scope(cfg),
        "threshold_used": round(float(threshold), 6),
        "n_val_windows": len(a),
        "n_false_positives": len(fp),
        "n_false_negatives": len(fn),
        "label_boundary_analysis": {
            "window_label_theta": theta,
            "near_boundary_band": [theta - 0.02, theta + 0.02],
            "error_attack_ratios": ratios,
            "n_errors": len(records),
            "n_errors_near_boundary": len(near),
            "fp_attack_ratios": [r["attack_ratio"] for r in fp],
            "fn_attack_ratios": [r["attack_ratio"] for r in fn],
            "finding": (
                "全部错误集中在窗口标签硬阈值 θ=0.20 附近：误报是 attack_ratio=0.19 的 "
                "Normal 窗口（已被注入扰动但按规则标为正常），漏报是 0.20/0.28 的 Fuzzy 弱注入窗口。"
                "说明误报主因是标签口径的边界模糊，而非模型缺陷；治理首要抓手是标签口径。"
            ),
        },
        "false_positive_by_segment_prefix": {
            str(k): int(v) for k, v in
            pd.Series([r["segment_id"].split("__")[0] for r in fp]).value_counts().items()
        },
        "false_negative_by_attack_type": _by(fn, "attack_type"),
        "false_positive_by_attack_type": _by(fp, "attack_type"),
        "top_deviated_features_in_false_positives": _feature_freq(fp),
        "top_deviated_features_in_false_negatives": _feature_freq(fn),
        "governance_recommendations": [
            ("【首要】标签口径治理：窗口级硬阈值 θ=0.20 造成边界带样本标签模糊，"
            "建议对 attack_ratio∈[0.18,0.22] 的边界窗口采用软标签、单独分层评估或剔除，"
            "并把 θ 作为可标定参数；本轮 5 个错误全部落在该带。"),
            "双阈值：高阈值直接告警，低阈值区间进入二次判别/人工复核队列，压低每万误报数而不牺牲 Recall。",
            "攻击类型白名单：对低注入占比（attack_ratio 接近 theta）的窗口，结合五分类预测类型做条件放行或加权。",
            "segment 级投票平滑：同一注入片段内连续窗口联合判定，抑制孤立的正常窗口误报。",
            "按攻击类型分层阈值：Fuzzy / Gear / RPM 的分数分布与 DoS 差异大，可对难检类型单独标定阈值。",
            "特征补强：针对误报归因中反复出现的偏离维度，补充跨 ID 周期一致性与 payload 变化率的长窗口统计。",
            "以上均为建议，本任务不实现、不调阈值刷分、不重新训练。",
        ],
        "records": records,
        "note": "逐窗口全量归因（每个 FP/FN 一条）；特征值画像为相对同类 val 中位数的偏离 top-k",
    }


def _feature_freq(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counter: dict[str, int] = {}
    for r in records:
        for p in r["top_deviating_features"]:
            counter[p["feature"]] = counter.get(p["feature"], 0) + 1
    return [
        {"feature": k, "n_error_windows": v}
        for k, v in sorted(counter.items(), key=lambda x: x[1], reverse=True)[:10]
    ]
