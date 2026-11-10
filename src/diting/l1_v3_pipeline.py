"""谛听 W3 · split-v3 重训 + 评估治理主流水线（独立入口，不改动 W2 任何代码）。

运行：``python -m diting.l1_v3_pipeline``

与 W2 ``l1_pipeline.py`` 的关系：复用 ``models/l1.py`` 的 fit_one/predict_proba、
``eval/l1_metrics.py`` 的阈值标定与指标、``eval/l1_ablation.py`` 的三组消融、
``eval/l1_importance.py`` 的特征重要性，仅替换数据加载（split-v3）与评估治理层。
所有产物一律新建文件并使用 ``l1_v3_`` 前缀 / 新目录，绝不覆盖 v2 产物。

纪律：train 只用于拟合、val 用于评估与阈值标定、test 全程封存且零载入。
"""

from __future__ import annotations

# HistGradientBoosting / OpenMP 在目标环境 8 GB 内存上多线程会触发内存暴涨被系统终止；
# 单线程实测训练仅需约 3 秒且指标完全确定。必须在 numpy/sklearn 导入前设置。
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from .common import (
    PROJECT_ROOT,
    git_revision,
    read_json,
    sha256_file,
    utc_now,
    write_json,
    write_manifest,
)
from .eval.l1_ablation import run_ablations
from .eval.l1_importance import compute_permutation_importance, render_importance_svg
from .eval.l1_metrics import binary_metrics_at_threshold, calibrate_threshold, multiclass_metrics
from .eval.l1_v3_governance import (
    analyze_gear_rpm_confusion,
    attribute_errors,
    build_l0_comparison,
    per_attack_type_binary_metrics,
    pr_curve_points,
    recompute_l0_on_v3,
)
from .features.build import FEATURE_COLUMNS
from .models.l1 import fit_one, positive_column, predict_proba, validate_ablation_groups
from .models.l1_v3 import load_l1_v3_config, load_l1_v3_dataset
from .split.make_split_v3 import build_split_v3

STAGE_NAME = "split-v3-rebuild"
REPORT_DIR = PROJECT_ROOT / "artifacts" / "reports" / "l1_v3"


def _out(cfg: dict[str, Any], key: str) -> Path:
    p = Path(cfg["outputs"][key])
    p = p if p.is_absolute() else PROJECT_ROOT / p
    p.mkdir(parents=True, exist_ok=True)
    return p


def _scope(cfg: dict[str, Any]) -> dict[str, str]:
    return {
        "evaluated_on": cfg["evaluated_on"],
        "data_scope": cfg["data_scope"],
        "data_scope_short": cfg["data_scope_short"],
        "dataset": cfg["dataset"],
        "test_set_status": cfg["test_set_status"],
        "split_version": cfg["split_version"],
        "level": "window",
    }


def _manifest_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        "project": {"name": "diting", "version": cfg["version"], "seed": int(cfg["seed"])},
        "_config_path": cfg["_config_path"],
        "_config_sha256": cfg["_config_sha256"],
        "paths": {},
    }


def run(cfg: dict[str, Any] | None = None, generate_split: bool = True) -> dict[str, Any]:
    """执行 split-v3 生成 + L1 重训 + 评估治理 + 落盘，返回汇总。"""
    t_start = time.perf_counter()
    cfg = cfg or load_l1_v3_config()
    model_dir = _out(cfg, "model_dir")
    eval_dir = _out(cfg, "eval_dir")
    ablation_dir = _out(cfg, "ablation_dir")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # 0) split-v3：若产物不存在则生成（生成逻辑自带 7 条断言）
    split_dir = PROJECT_ROOT / "data" / "processed" / "split" / "split-v3"
    if generate_split or not (split_dir / "split_manifest.json").is_file():
        build_split_v3()
    v3_man_path = PROJECT_ROOT / cfg["inputs"]["split_manifest"]
    v3_man = read_json(v3_man_path)
    v2_man_path = PROJECT_ROOT / cfg["inputs"]["split_v2_manifest"]
    v2_man = read_json(v2_man_path)

    # 1) 数据加载（test 零载入）
    group_check = validate_ablation_groups(cfg)
    dataset, aux = load_l1_v3_dataset(cfg)
    if "test" in set(dataset.split.tolist()):
        raise AssertionError("test 行进入 L1-v3 数据集，违反封存纪律")

    # 2) 训练（只用 train）
    binary_model, binary_cols = fit_one(cfg, dataset, task="binary")
    multi_model, multi_cols = fit_one(cfg, dataset, task="multiclass")

    val_mask = dataset.split == "val"
    y_true = dataset.y_binary[val_mask]
    y_multi_true = dataset.y_multiclass[val_mask]
    proba, classes = predict_proba(binary_model, dataset, binary_cols, split="val")
    y_score = proba[:, positive_column(classes)]
    X_val_multi = dataset.features.loc[val_mask, multi_cols].to_numpy(dtype=np.float64)
    y_multi_pred = multi_model.predict(X_val_multi)
    model_classes = [str(c) for c in multi_model.classes_]

    # 3) 二分类 / 阈值标定 / 五分类
    calib = calibrate_threshold(y_true, y_score, cfg)
    default_thr = float(cfg["threshold"]["default"])
    m_default = binary_metrics_at_threshold(y_true, y_score, default_thr, cfg)
    m_cal = binary_metrics_at_threshold(y_true, y_score, calib["selected_threshold"], cfg)
    m_multi = multiclass_metrics(y_multi_true, y_multi_pred, cfg, model_classes)

    # 4) 评估治理（本阶段核心）
    aux_val = aux[aux["split"] == "val"].reset_index(drop=True).copy()
    if len(aux_val) != len(y_true):
        raise AssertionError("归因元数据表与 val 指标样本数不一致")
    l0 = recompute_l0_on_v3(aux, cfg)
    comparison = build_l0_comparison(l0, m_default, m_cal, cfg)
    if comparison["l0_n_samples"] != comparison["l1_n_samples"]:
        raise AssertionError("L0 与 L1-v3 的 val 样本数不一致，同口径对照无效")
    pr = pr_curve_points(y_true, y_score, cfg)
    gear_rpm = analyze_gear_rpm_confusion(m_multi, cfg)
    per_type = per_attack_type_binary_metrics(
        aux_val, y_score, calib["selected_threshold"], default_thr, cfg
    )
    errors = attribute_errors(
        aux_val, y_score, y_true, calib["selected_threshold"], y_multi_pred, cfg
    )

    # 5) 消融（复用 W2 实现，阈值沿用上一轮 10% 崩塌判据）+ 特征重要性
    ablation = run_ablations(cfg, dataset, m_default, calib["selected_threshold"])
    X_val = dataset.features.loc[val_mask, binary_cols].to_numpy(dtype=np.float64)
    importance = compute_permutation_importance(binary_model, X_val, y_true, binary_cols, cfg)
    # 不调用既有 write_importance_table（它会写 v2 的 l1_feature_importance.* 造成覆盖）；
    # 一律写 l1_v3_ 前缀的新文件（表格在下方产物打包段统一写）。

    # 6) 产物打包
    bundle = {
        "binary_model": binary_model,
        "multiclass_model": multi_model,
        "binary_feature_columns": binary_cols,
        "multiclass_feature_columns": multi_cols,
        "binary_classes": classes.tolist(),
        "multiclass_classes": model_classes,
        "model_version": cfg["version"],
        "feature_version": cfg["_default_config_sha256"][:12],
        "split_version": cfg["split_version"],
        "random_seed": int(cfg["seed"]),
    }
    model_path = model_dir / "model.joblib"
    joblib.dump(bundle, model_path)

    version_bundle = {
        "model_version": cfg["version"],
        "feature_version": "l1-features-v1",
        "threshold_version": cfg["threshold"]["version"],
        "split_manifest_version": v3_man.get("version"),
        "code_version": git_revision(),
        "random_seed": int(cfg["seed"]),
        "l1_v3_config_sha256": cfg["_config_sha256"],
        "default_config_sha256": cfg["_default_config_sha256"],
        "model_joblib_sha256": sha256_file(model_path),
        "split_manifest_sha256": dataset.meta["split_manifest_sha256"],
    }

    write_json(model_dir / "features.json", {
        "version": "l1-features-v1",
        "source": cfg["inputs"]["features"],
        "source_sha256": dataset.meta["features_sha256"],
        "n_features": len(FEATURE_COLUMNS),
        "columns": list(FEATURE_COLUMNS),
        "feature_groups": dataset.meta["feature_groups"],
        "ablation_groups": group_check,
        "leakage_guard": "flag / label / attack_ratio / can_id==0x000 布尔等标签等价字段均未入特征",
        "label_source": dataset.meta["label_source"],
        **_scope(cfg),
    })
    write_json(model_dir / "threshold.json", {
        "version": calib["version"],
        "selected_threshold": calib["selected_threshold"],
        "default_threshold": calib["default_threshold"],
        "criterion": calib["criterion"],
        "calibrated_on": calib["calibrated_on"],
        "train_used_for": calib["train_used_for"],
        "grid_range": calib["grid_range"],
        "n_grid_points": calib["n_grid_points"],
        "selected_youden_j": calib["selected_youden_j"],
        "test_set_status": calib["test_set_status"],
        **_scope(cfg),
    })
    write_json(model_dir / "split_manifest_ref.json", {
        "split_manifest_path": str(v3_man_path.relative_to(PROJECT_ROOT)),
        "split_manifest_sha256": dataset.meta["split_manifest_sha256"],
        "split_manifest_version": v3_man.get("version"),
        "content_sha256": v3_man.get("content_sha256"),
        "split_windows": v3_man.get("split_windows"),
        "class_support_by_split": v3_man.get("class_support_by_split"),
        "test_set_status": v3_man.get("test_set", {}).get("status"),
        "note": "L1-v3 只读该清单用于版本绑定；test 集合原样沿用 split-v2，未读取任何 test 窗口",
    })

    metrics_all = {
        "version": cfg["version"],
        "task_binary": {
            "at_default_threshold": m_default,
            "at_calibrated_threshold": m_cal,
        },
        "task_multiclass": m_multi,
        "gear_rpm_confusion_analysis": gear_rpm,
        "l0_recomputed_on_split_v3": l0,
        "l1_v3_vs_l0_comparison": comparison,
        "per_attack_type_binary_metrics": per_type,
        "threshold_calibration": {
            "version": calib["version"],
            "selected_threshold": calib["selected_threshold"],
            "selected_youden_j": calib["selected_youden_j"],
            "n_grid_points": calib["n_grid_points"],
            "grid_range": calib["grid_range"],
            "calibrated_on": calib["calibrated_on"],
        },
        "pr_curve_summary": {
            "n_points": pr["n_points"],
            "average_precision": pr["average_precision"],
            "artifact": "artifacts/eval/l1_v3_pr_curve.json",
        },
        "error_attribution_summary": {
            "threshold_used": errors["threshold_used"],
            "n_false_positives": errors["n_false_positives"],
            "n_false_negatives": errors["n_false_negatives"],
            "false_negative_by_attack_type": errors["false_negative_by_attack_type"],
            "label_boundary_analysis": errors["label_boundary_analysis"],
            "artifact": "artifacts/eval/l1_v3_error_attribution.json",
        },
        "ablation": {
            "collapse_rule": ablation["collapse_rule"],
            "id_group_conclusion": ablation["id_group_conclusion"],
            "variants": ablation["variants"],
        },
        "class_coverage": dataset.meta["class_coverage"],
        "split_v3_manifest": {
            "version": v3_man.get("version"),
            "content_sha256": v3_man.get("content_sha256"),
            "split_windows": v3_man.get("split_windows"),
            "class_support_by_split": v3_man.get("class_support_by_split"),
            "assertions": {k: v.get("passed") for k, v in v3_man["assertions"].items()},
        },
        "split_v2_context": {
            "version": v2_man.get("version"),
            "split_windows": v2_man.get("split_windows"),
            "comparability_note": (
                "split-v2 与 split-v3 的 val 窗口集合不同，两轮数字不可直接比较；"
                "v2 回答『跨攻击类型泛化』，v3 回答『同分布检出与分类能力』"
            ),
        },
        **_scope(cfg),
    }
    write_json(model_dir / "metrics.json", metrics_all)
    write_json(REPORT_DIR / "l1v3_metrics.json", metrics_all)

    # 评估产物（全部 l1_v3_ 前缀，避免覆盖 v2 的 l1_* 文件）
    write_json(eval_dir / "l1_v3_binary_metrics.json", {
        "at_default_threshold": m_default, "at_calibrated_threshold": m_cal, **_scope(cfg),
    })
    write_json(eval_dir / "l1_v3_multiclass_metrics.json", m_multi)
    write_json(eval_dir / "l1_v3_threshold_curve.json", calib)
    write_json(eval_dir / "l1_v3_pr_curve.json", pr)
    write_json(eval_dir / "l1_v3_gear_rpm_analysis.json", gear_rpm)
    write_json(eval_dir / "l1_v3_l0_recomputed.json", l0)
    write_json(eval_dir / "l1_v3_vs_l0.json", comparison)
    write_json(eval_dir / "l1_v3_per_attack_type.json", per_type)
    write_json(eval_dir / "l1_v3_error_attribution.json", errors)
    write_json(eval_dir / "l1_v3_ablation_metrics.json", ablation)
    write_json(eval_dir / "l1_v3_feature_importance.json", importance)
    (eval_dir / "l1_v3_feature_importance.svg").write_text(
        render_importance_svg(importance, cfg), encoding="utf-8"
    )
    _write_v3_importance_table(importance, eval_dir, cfg)
    (eval_dir / "l1_v3_ablation_comparison.svg").write_text(
        _render_ablation_svg(ablation), encoding="utf-8"
    )

    # 五类版本号 manifest（模型 / 特征 / 阈值 / 划分 / 代码+种子）
    write_manifest(
        model_dir, "l1_v3_model", _manifest_cfg(cfg),
        [PROJECT_ROOT / cfg["inputs"]["features"], PROJECT_ROOT / cfg["inputs"]["labels"],
         v3_man_path, PROJECT_ROOT / cfg["inputs"]["split_v3_windows"]],
        extra={
            "versions": version_bundle,
            "evaluated_on": cfg["evaluated_on"],
            "data_scope": cfg["data_scope"],
            "split_version": cfg["split_version"],
            "test_set_status": cfg["test_set_status"],
            "test_rows_loaded": 0,
            "split_filter": dataset.meta["split_filter"],
            "class_coverage": dataset.meta["class_coverage"],
            "outputs": [
                "model.joblib", "features.json", "threshold.json", "metrics.json",
                "split_manifest_ref.json", "manifest.json",
            ],
        },
    )
    write_json(eval_dir / "l1_v3_eval_manifest.json", {
        "stage": "l1_v3_eval", **version_bundle,
        "created_at": utc_now(),
        "evaluated_on": cfg["evaluated_on"],
        "data_scope": cfg["data_scope"],
        "split_version": cfg["split_version"],
        "test_set_status": cfg["test_set_status"],
        "test_rows_loaded": 0,
        "outputs": [
            "l1_v3_binary_metrics.json", "l1_v3_multiclass_metrics.json",
            "l1_v3_threshold_curve.json", "l1_v3_pr_curve.json",
            "l1_v3_gear_rpm_analysis.json", "l1_v3_l0_recomputed.json",
            "l1_v3_vs_l0.json", "l1_v3_per_attack_type.json",
            "l1_v3_error_attribution.json", "l1_v3_ablation_metrics.json",
            "l1_v3_feature_importance.json", "l1_v3_feature_importance.svg",
            "l1_v3_ablation_comparison.svg", "l1_v3_eval_manifest.json",
        ],
    })
    # 写到新子目录，避免覆盖 W2 的 artifacts/ablation/manifest.json
    write_manifest(
        ablation_dir / "l1_v3", "l1_v3_ablation", _manifest_cfg(cfg),
        [PROJECT_ROOT / cfg["inputs"]["features"]],
        extra={
            "versions": version_bundle,
            "evaluated_on": cfg["evaluated_on"],
            "data_scope": cfg["data_scope"],
            "split_version": cfg["split_version"],
            "groups": group_check["groups"],
            "id_group_conclusion": ablation["id_group_conclusion"]["conclusion"],
            "outputs": ["../l1_v3_ablation_metrics.json", "../l1_v3_ablation_comparison.svg"],
        },
    )

    # 7) 报告与对照表（公开报告目录 + 评估产物副本）
    from .l1_v3_report import render_comparison_md, render_report

    report = render_report(
        cfg, dataset, v3_man, v2_man, m_default, m_cal, m_multi, calib,
        l0, comparison, gear_rpm, per_type, errors, ablation, importance, pr,
    )
    comparison_md = render_comparison_md(cfg, comparison, m_default, m_cal, l0)
    (REPORT_DIR / "W3评估治理报告.md").write_text(report, encoding="utf-8")
    (REPORT_DIR / "L1v3_vs_L0对照表.md").write_text(comparison_md, encoding="utf-8")
    (eval_dir / "l1_v3_report.md").write_text(report, encoding="utf-8")
    (eval_dir / "l1_v3_vs_l0.md").write_text(comparison_md, encoding="utf-8")

    elapsed = round(time.perf_counter() - t_start, 2)
    summary = {
        "elapsed_seconds": elapsed,
        **version_bundle,
        "split_v3_windows": v3_man["split_windows"],
        "class_support_train": v3_man["class_support_by_split"]["train"],
        "class_support_val": v3_man["class_support_by_split"]["val"],
        "binary_calibrated": {k: m_cal[k] for k in ("recall", "fpr", "precision", "f1", "threshold")},
        "binary_default": {k: m_default[k] for k in ("recall", "fpr", "precision", "f1", "threshold")},
        "l0_fpr": l0["primary_metrics"]["fpr"],
        "fpr_dropped_significantly": comparison["fpr_dropped_significantly"],
        "macro_f1": m_multi["macro_f1"],
        "gear_rpm_bidirectional": gear_rpm["total_bidirectional"],
        "gear_rpm_observable": gear_rpm["observable_on_val"],
        "id_group_collapsed": ablation["id_group_conclusion"]["collapsed"],
        "n_false_positives": errors["n_false_positives"],
        "n_false_negatives": errors["n_false_negatives"],
        "pr_curve_points": pr["n_points"],
        "test_rows_loaded": 0,
        "report_dir": str(REPORT_DIR),
    }
    write_json(eval_dir / "l1_v3_run_summary.json", summary)
    print(f"[diting][l1_v3] 完成，耗时 {elapsed}s；val Recall={m_cal['recall']:.4f} "
          f"FPR={m_cal['fpr']:.6f} (L0 FPR={l0['primary_metrics']['fpr']:.6f})；"
          f"macro-F1={m_multi['macro_f1']:.4f}；Gear↔RPM 互混={gear_rpm['total_bidirectional']}")
    return summary


def _write_v3_importance_table(importance: dict[str, Any], out_dir: Path, cfg: dict[str, Any]) -> None:
    """输出 l1_v3_ 前缀的特征重要性 CSV / Markdown（不覆盖 W2 的 l1_feature_importance.*）。"""
    import csv

    with (out_dir / "l1_v3_feature_importance.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["rank", "feature", "group", "importance_mean", "importance_std"])
        for r in importance["ranked"]:
            w.writerow([r["rank"], r["feature"], r["group"],
                        r["importance_mean"], r["importance_std"]])
    lines = [
        "# L1-v3 特征重要性（permutation importance, split-v3 val）",
        "",
        f"- 评估集合：val；评分：{importance['scoring']}；重复：{importance['n_repeats']}；种子：{importance['random_state']}",
        f"- 口径：{cfg['data_scope']}；test 封存。",
        "",
        "| 排名 | 特征 | 组 | importance 均值 | 标准差 |",
        "|---:|---|---|---:|---:|",
    ]
    for r in importance["ranked"]:
        lines.append(f"| {r['rank']} | `{r['feature']}` | {r['group']} | "
                     f"{r['importance_mean']:.6f} | {r['importance_std']:.6f} |")
    lines += ["", "## 按 G1–G6 聚合", "", "| 组 | 特征数 | 重要性之和 | 组内均值 |", "|---|---:|---:|---:|"]
    for g in importance["by_group"]:
        lines.append(f"| {g['group']} | {g['n_features']} | "
                     f"{g['sum_importance']:.6f} | {g['mean_importance']:.6f} |")
    lines.append("")
    (out_dir / "l1_v3_feature_importance.md").write_text("\n".join(lines), encoding="utf-8")


def _render_ablation_svg(ablation: dict[str, Any]) -> str:
    variants = ablation["variants"]
    labels = [v["label"] for v in variants]
    aps = [v["average_precision"] for v in variants]
    f1s = [v["metrics_at_0_5"]["f1"] for v in variants]
    width, height, left, top, chart_w, row_h = 820, 300, 300, 70, 420, 44
    mx = max(aps + f1s + [1e-9])
    colors = ["#4C78A8", "#E45756", "#F58518", "#54A24B"]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="24" y="28" font-family="-apple-system,Helvetica,Arial,sans-serif" font-size="17" font-weight="700" fill="#1f2933">W3 · 三组特征消融（split-v3 val） · AP 与 F1@0.5</text>',
        '<text x="24" y="48" font-family="Helvetica,Arial,sans-serif" font-size="12" fill="#52606d">实色=AP（阈值无关，崩塌主判据）；浅色=F1@0.5</text>',
    ]
    for i, (lab, ap, f1) in enumerate(zip(labels, aps, f1s)):
        y = top + i * row_h
        w1 = chart_w * ap / mx
        w2 = chart_w * f1 / mx
        parts.append(f'<text x="{left-12}" y="{y+15}" font-family="Helvetica,Arial,sans-serif" font-size="12" fill="#323f4b" text-anchor="end">{_esc(lab)}</text>')
        parts.append(f'<rect x="{left}" y="{y+2}" width="{w1:.1f}" height="13" rx="2" fill="{colors[i]}"/>')
        parts.append(f'<rect x="{left}" y="{y+18}" width="{w2:.1f}" height="10" rx="2" fill="{colors[i]}" opacity="0.45"/>')
        parts.append(f'<text x="{left+w1+6:.1f}" y="{y+13}" font-family="Helvetica,Arial,sans-serif" font-size="11" fill="#3e4c59">{ap:.4f}</text>')
    parts.append('</svg>')
    return "\n".join(parts)


def _esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


if __name__ == "__main__":
    out = run()
    print(json.dumps(out, ensure_ascii=False, indent=2))
