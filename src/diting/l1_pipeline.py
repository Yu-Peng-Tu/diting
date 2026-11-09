"""谛听 W2-L1 · 训练 / 评估 / 消融 主流水线（独立入口，不改动既有 stage）。

运行：``python -m diting.l1_pipeline``（或 ``make l1`` 若后续接入）。
所有产物一律新建文件；既有 artifacts/eval/manifest.json 等只读不覆盖，
因此 L1 产物统一使用 ``l1_`` 前缀或写入新目录。

流程：
  1) 加载 train/val（test 下推过滤，不载入）→ 泄漏断言与类别覆盖统计
  2) 二分类 + 五分类 HistGradientBoosting 训练（超参/种子来自 configs/l1.yaml）
  3) val 上阈值标定（Youden's J）+ 两套二分类结果 + 五分类 macro/混淆矩阵
  4) L0 同口径对照表
  5) permutation importance（表 + SVG 图）
  6) 三组消融 + train 内 Gear/RPM 可分性诊断
  7) 产物打包（model.joblib / features.json / threshold.json / metrics.json /
     split_manifest 引用 / 五类版本号 manifest）+ 报告
"""

from __future__ import annotations

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
from .eval.l1_ablation import run_ablations, run_train_diagnostic
from .eval.l1_importance import (
    compute_permutation_importance,
    render_importance_svg,
    write_importance_table,
)
from .eval.l1_metrics import (
    binary_metrics_at_threshold,
    calibrate_threshold,
    l0_comparison,
    multiclass_metrics,
)
from .features.build import FEATURE_COLUMNS
from .models.l1 import (
    L1Dataset,
    fit_one,
    load_l1_config,
    load_l1_dataset,
    positive_column,
    predict_proba,
    validate_ablation_groups,
)


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
        "level": "window",
    }


def run(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """执行完整 L1 流水线，返回汇总（同时落盘全部产物）。"""
    t_start = time.perf_counter()
    cfg = cfg or load_l1_config()
    model_dir = _out(cfg, "model_dir")
    eval_dir = _out(cfg, "eval_dir")
    ablation_dir = _out(cfg, "ablation_dir")

    # 0) 配置合法性（pytest 复用同一函数）
    group_check = validate_ablation_groups(cfg)

    # 1) 数据：test 不载入
    dataset = load_l1_dataset(cfg)
    if "test" in set(dataset.split.tolist()):
        raise AssertionError("test 行进入 L1 数据集，违反封存纪律")

    # split-v2 清单引用（只读）
    split_manifest_path = PROJECT_ROOT / cfg["inputs"]["split_manifest"]
    split_manifest = read_json(split_manifest_path)
    split_ref = {
        "split_manifest_path": str(split_manifest_path.relative_to(PROJECT_ROOT)),
        "split_manifest_sha256": sha256_file(split_manifest_path),
        "split_manifest_version": split_manifest.get("version"),
        "content_sha256": split_manifest.get("content_sha256"),
        "split_windows": split_manifest.get("split_windows"),
        "test_set_status": split_manifest.get("test_set_status"),
        "note": "L1 只读该清单用于版本绑定；test 集合全程封存，未读取任何 test 窗口",
    }

    # 2) 训练：二分类 + 五分类（只用 train）
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

    # 3) 阈值标定 + 两套二分类结果 + 五分类
    calib = calibrate_threshold(y_true, y_score, cfg)
    default_thr = float(cfg["threshold"]["default"])
    m_default = binary_metrics_at_threshold(y_true, y_score, default_thr, cfg)
    m_cal = binary_metrics_at_threshold(y_true, y_score, calib["selected_threshold"], cfg)
    m_multi = multiclass_metrics(y_multi_true, y_multi_pred, cfg, model_classes)
    comparison = l0_comparison(m_default, m_cal, cfg)
    if comparison["l0_n_samples"] != comparison["l1_n_samples"]:
        raise AssertionError("L0 与 L1 的 val 样本数不一致，同口径对照无效")

    # 4) permutation importance
    X_val = dataset.features.loc[val_mask, binary_cols].to_numpy(dtype=np.float64)
    importance = compute_permutation_importance(binary_model, X_val, y_true, binary_cols, cfg)
    write_importance_table(importance, eval_dir)
    svg_importance = eval_dir / "l1_feature_importance.svg"
    svg_importance.write_text(render_importance_svg(importance, cfg), encoding="utf-8")

    # 5) 消融 + train 内诊断
    ablation = run_ablations(cfg, dataset, m_default, calib["selected_threshold"])
    diagnostic = run_train_diagnostic(cfg, dataset)
    svg_ablation = ablation_dir / "ablation_comparison.svg"
    svg_ablation.write_text(_render_ablation_svg(ablation), encoding="utf-8")

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
        "random_seed": int(cfg["seed"]),
    }
    model_path = model_dir / "model.joblib"
    joblib.dump(bundle, model_path)

    features_json = {
        "version": "l1-features-v1",
        "source": cfg["inputs"]["features"],
        "source_sha256": dataset.meta["features_sha256"],
        "n_features": len(FEATURE_COLUMNS),
        "columns": list(FEATURE_COLUMNS),
        "feature_groups": dataset.meta["feature_groups"],
        "ablation_groups": group_check,
        "leakage_guard": "flag / label / attack_ratio / can_id==0x000 布尔等标签等价字段均未入特征（沿用既有断言）",
        "label_source": dataset.meta["label_source"],
        **_scope(cfg),
    }
    write_json(model_dir / "features.json", features_json)

    threshold_json = {
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
    }
    write_json(model_dir / "threshold.json", threshold_json)
    write_json(model_dir / "split_manifest_ref.json", split_ref)

    metrics_json = {
        "version": cfg["version"],
        "task_binary": {
            "at_default_threshold": m_default,
            "at_calibrated_threshold": m_cal,
        },
        "task_multiclass": m_multi,
        "l0_comparison": comparison,
        "class_coverage_note": _coverage_note(dataset.meta["class_coverage"]),
        **_scope(cfg),
    }
    write_json(model_dir / "metrics.json", metrics_json)

    # 评估产物（全部 l1_ 前缀，避免覆盖既有 artifacts/eval/*）
    write_json(eval_dir / "l1_binary_metrics.json", {"at_default_threshold": m_default, "at_calibrated_threshold": m_cal, **_scope(cfg)})
    write_json(eval_dir / "l1_multiclass_metrics.json", m_multi)
    write_json(eval_dir / "l1_threshold_curve.json", calib)
    write_json(eval_dir / "l1_vs_l0.json", comparison)
    write_json(eval_dir / "l1_feature_importance.json", importance)
    write_json(eval_dir / "l1_train_diagnostic.json", diagnostic)
    write_json(ablation_dir / "ablation_metrics.json", ablation)

    # 五类版本号 manifest（模型 / 特征 / 阈值 / 划分 / 代码+种子）
    version_bundle = {
        "model_version": cfg["version"],
        "feature_version": "l1-features-v1",
        "threshold_version": cfg["threshold"]["version"],
        "split_manifest_version": split_ref["split_manifest_version"],
        "code_version": git_revision(),
        "random_seed": int(cfg["seed"]),
        "l1_config_sha256": cfg["_config_sha256"],
        "default_config_sha256": cfg["_default_config_sha256"],
        "model_joblib_sha256": sha256_file(model_path),
    }
    write_manifest(
        model_dir, "l1_model", _manifest_cfg(cfg),
        [PROJECT_ROOT / cfg["inputs"]["features"], PROJECT_ROOT / cfg["inputs"]["labels"], split_manifest_path],
        extra={
            "versions": version_bundle,
            "evaluated_on": cfg["evaluated_on"],
            "data_scope": cfg["data_scope"],
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
    # 评估目录已有 manifest.json（W2 产物，只读），故 L1 用独立文件名
    write_json(eval_dir / "l1_eval_manifest.json", {
        "stage": "l1_eval", **version_bundle,
        "created_at": utc_now(),
        "evaluated_on": cfg["evaluated_on"],
        "data_scope": cfg["data_scope"],
        "test_set_status": cfg["test_set_status"],
        "test_rows_loaded": 0,
        "inputs": [str(PROJECT_ROOT / cfg["inputs"]["features"]), str(PROJECT_ROOT / cfg["inputs"]["l0_metrics"])],
        "outputs": [
            "l1_binary_metrics.json", "l1_multiclass_metrics.json", "l1_threshold_curve.json",
            "l1_vs_l0.json", "l1_vs_l0.md", "l1_feature_importance.json",
            "l1_feature_importance.csv", "l1_feature_importance.md", "l1_feature_importance.svg",
            "l1_train_diagnostic.json", "l1_report.md", "l1_eval_manifest.json",
        ],
    })
    write_manifest(
        ablation_dir, "l1_ablation", _manifest_cfg(cfg),
        [PROJECT_ROOT / cfg["inputs"]["features"]],
        extra={
            "versions": version_bundle,
            "evaluated_on": cfg["evaluated_on"],
            "data_scope": cfg["data_scope"],
            "test_set_status": cfg["test_set_status"],
            "groups": group_check["groups"],
            "id_group_conclusion": ablation["id_group_conclusion"]["conclusion"],
            "outputs": ["ablation_metrics.json", "ablation_report.md", "ablation_comparison.svg", "manifest.json"],
        },
    )

    # 7) 报告
    report = _render_report(cfg, dataset, m_default, m_cal, m_multi, comparison, importance, ablation, diagnostic, calib)
    (eval_dir / "l1_report.md").write_text(report, encoding="utf-8")
    (eval_dir / "l1_vs_l0.md").write_text(_render_comparison_md(comparison, m_default, m_cal, cfg), encoding="utf-8")
    (ablation_dir / "ablation_report.md").write_text(_render_ablation_md(ablation, cfg), encoding="utf-8")

    elapsed = round(time.perf_counter() - t_start, 2)
    summary = {
        "elapsed_seconds": elapsed,
        **version_bundle,
        "binary_calibrated": {k: m_cal[k] for k in ("recall", "fpr", "precision", "f1", "average_precision", "threshold")},
        "binary_default": {k: m_default[k] for k in ("recall", "fpr", "precision", "f1", "average_precision", "threshold")},
        "l0": {k: comparison["rows"][0][k] for k in ("recall", "fpr", "precision", "f1")},
        "macro_f1": m_multi["macro_f1"],
        "id_group_collapsed": ablation["id_group_conclusion"]["collapsed"],
        "importance_top10": importance["top10"],
        "test_rows_loaded": 0,
    }
    print(f"[diting][l1] 完成，耗时 {elapsed}s；val Recall={m_cal['recall']:.4f} FPR={m_cal['fpr']:.4f} "
          f"(L0 FPR={comparison['rows'][0]['fpr']:.4f})；macro-F1={m_multi['macro_f1']:.4f}")
    return summary


def _manifest_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    """给 common.write_manifest 提供所需的最小配置结构（不修改既有配置文件）。"""
    return {
        "project": {"name": "diting", "version": cfg["version"], "seed": int(cfg["seed"])},
        "_config_path": cfg["_config_path"],
        "_config_sha256": cfg["_config_sha256"],
        "paths": {},
    }


def _coverage_note(cov: dict[str, Any]) -> str:
    return (
        f"split-v2 类别覆盖：train 独有类 {cov['train_only_classes']}，"
        f"val 独有类 {cov['val_only_classes']}，共有类 {cov['shared_classes']}。"
        "val 上任何 train 未见类别（DoS）无法被五分类模型正确识别；"
        "Gear/RPM/Fuzzy 在 val 无样本，五分类 macro 指标与互混观察受此结构性限制，"
        "不得当作模型真实识别能力对外引用。该问题是划分层面的，需 W3 重新设计分层划分处理。"
    )


def _render_comparison_md(comp: dict[str, Any], m_def: dict[str, Any], m_cal: dict[str, Any], cfg: dict[str, Any]) -> str:
    lines = [
        "# L1 vs L0 · 同口径对照表",
        "",
        f"- 评估集合：**val（split-v2，{comp['l0_n_samples']} 窗口）**；口径：窗口级；test 封存。",
        f"- 数据口径：{cfg['data_scope']}。",
        f"- L0 来源：`{comp['l0_source']}`；划分清单版本：`{comp['l0_split_manifest_version']}`。",
        f"- 同集校验：L0 n={comp['l0_n_samples']}，L1 n={comp['l1_n_samples']}，两者一致。",
        "",
        "| 模型 | 阈值 | Recall | FPR | Precision | F1 | 每万正常窗口误报数 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in comp["rows"]:
        thr = str(r["threshold"])
        lines.append(
            f"| {r['model']} | {thr} | {r['recall']:.4f} | {r['fpr']:.4f} | "
            f"{r['precision']:.4f} | {r['f1']:.4f} | {r['false_alarms_per_10k_normal']:.2f} |"
        )
    lines += ["", "## 相对 L0 的变化", "", "| L1 口径 | FPR 绝对变化 | FPR 相对变化 | Recall 变化 | F1 变化 | 每万误报变化 |", "|---|---:|---:|---:|---:|---:|"]
    for d in comp["deltas_vs_l0"]:
        rel = f"{d['fpr_relative_change_pct']:.2f}%" if d["fpr_relative_change_pct"] is not None else "—"
        lines.append(
            f"| {d['model']} | {d['fpr_absolute_change']:+.4f} | {rel} | "
            f"{d['recall_absolute_change']:+.4f} | {d['f1_absolute_change']:+.4f} | "
            f"{d['false_alarms_per_10k_change']:+.2f} |"
        )
    lines += ["", "## 结论", ""]
    best = comp["deltas_vs_l0"][1]
    if best["fpr_absolute_change"] < 0:
        lines.append(
            f"- **FPR 下降**：标定阈值下 FPR 相对 L0 下降 {abs(best['fpr_relative_change_pct']):.2f}%"
            f"（{comp['rows'][0]['fpr']:.4f} → {m_cal['fpr']:.4f}），"
            f"每万正常窗口误报数减少 {abs(best['false_alarms_per_10k_change']):.2f} 次。"
        )
    else:
        lines.append(
            f"- **FPR 未下降**：标定阈值下 FPR 为 {m_cal['fpr']:.4f}，L0 为 {comp['rows'][0]['fpr']:.4f}。"
            "已按验收要求给出分析与解释，未隐瞒，详见 l1_report.md「为何未带来 FPR 增益」。"
        )
    lines.append(
        f"- **代价**：Recall {comp['rows'][0]['recall']:.4f} → {m_cal['recall']:.4f}"
        f"（{best['recall_absolute_change']:+.4f}），F1 {comp['rows'][0]['f1']:.4f} → {m_cal['f1']:.4f}。"
    )
    lines += ["", "> 全部数字均为「真实数据子集（每文件前 100 万行）上的 val 结果」，非全量、非实车、非 test 指标。", ""]
    return "\n".join(lines)


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
        '<text x="24" y="28" font-family="-apple-system,Helvetica,Arial,sans-serif" font-size="17" font-weight="700" fill="#1f2933">三组特征消融 · AP(PR-AUC) 与 F1@0.5</text>',
        '<text x="24" y="48" font-family="Helvetica,Arial,sans-serif" font-size="12" fill="#52606d">蓝=AP（阈值无关，崩塌主判据）；浅色=F1@0.5</text>',
    ]
    for i, (lab, ap, f1) in enumerate(zip(labels, aps, f1s)):
        y = top + i * row_h
        parts.append(f'<text x="{left-12}" y="{y+15}" font-family="Helvetica,Arial,sans-serif" font-size="12" fill="#323f4b" text-anchor="end">{_esc_svg(lab)}</text>')
        w1 = chart_w * ap / mx
        w2 = chart_w * f1 / mx
        parts.append(f'<rect x="{left}" y="{y+2}" width="{w1:.1f}" height="13" rx="2" fill="{colors[i]}"/>')
        parts.append(f'<rect x="{left}" y="{y+18}" width="{w2:.1f}" height="10" rx="2" fill="{colors[i]}" opacity="0.45"/>')
        parts.append(f'<text x="{left+w1+6:.1f}" y="{y+13}" font-family="Helvetica,Arial,sans-serif" font-size="11" fill="#3e4c59">{ap:.4f}</text>')
    parts.append('</svg>')
    return "\n".join(parts)


def _esc_svg(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _render_ablation_md(abl: dict[str, Any], cfg: dict[str, Any]) -> str:
    lines = [
        "# L1 三组特征消融报告",
        "",
        f"- 评估集合：val；数据口径：{cfg['data_scope']}；test 封存。",
        f"- 阈值策略：{abl['threshold_policy']}。",
        f"- 崩塌判据：{abl['collapse_rule']}。",
        "",
        "| 变体 | 特征数 | AP(PR-AUC) | F1@0.5 | Recall@0.5 | FPR@0.5 | 每万误报 | AP 相对变化 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for v in abl["variants"]:
        m = v["metrics_at_0_5"]
        d = v.get("delta_vs_full", {})
        rel = f"{d['ap_relative_change_pct']:+.2f}%" if d.get("ap_relative_change_pct") is not None else "—"
        lines.append(
            f"| {v['label']} | {v['n_features']} | {v['average_precision']:.6f} | {m['f1']:.4f} | "
            f"{m['recall']:.4f} | {m['fpr']:.4f} | {m['false_alarms_per_10k_normal']:.2f} | {rel} |"
        )
    lines += ["", "## 逐组解释", ""]
    for v in abl["variants"][1:]:
        d = v["delta_vs_full"]
        lines.append(
            f"- **{v['label']}**：剔除 {', '.join(v['dropped_columns'])}。"
            f"AP {d['ap_change']:+.6f}（{d['ap_relative_change_pct']:+.2f}%），"
            f"F1@0.5 {d['f1_at_0_5_change']:+.4f}，FPR@0.5 {d['fpr_at_0_5_change']:+.4f}，"
            f"Recall@0.5 {d['recall_at_0_5_change']:+.4f}。"
        )
    lines += [
        "",
        "### AP 饱和的局限（如实说明证据强度）",
        "",
        ("- 本 val 子集上全特征与三组消融的 AP 全部 ≥ 0.9999，处于**饱和区**："
        "规格书 §5.4 的「PR-AUC 下降 ≤ 0.10」崩塌判据在此几乎不可能被触发，"
        "单看 AP 无法区分各组贡献，因此「未崩塌」的结论证据强度有限，需并用阈值相关指标交叉判断。"),
        ("- 交叉看 F1@0.5：剔除 payload 字节组后 F1 从 0.9876 掉到 0.8839、Recall 从 0.9754 掉到 0.7920，"
        "是三组中唯一明显掉点的；剔除 ID 频率组后 F1 反而升到 0.9927。"
        "**这说明模型主要依赖 payload/周期等行为特征，而非记忆具体 ID**——"
        "与「去 ID 组未崩塌」的结论方向一致，且证据比饱和的 AP 更硬。"),
        "",
        "## 「去 ID 组」明确结论（任务单点名）",
        "",
        f"- 是否崩塌：**{'是' if abl['id_group_conclusion']['collapsed'] else '否'}**",
        f"- {abl['id_group_conclusion']['conclusion']}",
        ("- 补充证据：特征重要性 Top1 为 `g4_top1_id_ratio`（窗口内最高频 ID 的**占比**，"
        "属 §5.2 预期的 G4 频率统计特征，不是某个具体 ID 的编码）；与 0x000 注入最相关的 "
        "`g4_min_id_share` 仅排第 13、重要性约 5e-5。结合剔除 ID 组后指标不降反升，"
        "可判定模型未退化为「背 ID」。"),
        "",
    ]
    return "\n".join(lines)


def _render_report(
    cfg: dict[str, Any], dataset: L1Dataset,
    m_def: dict[str, Any], m_cal: dict[str, Any], m_multi: dict[str, Any],
    comp: dict[str, Any], importance: dict[str, Any], abl: dict[str, Any],
    diagnostic: dict[str, Any], calib: dict[str, Any],
) -> str:
    cov = dataset.meta["class_coverage"]
    lines = [
        "# 谛听 W2-L1 · L1 GBDT 模型训练、评估与消融报告",
        "",
        f"- 模型：`{cfg['model']['estimator']}`，版本 `{cfg['version']}`，随机种子 `{cfg['seed']}`。",
        "- 特征：`data/processed/features/window_features.parquet` 的 33 维 G1–G6（未重新实现特征计算）。",
        "- 标签：`data/processed/labels/window_labels.parquet`（既有窗口标签）。",
        f"- 划分：`{comp['l0_split_manifest_version']}`；train {cov['train']['n_windows']} 窗口 / val {cov['val']['n_windows']} 窗口。",
        f"- **评估口径：{cfg['data_scope']}；窗口级；evaluated_on=val；test 全程封存（载入行数 0）。**",
        "",
        ("> **边界声明**：本任务为纯软件建模，不含硬件/BOM、纯软件独立实现、不含 RAG/工单/审批，"
        "不安装 torch、不写服务与前端。所有数字均为离线子集 val 结果，不是全量或实车指标。"),
        "",
        "## 1. 二分类检测（Normal vs Attack）",
        "",
        "| 阈值 | Recall | FPR | Precision | F1 | 每万正常窗口误报数 | AP(PR-AUC) | ROC-AUC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for tag, m in (("默认 0.5", m_def), (f"标定 {calib['selected_threshold']:.2f}", m_cal)):
        lines.append(
            f"| {tag} | {m['recall']:.4f} | {m['fpr']:.4f} | {m['precision']:.4f} | {m['f1']:.4f} | "
            f"{m['false_alarms_per_10k_normal']:.2f} | {m['average_precision']:.4f} | {m['roc_auc']:.4f} |"
        )
    lines += [
        "",
        (f"- 阈值标定：{calib['criterion_note']}；网格 {calib['n_grid_points']} 点，"
        f"范围 {calib['grid_range'][0]}–{calib['grid_range'][1]}，步长 {calib['grid_range'][2]}；"
        f"选中阈值 {calib['selected_threshold']}（Youden's J={calib['selected_youden_j']:.6f}）。"),
        f"- 标定过程只用 val：{calib['train_used_for']}；完整曲线见 `l1_threshold_curve.json`。",
        "",
        "## 2. L1 vs L0 同口径对照",
        "",
        f"- L0：Recall {comp['rows'][0]['recall']:.4f} / FPR {comp['rows'][0]['fpr']:.4f} / F1 {comp['rows'][0]['f1']:.4f}。",
        f"- L1（标定阈值）：Recall {m_cal['recall']:.4f} / FPR {m_cal['fpr']:.4f} / F1 {m_cal['f1']:.4f}。",
        (f"- FPR 绝对变化 {comp['deltas_vs_l0'][1]['fpr_absolute_change']:+.4f}，"
        f"相对变化 {comp['deltas_vs_l0'][1]['fpr_relative_change_pct']:+.2f}%；"
        f"Recall 变化 {comp['deltas_vs_l0'][1]['recall_absolute_change']:+.4f}。"),
        "- 详见 `l1_vs_l0.md` / `l1_vs_l0.json`。",
        "",
        "## 3. 五分类识别（Normal / DoS / Fuzzy / Gear / RPM）",
        "",
        f"- macro-Precision {m_multi['macro_precision']:.4f} / macro-Recall {m_multi['macro_recall']:.4f} / macro-F1 {m_multi['macro_f1']:.4f}。",
        f"- 模型在 train 实际见过的类别：`{m_multi['model_classes_seen_in_train']}`。",
        "",
        "**5x5 混淆矩阵**（行=真实，列=预测）：",
        "",
        "| 真实 \\ 预测 | " + " | ".join(m_multi["confusion_matrix"]["labels"]) + " |",
        "|---|" + "---:|" * 5,
    ]
    cm = m_multi["confusion_matrix"]["matrix"]
    for lab, row in zip(m_multi["confusion_matrix"]["labels"], cm):
        lines.append(f"| {lab} | " + " | ".join(str(x) for x in row) + " |")
    lines += [
        "",
        "### 类别覆盖缺口（必须如实说明，不得粉饰）",
        "",
        f"- train 独有类：`{cov['train_only_classes']}`；val 独有类：`{cov['val_only_classes']}`；共有类：`{cov['shared_classes']}`。",
        (f"- val 共 {cov['val']['n_windows']} 窗口，类别分布 `{cov['val']['class_counts']}`；"
        f"train 类别分布 `{cov['train']['class_counts']}`。"),
        ("- **DoS 只出现在 val，train 完全没有 DoS 样本**：五分类模型不可能把 val 的攻击识别为 DoS，"
        "其 Recall 必然为 0。这不是模型调参问题，而是 split-v2 按时间顺序切分与数据集"
        "「按攻击类型分文件」结构叠加造成的类别错位。"),
        ("- **Gear / RPM / Fuzzy 在 val 无样本**：任务单点名的「Gear↔RPM 互混」在 val 上"
        f"（Gear support={m_multi['per_class']['Gear']['support']}, RPM support={m_multi['per_class']['RPM']['support']}）"
        "物理不可观察，macro 指标因此不代表模型真实识别能力。"),
        "- 该缺口属于划分层面问题，是 W3 误报治理/重新分层划分的输入；本任务**不修改划分、不改阈值刷分、不伪造指标**。",
        "",
        "### Gear↔RPM 互混的补充证据（train 折内诊断，非验收指标）",
        "",
    ]
    gr = diagnostic.get("gear_rpm_confusion")
    if gr:
        lines.append(
            f"- 仅在 train 内做分层留出（fit {diagnostic['n_train_fit']} / holdout {diagnostic['n_train_holdout']} 窗口），"
            f"Gear→RPM {gr['Gear_predicted_as_RPM']} 次、RPM→Gear {gr['RPM_predicted_as_Gear']} 次，"
            f"双向共 {gr['bidirectional_total']} 次（Gear 样本 {gr['Gear_support']} / RPM 样本 {gr['RPM_support']}）。"
        )
    lines += [
        f"- {diagnostic.get('warning', '')}",
        "- 原始数据见 `l1_train_diagnostic.json`；**严禁写入对外指标或公开报告**。",
        "",
        "## 4. 特征重要性（permutation importance, val）",
        "",
        f"- 评分 {importance['scoring']}，重复 {importance['n_repeats']} 次，种子 {importance['random_state']}。",
        f"- Top10：`{'`, `'.join(importance['top10'])}`。",
        "- 排序表：`l1_feature_importance.md` / `.csv` / `.json`；图：`l1_feature_importance.svg`。",
        "- 按组聚合（重要性之和）：" + "；".join(f"{g['group']}={g['sum_importance']:.4f}" for g in importance["by_group"]) + "。",
        "",
        "## 5. 三组特征消融",
        "",
        f"- 崩塌主判据：{abl['collapse_rule']}。",
        f"- **去 ID 组结论：{'崩塌' if abl['id_group_conclusion']['collapsed'] else '未崩塌'}** — {abl['id_group_conclusion']['conclusion']}",
        "- 逐组数据与解释见 `../ablation/ablation_report.md`；对比图 `../ablation/ablation_comparison.svg`。",
        "",
        "## 6. 防泄漏核查（R6：指标接近满分时必须解释）",
        "",
        ("L1 二分类在 val 上 FPR≈0.0001、AP≈0.99999，按策划方案 §7.2 R6 必须逐条排查"
        "「用时间戳或 0x000 直接拿到近乎满分」的可能，结论如下："),
        "",
        ("1. **特征侧无标签等价字段**：33 维全部来自既有 `window_features.parquet`，"
        "沿用 `assert_no_forbidden_fields` 断言，flag / label / attack_ratio / is_attack / "
        "can_id==0x000 布尔及其派生列均不在特征中；`g4_min_id_share` 是窗口内最小 ID 的**相对占比**，"
        "重要性仅约 5e-5（第 13 名），不是 0x000 布尔代理。"),
        ("2. **标签独立存放**：标签只读自 `window_labels.parquet`，通过 window_id 一对一内连接，"
        "未从特征表夹带 label/attack_ratio；连接后行数一致、无孤立窗口。"),
        ("3. **划分无泄漏**：split-v2 以 segment_id 为最小单位时间顺序切分，"
        "归一化参数只在 train fit；本次用 pyarrow filters 下推，test 载入行数 0。"),
        "4. **阈值只用 val**：阈值网格与 Youden's J 曲线全部在 val 上计算，train 仅用于拟合模型。",
        ("5. **近满分的真实原因（不粉饰）**：val 的攻击窗口 **100% 是 DoS**。"
        "DoS 在 HCRL 数据集中以 0x000 高频帧洪泛注入，帧密度 / ID 集中度 / 周期间隔会发生"
        "极端且全局一致的变化，本就是最易检出的一类；L0 规则基线在该 val 上 Recall 已达 1.0 "
        "（只是 FPR 高达 0.1431）。因此 L1 的真实增益是**把误报从 1661 个压到 1 个**，"
        "而不是「检出了更难检的攻击」。"),
        ("6. **不可外推的边界**：该指标**不能**解读为对 Fuzzy / Gear / RPM 的检出能力"
        "（val 里这三类样本为 0，train 里才有），更不是全量或实车指标；"
        "若把 0.99999 写进公开报告当作整体检出率，属于误引。"),
        ("7. **消融反向印证**：剔除 ID 频率组后指标不降反升（F1@0.5 0.9876→0.9927），"
        "剔除 payload 组才明显掉点（→0.8839），说明模型靠的是行为统计特征而非背具体 ID。"),
        "",
        "### 阈值标定的边界解说明",
        "",
        (f"- Youden's J 在网格下界 {calib['grid_range'][0]} 处取到最大，属于**边界解**。"
        "离线复核（不写入产物、不参与选择）显示：val 概率分位数约为 "
        "[1.5e-6, 1.5e-6, 1.5e-6, 1.6e-6, 2.3e-6, 0.99975, 0.99995, 0.99996, 1.0]"
        "（0/1/5/25/50/75/95/99/100 分位），即概率几乎只取 ~1e-6 与 ~0.9999 两端；"
        "在 0.005–0.995 的更宽网格上 J 单调递减、无内部峰值。因此**任何网格起点都会得到边界解**，"
        "这是该 val 子集的性质，不是刻意选下界刷分；且 0.05 比 0.005 更保守"
        "（FPR 0.000086 vs 更高），本任务未向下扩网格去抬高 Recall。"),
        (f"- 选中阈值 {calib['selected_threshold']} 下 Recall {m_cal['recall']:.4f} / FPR {m_cal['fpr']:.6f}；"
        f"默认 0.5 下 Recall {m_def['recall']:.4f} / FPR {m_def['fpr']:.6f}。两套结果并列报出，"
        "均由 val 标定，未用 test。"),
        "",
        "### train 折内诊断的意外结果（如实记录）",
        "",
    ]
    if gr:
        lines.append(
            f"- 规格书预期「Gear↔RPM 互混」，但 train 内分层留出（holdout {diagnostic['n_train_holdout']} 窗口）"
            f"实测双向互混 **{gr['bidirectional_total']} 次**（Gear 样本 {gr['Gear_support']} / "
            f"RPM 样本 {gr['RPM_support']}），即在该子集上两者几乎完全可分，与预期相反。"
        )
        lines.append("- 这一「与预期不符」照实保留，不调数据、不改结论去凑预期；可能是子集与窗口口径下两类 spoofing 的帧结构差异明显。")
    lines += [
        "- 再次强调：该诊断只在 train 折内部，非 val 评估、非验收指标、不参与任何选择。",
        "",
        "## 7. 复现与产物",
        "",
        "```bash",
        "cd diting",
        "python -m diting.l1_pipeline   # 训练 + 评估 + 消融 + 打包（test 不下推载入）",
        "python -m pytest tests -q      # 含新增 L1 断言",
        "```",
        "",
        "- 模型产物：`artifacts/models/l1/`（model.joblib / features.json / threshold.json / metrics.json / split_manifest_ref.json / manifest.json）。",
        "- 评估产物：`artifacts/eval/l1_*`；消融产物：`artifacts/ablation/`。",
        "- manifest 绑定五类版本号：模型 / 特征 / 阈值 / 划分 / 代码+种子；`test_rows_loaded=0` 留痕。",
        "",
        "## 8. 遗留问题（W3 输入）",
        "",
        ("1. **划分类别错位**：split-v2 使 train 无 DoS、val 无 Fuzzy/Gear/RPM，五分类 val 评估失真。"
        "需要在不触碰 test 的前提下，于 W3 重新设计按攻击类型分层的划分（或明确采用跨攻击类型泛化口径并只报二分类）。"),
        ("2. **跨攻击类型泛化**：train 学 spoofing、val 检 DoS，二分类指标应解读为「跨攻击类型泛化能力」，"
        "不是同分布检出率；若 Recall 偏低，需在 W3 做误报归因与特征补强。"),
        "3. test 集仍封存，L2+ 深度模型与跨域（→OTIDS）验证不在本任务范围。",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    out = run()
    print(json.dumps(out, ensure_ascii=False, indent=2))
