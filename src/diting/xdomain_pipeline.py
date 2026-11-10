"""谛听 W4 · OTIDS 跨域零样本验证主流水线（独立入口，不改动任何既有代码）。

运行：
    python -m diting.xdomain_pipeline run        # 数据接入 + 评估 + 全部产物
    python -m diting.xdomain_pipeline finalize --pytest-passed N --pytest-total M

纪律：
- 冻结模型/阈值/特征参数/ID 频次表全部来自 HCRL，OTIDS 侧零拟合、零训练；
- 所有产物为新文件/新目录（otids / xdomain 前缀），既有文件零修改、零删除、零重命名；
- HCRL test 折全程封存、零载入；不访问外部服务；不装重型依赖。
"""

from __future__ import annotations

# HistGradientBoosting（消融重训）在目标环境 8 GB 内存上多线程会被 SIGTERM(137)，
# 必须在 numpy / sklearn 导入前设为单线程。
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import time
from pathlib import Path
from typing import Any

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
from .data.otids_raw import (
    build_otids_features,
    build_otids_windows,
    load_xdomain_config,
    verify_input_zip,
    write_otids_frames_streaming,
    write_otids_manifest,
)
from .eval.xdomain_ablation import cross_domain_id_ablation, load_hcrl_val_norm_features
from .eval.xdomain_eval import (
    adaptation_comparison,
    binary_metrics,
    feature_group_drift,
    five_class_mapping,
    frozen_scores,
    hcrl_artifact_snapshot,
    load_frozen_assets,
    per_attack_type_metrics,
    verify_hcrl_untouched,
)
from .xdomain_report import render_report

REPORT_DIR = PROJECT_ROOT / "artifacts" / "reports" / "xdomain"
CTX_PATH = PROJECT_ROOT / "artifacts" / "eval" / "xdomain_report_ctx.json"


def _out(cfg: dict[str, Any], key: str) -> Path:
    p = Path(cfg["outputs"][key])
    p = p if p.is_absolute() else PROJECT_ROOT / p
    p.mkdir(parents=True, exist_ok=True)
    return p


def _w3_same_dist() -> dict[str, Any]:
    """只读加载 W3 同分布结果，供对照表使用（不重算、不改写）。"""
    p = PROJECT_ROOT / "artifacts" / "models" / "l1_v3" / "metrics.json"
    d = read_json(p)
    m = d["task_binary"]["at_calibrated_threshold"]
    return {
        "source": str(p),
        "source_sha256": sha256_file(p),
        "evaluated_on": "hcrl_split_v3_val",
        "data_scope": d["data_scope"],
        "threshold": m["threshold"],
        "recall": m["recall"],
        "fpr": m["fpr"],
        "precision": m["precision"],
        "f1": m["f1"],
        "average_precision": m["average_precision"],
        "roc_auc": m["roc_auc"],
        "false_alarms_per_10k_normal": m["false_alarms_per_10k_normal"],
        "macro_f1": d["task_multiclass"]["macro_f1"],
        "note": "同分布上限，已明确不可外推；仅作对照，非同一测试集 A/B",
    }


def _official_sensitivity(
    windows, y_score: np.ndarray, cfg: dict[str, Any]
) -> dict[str, Any]:
    """官方描述口径的敏感性对照：同一模型/特征/窗口，只换标签。"""
    y = windows["official_label"].to_numpy(dtype=int)
    rows = []
    for tag, thr in (("0_5", float(cfg["frozen"]["default_threshold"])),
                     ("0_54", float(cfg["frozen"]["frozen_threshold"]))):
        m = binary_metrics(
            y, y_score, thr, cfg,
            evaluated_on="otids_zeroshot_official_label_sensitivity",
            label_policy="official_description_sensitivity",
        )
        rows.append(
            {
                "tag": tag,
                "threshold": thr,
                "n_positive": m["n_positive"],
                "n_negative": m["n_negative"],
                "recall": m["recall"],
                "fpr": m["fpr"],
                "f1": m["f1"],
                "average_precision": m["average_precision"],
                "evaluated_on": m["evaluated_on"],
                "data_scope": m["data_scope"],
                "label_policy": m["label_policy"],
            }
        )
    primary = read_json(CTX_PATH)["binary_0_54"] if CTX_PATH.is_file() else None
    delta = None
    if primary is not None:
        delta = {
            "recall_delta_vs_primary": round(rows[1]["recall"] - primary["recall"], 8),
            "f1_delta_vs_primary": round(rows[1]["f1"] - primary["f1"], 8),
            "fpr_delta_vs_primary": round(rows[1]["fpr"] - primary["fpr"], 8),
        }
    return {
        "evaluated_on": "otids_zeroshot_official_label_sensitivity",
        "data_scope": cfg["data_scope"],
        "level": cfg["level"],
        "label_policy": "official_description_sensitivity",
        "role": "敏感性对照，不参与主结论",
        "description": cfg["label_policy"]["official_sensitivity"],
        "rows": rows,
        "delta_vs_primary_at_0_54": delta,
        "interpretation": (
            "标签口径从「文件级」换成「官方描述」（DoS 按 ID=0x000、"
            "Fuzzy/Impersonation 按 250 秒攻击段）后，正/负窗口集合发生变化，"
            "Recall / FPR / F1 随之改变——这说明跨域数字对标签口径敏感。"
            "两套口径的差异是 OTIDS 镜像缺乏逐帧真值带来的固有不确定性，"
            "不得择优选一套对外报数；主结论一律以第 2 节 primary 口径为准。"
        ),
    }


def run() -> dict[str, Any]:
    t0 = time.perf_counter()
    cfg = load_xdomain_config()
    processed_dir = _out(cfg, "processed_dir")
    eval_dir = _out(cfg, "eval_dir")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # 0) 冻结资产零触碰快照（运行前）
    before = hcrl_artifact_snapshot(cfg)
    frozen = load_frozen_assets(cfg)

    # 1) 输入 zip 校验 + 数据接入
    zip_check = verify_input_zip(cfg)
    parse_stats = write_otids_frames_streaming(cfg, processed_dir)
    windows, window_stats = build_otids_windows(cfg, processed_dir)
    features_norm, feature_meta = build_otids_features(cfg, processed_dir, windows)
    manifest = write_otids_manifest(
        cfg, processed_dir, zip_check, parse_stats, window_stats, feature_meta
    )

    # 2) 冻结模型零样本推理
    y_true = windows["label"].to_numpy(dtype=int)
    y_score, y_multi = frozen_scores(frozen["bundle"], features_norm)
    thr054 = float(cfg["frozen"]["frozen_threshold"])
    thr05 = float(cfg["frozen"]["default_threshold"])
    m05 = binary_metrics(y_true, y_score, thr05, cfg)
    m054 = binary_metrics(y_true, y_score, thr054, cfg)

    # 先写一份最小 ctx，供敏感性对照读取 primary 数字
    write_json(CTX_PATH, {"binary_0_54": m054})

    per_type = per_attack_type_metrics(
        windows, y_score, {"t0_5": thr05, "frozen": thr054}, cfg
    )
    five = five_class_mapping(windows, y_multi, cfg)
    official = _official_sensitivity(windows, y_score, cfg)
    adapt = adaptation_comparison(windows, y_score, thr054, cfg)

    # 3) 漂移归因（HCRL val 归一化特征只读下推）
    hcrl_val_norm = load_hcrl_val_norm_features(cfg)
    drift = feature_group_drift(hcrl_val_norm, features_norm, cfg)

    # 4) 跨域去 ID 消融（在 HCRL train 上重训，仅用于本评估）
    abl = cross_domain_id_ablation(cfg, features_norm, y_true, y_score, eval_dir)

    # 5) 冻结资产零触碰快照（运行后，含全部产物）
    after = hcrl_artifact_snapshot(cfg)
    integrity = verify_hcrl_untouched(before, after)
    if not integrity["verified"]:
        raise AssertionError(f"HCRL 既有产物发生变更：{integrity}")

    w3 = _w3_same_dist()
    ctx: dict[str, Any] = {
        "cfg": cfg,
        "finished_at": utc_now(),
        "git_revision": git_revision(),
        "pytest_passed": 0,
        "pytest_total": 0,
        "zip_check": zip_check,
        "parse_stats": parse_stats,
        "window_stats": window_stats,
        "feature_meta": feature_meta,
        "manifest": manifest,
        "frozen_record": frozen["record"],
        "binary_0_5": m05,
        "binary_0_54": m054,
        "per_attack_type": per_type,
        "five_class": five,
        "official_sensitivity": official,
        "adaptation": adapt,
        "id_ablation": abl,
        "drift": drift,
        "w3_same_dist": w3,
        "integrity": integrity,
    }
    write_json(CTX_PATH, ctx)

    # 6) 公开报告目录 / 仓库内评估产物打包
    metrics_all = {
        "version": cfg["version"],
        "task": cfg["task"],
        "created_at": utc_now(),
        "data_scope": cfg["data_scope"],
        "dataset": cfg["dataset"],
        "level": cfg["level"],
        "source_tag": cfg["source_tag"],
        "test_set_status": cfg["test_set_status"],
        "frozen_assets": frozen["record"],
        "input_zip": zip_check,
        "label_policy": cfg["label_policy"],
        "window_policy": manifest["window_policy"],
        "w3_same_dist_reference": w3,
        "metrics": {
            "otids_zeroshot_binary_at_0_5": m05,
            "otids_zeroshot_binary_at_frozen_0_54": m054,
            "otids_zeroshot_per_attack_type": per_type,
            "otids_zeroshot_five_class_mapping": five,
            "otids_zeroshot_official_label_sensitivity": official,
            "otids_adapted_vs_zeroshot": adapt,
            "otids_zeroshot_id_ablation": abl,
            "hcrl_val_vs_otids_feature_drift": drift,
        },
        "hcrl_untouched_verification": integrity,
    }
    write_json(REPORT_DIR / "xdomain_metrics.json", metrics_all)
    write_json(eval_dir / "xdomain_metrics.json", metrics_all)
    write_json(eval_dir / "xdomain_binary_metrics.json", {"at_0_5": m05, "at_frozen_0_54": m054})
    write_json(eval_dir / "xdomain_per_attack_type.json", per_type)
    write_json(eval_dir / "xdomain_five_class_mapping.json", five)
    write_json(eval_dir / "xdomain_official_label_sensitivity.json", official)
    write_json(eval_dir / "xdomain_adaptation.json", adapt)
    write_json(eval_dir / "xdomain_feature_drift.json", drift)
    write_json(eval_dir / "xdomain_frozen_assets.json", frozen["record"])
    write_json(eval_dir / "xdomain_hcrl_integrity.json", {
        "before_n_files": len(before),
        "after_n_files_in_scope": len(after),
        "verification": integrity,
        "before_snapshot_sha256_of_keys": None,
        "note": "逐文件 SHA-256 快照运行前后比对；完整快照见 evaluate 进程内比对结论",
    })
    write_json(eval_dir / "xdomain_window_stats.json", window_stats)
    write_json(eval_dir / "xdomain_parse_stats.json", parse_stats)

    # 五类版本号 manifest（模型 / 特征 / 阈值 / 划分 / 代码+种子）
    write_manifest(
        eval_dir / "xdomain_eval",
        "xdomain_eval",
        {
            "project": {"name": "diting", "version": cfg["version"], "seed": int(cfg["seed"])},
            "_config_path": cfg["_config_path"],
            "_config_sha256": cfg["_config_sha256"],
            "paths": {},
        },
        [
            PROJECT_ROOT / cfg["frozen"]["model_joblib"],
            PROJECT_ROOT / cfg["frozen"]["threshold_json"],
            PROJECT_ROOT / cfg["frozen"]["feature_params"],
            PROJECT_ROOT / cfg["frozen"]["id_frequency"],
            PROJECT_ROOT / cfg["input_zip"]["path"],
        ],
        extra={
            "versions": {
                "model_version": "l1-gbdt-v3 (frozen)",
                "feature_version": "l1-features-v1 (frozen)",
                "threshold_version": "l1-v3-thr-v1 (frozen)",
                "split_manifest_version": "split-v3 (HCRL, read-only)",
                "code_version": git_revision(),
                "random_seed": int(cfg["seed"]),
            },
            "created_at": utc_now(),
            "evaluated_on": "otids_zeroshot",
            "data_scope": cfg["data_scope"],
            "test_set_status": cfg["test_set_status"],
            "otids_rows_loaded": int(parse_stats["parquet_rows_total"]),
            "otids_windows": int(window_stats["n_windows"]),
            "hcrl_test_rows_loaded": 0,
            "outputs": [
                "../xdomain_metrics.json", "../xdomain_binary_metrics.json",
                "../xdomain_per_attack_type.json", "../xdomain_five_class_mapping.json",
                "../xdomain_official_label_sensitivity.json", "../xdomain_adaptation.json",
                "../xdomain_id_ablation.json", "../xdomain_feature_drift.json",
                "../xdomain_frozen_assets.json", "../xdomain_hcrl_integrity.json",
            ],
        },
    )

    report = render_report(ctx)
    (REPORT_DIR / "W4跨域验证报告.md").write_text(report, encoding="utf-8")
    (eval_dir / "xdomain_report.md").write_text(report, encoding="utf-8")

    elapsed = round(time.perf_counter() - t0, 2)
    summary = {
        "elapsed_seconds": elapsed,
        "otids_frames": int(parse_stats["parquet_rows_total"]),
        "otids_windows": int(window_stats["n_windows"]),
        "zeroshot_f1_at_0_54": m054["f1"],
        "zeroshot_recall_at_0_54": m054["recall"],
        "zeroshot_fpr_at_0_54": m054["fpr"],
        "zeroshot_ap": m054["average_precision"],
        "w3_same_dist_f1": w3["f1"],
        "f1_drop": round(w3["f1"] - m054["f1"], 6),
        "adapted_threshold": adapt["threshold_calibration"]["selected_threshold"],
        "id_ablation_collapsed": abl["collapsed"],
        "id_ablation_ap_delta": abl["ap_absolute_delta"],
        "dominant_drift_group": drift["dominant_drift_group"],
        "hcrl_untouched": integrity["verified"],
        "report_dir": str(REPORT_DIR),
        "result_md_written": False,
    }
    write_json(eval_dir / "xdomain_run_summary.json", summary)
    return summary


def finalize(pytest_passed: int, pytest_total: int) -> dict[str, Any]:
    """git 提交与 pytest 完成后，用最终版本号/测试数字重渲染报告。"""
    ctx = read_json(CTX_PATH)
    ctx["git_revision"] = git_revision()
    ctx["pytest_passed"] = int(pytest_passed)
    ctx["pytest_total"] = int(pytest_total)
    report = render_report(ctx)
    (REPORT_DIR / "W4跨域验证报告.md").write_text(report, encoding="utf-8")
    eval_dir = _out(ctx["cfg"], "eval_dir")
    (eval_dir / "xdomain_report.md").write_text(report, encoding="utf-8")
    write_json(CTX_PATH, ctx)
    return {
        "report": str(REPORT_DIR / "W4跨域验证报告.md"),
        "git_revision": ctx["git_revision"],
        "pytest": f"{pytest_passed}/{pytest_total}",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="谛听 W4 OTIDS 跨域验证流水线")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("run")
    fin = sub.add_parser("finalize")
    fin.add_argument("--pytest-passed", type=int, required=True)
    fin.add_argument("--pytest-total", type=int, required=True)
    args = ap.parse_args()
    if args.cmd == "run":
        out = run()
        print(json.dumps(out, ensure_ascii=False, indent=2))
    else:
        out = finalize(args.pytest_passed, args.pytest_total)
        print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
