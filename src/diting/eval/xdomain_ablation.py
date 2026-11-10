"""谛听 W4 · 跨域「去 ID 消融」重做（跨域去 ID 消融）。

W3 的三组消融变体只有指标 JSON，**没有保存模型 joblib**
（见 ``artifacts/ablation/l1_v3/`` 与 ``artifacts/eval/l1_v3_ablation_metrics.json``）。
按允许的替代路径：用同配置、同种子、同 HCRL split-v3 train 折，
重训一版「去 ID 频率组」二分类模型，**仅用于本次跨域评估**。

纪律：
- 重训只在 HCRL train 折上进行，**不使用任何 OTIDS 数据训练/微调权重**；
- 重训模型是新产物（新目录 xdomain_*），不覆盖、不修改冻结的 L1-v3；
- 冻结的全特征模型与重训的去 ID 模型都在同一 OTIDS 全量窗口上零样本推理，
  阈值一律沿用冻结 0.54（并同时报 @0.5），保证可比；
- 崩塌判据沿用 W3：AP 相对下降 ≥ 10% 判为崩塌；跨域场景额外量化
  「去掉 ID 组后性能是否改善」，作为 ID 空间漂移贡献的直接证据。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import average_precision_score

from ..common import (
    PROJECT_ROOT,
    git_revision,
    sha256_file,
    utc_now,
    write_json,
    write_manifest,
)
from ..features.build import FEATURE_COLUMNS
from ..models.l1 import fit_one, positive_column, select_columns
from ..models.l1_v3 import load_l1_v3_config, load_l1_v3_dataset
from .metrics import compute_metrics

DROP_GROUP = "id_frequency"
COLLAPSE_REL_PCT = 10.0


def _resolve(cfg: dict[str, Any], relative: str) -> Path:
    p = Path(relative)
    return p if p.is_absolute() else PROJECT_ROOT / p


def load_hcrl_val_norm_features(cfg: dict[str, Any]) -> pd.DataFrame:
    """只读下推加载 HCRL split-v3 val 的归一化 33 维特征（供漂移归因对照）。"""
    feat_path = _resolve(cfg, cfg["frozen"]["hcrl_features_norm"])
    win_path = _resolve(cfg, cfg["frozen"]["hcrl_split_v3_windows"])
    v3 = pq.read_table(win_path, columns=["window_id", "split"]).to_pandas()
    val_ids = v3.loc[v3["split"].astype(str) == "val", "window_id"].astype(str).tolist()
    if not val_ids:
        raise AssertionError("split-v3 val 窗口为空")
    # split-v3/windows.parquet 只含 train/val；test 折只以标识符形式存在于 split-v2，
    # 本函数按 window_id in-list 下推，test 窗口特征零载入
    cols = ["window_id", *FEATURE_COLUMNS]
    feat = pq.read_table(
        feat_path, columns=cols, filters=[("window_id", "in", val_ids)]
    ).to_pandas()
    if len(feat) != len(val_ids):
        raise AssertionError(f"HCRL val 特征下推行数 {len(feat)} != val 窗口数 {len(val_ids)}")
    return feat


def cross_domain_id_ablation(
    cfg: dict[str, Any],
    otids_norm: pd.DataFrame,
    y_true: np.ndarray,
    frozen_score: np.ndarray,
    out_dir: Path,
) -> dict[str, Any]:
    """重训去 ID 组模型并在 OTIDS 上零样本评估，返回明确结论。

    返回结构含：重训溯源、两个模型在 OTIDS 上的同口径指标、崩塌判据结论、
    「去 ID 是否改善跨域性能」的量化幅度。
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    v3cfg = load_l1_v3_config()
    dataset, _aux = load_l1_v3_dataset(v3cfg)

    # 与 W3 消融完全相同的列选择逻辑
    used_cols = select_columns(v3cfg, DROP_GROUP)
    dropped = [c for c in FEATURE_COLUMNS if c not in used_cols]
    if len(used_cols) != 26 or len(dropped) != 7:
        raise AssertionError(f"去 ID 组列数异常：used={len(used_cols)} dropped={len(dropped)}")

    model, _ = fit_one(v3cfg, dataset, task="binary", drop_group=DROP_GROUP)
    if int(model.n_features_in_) != len(used_cols):
        raise AssertionError("重训模型输入维度与去 ID 组列数不一致")

    model_path = out_dir / "xdomain_drop_id_frequency_model.joblib"
    joblib.dump(
        {
            "model": model,
            "feature_columns": used_cols,
            "dropped_group": DROP_GROUP,
            "dropped_columns": dropped,
            "trained_on": "HCRL split-v3 train（仅用于 W4 跨域消融评估）",
            "config_sha256": v3cfg["_config_sha256"],
            "seed": int(v3cfg["seed"]),
            "model_version": "xdomain-drop-id-v1",
        },
        model_path,
    )

    X = otids_norm[used_cols].to_numpy(dtype=np.float64)
    classes = np.asarray(model.classes_)
    y_score = model.predict_proba(X)[:, positive_column(classes)]

    frozen_ap = float(average_precision_score(y_true, frozen_score))
    ablated_ap = float(average_precision_score(y_true, y_score))

    def _metrics(score: np.ndarray, thr: float, tag: str) -> dict[str, Any]:
        pred = (score >= thr).astype(int)
        m = compute_metrics(np.asarray(y_true, dtype=int), pred)
        return {
            "tag": tag,
            "threshold": round(float(thr), 6),
            "recall": m["recall"],
            "fpr": m["fpr"],
            "precision": m["precision"],
            "f1": m["f1"],
            "false_alarms_per_10k_normal": m["false_alarms_per_10k_normal"],
            "average_precision": round(float(average_precision_score(y_true, score)), 8),
            "tp": m["tp"], "fp": m["fp"], "tn": m["tn"], "fn": m["fn"],
            "evaluated_on": "otids_zeroshot",
            "data_scope": cfg["data_scope"],
            "level": cfg["level"],
            "label_policy": "primary_file_level_theta_0_20",
            "model_weights_source": "HCRL train only（OTIDS 零参与训练）",
        }

    rows = []
    for thr in (float(cfg["frozen"]["default_threshold"]), float(cfg["frozen"]["frozen_threshold"])):
        rows.append(_metrics(frozen_score, thr, f"full_33_features@{thr}"))
        rows.append(_metrics(y_score, thr, f"drop_id_frequency@{thr}"))

    frozen_ap_r = round(frozen_ap, 8)
    ablated_ap_r = round(ablated_ap, 8)
    ap_drop = frozen_ap_r - ablated_ap_r
    rel_drop_pct = round(100.0 * ap_drop / frozen_ap_r, 4) if frozen_ap_r else None
    collapsed = bool(rel_drop_pct is not None and rel_drop_pct >= COLLAPSE_REL_PCT)
    improved = bool(ap_drop < 0)

    if collapsed:
        verdict = (
            f"崩塌：去掉 ID 频率组后跨域 AP 从 {frozen_ap_r:.6f} 降到 {ablated_ap_r:.6f}"
            f"（相对下降 {rel_drop_pct:.2f}%，≥10% 判据），模型在跨域场景下仍依赖 ID 类特征。"
        )
    elif improved:
        verdict = (
            f"未崩塌且性能反而改善：去掉 ID 频率组后跨域 AP 从 {frozen_ap_r:.6f} "
            f"升到 {ablated_ap_r:.6f}（相对变化 {-rel_drop_pct:.2f}%）。"
            "这是「ID 特征在跨车型 ID 空间上错配、模型背了 ID」的直接证据："
            "HCRL train 拟合的 ID 频次表应用到 Kia Soul 上成为噪声/误导特征。"
        )
    else:
        verdict = (
            f"未崩塌：去掉 ID 频率组后跨域 AP 从 {frozen_ap_r:.6f} 到 {ablated_ap_r:.6f}"
            f"（相对下降 {rel_drop_pct:.2f}%，未达 10% 崩塌线），幅度有限。"
        )

    result = {
        
            "evaluated_on": "otids_zeroshot",
            "data_scope": cfg["data_scope"],
            "level": cfg["level"],
            "label_policy": "primary_file_level_theta_0_20",
            "test_set_status": cfg["test_set_status"]
        ,
        "version": "xdomain-id-abl-v1",
        "trained_on": "HCRL split-v3 train（同配置同种子重训，仅用于本次跨域评估）",
        "model_source_note": (
            "W3 消融变体未保存模型 joblib（仅指标 JSON），按允许的路径重训；"
            "重训完全不使用 OTIDS 数据，冻结的 L1-v3 全特征模型未被修改"
        ),
        "hcrl_train_windows": int((dataset.split == "train").sum()),
        "hcrl_val_windows": int((dataset.split == "val").sum()),
        "config_sha256": v3cfg["_config_sha256"],
        "seed": int(v3cfg["seed"]),
        "dropped_group": DROP_GROUP,
        "dropped_columns": dropped,
        "n_features_full": len(FEATURE_COLUMNS),
        "n_features_ablated": len(used_cols),
        "model_artifact": str(model_path),
        "model_artifact_sha256": sha256_file(model_path),
        "collapse_rule": f"AP 相对下降 ≥ {COLLAPSE_REL_PCT:.1f}% 判为崩塌（沿用同一判据）",
        "ap_full_33": frozen_ap_r,
        "ap_drop_id": ablated_ap_r,
        "ap_absolute_delta": round(ablated_ap_r - frozen_ap_r, 8),
        "ap_relative_drop_pct": rel_drop_pct,
        "collapsed": collapsed,
        "performance_improved_after_dropping_id": improved,
        "conclusion": verdict,
        "rows": rows,
        "same_otids_windows": True,
        "same_thresholds": [float(cfg["frozen"]["default_threshold"]), float(cfg["frozen"]["frozen_threshold"])],
        "note": (
            "两模型在同一 OTIDS 全量窗口、同一阈值集合、同一标签口径下对比；"
            "阈值均沿用冻结值，未针对消融模型重新标定"
        ),
    }
    write_json(out_dir / "xdomain_id_ablation.json", result)

    # 五类版本号 manifest（同前规范）
    write_manifest(
        out_dir / "xdomain_id_ablation",
        "xdomain_id_ablation",
        {
            "project": {"name": "diting", "version": cfg["version"], "seed": int(cfg["seed"])},
            "_config_path": cfg["_config_path"],
            "_config_sha256": cfg["_config_sha256"],
            "paths": {},
        },
        [
            _resolve(cfg, cfg["frozen"]["hcrl_features_norm"]),
            _resolve(cfg, cfg["frozen"]["hcrl_split_v3_windows"]),
            _resolve(cfg, cfg["input_zip"]["path"]),
        ],
        extra={
            "created_at": utc_now(),
            "git_revision": git_revision(),
            "versions": {
                "model_version": "xdomain-drop-id-v1",
                "feature_version": "l1-features-v1",
                "threshold_version": "l1-v3-thr-v1 (frozen, reused)",
                "split_manifest_version": "split-v3 (HCRL train only)",
                "code_version": git_revision(),
                "random_seed": int(v3cfg["seed"]),
            },
            "trained_on": "HCRL split-v3 train",
            "otids_used_for_training": False,
            "outputs": ["../xdomain_drop_id_frequency_model.joblib", "../xdomain_id_ablation.json"],
        },
    )
    return result
