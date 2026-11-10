"""谛听 · OTIDS 跨域零样本评估。

口径纪律：
- 冻结模型 + 冻结阈值 0.54 在 HCRL 上训练完毕后**完全冻结**，OTIDS 侧只做推理；
  归一化参数 / ID 频次表同样冻结复用，OTIDS 侧零拟合。
- 零样本（otids_zeroshot）与「仅调阈值」适配上限（otids_adapted）**并列报出、互不混淆**；
  适配只在 OTIDS 适配折上重标定阈值，模型权重不动，评估在另一折。
- 官方描述敏感性口径只作对照，不参与主结论；性能下降照实报，不做任何口径美化。
- 每一条指标都带 evaluated_on / data_scope / level / label_policy 口径标注。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from ..common import PROJECT_ROOT, read_json, sha256_file
from ..features.build import FEATURE_COLUMNS
from .metrics import compute_metrics

# 冻结阈值与全特征模型为 W3 产物，版本号绑定用
FROZEN_MODEL_VERSION = "l1-gbdt-v3"
FROZEN_THRESHOLD_VERSION = "l1-v3-thr-v1"


def _resolve(cfg: dict[str, Any], relative: str) -> Path:
    p = Path(relative)
    return p if p.is_absolute() else PROJECT_ROOT / p


def scope(cfg: dict[str, Any], evaluated_on: str, label_policy: str, threshold: float) -> dict[str, Any]:
    """统一的口径标注块，写进每一条指标。"""
    return {
        "evaluated_on": evaluated_on,
        "data_scope": cfg["data_scope"],
        "dataset": cfg["dataset"],
        "level": cfg["level"],
        "source_tag": cfg["source_tag"],
        "label_policy": label_policy,
        "threshold": round(float(threshold), 6),
        "test_set_status": cfg["test_set_status"],
        "model_version": FROZEN_MODEL_VERSION,
        "threshold_version": (
            FROZEN_THRESHOLD_VERSION if evaluated_on != "otids_adapted" else "otids-adapted-thr-v1"
        ),
    }


def load_frozen_assets(cfg: dict[str, Any]) -> dict[str, Any]:
    """加载冻结模型与阈值，逐项校验 SHA-256 / 版本 / 33 维 / 阈值=0.54。"""
    f = cfg["frozen"]
    paths = {
        "model_joblib": _resolve(cfg, f["model_joblib"]),
        "threshold_json": _resolve(cfg, f["threshold_json"]),
        "features_json": _resolve(cfg, f["features_json"]),
        "manifest_json": _resolve(cfg, f["manifest_json"]),
        "feature_params": _resolve(cfg, f["feature_params"]),
        "id_frequency": _resolve(cfg, f["id_frequency"]),
    }
    sha = {k: (sha256_file(p) if p.is_file() else None) for k, p in paths.items()}
    for k, p in paths.items():
        if not p.is_file():
            raise FileNotFoundError(f"冻结资产缺失：{k} -> {p}")

    manifest = read_json(paths["manifest_json"])
    declared = manifest["versions"]["model_joblib_sha256"]
    if sha["model_joblib"] != declared:
        raise AssertionError(
            f"模型 joblib SHA-256 与 manifest 不一致：{sha['model_joblib']} != {declared}"
        )
    thr = read_json(paths["threshold_json"])
    features = read_json(paths["features_json"])
    if float(thr["selected_threshold"]) != float(f["frozen_threshold"]):
        raise AssertionError("冻结阈值与配置不符")
    if thr["version"] != FROZEN_THRESHOLD_VERSION:
        raise AssertionError("冻结阈值版本不符")
    if list(features["columns"]) != list(FEATURE_COLUMNS) or int(features["n_features"]) != 33:
        raise AssertionError("冻结特征清单与 33 维 G1–G6 不一致")
    if features["source_sha256"] != sha256_file(_resolve(cfg, f["hcrl_features_norm"])):
        raise AssertionError("HCRL 归一化特征表 SHA-256 与冻结 features.json 不一致")

    bundle = joblib.load(paths["model_joblib"])
    if bundle["model_version"] != FROZEN_MODEL_VERSION:
        raise AssertionError("加载的模型版本不是 l1-gbdt-v3")
    if list(bundle["binary_feature_columns"]) != list(FEATURE_COLUMNS):
        raise AssertionError("模型特征列顺序与冻结 33 维不一致")
    if int(bundle["binary_model"].n_features_in_) != 33:
        raise AssertionError("模型输入维度不是 33")

    record = {
        "paths": {k: str(v) for k, v in paths.items()},
        "sha256": sha,
        "model_joblib_sha256_matches_manifest": True,
        "declared_model_joblib_sha256": declared,
        "frozen_threshold": float(thr["selected_threshold"]),
        "default_threshold": float(f["default_threshold"]),
        "threshold_version": thr["version"],
        "calibrated_on": thr["calibrated_on"],
        "model_version": bundle["model_version"],
        "feature_version": features["version"],
        "split_version": bundle["split_version"],
        "random_seed": int(bundle["random_seed"]),
        "multiclass_classes": list(bundle["multiclass_classes"]),
    }
    return {"bundle": bundle, "threshold": float(thr["selected_threshold"]), "record": record}


def frozen_scores(bundle: dict[str, Any], features_norm: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """冻结模型在 OTIDS 全量窗口上推理：二分类正类分数 + 五分类预测类别。"""
    cols = list(bundle["binary_feature_columns"])
    X = features_norm[cols].to_numpy(dtype=np.float64)
    if not np.isfinite(X).all():
        raise AssertionError("OTIDS 特征矩阵含非有限值")
    classes = np.asarray(bundle["binary_classes"])
    pos = int(np.where(classes == 1)[0][0])
    y_score = bundle["binary_model"].predict_proba(X)[:, pos]
    y_multi = bundle["multiclass_model"].predict(X)
    return np.asarray(y_score, dtype=np.float64), np.asarray(y_multi)


def binary_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
    cfg: dict[str, Any],
    evaluated_on: str = "otids_zeroshot",
    label_policy: str = "primary_file_level_theta_0_20",
) -> dict[str, Any]:
    """二分类全套指标（复用既有 compute_metrics 保证口径一致）+ AP / ROC-AUC。"""
    y_pred = (np.asarray(y_score) >= float(threshold)).astype(int)
    m = compute_metrics(np.asarray(y_true, dtype=int), y_pred)
    m["average_precision"] = round(float(average_precision_score(y_true, y_score)), 8)
    m["roc_auc"] = (
        round(float(roc_auc_score(y_true, y_score)), 8) if len(np.unique(y_true)) > 1 else None
    )
    m.update(scope(cfg, evaluated_on, label_policy, threshold))
    m["confusion_matrix_note"] = "行=真实，列=预测；labels=[Normal, Attack]"
    return m


def per_attack_type_metrics(
    windows: pd.DataFrame,
    y_score: np.ndarray,
    thresholds: dict[str, float],
    cfg: dict[str, Any],
    label_col: str = "label",
    evaluated_on: str = "otids_zeroshot",
    label_policy: str = "primary_file_level_theta_0_20",
) -> dict[str, Any]:
    """按攻击类型分解：每类 Recall + AP（该类 vs 全部 Normal），Normal 单独计 FPR。

    AP 口径沿用 W3 ``per_attack_type_binary_metrics``：把该类攻击窗口与全部
    Normal 窗口拼成二分类集合，用平均精度衡量「该类相对正常流量」的可分性。
    """
    label = windows[label_col].to_numpy(dtype=int)
    cls = windows["dominant_class"].astype(str).to_numpy()
    score = np.asarray(y_score)
    normal_mask = label == 0
    out: dict[str, Any] = {
        **scope(cfg, evaluated_on, label_policy, thresholds.get("frozen", 0.54)),
        "thresholds": {k: round(float(v), 6) for k, v in thresholds.items()},
        "by_type": {},
    }
    for tname, thr in thresholds.items():
        normal_pred = (score[normal_mask] >= thr).astype(int)
        out["by_type"].setdefault("Normal", {})[f"fpr_at_{tname}"] = round(
            compute_metrics(np.zeros(int(normal_mask.sum()), dtype=int), normal_pred)["fpr"], 8
        )
        out["by_type"]["Normal"][f"false_alarms_at_{tname}"] = int(normal_pred.sum())
    out["by_type"]["Normal"]["support"] = int(normal_mask.sum())
    out["by_type"]["Normal"]["role"] = "负类（只评估 FPR / 误报）"

    for attack in ("DoS", "Fuzzy", "Impersonation"):
        mask = cls == attack
        if mask.sum() == 0:
            out["by_type"][attack] = {"support": 0, "recall": None}
            continue
        ap = float(
            average_precision_score(
                np.r_[np.ones(int(mask.sum()), dtype=int), np.zeros(int(normal_mask.sum()), dtype=int)],
                np.r_[score[mask], score[normal_mask]],
            )
        )
        rec: dict[str, Any] = {
            "role": "正类（只评估 Recall / 检出）",
            "support": int(mask.sum()),
            "average_precision_vs_normal": round(ap, 8),
            "positive_windows_under_label_policy": int(label[mask].sum()),
        }
        for tname, thr in thresholds.items():
            pred = (score[mask] >= thr).astype(int)
            rec[f"recall_at_{tname}"] = round(
                compute_metrics(np.ones(int(mask.sum()), dtype=int), pred)["recall"], 8
            )
            rec[f"missed_at_{tname}"] = int((1 - pred).sum())
        out["by_type"][attack] = rec
    return out


def five_class_mapping(
    windows: pd.DataFrame,
    y_multi_pred: np.ndarray,
    cfg: dict[str, Any],
    evaluated_on: str = "otids_zeroshot",
) -> dict[str, Any]:
    """五分类映射次级口径：Normal/DoS/Fuzzy 直接映射；Impersonation 单独报分布。

    OTIDS 无 Gear/RPM 两类，且 Impersonation 属伪装注入（官方描述针对 ID 0x164），
    不在 HCRL 五分类标签空间内，故**不强行并入 macro-F1**，只报被判为
    Gear / RPM / Normal 等类别的分布，并给出口径理由。
    """
    cls = windows["dominant_class"].astype(str).to_numpy()
    pred = np.asarray(y_multi_pred).astype(str)
    order = ["Normal", "DoS", "Fuzzy", "Gear", "RPM"]
    mapped = {"Normal": "Normal", "DoS": "DoS", "Fuzzy": "Fuzzy"}
    per_true: dict[str, Any] = {}
    for truth in ("Normal", "DoS", "Fuzzy", "Impersonation"):
        mask = cls == truth
        if mask.sum() == 0:
            per_true[truth] = {"support": 0}
            continue
        dist = {c: int((pred[mask] == c).sum()) for c in order}
        n = int(mask.sum())
        item: dict[str, Any] = {
            "support": n,
            "predicted_distribution": dist,
            "predicted_distribution_pct": {c: round(100.0 * dist[c] / n, 4) for c in order},
        }
        if truth in mapped:
            item["mapped_class"] = mapped[truth]
            item["mapping_accuracy"] = round(dist[mapped[truth]] / n, 8)
        per_true[truth] = item
    imp = per_true.get("Impersonation", {})
    imp_dist = imp.get("predicted_distribution", {})
    return {
        **scope(cfg, evaluated_on, "secondary_multiclass_mapping", 0.54),
        "model_classes": order,
        "direct_mapping": mapped,
        "per_true_type": per_true,
        "impersonation_analysis": {
            "judged_as_Gear": imp_dist.get("Gear", None),
            "judged_as_RPM": imp_dist.get("RPM", None),
            "judged_as_Normal": imp_dist.get("Normal", None),
            "judged_as_DoS": imp_dist.get("DoS", None),
            "judged_as_Fuzzy": imp_dist.get("Fuzzy", None),
            "note": (
                "Impersonation 为伪装注入，HCRL 五分类标签空间无对应类别；"
                "按任务单要求单独报告其被判为 Gear/RPM/Normal 的分布，不强行并入 macro-F1。"
                "被判为 Normal 的比例即伪装类漏检的定性参考。"
            ),
        },
        "macro_f1_reported": False,
        "macro_f1_reason": (
            "OTIDS 无 Gear/RPM 真值类、Impersonation 无对应类别，"
            "五分类 macro-F1 在该跨域集上不可定义；强行计算会造成口径误导"
        ),
    }


def allocate_adaptation_segments(
    windows: pd.DataFrame, cfg: dict[str, Any]
) -> tuple[dict[str, str], dict[str, Any]]:
    """按攻击类型分层、以 segment 为单元把 OTIDS 窗口切成适配折 / 评估折（各约一半）。

    分层键为 segment 前缀（session_id，等价攻击类型）；同一 segment 不跨折，
    抽样单元是 segment 而非窗口，避免片段泄漏。固定种子保证可复现。
    """
    ratio = float(cfg["adaptation"]["eval_ratio"])
    seed = int(cfg["seed"])
    seg = (
        windows.groupby("segment_id", sort=False)
        .agg(n_windows=("window_id", "count"), prefix=("session_id", "first"))
        .reset_index()
    )
    rng = np.random.default_rng(seed)
    assignment: dict[str, str] = {}
    plan: dict[str, Any] = {}
    for prefix, grp in seg.groupby("prefix", sort=True):
        order = rng.permutation(len(grp))
        shuffled = grp.iloc[order]
        target_adapt = int(grp["n_windows"].sum() * (1.0 - ratio))
        adapt_ids: list[str] = []
        cum = 0
        for row in shuffled.itertuples(index=False):
            if len(adapt_ids) >= len(shuffled) - 1:
                break
            if cum + row.n_windows / 2.0 > target_adapt:
                break
            adapt_ids.append(row.segment_id)
            cum += row.n_windows
        adapt_set = set(adapt_ids)
        for sid in grp["segment_id"]:
            assignment[str(sid)] = "adapt" if sid in adapt_set else "eval"
        plan[str(prefix)] = {
            "n_segments": len(grp),
            "n_windows": int(grp["n_windows"].sum()),
            "n_segments_adapt": len(adapt_ids),
            "n_segments_eval": int(len(grp) - len(adapt_ids)),
            "target_adapt_windows": target_adapt,
            "selected_adapt_windows": int(cum),
        }
    return assignment, plan


def calibrate_threshold_on_adapt(
    y_true: np.ndarray, y_score: np.ndarray, cfg: dict[str, Any]
) -> dict[str, Any]:
    """与 W3 同判据的 Youden's J 阈值重标定（只在 OTIDS 适配折上，模型权重不动）。"""
    g = cfg["adaptation"]["threshold_grid"]
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
        curve.append(
            {
                "threshold": round(float(thr), 4),
                "recall": round(recall, 6),
                "fpr": round(fpr, 6),
                "youden_j": round(recall - fpr, 6),
                "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            }
        )
    best_j = max(c["youden_j"] for c in curve)
    candidates = [c for c in curve if abs(c["youden_j"] - best_j) < 1e-12]
    best = max(candidates, key=lambda c: c["threshold"])  # 平局取更高阈值，与 W3 一致
    return {
        "criterion": cfg["adaptation"]["criterion"],
        "calibrated_on": "otids_adapt_fold",
        "model_weights_modified": False,
        "adapt_fold_used_for": "threshold calibration only",
        "grid_range": [float(g["start"]), float(g["stop"]), float(g["step"])],
        "n_grid_points": len(curve),
        "selected_threshold": best["threshold"],
        "selected_youden_j": best["youden_j"],
        "tie_breaker": "prefer_higher_threshold",
        "curve": curve,
    }


def adaptation_comparison(
    windows: pd.DataFrame,
    y_score: np.ndarray,
    frozen_threshold: float,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """「仅调阈值」适配上限：适配折标定阈值，评估折评估；与零样本并列、互不混淆。"""
    assignment, plan = allocate_adaptation_segments(windows, cfg)
    fold = windows["segment_id"].astype(str).map(assignment).to_numpy()
    if pd.isna(pd.Series(fold)).any():
        raise AssertionError("存在未分配适配/评估折的 segment")
    cross = pd.DataFrame({"segment_id": windows["segment_id"].astype(str), "fold": fold})
    if int((cross.groupby("segment_id")["fold"].nunique() > 1).sum()) > 0:
        raise AssertionError("同一 segment 跨适配/评估折")

    y_true = windows["label"].to_numpy(dtype=int)
    adapt_mask = fold == "adapt"
    eval_mask = fold == "eval"
    if not (y_true[adapt_mask].sum() > 0 and (1 - y_true[adapt_mask]).sum() > 0):
        raise AssertionError("适配折缺正负样本，无法标定阈值")
    if not (y_true[eval_mask].sum() > 0 and (1 - y_true[eval_mask]).sum() > 0):
        raise AssertionError("评估折缺正负样本，无法评估")

    calib = calibrate_threshold_on_adapt(y_true[adapt_mask], y_score[adapt_mask], cfg)
    adapted_thr = float(calib["selected_threshold"])
    frozen = float(frozen_threshold)

    rows = [
        {
            "policy": "otids_zeroshot",
            "evaluated_on": "otids_zeroshot",
            "fold": "adapt+eval 全量 OTIDS 窗口",
            "threshold": frozen,
            "threshold_source": "HCRL split-v3 val 冻结阈值 0.54",
            "model_weights_modified": False,
            "metrics": binary_metrics(y_true, y_score, frozen, cfg, "otids_zeroshot"),
        },
        {
            "policy": "otids_adapted",
            "evaluated_on": "otids_adapted",
            "fold": "eval 折（另一半 OTIDS 窗口）",
            "threshold": adapted_thr,
            "threshold_source": "OTIDS adapt 折上重新标定（仅阈值，模型权重不动）",
            "model_weights_modified": False,
            "metrics": binary_metrics(y_true[eval_mask], y_score[eval_mask], adapted_thr, cfg, "otids_adapted"),
        },
        {
            "policy": "otids_zeroshot_on_eval_fold",
            "evaluated_on": "otids_zeroshot",
            "fold": "eval 折（同一半窗口，用于隔离阈值贡献）",
            "threshold": frozen,
            "threshold_source": "HCRL split-v3 val 冻结阈值 0.54",
            "model_weights_modified": False,
            "metrics": binary_metrics(y_true[eval_mask], y_score[eval_mask], frozen, cfg, "otids_zeroshot"),
        },
    ]
    zs_eval = rows[2]["metrics"]
    adapted = rows[1]["metrics"]
    return {
        **scope(cfg, "otids_adapted_vs_zeroshot_comparison", "primary_file_level_theta_0_20", adapted_thr),
        "policy": "并列报出，严禁混报：适配上限明确标注为非零样本",
        "stratification": {
            "sampling_unit": "segment_id",
            "stratification_key": "session_id 前缀（攻击类型）",
            "seed": int(cfg["seed"]),
            "per_stratum_plan": plan,
        },
        "fold_sizes": {
            "adapt": int(adapt_mask.sum()),
            "eval": int(eval_mask.sum()),
            "total": len(y_true),
            "adapt_positive": int(y_true[adapt_mask].sum()),
            "eval_positive": int(y_true[eval_mask].sum()),
        },
        "threshold_calibration": calib,
        "rows": rows,
        "same_eval_fold_delta_vs_frozen": {
            "threshold_frozen": frozen,
            "threshold_adapted": round(adapted_thr, 6),
            "f1_delta": round(adapted["f1"] - zs_eval["f1"], 8),
            "recall_delta": round(adapted["recall"] - zs_eval["recall"], 8),
            "fpr_delta": round(adapted["fpr"] - zs_eval["fpr"], 8),
            "note": (
                "仅在完全相同的 eval 折上对比两个阈值，隔离『换阈值』的净贡献；"
                "该差值不含评估集变化，是适配上限最干净的证据"
            ),
        },
    }


def hcrl_artifact_snapshot(cfg: dict[str, Any]) -> dict[str, str]:
    """对 HCRL 全部既有产物做只读 SHA-256 快照，用于「零修改」校验证据。"""
    roots = [
        PROJECT_ROOT / "data" / "processed" / "features",
        PROJECT_ROOT / "data" / "processed" / "labels",
        PROJECT_ROOT / "data" / "processed" / "split",
        PROJECT_ROOT / "artifacts" / "models",
        PROJECT_ROOT / "artifacts" / "features",
        PROJECT_ROOT / "artifacts" / "eval",
        PROJECT_ROOT / "artifacts" / "ablation",
        PROJECT_ROOT / "artifacts" / "baseline",
        PROJECT_ROOT / "configs",
    ]
    out: dict[str, str] = {}
    for root in roots:
        if not root.is_dir():
            continue
        for p in sorted(root.rglob("*")):
            if p.is_file() and not p.name.endswith((".pyc", ".DS_Store")):
                # 排除本次 W4 新增的 xdomain_* 产物
                if "xdomain" in p.name or "otids" in str(p).lower():
                    continue
                out[str(p.relative_to(PROJECT_ROOT))] = sha256_file(p)
    return out


def verify_hcrl_untouched(before: dict[str, str], after: dict[str, str]) -> dict[str, Any]:
    """逐文件比对快照，返回零触碰校验证据（含新增文件清单）。"""
    changed = sorted(k for k in before if k in after and before[k] != after[k])
    removed = sorted(k for k in before if k not in after)
    added = sorted(k for k in after if k not in before)
    return {
        "verified": not changed and not removed,
        "n_files_before": len(before),
        "n_files_after_in_scope": len(after),
        "changed_files": changed,
        "removed_files": removed,
        "new_files_excluded_from_scope": added,
        "conclusion": (
            "HCRL 侧全部既有产物（含 test 封存、split-v2/v3、L1-v2/v3、特征参数）"
            "校验和前后完全一致；本次只新增 xdomain_/otids 前缀文件"
            if not changed and not removed
            else "存在变更，需停工核查"
        ),
    }


def feature_group_drift(
    hcrl_val_norm: pd.DataFrame, otids_norm: pd.DataFrame, cfg: dict[str, Any]
) -> dict[str, Any]:
    """按特征组量化 HCRL val 与 OTIDS 的分布漂移（归一化尺度，单位≈train std）。

    分组归因映射：
    - G1 + G4 的 ID 相关维度 -> ID 空间漂移
    - G2 + G5 -> 周期间隔分布漂移
    - G3 + G6 -> payload 字节分布漂移
    """
    groups = {
        "id_space": [c for c in FEATURE_COLUMNS if c.startswith(("g1_id", "g4_"))],
        "interval": [c for c in FEATURE_COLUMNS if c.startswith(("g2_", "g5_"))],
        "payload": [c for c in FEATURE_COLUMNS if c.startswith(("g3_", "g6_"))],
        "dlc": [c for c in FEATURE_COLUMNS if c.startswith("g1_dlc") or c == "g4_frame_count"],
    }
    out: dict[str, Any] = {
        **scope(cfg, "hcrl_val_vs_otids_feature_drift", "n/a (无标签参与)", 0.54),
        "unit": "归一化空间（HCRL train 均值/标准差），|Δmean| 单位≈1 个 train std",
        "hcrl_val_n_windows": len(hcrl_val_norm),
        "otids_n_windows": len(otids_norm),
        "by_group": {},
        "per_feature": {},
    }
    for gname, cols in groups.items():
        deltas = []
        for c in cols:
            a = float(hcrl_val_norm[c].mean())
            b = float(otids_norm[c].mean())
            sa = float(hcrl_val_norm[c].std())
            sb = float(otids_norm[c].std())
            deltas.append(abs(b - a))
            out["per_feature"][c] = {
                "group": gname,
                "hcrl_val_mean": round(a, 6),
                "otids_mean": round(b, 6),
                "abs_mean_delta": round(abs(b - a), 6),
                "hcrl_val_std": round(sa, 6),
                "otids_std": round(sb, 6),
                "std_ratio": round(sb / sa, 6) if sa > 1e-9 else None,
            }
        out["by_group"][gname] = {
            "n_features": len(cols),
            "mean_abs_delta": round(float(np.mean(deltas)), 6),
            "max_abs_delta": round(float(np.max(deltas)), 6),
            "sum_abs_delta": round(float(np.sum(deltas)), 6),
        }
    ranked = sorted(out["by_group"].items(), key=lambda x: x[1]["mean_abs_delta"], reverse=True)
    out["dominant_drift_group"] = ranked[0][0]
    out["by_group_note"] = {
        "id_space": "G1 ID 频次 + G4 ID 分布（跨车型 ID 空间 / 频次表错配）",
        "interval": "G2/G5 帧间隔与同 ID 周期（采集环境、总线负载差异）",
        "payload": "G3/G6 payload 字节统计（车型信号编码差异）",
        "dlc": "G1 DLC 与窗口帧数（对照组）",
    }
    return out
