"""W3 pytest · split-v3 分层/防泄漏 + L1-v3 重训与评估治理产物断言。

只读已落盘产物与配置做断言，不重训、不联网、不触碰 test 集。
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import joblib
import numpy as np
import pytest

from diting.common import PROJECT_ROOT, read_json, sha256_obj
from diting.eval.metrics import compute_metrics
from diting.features.build import FEATURE_COLUMNS
from diting.models.l1_v3 import load_l1_v3_config, load_l1_v3_dataset

V3_SPLIT = PROJECT_ROOT / "data" / "processed" / "split" / "split-v3"
V2_SPLIT = PROJECT_ROOT / "data" / "processed" / "split" / "split-v2"
MODEL_DIR = PROJECT_ROOT / "artifacts" / "models" / "l1_v3"
EVAL_DIR = PROJECT_ROOT / "artifacts" / "eval"
ABL_DIR = PROJECT_ROOT / "artifacts" / "ablation"
CLASS_ORDER = ["Normal", "DoS", "Fuzzy", "Gear", "RPM"]


@pytest.fixture(scope="module")
def v3_cfg():
    return load_l1_v3_config()


@pytest.fixture(scope="module")
def v3_man():
    return read_json(V3_SPLIT / "split_manifest.json")


@pytest.fixture(scope="module")
def v3_windows():
    import pandas as pd

    return pd.read_parquet(V3_SPLIT / "windows.parquet", engine="pyarrow")


# --- 1) split-v3 分层与类别覆盖 ------------------------------------------------

def test_split_v3_version_and_seed(v3_man):
    assert v3_man["version"] == "split-v3"
    assert v3_man["random_seed"] == 20261108
    assert v3_man["grouping_key"] == "segment_id"
    assert v3_man["source_split"] == "split-v2"
    assert v3_man["code_version"]


def test_every_split_covers_all_five_classes(v3_man):
    for s in ("train", "val"):
        sp = v3_man["class_support_by_split"][s]
        assert all(sp[c] > 0 for c in CLASS_ORDER), f"{s} 折缺类别：{sp}"
    assert v3_man["assertions"]["A2_all_five_classes_in_train_and_val"]["passed"] is True


def test_assertions_all_pass(v3_man):
    for key, rec in v3_man["assertions"].items():
        assert rec.get("passed") is True, f"断言 {key} 未通过"


def test_train_val_ratio_near_75_25(v3_man):
    n = v3_man["split_windows"]
    ratio = n["val"] / (n["train"] + n["val"])
    assert 0.22 <= ratio <= 0.28
    assert n["train"] + n["val"] == v3_man["pool_windows"] == 79533


def test_no_segment_across_train_val(v3_windows):
    cross = v3_windows.groupby("segment_id")["split"].nunique()
    assert int((cross > 1).sum()) == 0
    assert set(v3_windows["split"].astype(str).unique()) == {"train", "val"}


def test_test_set_identical_to_v2_and_sealed(v3_man, v3_windows):
    import pyarrow.parquet as pq

    v2_man = read_json(V2_SPLIT / "split_manifest.json")
    test_ids = (
        pq.read_table(V2_SPLIT / "windows.parquet", columns=["window_id"],
                      filters=[("split", "=", "test")])
        .to_pandas()["window_id"].astype(str).tolist()
    )
    fp = sha256_obj(sorted(test_ids))
    assert len(test_ids) == v3_man["split_windows"]["test"] == v2_man["split_windows"]["test"]
    assert v3_man["test_set"]["window_ids_sorted_sha256"] == fp
    assert v3_man["test_set"]["rows_in_split_v3_artifact"] == 0
    assert v3_man["test_set"]["feature_rows_loaded"] == 0
    assert v3_man["test_set"]["label_rows_loaded"] == 0
    assert "test" not in set(v3_windows["split"].astype(str).unique())
    ids_v3 = set(v3_windows["window_id"].astype(str))
    assert not (ids_v3 & set(test_ids))
    assert len(ids_v3 | set(test_ids)) == v2_man["total_windows"]


def test_split_manifest_content_hash(v3_man):
    body = {k: v for k, v in v3_man.items() if k != "content_sha256"}
    assert sha256_obj(body) == v3_man["content_sha256"]


def test_split_v2_artifacts_untouched():
    """split-v2 清单仍可读、version 未变（禁止修改既有产物）。"""
    m = read_json(V2_SPLIT / "split_manifest.json")
    assert m["version"] == "split-v2"
    assert m["split_windows"] == {"train": 59831, "val": 19702, "test": 19952}


# --- 2) 数据加载：test 零载入、标签不夹带 --------------------------------------

def test_dataset_test_rows_zero(v3_cfg):
    ds, aux = load_l1_v3_dataset(v3_cfg)
    assert set(ds.split.tolist()) == {"train", "val"}
    assert "test" not in set(ds.split.tolist())
    assert ds.meta["test_rows_loaded"] == 0
    assert len(ds.features) == 79533
    assert list(ds.features.columns) == list(FEATURE_COLUMNS)
    assert len(aux) == len(ds.features)
    assert "label" not in ds.features.columns


def test_dataset_all_five_classes_both_splits(v3_cfg):
    ds, _ = load_l1_v3_dataset(v3_cfg)
    for s in ("train", "val"):
        cc = ds.meta["class_coverage"][s]["class_counts"]
        assert all(cc[c] > 0 for c in CLASS_ORDER)
    assert ds.meta["class_coverage"]["shared_classes"] == sorted(CLASS_ORDER)


def test_dataset_segment_no_cross_split(v3_cfg):
    _, aux = load_l1_v3_dataset(v3_cfg)
    cross = aux.groupby("segment_id")["split"].nunique()
    assert int((cross > 1).sum()) == 0


# --- 3) L1-v3 模型与指标 -------------------------------------------------------

@pytest.fixture(scope="module")
def bundle():
    p = MODEL_DIR / "model.joblib"
    assert p.is_file(), "L1-v3 模型产物缺失，请先运行 python -m diting.l1_v3_pipeline"
    return joblib.load(p)


def test_model_bundle_loadable(bundle):
    assert hasattr(bundle["binary_model"], "predict_proba")
    assert hasattr(bundle["multiclass_model"], "predict_proba")
    assert set(bundle["binary_classes"]) == {0, 1}
    assert set(bundle["binary_feature_columns"]) == set(FEATURE_COLUMNS)
    assert set(bundle["multiclass_classes"]) == set(CLASS_ORDER), "v3 五分类应见过全 5 类"
    assert bundle["split_version"] == "split-v3"
    assert bundle["random_seed"] == 20261108


def test_prediction_shapes(bundle):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(11, 33)).astype(np.float64)
    p2 = bundle["binary_model"].predict_proba(X)
    assert p2.shape == (11, 2)
    assert np.allclose(p2.sum(axis=1), 1.0, atol=1e-6)
    p5 = bundle["multiclass_model"].predict_proba(X)
    assert p5.shape == (11, 5)


def test_model_files_complete():
    for name in ("model.joblib", "features.json", "threshold.json", "metrics.json",
                 "split_manifest_ref.json", "manifest.json"):
        assert (MODEL_DIR / name).is_file(), f"缺产物 {name}"


@pytest.mark.parametrize("key", ["at_default_threshold", "at_calibrated_threshold"])
def test_binary_metrics_recompute(key):
    m = read_json(EVAL_DIR / "l1_v3_binary_metrics.json")[key]
    cm = m["confusion_matrix"]["matrix"]
    tn, fp = cm[0]
    fn, tp = cm[1]
    y_true = np.array([0] * (tn + fp) + [1] * (fn + tp))
    y_pred = np.array([0] * tn + [1] * fp + [0] * fn + [1] * tp)
    rec = compute_metrics(y_true, y_pred)
    for f in ("recall", "precision", "fpr", "f1", "false_alarms_per_10k_normal",
              "tp", "fp", "tn", "fn", "n_samples"):
        assert rec[f] == pytest.approx(m[f], abs=1e-5), f"{key}.{f} 复算不一致"


def test_binary_metrics_scope_and_manifest():
    b = read_json(EVAL_DIR / "l1_v3_binary_metrics.json")
    assert b["evaluated_on"] == "val"
    assert b["split_version"] == "split-v3"
    assert "subset" in b["data_scope"] or "子集" in b["data_scope"]
    for f in ("features.json", "threshold.json", "metrics.json"):
        d = read_json(MODEL_DIR / f)
        assert d.get("data_scope") and d.get("evaluated_on")
    man = read_json(MODEL_DIR / "manifest.json")
    assert man["test_rows_loaded"] == 0
    assert man["split_version"] == "split-v3"


def test_multiclass_5x5_all_class_support_positive():
    m = read_json(EVAL_DIR / "l1_v3_multiclass_metrics.json")
    assert m["confusion_matrix"]["labels"] == CLASS_ORDER
    cm = m["confusion_matrix"]["matrix"]
    assert len(cm) == 5 and all(len(r) == 5 for r in cm)
    for c in CLASS_ORDER:
        assert m["per_class"][c]["support"] > 0, f"{c} 在 val 上 support 为 0"
    assert m["gear_rpm_confusion"]["observable_on_val"] is True


def test_gear_rpm_analysis_observable():
    g = read_json(EVAL_DIR / "l1_v3_gear_rpm_analysis.json")
    assert g["observable_on_val"] is True
    assert g["gear_support"] > 0 and g["rpm_support"] > 0
    assert g["total_bidirectional"] == (
        g["Gear_predicted_as_RPM"] + g["RPM_predicted_as_Gear"]
    )
    assert g["verdict"]


# --- 4) L0 同口径对照 ----------------------------------------------------------

def test_l0_recomputed_same_val():
    l0 = read_json(EVAL_DIR / "l1_v3_l0_recomputed.json")
    c = read_json(EVAL_DIR / "l1_v3_vs_l0.json")
    assert l0["n_val_windows"] == 19277
    assert l0["primary_metrics"]["n_samples"] == c["l0_n_samples"] == c["l1_n_samples"]
    assert c["evaluated_on"] == "val"
    assert len(c["rows"]) == 3 and len(c["deltas_vs_l0"]) == 2
    assert c["rows"][0]["model"].startswith("L0")
    assert c["fpr_conclusion"]
    assert isinstance(c["fpr_dropped_significantly"], bool)


def test_l0_primary_is_same_scope_refit():
    """主对照行必须用 split-v3 train 重新 fit 的阈值（与 L1 同口径）。"""
    l0 = read_json(EVAL_DIR / "l1_v3_l0_recomputed.json")
    assert "split-v3 train" in l0["primary_threshold_source"]
    assert l0["primary_rule"]["fit_on"] == "train_normal_windows"
    assert l0["reference_metrics"]["fpr"] != l0["primary_metrics"]["fpr"]
    assert "敏感性" in l0["reference_policy"]


# --- 5) 阈值 / PR 曲线 ---------------------------------------------------------

def test_threshold_curve_consistent():
    c = read_json(EVAL_DIR / "l1_v3_threshold_curve.json")
    assert c["calibrated_on"] == "val"
    assert c["n_grid_points"] == len(c["curve"]) == 91
    for r in c["curve"]:
        # youden_j/recall/fpr 各自 round(6) 后相减可能有 1e-6 量级浮点残差
        assert r["youden_j"] == pytest.approx(r["recall"] - r["fpr"], abs=2e-6)
    best = max(r["youden_j"] for r in c["curve"])
    assert c["selected_youden_j"] == pytest.approx(best, abs=1e-6)
    plateau = [r["threshold"] for r in c["curve"] if abs(r["youden_j"] - best) < 1e-12]
    assert c["selected_threshold"] == max(plateau), "平局应取更高阈值"
    assert "sealed" in c["test_set_status"]


def test_pr_curve_points_exported():
    p = read_json(EVAL_DIR / "l1_v3_pr_curve.json")
    assert p["n_points"] == len(p["points"]) > 50
    assert p["evaluated_on"] == "val"
    for pt in p["points"][:5]:
        assert 0.0 <= pt["precision"] <= 1.0
        assert 0.0 <= pt["recall"] <= 1.0
    assert p["points"][-1]["threshold"] is None
    assert p["average_precision"] > 0.9


# --- 6) 消融 / 误报归因 --------------------------------------------------------

def test_ablation_three_groups_and_conclusion():
    a = read_json(EVAL_DIR / "l1_v3_ablation_metrics.json")
    assert len(a["variants"]) == 4
    assert {v["dropped_group"] for v in a["variants"][1:]} == {
        "id_frequency", "interval", "payload"
    }
    assert a["evaluated_on"] == "val"
    assert "collapsed" in a["id_group_conclusion"]
    assert "10%" in a["collapse_rule"]


def test_error_attribution_per_window():
    e = read_json(EVAL_DIR / "l1_v3_error_attribution.json")
    assert e["n_false_positives"] + e["n_false_negatives"] == len(e["records"])
    assert e["evaluated_on"] == "val"
    assert e["governance_recommendations"]
    for r in e["records"]:
        assert r["error_kind"] in ("false_positive", "false_negative")
        assert r["window_id"] and r["segment_id"] and r["attack_type"]
        assert r["possible_cause"]
        assert len(r["top_deviating_features"]) == 6
    lb = e["label_boundary_analysis"]
    assert lb["n_errors"] == len(e["records"])
    assert lb["n_errors_near_boundary"] >= 1
    assert lb["fp_attack_ratios"] and lb["fn_attack_ratios"]


def test_per_attack_type_metrics_four_attacks():
    p = read_json(EVAL_DIR / "l1_v3_per_attack_type.json")
    for c in ("DoS", "Fuzzy", "Gear", "RPM"):
        assert p["by_type"][c]["support"] > 0
        assert p["by_type"][c]["recall"] is not None
    assert p["by_type"]["Normal"]["support"] > 0
    assert p["evaluated_on"] == "val"


def test_svg_wellformed():
    for p in (EVAL_DIR / "l1_v3_feature_importance.svg",
              EVAL_DIR / "l1_v3_ablation_comparison.svg"):
        assert p.is_file() and p.stat().st_size > 500
        assert ET.parse(p).getroot().tag.endswith("svg")


# --- 7) 公开评估产物 / 工程纪律 -----------------------------------------------------

def test_public_evaluation_deliverables_exist():
    for name in ("l1_v3_feature_importance.json", "l1_v3_l0_recomputed.json"):
        assert (EVAL_DIR / name).is_file(), f"缺公开评估产物 {name}"
    assert (MODEL_DIR / "metrics.json").is_file()


def test_model_metrics_scope_and_sealing():
    metrics = read_json(MODEL_DIR / "metrics.json")
    assert "task_binary" in metrics
    assert "task_multiclass" in metrics
    assert metrics["evaluated_on"] == "val"
    assert metrics["split_version"] == "split-v3"
    assert metrics["test_set_status"].startswith("sealed")


def test_manifest_five_version_classes():
    v = read_json(MODEL_DIR / "manifest.json")["versions"]
    for k in ("model_version", "feature_version", "threshold_version",
              "split_manifest_version", "code_version", "random_seed"):
        assert v.get(k), f"manifest 缺版本号 {k}"


def test_inputs_unchanged_checksums(v3_cfg):
    from diting.common import sha256_file

    feat = read_json(MODEL_DIR / "features.json")
    fpath = PROJECT_ROOT / v3_cfg["inputs"]["features"]
    assert sha256_file(fpath) == feat["source_sha256"]


def test_v2_artifacts_not_overwritten():
    """W2 的评估产物仍在且未被 v3 覆盖（检查关键指标仍是 v2 的 split-v2 口径）。"""
    b = read_json(EVAL_DIR / "l1_binary_metrics.json")
    assert b["at_default_threshold"]["n_samples"] == 19702
    m = read_json(EVAL_DIR / "l1_multiclass_metrics.json")
    assert m["per_class"]["DoS"]["support"] > 0
    assert m["per_class"]["Gear"]["support"] == 0
    assert (ABL_DIR / "ablation_metrics.json").is_file()
    assert (ABL_DIR / "manifest.json").is_file()
