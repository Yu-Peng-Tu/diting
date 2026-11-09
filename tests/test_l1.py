"""W2-L1 pytest · 模型可加载 / 预测形状 / 指标复算一致 / 消融配置合法。

只读已落盘的 L1 产物与配置做断言，不重训、不联网、不触碰 test 集。
"""

from __future__ import annotations

import joblib
import numpy as np
import pytest

from diting.common import PROJECT_ROOT, read_json
from diting.eval.metrics import compute_metrics
from diting.features.build import FEATURE_COLUMNS
from diting.models.l1 import load_l1_config, validate_ablation_groups

MODEL_DIR = PROJECT_ROOT / "artifacts" / "models" / "l1"
EVAL_DIR = PROJECT_ROOT / "artifacts" / "eval"
ABL_DIR = PROJECT_ROOT / "artifacts" / "ablation"


@pytest.fixture(scope="module")
def l1_cfg():
    return load_l1_config()


@pytest.fixture(scope="module")
def bundle():
    path = MODEL_DIR / "model.joblib"
    assert path.is_file(), "L1 模型产物缺失，请先运行 python -m diting.l1_pipeline"
    return joblib.load(path)


# 1) 模型可加载 ---------------------------------------------------------------
def test_model_bundle_loadable(bundle):
    assert "binary_model" in bundle and "multiclass_model" in bundle
    assert hasattr(bundle["binary_model"], "predict_proba")
    assert hasattr(bundle["multiclass_model"], "predict")
    assert len(bundle["binary_feature_columns"]) == 33
    assert set(bundle["binary_feature_columns"]) == set(FEATURE_COLUMNS)
    assert set(bundle["binary_classes"]) == {0, 1}
    assert "DoS" not in bundle["multiclass_classes"], "train 无 DoS，模型类别不应包含 DoS"
    assert bundle["random_seed"] == 20261108


def test_model_files_complete():
    for name in ("model.joblib", "features.json", "threshold.json", "metrics.json",
                 "split_manifest_ref.json", "manifest.json"):
        assert (MODEL_DIR / name).is_file(), f"缺产物 {name}"


# 2) 预测形状 -----------------------------------------------------------------
def test_prediction_shapes(bundle):
    rng = np.random.default_rng(0)
    n = 17
    X = rng.normal(size=(n, 33)).astype(np.float64)
    p2 = bundle["binary_model"].predict_proba(X)
    assert p2.shape == (n, 2)
    assert np.allclose(p2.sum(axis=1), 1.0, atol=1e-6)
    yhat2 = bundle["binary_model"].predict(X)
    assert yhat2.shape == (n,)
    p5 = bundle["multiclass_model"].predict_proba(X)
    assert p5.shape == (n, len(bundle["multiclass_classes"]))
    yhat5 = bundle["multiclass_model"].predict(X)
    assert yhat5.shape == (n,)
    assert set(np.unique(yhat5)).issubset(set(bundle["multiclass_classes"]))


# 3) 指标复算一致 -------------------------------------------------------------
@pytest.mark.parametrize("key", ["at_default_threshold", "at_calibrated_threshold"])
def test_binary_metrics_recompute(key):
    m = read_json(EVAL_DIR / "l1_binary_metrics.json")[key]
    cm = m["confusion_matrix"]["matrix"]
    tn, fp = cm[0]
    fn, tp = cm[1]
    n = tn + fp + fn + tp
    y_true = np.array([0] * (tn + fp) + [1] * (fn + tp))
    y_pred = np.array([0] * tn + [1] * fp + [0] * fn + [1] * tp)
    recomputed = compute_metrics(y_true, y_pred)
    for field in ("recall", "precision", "fpr", "f1", "false_alarms_per_10k_normal",
                  "tp", "fp", "tn", "fn", "n_samples"):
        assert recomputed[field] == pytest.approx(m[field], abs=1e-5), f"{key}.{field} 复算不一致"
    assert n == m["n_samples"]


def test_threshold_curve_consistent():
    c = read_json(EVAL_DIR / "l1_threshold_curve.json")
    assert c["calibrated_on"] == "val"
    assert c["n_grid_points"] == len(c["curve"])
    for row in c["curve"]:
        assert row["youden_j"] == pytest.approx(row["recall"] - row["fpr"], abs=1e-6)
        assert row["tp"] + row["fn"] > 0 and row["fp"] + row["tn"] > 0
    best = max(r["youden_j"] for r in c["curve"])
    assert c["selected_youden_j"] == pytest.approx(best, abs=1e-6)
    assert c["default_threshold"] == 0.5
    assert "sealed" in c["test_set_status"]
    assert "val" in c["train_used_for"] or "train" in c["train_used_for"]


def test_threshold_json_only_val():
    t = read_json(MODEL_DIR / "threshold.json")
    assert t["calibrated_on"] == "val"
    assert t["evaluated_on"] == "val"
    lo, hi, _step = t["grid_range"]
    assert lo <= t["selected_threshold"] <= hi
    assert t["default_threshold"] == 0.5
    assert t["data_scope"]


def test_l0_comparison_same_val():
    comp = read_json(EVAL_DIR / "l1_vs_l0.json")
    assert comp["l0_n_samples"] == comp["l1_n_samples"]
    assert len(comp["rows"]) == 3
    assert len(comp["deltas_vs_l0"]) == 2
    assert comp["rows"][0]["model"].startswith("L0")
    assert comp["evaluated_on"] == "val"


def test_multiclass_confusion_is_5x5():
    m = read_json(EVAL_DIR / "l1_multiclass_metrics.json")
    assert m["confusion_matrix"]["labels"] == ["Normal", "DoS", "Fuzzy", "Gear", "RPM"]
    cm = m["confusion_matrix"]["matrix"]
    assert len(cm) == 5 and all(len(r) == 5 for r in cm)
    assert m["per_class"]["DoS"]["support"] > 0
    assert m["per_class"]["Gear"]["support"] == 0 and m["per_class"]["RPM"]["support"] == 0
    assert m["gear_rpm_confusion"]["observable_on_val"] is False


# 4) 消融配置合法 -------------------------------------------------------------
def test_ablation_config_valid(l1_cfg):
    report = validate_ablation_groups(l1_cfg)
    assert report["passed"] is True
    assert report["n_total"] == 33
    assert set(report["groups"]) == {"id_frequency", "interval", "payload"}
    assert report["retained_base_columns"]
    assert sum(g["n_columns"] for g in report["groups"].values()) + len(report["retained_base_columns"]) == 33


def test_ablation_results_three_groups():
    a = read_json(ABL_DIR / "ablation_metrics.json")
    assert len(a["variants"]) == 4  # full + 3 groups
    assert {v["dropped_group"] for v in a["variants"][1:]} == {"id_frequency", "interval", "payload"}
    for v in a["variants"][1:]:
        assert "delta_vs_full" in v
        assert v["n_features"] < 33
    assert "collapsed" in a["id_group_conclusion"]
    assert a["id_group_conclusion"]["conclusion"]
    assert a["evaluated_on"] == "val"


def test_importance_ranked_33():
    imp = read_json(EVAL_DIR / "l1_feature_importance.json")
    assert imp["n_features"] == 33
    assert len(imp["ranked"]) == 33
    assert [r["rank"] for r in imp["ranked"]] == list(range(1, 34))
    means = [r["importance_mean"] for r in imp["ranked"]]
    assert means == sorted(means, reverse=True)
    assert imp["evaluated_on"] == "val"


# 5) 纪律：test 封存、口径标注、版本号 -----------------------------------------
def test_test_set_sealed_and_scope():
    man = read_json(MODEL_DIR / "manifest.json")
    assert man["test_rows_loaded"] == 0
    assert man["evaluated_on"] == "val"
    assert "sealed" in man["test_set_status"]
    assert "train" in man["split_filter"] and "val" in man["split_filter"]
    for f in ("features.json", "threshold.json", "metrics.json"):
        d = read_json(MODEL_DIR / f)
        assert d.get("data_scope") or d.get("evaluated_on"), f"{f} 缺口径标注"


def test_manifest_five_version_classes():
    man = read_json(MODEL_DIR / "manifest.json")
    v = man["versions"]
    for key in ("model_version", "feature_version", "threshold_version",
                "split_manifest_version", "code_version", "random_seed"):
        assert v.get(key), f"manifest 缺版本号 {key}"


def test_class_coverage_recorded_honestly():
    man = read_json(MODEL_DIR / "manifest.json")
    cov = man["class_coverage"]
    assert cov["train"]["class_counts"]["DoS"] == 0
    assert cov["val"]["class_counts"]["Fuzzy"] == 0
    assert "DoS" in cov["val_only_classes"]
    assert set(cov["train_only_classes"]) == {"Fuzzy", "Gear", "RPM"}


def test_train_diagnostic_marked_non_metric():
    d = read_json(EVAL_DIR / "l1_train_diagnostic.json")
    assert d["role"] == "diagnostic_only"
    assert d["is_acceptance_metric"] is False
    assert d["warning"]
    assert set(d["classes_in_train"]) == {"Fuzzy", "Gear", "Normal", "RPM"}


def test_svg_artifacts_wellformed():
    import xml.etree.ElementTree as ET

    for path in (EVAL_DIR / "l1_feature_importance.svg", ABL_DIR / "ablation_comparison.svg"):
        assert path.is_file() and path.stat().st_size > 500
        root = ET.parse(path).getroot()
        assert root.tag.endswith("svg")


def test_inputs_unchanged_checksums(l1_cfg):
    """L1 只读输入：特征/标签文件校验和与 manifest 记录一致（未修改输入）。"""
    from diting.common import sha256_file

    feat = read_json(MODEL_DIR / "features.json")
    man = read_json(MODEL_DIR / "manifest.json")
    fpath = PROJECT_ROOT / l1_cfg["inputs"]["features"]
    assert sha256_file(fpath) == feat["source_sha256"]
    inp = {i["path"]: i for i in man["inputs"]}
    assert "data/processed/features/window_features.parquet" in inp
    assert inp["data/processed/features/window_features.parquet"]["exists"] is True
