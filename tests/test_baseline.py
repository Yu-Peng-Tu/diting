"""L0 · 规则基线、留出评估指标与产物 manifest 测试。"""

from __future__ import annotations

import json

import numpy as np
import pytest

from diting.eval.metrics import compute_metrics
from diting.models.l0 import fit_rule_thresholds, predict_l0


@pytest.fixture(scope="module")
def predicted(sample_frames, sample_windows, sample_split):
    from diting.features.build import compute_window_features, fit_id_frequency

    windows = sample_windows.copy()
    windows["split"] = windows["segment_id"].map(
        {seg: s for s, segs in sample_split["split_segments"].items() for seg in segs}
    )
    raw = compute_window_features(sample_frames, windows, fit_id_frequency(sample_frames))
    rule = fit_rule_thresholds(raw)
    preds = predict_l0(raw, rule, min_vote=1)
    return raw, rule, preds, windows


def test_l0_nonzero_recall_on_validation(predicted, sample_split):
    """L0 在留出验证集上给出非零 Recall（W1 验收口径）。"""
    from diting.eval.metrics import evaluate_l0

    _, _, preds, _ = predicted
    cfg = {
        "eval": {"version": "test"},
        "baseline": {"version": "l0-test"},
    }
    report = evaluate_l0(preds, sample_split, cfg)
    assert report["evaluated_on"].startswith("val")
    assert report["metrics"]["recall"] > 0, "L0 连 DoS 都检不出，应先查数据管线（§5.1）"
    assert 0.0 <= report["metrics"]["fpr"] <= 1.0
    assert report["test_set_status"].startswith("sealed")


def test_confusion_matrix_sums_correctly(predicted):
    _, _, preds, _ = predicted
    val = preds[preds["split"] == "val"]
    m = compute_metrics(val["label"].to_numpy(), val["pred"].to_numpy())
    assert m["tp"] + m["fp"] + m["tn"] + m["fn"] == m["n_samples"]
    assert m["confusion_matrix"]["matrix"] == [[m["tn"], m["fp"]], [m["fn"], m["tp"]]]


def test_thresholds_fit_only_on_train_normal(predicted):
    _, rule, _, _ = predicted
    assert rule["fit_on"] == "train_normal_windows"
    assert rule["n_train_normal_windows"] > 0
    assert "g4_min_id_share" in rule["excluded_from_rules"]
    assert len(rule["rule_features"]) == 4
    assert all(np.isfinite(v) for v in rule["thresholds"].values())


def test_metrics_edge_cases():
    m = compute_metrics(np.array([1, 1, 0, 0]), np.array([1, 0, 0, 1]))
    assert m["tp"] == 1 and m["fn"] == 1 and m["fp"] == 1 and m["tn"] == 1
    assert m["recall"] == 0.5 and m["fpr"] == 0.5
    all_zero = compute_metrics(np.array([1, 1, 0, 0]), np.array([0, 0, 0, 0]))
    assert all_zero["recall"] == 0.0 and all_zero["precision"] == 0.0


def test_artifacts_have_manifest(project_root):
    """每个产物目录必须带 manifest.json，含种子/输入校验和/代码版本。"""
    for stage in ("data", "split", "features", "baseline"):
        p = project_root / "artifacts" / stage / "manifest.json"
        if not p.is_file():
            pytest.skip(f"尚未运行 make {stage}")
        man = json.loads(p.read_text(encoding="utf-8"))
        assert man["stage"] == stage
        assert "random_seed" in man and "created_at" in man
        assert "code_version" in man and "inputs" in man
        for rec in man["inputs"]:
            assert "sha256" in rec and "exists" in rec
