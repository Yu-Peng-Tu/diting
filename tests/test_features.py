"""D4 · 特征 golden vector 一致性、泄漏断言与训练折拟合测试。"""

from __future__ import annotations

import numpy as np
import pytest

from diting.common import assert_no_forbidden_fields
from diting.features.build import (
    FEATURE_COLUMNS,
    apply_normalization,
    assert_feature_matrix_clean,
    compute_window_features,
    fit_id_frequency,
    fit_normalization,
)


@pytest.fixture(scope="module")
def built(sample_frames, sample_windows, sample_split):
    windows = sample_windows.copy()
    windows["split"] = windows["segment_id"].map(
        {seg: s for s, segs in sample_split["split_segments"].items() for seg in segs}
    )
    id_freq = fit_id_frequency(sample_frames)
    raw = compute_window_features(sample_frames, windows, id_freq)
    params = fit_normalization(raw[raw["split"] == "train"])
    return raw, params, id_freq, windows


def test_golden_vector_consistency(sample_frames, sample_windows, built):
    """固定样本输入 → 特征值快照逐值比对（训练/推理共用同一实现）。"""
    from diting.features.build import build_golden_vector

    _, _, id_freq, _ = built
    g1 = build_golden_vector(sample_frames, sample_windows, id_freq, n_frames=100)
    g2 = build_golden_vector(sample_frames, sample_windows, id_freq, n_frames=100)
    assert g1["values"] == g2["values"], "同一输入两次计算必须逐值一致"
    assert list(g1["values"].keys()) == FEATURE_COLUMNS
    assert all(np.isfinite(v) for v in g1["values"].values())


def test_golden_vector_artifact_matches_recompute(project_root, cfg, built):
    """落盘的 golden_vector.json 与当前代码重算结果一致。

    W2 起真实数据约 500 万帧，按 golden 记录的 sample segment 流式只读取该段，
    与生产阶段 `stage_features` 的取样本方式完全一致；断言强度不变（逐值比对）。
    """
    import json

    from diting.features.build import build_golden_vector
    from diting.features.stream import load_segment_frames
    from diting.split.make_split import load_split

    golden_path = project_root / "artifacts" / "features" / "golden_vector.json"
    if not golden_path.is_file():
        pytest.skip("尚未运行 make features")
    interim_frames = project_root / "data" / "interim" / "frames.parquet"
    if not interim_frames.is_file():
        pytest.skip("帧级中间表不随仓库分发；先跑数据阶段再跑 make features")
    golden = json.loads(golden_path.read_text(encoding="utf-8"))
    frames = load_segment_frames(
        project_root / "data" / "interim", golden["sample"]["segment_id"]
    )
    windows, _ = load_split(project_root / "data" / "processed")
    id_freq = json.loads(
        (project_root / "artifacts" / "features" / "id_frequency_train.json").read_text(encoding="utf-8")
    )["id_frequency"]
    recomputed = build_golden_vector(
        frames, windows, id_freq, int(cfg["features"]["golden_sample_frames"])
    )
    assert recomputed["sample"]["segment_id"] == golden["sample"]["segment_id"]
    for col in FEATURE_COLUMNS:
        assert abs(golden["values"][col] - recomputed["values"][col]) < 1e-9, f"{col} 不一致"


def test_no_label_leakage_in_feature_matrix(built):
    """flag / label / can_id==0x000 布尔不在特征矩阵中（代码断言）。"""
    raw, _, _, _ = built
    report = assert_feature_matrix_clean(raw)
    assert report["passed"] is True
    feature_cols = [c for c in raw.columns if c.startswith("g")]
    assert "flag" not in feature_cols and "label" not in feature_cols
    assert not any("zero" in c.lower() or "0x000" in c.lower() for c in feature_cols)
    assert report["n_feature_columns"] == len(FEATURE_COLUMNS)


def test_forbidden_field_guard_really_raises():
    """守卫必须对禁用字段抛错，不能是空实现。"""
    for bad in (["flag"], ["g1_is_0x000"], ["g1_can_id_zero"], ["label_window"], ["g1_zero_id_bool"]):
        with pytest.raises(AssertionError):
            assert_no_forbidden_fields(FEATURE_COLUMNS + bad)
    assert_no_forbidden_fields(FEATURE_COLUMNS)  # 正常特征通过


def test_normalization_fit_only_on_train(built):
    """归一化参数只在 train 折 fit，且不改变各分折行数。"""
    raw, params, _, _ = built
    assert params["fit_on"] == "train"
    assert set(params["columns"]) == set(FEATURE_COLUMNS)
    norm = apply_normalization(raw, params)
    assert len(norm) == len(raw)
    for col in FEATURE_COLUMNS:
        assert np.isfinite(norm[col]).all(), f"{col} 归一化后出现非有限值"
        assert abs(norm[col].abs().max()) <= params["clip"] + 1e-9


def test_feature_groups_g1_to_g6(built):
    raw, _, _, _ = built
    for g in range(1, 7):
        cols = [c for c in raw.columns if c.startswith(f"g{g}_")]
        assert len(cols) >= 4, f"G{g} 特征不足"
