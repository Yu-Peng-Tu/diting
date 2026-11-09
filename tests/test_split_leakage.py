"""D3 · 划分清单防泄漏断言测试（重点：同一注入片段不跨集合）。"""

from __future__ import annotations

import pytest

from diting.split.make_split import run_split_assertions


def test_no_segment_across_splits(sample_split, sample_windows):
    """核心验收：同一注入片段的窗口不得同时落入 train/val/test。"""
    manifest, windows = sample_split, sample_windows.copy()
    windows["split"] = windows["segment_id"].map(
        {seg: s for s, segs in manifest["split_segments"].items() for seg in segs}
    )
    assert windows["split"].notna().all()
    cross = windows.groupby("segment_id")["split"].nunique()
    assert (cross == 1).all(), f"跨集合片段：{cross[cross > 1].index.tolist()}"
    checks = run_split_assertions(windows, manifest)
    assert checks["A1_no_segment_across_splits"]["passed"] is True
    assert checks["A1_no_segment_across_splits"]["n_offenders"] == 0


def test_assertion_rejects_crossing_segment(sample_split, sample_windows):
    """人为制造一个跨集合片段，断言必须抛错（证明断言真的有效，不是摆设）。"""
    manifest = dict(sample_split)
    manifest["split_segments"] = {
        k: list(v) for k, v in sample_split["split_segments"].items()
    }
    windows = sample_windows.copy()
    windows["split"] = windows["segment_id"].map(
        {seg: s for s, segs in manifest["split_segments"].items() for seg in segs}
    )
    # 把 train 里某个窗口强行改成 val → 该片段跨两个集合
    train_seg = manifest["split_segments"]["train"][0]
    idx = windows.index[windows["segment_id"] == train_seg][0]
    windows.loc[idx, "split"] = "val"
    with pytest.raises(AssertionError, match="同一注入片段跨集合"):
        run_split_assertions(windows, manifest)


def test_all_three_splits_populated_and_time_ordered(sample_split):
    m = sample_split
    assert all(m["split_windows"][s] > 0 for s in ("train", "val", "test"))
    starts = [m["time_ranges"][s][0] for s in ("train", "val", "test")]
    assert starts == sorted(starts), "测试集应位于时间后段"
    assert m["assertions"]["A4_time_ordered"]["passed"] is True
    assert m["test_set_status"].startswith("sealed")


def test_windows_do_not_cross_segment_boundaries(sample_split, sample_windows):
    checks = sample_split["assertions"]
    assert checks["A2_no_window_crosses_segment"]["n_bad"] == 0
    seg_span = sample_windows.groupby("segment_id").agg(
        lo=("frame_start", "min"), hi=("frame_end", "max")
    )
    for w in sample_windows.itertuples(index=False):
        span = seg_span.loc[w.segment_id]
        assert span.lo <= w.frame_start <= w.frame_end <= span.hi
