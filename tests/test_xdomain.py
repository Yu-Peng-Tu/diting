"""谛听 W4 · OTIDS 跨域验证新增断言。

覆盖任务单要求的 4 类校验：OTIDS 解析、窗口标签口径、冻结模型加载校验、
HCRL 产物零触碰校验；另加口径标注完整性与防泄漏断言。
不依赖真实大数据的测试用内存小样本；依赖产物的测试读取 W4 已落盘产物。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from diting.common import PROJECT_ROOT, load_config, read_json
from diting.data.labels import build_windows
from diting.data.otids_raw import (
    OTIDS_SOURCE_TAG,
    load_xdomain_config,
    parse_otids_line,
)
from diting.features.build import FEATURE_COLUMNS

XD_SPEC_DOS = {"name": "dataset1.csv", "session_id": "OTIDS_DoS", "file_label": 1, "attack_kind": "DoS"}
XD_SPEC_FUZZY = {"name": "dataset2.csv", "session_id": "OTIDS_Fuzzy", "file_label": 2, "attack_kind": "Fuzzy"}
XD_SPEC_FREE = {"name": "dataset.csv", "session_id": "OTIDS_AttackFree", "file_label": 0, "attack_kind": "Normal"}


def _line(ts=0.0, id1="0316", id0="000", length=8, dlcs=None, target=1):
    if dlcs is None:
        dlcs = ["05", "20", "ea", "0a", "20", "1a", "00", "7f"]
    return ",".join([f"{ts:.6f}", id1, id0, str(length), *dlcs, str(target)])


# ---------------- 1. OTIDS 解析 ----------------
def test_parse_otids_line_primary_and_official_dos():
    rec = parse_otids_line(_line(id1="0080", target=1), XD_SPEC_DOS, 0.0, 250.0)
    assert rec is not None
    assert rec["can_id"] == 0x80
    assert rec["id0_token"] == "000"
    assert rec["dlc"] == 8
    assert rec["data_0"] == 0x05 and rec["data_7"] == 0x7F
    assert rec["source_tag"] == OTIDS_SOURCE_TAG
    assert rec["flag"] == "NA"
    # 主口径：文件级标签，正常 ID 的帧同样标攻击
    assert rec["label"] == 1
    assert rec["file_label"] == 1
    # 官方描述口径：DoS 只有 ID=0x000 为注入帧
    assert rec["official_label"] == 0
    zero = parse_otids_line(_line(id1="0000", target=1), XD_SPEC_DOS, 0.0, 250.0)
    assert zero["can_id"] == 0 and zero["official_label"] == 1


def test_parse_otids_line_short_dlc_padding():
    # LEN=5：后 3 个 DLC 列为空 -> data_5..7 = -1
    rec = parse_otids_line(
        _line(length=5, dlcs=["3a", "ff", "00", "07", "2c", "", "", ""], target=2),
        XD_SPEC_FUZZY, 0.0, 250.0,
    )
    assert rec is not None
    assert rec["dlc"] == 5
    assert [rec[f"data_{i}"] for i in range(8)] == [0x3A, 0xFF, 0x00, 0x07, 0x2C, -1, -1, -1]
    assert rec["label"] == 1


def test_parse_otids_line_official_time_rule():
    # Fuzzy / Impersonation：250 秒后为攻击段
    before = parse_otids_line(_line(ts=100.0, target=2), XD_SPEC_FUZZY, 0.0, 250.0)
    after = parse_otids_line(_line(ts=300.0, target=2), XD_SPEC_FUZZY, 0.0, 250.0)
    assert before["official_label"] == 0 and after["official_label"] == 1
    # Attack-Free 恒正常
    free = parse_otids_line(_line(ts=999.0, target=0), XD_SPEC_FREE, 0.0, 250.0)
    assert free["label"] == 0 and free["official_label"] == 0


def test_parse_otids_line_rejects_bad_rows_without_silent_drop():
    bad_col = parse_otids_line("1.0,0316,000,8,aa,bb,cc", XD_SPEC_DOS, 0.0, 250.0)
    bad_target = parse_otids_line(_line(target=3), XD_SPEC_DOS, 0.0, 250.0)
    bad_number = parse_otids_line("abc,0316,000,8,00", XD_SPEC_DOS, 0.0, 250.0)
    assert bad_col is None and bad_target is None and bad_number is None


# ---------------- 2. 窗口标签口径 ----------------
def test_window_policy_matches_hcrl():
    cfg = load_xdomain_config()
    base = load_config()
    w = cfg["window"]
    assert w["size_frames"] == int(base["window"]["size_frames"])
    assert w["stride_frames"] == int(base["window"]["stride_frames"])
    assert w["attack_ratio_theta"] == float(base["window"]["attack_ratio_theta"])
    assert w["gap_threshold"] == float(base["local_data"]["gap_threshold"])
    assert w["segment_max_frames"] == int(base["local_data"]["segment_max_frames"])
    assert len(FEATURE_COLUMNS) == 33


def test_build_windows_theta_0_20_and_no_cross_segment():
    # 每个 segment 内攻击帧「均匀」分布（不能堆在开头，否则首个 100 帧窗口会命中 θ），
    # 这样每个窗口的 attack_ratio 才等于该段的设定比例，可验证 θ=0.20 边界与「窗口不跨段」。
    rows = []
    fi = 0
    for seg, is_attack in (
        ("S__seg0001", lambda i: i % 10 == 0),   # 0.10 < θ
        ("S__seg0002", lambda i: i % 5 == 0),    # 0.20 == θ（边界，应判正）
        ("S__seg0003", lambda i: i % 10 != 0),   # 0.90 > θ
    ):
        for i in range(200):
            rows.append(
                {
                    "frame_index": fi,
                    "timestamp": fi * 0.001,
                    "label": int(is_attack(i)),
                    "segment_id": seg,
                    "session_id": "S",
                    "source_tag": "simulated",
                    "attack_kind": "DoS",
                }
            )
            fi += 1
    frames = pd.DataFrame(rows)
    win = build_windows(frames, 100, 50, 0.20)
    by_seg = win.groupby("segment_id")["label"].agg(["min", "max"]).to_dict("index")
    assert by_seg["S__seg0001"]["max"] == 0  # attack_ratio=0.10 < 0.20
    assert by_seg["S__seg0002"]["min"] == 1  # attack_ratio=0.20 >= 0.20
    assert by_seg["S__seg0003"]["min"] == 1
    for w in win.itertuples(index=False):
        sub = frames[(frames.frame_index >= w.frame_start) & (frames.frame_index <= w.frame_end)]
        assert sub["segment_id"].nunique() == 1


def test_label_policy_artifact():
    p = PROJECT_ROOT / "data" / "processed" / "otids" / "label_policy.json"
    if not p.is_file():
        return
    d = read_json(p)
    assert d["theta"] == 0.20
    assert d["n_windows"] > 0
    assert set(d["by_dominant_class"]) >= {"Normal", "DoS", "Fuzzy", "Impersonation"}


# ---------------- 3. 冻结模型加载校验 ----------------
def test_frozen_assets_verification():
    from diting.eval.xdomain_eval import load_frozen_assets

    out = load_frozen_assets(load_xdomain_config())
    assert out["threshold"] == 0.54
    assert out["record"]["model_joblib_sha256_matches_manifest"] is True
    assert out["record"]["model_version"] == "l1-gbdt-v3"
    assert len(out["bundle"]["binary_feature_columns"]) == 33
    assert out["bundle"]["binary_model"].n_features_in_ == 33
    assert int(out["record"]["random_seed"]) == 20261108


def test_frozen_scores_are_finite_and_bounded():
    from diting.eval.xdomain_eval import frozen_scores, load_frozen_assets

    cfg = load_xdomain_config()
    bundle = load_frozen_assets(cfg)["bundle"]
    feat = pd.DataFrame(
        np.zeros((6, len(FEATURE_COLUMNS)), dtype=np.float64), columns=FEATURE_COLUMNS
    )
    score, multi = frozen_scores(bundle, feat)
    assert np.isfinite(score).all() and ((score >= 0) & (score <= 1)).all()
    assert len(multi) == 6


# ---------------- 4. HCRL 产物零触碰校验 ----------------
def test_hcrl_artifacts_untouched():
    p = PROJECT_ROOT / "artifacts" / "eval" / "xdomain_hcrl_integrity.json"
    if not p.is_file():
        return
    d = read_json(p)["verification"]
    assert d["verified"] is True
    assert d["changed_files"] == []
    assert d["removed_files"] == []
    assert d["n_files_before"] > 0


def test_split_v3_has_no_test_rows():
    import pyarrow.parquet as pq

    p = PROJECT_ROOT / "data" / "processed" / "split" / "split-v3" / "windows.parquet"
    t = pq.read_table(p, columns=["split"]).to_pandas()
    assert set(t["split"].astype(str).unique()) == {"train", "val"}


# ---------------- 5. 口径标注完整性与防泄漏 ----------------
def test_every_metric_block_has_scope_annotation():
    metrics = PROJECT_ROOT / "artifacts" / "eval" / "xdomain_metrics.json"
    if not metrics.is_file():
        return
    d = read_json(metrics)
    for key, block in d["metrics"].items():
        assert isinstance(block, dict), key
        assert "evaluated_on" in block, f"{key} 缺 evaluated_on"
        assert "data_scope" in block, f"{key} 缺 data_scope"
        assert "level" in block, f"{key} 缺 level"
    m054 = d["metrics"]["otids_zeroshot_binary_at_frozen_0_54"]
    assert m054["evaluated_on"] == "otids_zeroshot"
    assert m054["threshold"] == 0.54
    assert m054["confusion_matrix"]["labels"] == ["Normal", "Attack"]


def test_otids_features_leakage_guard():
    p = PROJECT_ROOT / "data" / "processed" / "otids" / "feature_meta.json"
    if not p.is_file():
        return
    d = read_json(p)
    assert d["n_features"] == 33
    assert d["leakage_assertions"]["passed"] is True
    assert d["normalization_fit_on"] == "train"
    assert d["id_frequency_fit_on"].startswith("HCRL train")


def test_adaptation_and_zeroshot_not_mixed():
    p = PROJECT_ROOT / "artifacts" / "eval" / "xdomain_adaptation.json"
    if not p.is_file():
        return
    d = read_json(p)
    tags = {r["policy"] for r in d["rows"]}
    assert {"otids_zeroshot", "otids_adapted", "otids_zeroshot_on_eval_fold"} <= tags
    assert d["threshold_calibration"]["model_weights_modified"] is False
    assert all(r["model_weights_modified"] is False for r in d["rows"])


def test_id_ablation_has_explicit_conclusion():
    p = PROJECT_ROOT / "artifacts" / "eval" / "xdomain_id_ablation.json"
    if not p.is_file():
        return
    d = read_json(p)
    assert isinstance(d["collapsed"], bool)
    assert isinstance(d["ap_relative_drop_pct"], float)
    assert len(d["conclusion"]) > 20
    assert d["n_features_ablated"] == 26 and d["n_features_full"] == 33
    assert d["trained_on"].startswith("HCRL")
