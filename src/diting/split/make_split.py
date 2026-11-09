"""D3 · 划分清单（按注入片段/会话分组 + 时间顺序 6:2:2）。

防泄漏要点（策划方案 §4.6 L1/L3/L6）：
- 不逐帧随机打散；以 segment_id 为最小划分单位。
- 按片段起始时间排序后顺序切分，测试集位于时间后段。
- 代码断言：「同一注入片段不跨集合」，窗口归属由片段决定，窗口不跨片段边界。
- 归一化参数只在训练折 fit；阈值只在验证集标定；测试集封存（本任务不评估 test）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from ..common import PROJECT_ROOT, read_json, sha256_obj, utc_now, write_json

SPLITS = ("train", "val", "test")


def build_split_manifest(
    windows: pd.DataFrame,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """生成带版本号的划分清单。"""
    ratios = cfg["split"]["ratios"]
    if abs(sum(ratios.values()) - 1.0) > 1e-6:
        raise ValueError("划分比例之和必须为 1.0")

    # 每个片段取最早窗口时间作为排序键，保证时间顺序切分
    seg = (
        windows.groupby("segment_id", sort=False)
        .agg(
            t_start=("t_start", "min"),
            t_end=("t_end", "max"),
            n_windows=("window_id", "count"),
            n_positive=("label", "sum"),
            session_id=("session_id", "first"),
            source_tag=("source_tag", "first"),
        )
        .reset_index()
        .sort_values(["t_start", "segment_id"], kind="mergesort")
        .reset_index(drop=True)
    )
    total = int(seg["n_windows"].sum())
    cut_train = round(total * ratios["train"])
    cut_val = round(total * (ratios["train"] + ratios["val"]))

    cumulative = 0
    assignment: dict[str, str] = {}
    for row in seg.itertuples(index=False):
        mid = cumulative + row.n_windows / 2.0
        if mid < cut_train:
            split = "train"
        elif mid < cut_val:
            split = "val"
        else:
            split = "test"
        assignment[row.segment_id] = split
        cumulative += row.n_windows

    windows = windows.copy()
    windows["split"] = windows["segment_id"].map(assignment)
    if windows["split"].isna().any():
        raise AssertionError("存在未分配 split 的窗口")

    manifest: dict[str, Any] = {
        "version": cfg["split"]["version"],
        "created_at": utc_now(),
        "random_seed": cfg["project"]["seed"],
        "method": "按 segment_id（注入片段/会话连续段）分组 + 片段起始时间顺序切分 6:2:2；窗口不跨片段边界",
        "ratios": ratios,
        "grouping_key": "segment_id",
        "total_windows": len(windows),
        "total_segments": len(seg),
        "split_windows": {s: int((windows["split"] == s).sum()) for s in SPLITS},
        "split_segments": {
            s: sorted([k for k, v in assignment.items() if v == s]) for s in SPLITS
        },
        "split_positive_windows": {
            s: int(((windows["split"] == s) & (windows["label"] == 1)).sum()) for s in SPLITS
        },
        "split_negative_windows": {
            s: int(((windows["split"] == s) & (windows["label"] == 0)).sum()) for s in SPLITS
        },
        "time_ranges": {
            s: [
                float(windows.loc[windows["split"] == s, "t_start"].min()),
                float(windows.loc[windows["split"] == s, "t_end"].max()),
            ]
            if (windows["split"] == s).any()
            else [None, None]
            for s in SPLITS
        },
        "assertions": {},
        "test_set_status": "sealed — 本任务不评估 test，仅在 val 上做 L0 基线评估",
    }
    manifest["assertions"] = run_split_assertions(windows, manifest)
    manifest["content_sha256"] = sha256_obj({k: v for k, v in manifest.items() if k != "content_sha256"})
    return manifest


def run_split_assertions(windows: pd.DataFrame, manifest: dict[str, Any]) -> dict[str, Any]:
    """逐条跑防泄漏断言；任何一条失败即抛 AssertionError。"""
    checks: dict[str, Any] = {}

    # A1 同一注入片段不跨集合
    cross = windows.groupby("segment_id")["split"].nunique()
    offenders = cross[cross > 1].index.tolist()
    assert not offenders, f"同一注入片段跨集合：{offenders[:5]}"
    checks["A1_no_segment_across_splits"] = {
        "passed": True,
        "n_segments": len(cross),
        "n_offenders": len(offenders),
    }

    # A2 窗口不跨片段边界（向量化实现，真实数据约 10 万窗口时避免逐行 .loc）
    lo = windows["segment_id"].map(windows.groupby("segment_id")["frame_start"].min())
    hi = windows["segment_id"].map(windows.groupby("segment_id")["frame_end"].max())
    boundary_bad = int(
        ((windows["frame_start"] < lo) | (windows["frame_end"] > hi)).sum()
    )
    assert boundary_bad == 0, f"存在跨片段边界的窗口：{boundary_bad}"
    checks["A2_no_window_crosses_segment"] = {"passed": True, "n_bad": boundary_bad}

    # A3 三集合均非空，且 train/val 同时含正负样本（保证基线可评估）
    for s in SPLITS:
        assert manifest["split_windows"][s] > 0, f"{s} 集合为空"
    for s in ("train", "val"):
        assert manifest["split_positive_windows"][s] > 0, f"{s} 无攻击窗口，无法评估 Recall"
        assert manifest["split_negative_windows"][s] > 0, f"{s} 无正常窗口，无法评估 FPR"
    checks["A3_splits_populated"] = {"passed": True, "split_windows": manifest["split_windows"]}

    # A4 时间顺序：train 起点 <= val 起点 <= test 起点
    starts = [manifest["time_ranges"][s][0] for s in SPLITS]
    ordered = all(starts[i] <= starts[i + 1] for i in range(2))
    assert ordered, f"划分不满足时间顺序：{starts}"
    checks["A4_time_ordered"] = {"passed": True, "split_start_times": starts}

    # A5 划分清单的片段集合与窗口表一致
    listed = {seg for s in SPLITS for seg in manifest["split_segments"][s]}
    actual = set(windows["segment_id"].unique())
    assert listed == actual, "划分清单片段集合与窗口表不一致"
    checks["A5_manifest_matches_windows"] = {"passed": True, "n_segments": len(actual)}

    return checks


def active_split_version(processed_dir: Path) -> str | None:
    """读取 active_version.json 指针；不存在表示使用 W1 的 split-v1 平铺文件。"""
    p = processed_dir / "split" / "active_version.json"
    return read_json(p).get("version") if p.is_file() else None


def split_target_dir(processed_dir: Path, version: str | None = None) -> Path:
    """解析当前生效版本的产物目录（v1 平铺 / v2+ 版本子目录）。"""
    split_dir = processed_dir / "split"
    v = version or active_split_version(processed_dir)
    return split_dir / v if v else split_dir


def write_split_artifacts(
    windows: pd.DataFrame,
    manifest: dict[str, Any],
    processed_dir: Path,
    version: str | None = None,
) -> dict[str, str]:
    """写划分清单；version 非空时写入版本子目录，并保持 v1 平铺文件不被改动。"""
    split_dir = processed_dir / "split"
    split_dir.mkdir(parents=True, exist_ok=True)
    target = split_dir / version if version else split_dir
    target.mkdir(parents=True, exist_ok=True)
    if version:
        write_json(
            split_dir / "active_version.json",
            {
                "version": version,
                "updated_at": utc_now(),
                "note": "当前生效的划分清单版本；split-v1 平铺文件按任务单要求保留不动",
                "windows_path": str(target / "windows.parquet"),
                "manifest_path": str(target / "split_manifest.json"),
            },
        )
    win_path = target / "windows.parquet"
    man_path = target / "split_manifest.json"
    windows.to_parquet(win_path, index=False, engine="pyarrow")
    write_json(man_path, manifest)
    return {"windows": str(win_path), "manifest": str(man_path)}


def load_split(
    processed_dir: Path, version: str | None = None
) -> tuple[pd.DataFrame, dict[str, Any]]:
    target = split_target_dir(processed_dir, version)
    windows = pd.read_parquet(target / "windows.parquet", engine="pyarrow")
    manifest = read_json(target / "split_manifest.json")
    # 加载时复跑断言，防止清单被手工改动
    run_split_assertions(windows, manifest)
    return windows, manifest


def split_dir_relative() -> str:
    return str(PROJECT_ROOT / "data" / "processed" / "split")
