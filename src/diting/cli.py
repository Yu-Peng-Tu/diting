"""谛听 W1 · 命令行入口（Makefile 各目标对应一个 stage）。

    python -m diting.cli data|split|features|baseline|eval --config configs/default.yaml

每个 stage 只依赖上游已落盘的产物，不隐式重跑；产物一律新建，输入只读。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from .common import (
    load_config,
    read_json,
    set_seed,
    subdir,
    write_json,
    write_manifest,
)


def _duckdb_stats(parquet_path: Path) -> dict[str, Any]:
    """用 DuckDB 查询归一化后的 Parquet（不用 CSV 做中间格式，不用 SQLite）。"""
    import duckdb

    con = duckdb.connect(database=":memory:")
    try:
        rel = con.execute(
            "SELECT count(*) AS n, count(DISTINCT can_id) AS ids, "
            "count(DISTINCT segment_id) AS segs, min(timestamp) AS t0, max(timestamp) AS t1, "
            "sum(CASE WHEN label = 1 THEN 1 ELSE 0 END) AS injected "
            f"FROM read_parquet('{parquet_path.as_posix()}')"
        )
        row = rel.fetchone()
        return {
            "engine": "duckdb",
            "query": "SELECT count(*), count(DISTINCT can_id), count(DISTINCT segment_id), min/max(timestamp), sum(label=1) FROM frames.parquet",
            "n_rows": int(row[0]),
            "n_unique_can_ids": int(row[1]),
            "n_segments": int(row[2]),
            "t_start": float(row[3]),
            "t_end": float(row[4]),
            "n_injected": int(row[5]),
        }
    finally:
        con.close()


def _duckdb_source_stats(parquet_path: Path) -> dict[str, Any]:
    """用 DuckDB 只扫 Parquet 元数据列，核对 source_tag 与 injected 帧数（不载入全表）。"""
    import duckdb

    con = duckdb.connect(database=":memory:")
    try:
        rows = con.execute(
            "SELECT source_tag, count(*) AS n, "
            "sum(CASE WHEN label = 1 THEN 1 ELSE 0 END) AS injected "
            f"FROM read_parquet('{parquet_path.as_posix()}') GROUP BY source_tag ORDER BY source_tag"
        ).fetchall()
        return {
            "by_source_tag": [
                {"source_tag": str(r[0]), "n_frames": int(r[1]), "n_injected": int(r[2])}
                for r in rows
            ],
            "is_real_data": any(str(r[0]) != "simulated" for r in rows),
        }
    finally:
        con.close()


def stage_data(cfg: dict[str, Any]) -> dict[str, Any]:
    """D1 数据接入/校验/归一 + D2 标签生成。

    默认走 `local_data.enabled` 的本地真实数据路径（流式、不联网）；
    W1 的 simulated 替代路径保留用于测试，仅在显式关闭时启用。
    """
    from .data.download import write_provenance
    from .data.labels import build_windows, write_labels
    from .data.local_raw import (
        build_windows_and_labels_streaming,
        inventory_raw_files,
        verify_zip,
        write_frames_streaming,
        write_real_provenance,
    )
    from .data.schema import normalize_all
    from .data.simulate import generate_simulated_raw

    set_seed(int(cfg["project"]["seed"]))
    raw_dir = subdir(cfg, "raw")
    interim_dir = subdir(cfg, "interim")
    processed_dir = subdir(cfg, "processed")
    art_data = subdir(cfg, "artifacts", "data")
    use_local = bool(cfg.get("local_data", {}).get("enabled"))

    if use_local:
        # 闸门 1：zip 校验和/字节数不符 → 立即停工，不猜测不继续
        zip_check = verify_zip(cfg)
        if not zip_check.get("verified"):
            raise AssertionError(
                f"输入 zip 校验失败，按任务单要求立即停工：{zip_check}"
            )
        real_dir = subdir(cfg, "raw", "car-hacking")
        inventory = inventory_raw_files(cfg, real_dir)
        # 闸门 2：逐文件行数须与官方计数在 ±5% 内
        if not inventory["all_within_5pct"]:
            offenders = [f["file"] for f in inventory["files"] if not f["within_5pct"]]
            raise AssertionError(f"真实文件行数与官方计数偏差超 ±5%：{offenders}")
        prov = write_real_provenance(cfg, art_data, inventory, zip_check)
        stats = write_frames_streaming(cfg, real_dir, interim_dir, inventory)
        duck = _duckdb_stats(interim_dir / "frames.parquet")
        if duck["n_rows"] != stats["parquet_rows_total"] or not stats["row_count_match"]:
            raise AssertionError("Parquet 行数与原始文件行数不一致")
        source_stats = _duckdb_source_stats(interim_dir / "frames.parquet")
        if duck["n_injected"] != stats["n_injected"]:
            raise AssertionError("DuckDB 注入帧数与流式统计不一致")
        write_json(
            art_data / "normalization_stats.json",
            {**stats, "duckdb_verification": duck, "source_tag_verification": source_stats},
        )
        write_json(art_data / "raw_file_inventory.json", inventory)
        windows, label_stats = build_windows_and_labels_streaming(cfg, interim_dir, processed_dir)
        print(
            f"[diting][data] 真实数据（Kaggle 镜像）流式接入："
            f"{stats['parquet_rows_total']} 帧 / {stats['n_segments']} 段 / {len(windows)} 窗口"
        )
    else:
        # W1 保留路径：仅用于管线测试，全程离线（allow_network=False）
        prov = write_provenance(cfg, art_data, allow_network=False)
        if not prov["real_data_available"]:
            sim_meta = generate_simulated_raw(cfg, raw_dir)
            write_json(art_data / "simulated_generation.json", sim_meta)
            print("[diting][data] 本地真实数据路径关闭 → 启用 source_tag=simulated 测试路径（非真实数据）")
        frames, stats = normalize_all(
            raw_dir, interim_dir, source_tag="public:car-hacking",
            exclude_dirs=("car-hacking",),  # simulated 测试路径不混入本地真实数据
        )
        duck = _duckdb_stats(interim_dir / "frames.parquet")
        if duck["n_rows"] != stats["parquet_rows_total"] or not stats["row_count_match"]:
            raise AssertionError("Parquet 行数与原始文件行数不一致")
        write_json(art_data / "normalization_stats.json", {**stats, "duckdb_verification": duck})
        windows = build_windows(
            frames,
            window_size=int(cfg["window"]["size_frames"]),
            stride=int(cfg["window"]["stride_frames"]),
            theta=float(cfg["window"]["attack_ratio_theta"]),
        )
        label_stats = write_labels(
            frames, windows, processed_dir / "labels", float(cfg["window"]["attack_ratio_theta"])
        )
        windows.to_parquet(processed_dir / "labels" / "windows_full.parquet", index=False, engine="pyarrow")

    man = write_manifest(
        art_data,
        "data",
        cfg,
        [interim_dir / "frames.parquet"],
        extra={
            "dataset_status": prov["status"],
            "real_data_available": prov["real_data_available"],
            "is_real_data": stats["is_real_data"],
            "source_tag": "public:car-hacking" if use_local else "mixed",
            "data_mode": "local_real_streaming" if use_local else "simulated_test_only",
            "zip_sha256_verified": zip_check["verified"] if use_local else None,
            "row_count_match": stats["row_count_match"],
            "parquet_rows": stats["parquet_rows_total"],
            "raw_data_lines": stats["raw_data_lines_total"],
            "duckdb_verification": duck,
            "n_windows": len(windows),
            "label_stats": label_stats,
            "subset_scope": stats.get("subset_scope"),
            "outputs": [
                "data/interim/frames.parquet",
                "data/interim/segment_index.json",
                "data/processed/labels/frame_labels.parquet",
                "data/processed/labels/window_labels.parquet",
                "data/processed/labels/windows_full.parquet",
                "artifacts/data/dataset_provenance.json",
                "artifacts/data/raw_file_inventory.json",
            ],
        },
    )
    print(
        f"[diting][data] rows={stats['parquet_rows_total']} (raw={stats['raw_data_lines_total']}) "
        f"segments={stats['n_segments']} windows={len(windows)} real_data={stats['is_real_data']}"
    )
    return {"manifest": man, "provenance": prov, "stats": stats, "windows": len(windows)}


def stage_split(cfg: dict[str, Any]) -> dict[str, Any]:
    """D3 划分清单 + 防泄漏断言。"""
    from .split.make_split import build_split_manifest, write_split_artifacts

    set_seed(int(cfg["project"]["seed"]))
    processed_dir = subdir(cfg, "processed")
    art = subdir(cfg, "artifacts", "split")
    windows = pd.read_parquet(processed_dir / "labels" / "windows_full.parquet", engine="pyarrow")
    manifest = build_split_manifest(windows, cfg)
    # build_split_manifest 在副本上赋值，此处把划分结果回写到落盘的窗口表
    windows["split"] = windows["segment_id"].map(
        {seg: s for s, segs in manifest["split_segments"].items() for seg in segs}
    )
    assert windows["split"].notna().all(), "存在未分配 split 的窗口"
    version = str(cfg["split"]["version"])
    paths = write_split_artifacts(windows, manifest, processed_dir, version=version)
    write_manifest(
        art, "split", cfg,
        [processed_dir / "labels" / "windows_full.parquet", paths["manifest"]],
        extra={
            "split_manifest_version": manifest["version"],
            "split_windows": manifest["split_windows"],
            "assertions": manifest["assertions"],
            "test_set_status": manifest["test_set_status"],
            "preserved_v1_files": [
                "data/processed/split/split_manifest.json",
                "data/processed/split/windows.parquet",
            ],
            "outputs": [paths["windows"], paths["manifest"], "data/processed/split/active_version.json"],
        },
    )
    print(
        f"[diting][split] {manifest['split_windows']} "
        f"断言 {len(manifest['assertions'])} 项全通过；test 封存"
    )
    return {"manifest": manifest}


def stage_features(cfg: dict[str, Any]) -> dict[str, Any]:
    """D4 G1–G6 特征 / 训练折参数 / golden vector / 泄漏断言。"""
    from .features.build import (
        apply_normalization,
        assert_feature_matrix_clean,
        build_golden_vector,
        compute_window_features,
        fit_id_frequency,
        fit_normalization,
        write_feature_artifacts,
    )
    from .features.stream import (
        compute_window_features_streaming,
        fit_id_frequency_streaming,
        load_segment_frames,
    )
    from .split.make_split import load_split, split_target_dir

    set_seed(int(cfg["project"]["seed"]))
    interim_dir = subdir(cfg, "interim")
    processed_dir = subdir(cfg, "processed")
    art = subdir(cfg, "artifacts", "features")
    use_local = bool(cfg.get("local_data", {}).get("enabled"))
    chunk_size = int(cfg.get("local_data", {}).get("chunk_size", 200000))

    windows, split_manifest = load_split(processed_dir)
    if use_local:
        # W2 真实数据：按 segment 流式，不把全量帧表载入内存
        train_segments = set(
            windows.loc[windows["split"] == "train", "segment_id"].astype(str).unique()
        )
        if not train_segments:
            raise ValueError("train 折无片段，无法 fit 参数")
        id_freq = fit_id_frequency_streaming(interim_dir, train_segments, chunk_size)
        raw = compute_window_features_streaming(interim_dir, windows, id_freq, chunk_size)
        seg_index = read_json(interim_dir / "segment_index.json")
        sample_seg = min(s["segment_id"] for s in seg_index["segments"])
        sample_frames = load_segment_frames(interim_dir, sample_seg)
        fitting_mode = "streaming_by_segment"
        n_train_segments = len(train_segments)
    else:
        # W1 simulated 测试路径保留：小样本一次性载入
        from .data.schema import load_frames

        frames = load_frames(interim_dir)
        train_ranges = windows[windows["split"] == "train"][["frame_start", "frame_end"]]
        mask = pd.Series(False, index=frames.index)
        for r in train_ranges.itertuples(index=False):
            mask |= (frames["frame_index"] >= r.frame_start) & (frames["frame_index"] <= r.frame_end)
        id_freq = fit_id_frequency(frames[mask])
        raw = compute_window_features(frames, windows, id_freq)
        sample_frames = frames
        sample_seg = str(min(frames["segment_id"].unique()))
        fitting_mode = "in_memory_simulated_test"
        n_train_segments = None

    guard = assert_feature_matrix_clean(raw)
    params = fit_normalization(raw[raw["split"] == "train"], float(cfg["features"]["numeric_clip"]))
    norm = apply_normalization(raw, params)

    # golden vector 用固定样本；只读取该 segment 的帧，避免全表载入
    golden = build_golden_vector(
        sample_frames, windows, id_freq, int(cfg["features"]["golden_sample_frames"])
    )
    golden["sample"]["sha256_of_input"] = _sha_of_golden_input(sample_frames, golden)
    golden["sample"]["source_tag"] = "public:car-hacking" if use_local else "simulated"
    if use_local:
        golden["version"] = "golden-v2-public-car-hacking"

    paths = write_feature_artifacts(raw, norm, params, golden, id_freq, cfg, processed_dir, art, guard)
    write_manifest(
        art, "features", cfg,
        [interim_dir / "frames.parquet", split_target_dir(processed_dir) / "windows.parquet"],
        extra={
            "feature_version": params["version"],
            "n_feature_columns": guard["n_feature_columns"],
            "feature_groups": guard["feature_groups"],
            "leakage_assertions": guard,
            "fit_on": "train",
            "fitting_mode": fitting_mode,
            "n_train_segments_used_for_id_freq": n_train_segments,
            "chunk_size": chunk_size if use_local else None,
            "golden_sample_segment": sample_seg,
            "split_manifest_version": split_manifest["version"],
            "outputs": list(paths.values()),
        },
    )
    print(
        f"[diting][features] {guard['n_feature_columns']} 维 G1–G6 特征；"
        f"参数仅在 train fit；泄漏断言通过；golden vector 已生成"
    )
    return {"params": params, "golden": golden, "guard": guard}


def _sha_of_golden_input(frames: pd.DataFrame, golden: dict[str, Any]) -> str:
    from .common import sha256_obj

    s = golden["sample"]
    seg = frames[frames["segment_id"] == s["segment_id"]].head(s["n_frames"])
    return sha256_obj(seg[["timestamp", "can_id", "dlc"]].to_dict(orient="records"))


def stage_baseline(cfg: dict[str, Any]) -> dict[str, Any]:
    """L0 规则基线 + val 留出评估。"""
    from .eval.metrics import evaluate_l0, write_eval_artifacts
    from .models.l0 import fit_rule_thresholds, predict_l0
    from .split.make_split import load_split

    set_seed(int(cfg["project"]["seed"]))
    processed_dir = subdir(cfg, "processed")
    art = subdir(cfg, "artifacts", "baseline")
    raw = pd.read_parquet(processed_dir / "features" / "window_features_raw.parquet", engine="pyarrow")
    _, split_manifest = load_split(processed_dir)

    rule = fit_rule_thresholds(
        raw,
        float(cfg["baseline"]["id_frequency_percentile"]),
        float(cfg["baseline"]["interval_cv_percentile"]),
    )
    rule["version"] = cfg["baseline"]["version"]
    preds = predict_l0(raw, rule, int(cfg["baseline"]["min_vote"]))
    report = evaluate_l0(preds, split_manifest, cfg)
    paths = write_eval_artifacts(preds, report, rule, art)
    write_manifest(
        art, "baseline", cfg,
        [processed_dir / "features" / "window_features_raw.parquet"],
        extra={
            "model_version": rule["version"],
            "evaluated_on": "val",
            "recall": report["metrics"]["recall"],
            "fpr": report["metrics"]["fpr"],
            "confusion_matrix": report["metrics"]["confusion_matrix"],
            "is_real_data": report["is_real_data"],
            "test_set_status": report["test_set_status"],
            "outputs": list(paths.values()),
        },
    )
    m = report["metrics"]
    print(
        f"[diting][baseline] val Recall={m['recall']:.4f} FPR={m['fpr']:.4f} "
        f"TP/FP/TN/FN={m['tp']}/{m['fp']}/{m['tn']}/{m['fn']} "
        f"(数据来源={'真实' if report['is_real_data'] else 'simulated，仅验证管线'})"
    )
    return {"report": report, "rule": rule, "predictions": preds}


def stage_eval(cfg: dict[str, Any]) -> dict[str, Any]:
    """汇总验收自检：产物齐备、断言复跑、manifest 校验、证据摘要。"""
    from .features.build import assert_feature_matrix_clean
    from .split.make_split import load_split, split_target_dir

    set_seed(int(cfg["project"]["seed"]))
    processed_dir = subdir(cfg, "processed")
    interim_dir = subdir(cfg, "interim")
    art = subdir(cfg, "artifacts", "eval")

    checks: dict[str, Any] = {}
    required = {
        "frames_parquet": interim_dir / "frames.parquet",
        "split_manifest": split_target_dir(processed_dir) / "split_manifest.json",
        "feature_params": subdir(cfg, "artifacts", "features") / "feature_params.json",
        "golden_vector": subdir(cfg, "artifacts", "features") / "golden_vector.json",
        "l0_metrics": subdir(cfg, "artifacts", "baseline") / "l0_metrics.json",
        "confusion_matrix": subdir(cfg, "artifacts", "baseline") / "confusion_matrix.json",
        "baseline_report": subdir(cfg, "artifacts", "baseline") / "baseline_report.md",
    }
    checks["E1_artifacts_exist"] = {
        "passed": all(p.is_file() for p in required.values()),
        "missing": [str(p) for p in required.values() if not p.is_file()],
    }
    assert checks["E1_artifacts_exist"]["passed"], f"产物缺失：{checks['E1_artifacts_exist']['missing']}"

    # 不再把全量帧表载入内存：用 DuckDB 只读 Parquet 元数据列做统计
    duck = _duckdb_stats(interim_dir / "frames.parquet")
    source_stats = _duckdb_source_stats(interim_dir / "frames.parquet")
    windows, split_manifest = load_split(processed_dir)
    from .split.make_split import run_split_assertions

    checks["E2_split_assertions_rerun"] = run_split_assertions(windows, split_manifest)
    raw = pd.read_parquet(processed_dir / "features" / "window_features_raw.parquet", engine="pyarrow")
    checks["E3_leakage_assertions_rerun"] = assert_feature_matrix_clean(raw)

    params = read_json(subdir(cfg, "artifacts", "features") / "feature_params.json")
    checks["E4_params_fit_on_train"] = {"passed": params.get("fit_on") == "train", "fit_on": params.get("fit_on")}
    assert checks["E4_params_fit_on_train"]["passed"]

    metrics = read_json(subdir(cfg, "artifacts", "baseline") / "l0_metrics.json")
    is_real = bool(metrics.get("is_real_data")) and source_stats["is_real_data"]
    recall = metrics["metrics"]["recall"]
    allow_zero = bool(cfg["eval"].get("allow_zero_recall_on_real_data")) and is_real
    checks["E5_nonzero_recall"] = {
        "passed": recall > 0 or allow_zero,
        "recall": recall,
        "mode": "真实数据弱基线：如实记录，不作为阻断项" if allow_zero else "严格模式",
        "note": "L0 是不训练的统计规则基线，真实数据上可能 Recall=0，不做阈值/划分调整",
    }
    assert checks["E5_nonzero_recall"]["passed"], "L0 在 val 上 Recall 为 0"

    manifests = {}
    for stage in ("data", "split", "features", "baseline"):
        p = subdir(cfg, "artifacts", stage) / "manifest.json"
        manifests[stage] = read_json(p) if p.is_file() else None
    checks["E6_manifests_present"] = {
        "passed": all(m and "random_seed" in m and "inputs" in m for m in manifests.values()),
        "stages": list(manifests),
    }
    assert checks["E6_manifests_present"]["passed"]

    summary = {
        "version": cfg["eval"]["version"],
        "evaluated_at": __import__("datetime").datetime.now().astimezone().isoformat(timespec="seconds"),
        "checks": checks,
        "n_frames": duck["n_rows"],
        "n_windows": len(windows),
        "split_windows": split_manifest["split_windows"],
        "split_manifest_version": split_manifest["version"],
        "is_real_data": is_real,
        "source_tag_counts": source_stats["by_source_tag"],
        "real_l0_metrics": {
            "recall": metrics["metrics"]["recall"],
            "fpr": metrics["metrics"]["fpr"],
            "confusion_matrix": metrics["metrics"]["confusion_matrix"],
        },
        "superseded_simulated_metrics": {
            "label": "仅管线验证，非项目指标（W1 simulated 数值已被真实数据结果取代）",
            "recall": 1.0,
            "fpr": 0.083333,
            "confusion_matrix": {"labels": ["Normal", "Attack"], "matrix": [[11, 1], [0, 15]]},
            "source": "早期 simulated 管线验证产物，仅用于说明管线可运行",
        },
    }
    write_json(art / "eval_summary.json", summary)
    write_manifest(art, "eval", cfg, [required[k] for k in required], extra={"checks": list(checks)})

    lines = [
        "# 谛听 W2 · 验收自检摘要（真实数据）",
        "",
        f"- 帧数：{summary['n_frames']}；窗口数：{summary['n_windows']}；划分：{summary['split_windows']}",
        f"- 划分清单版本：`{summary['split_manifest_version']}`（split-v1 文件保留不动）",
        f"- 数据来源：{'真实公开数据集' if summary['is_real_data'] else 'simulated 自生成小样本（非真实数据）'}",
        f"- source_tag 分布：{summary['source_tag_counts']}",
        "",
        "| 检查 | 结果 |",
        "|---|---|",
    ]
    for k, v in checks.items():
        lines.append(f"| {k} | {'PASS' if v.get('passed', all(x.get('passed') for x in v.values() if isinstance(x, dict))) else 'FAIL'} |")
    lines += [
        "",
        "## 真实数据 L0 留出验证集指标（本项目当前口径）",
        "",
        f"- Recall：{summary['real_l0_metrics']['recall']}；FPR：{summary['real_l0_metrics']['fpr']}",
        f"- 混淆矩阵：{summary['real_l0_metrics']['confusion_matrix']['matrix']}",
        "- 详见 `artifacts/baseline/l0_metrics.json` 与 `artifacts/baseline/baseline_report.md`。",
        "",
        "## simulated 数值已作废",
        "",
        "- W1 在 `source_tag=simulated` 小样本上跑出的 Recall=1.0 / FPR=0.0833 属**仅管线验证，非项目指标**；",
        "  它只证明评估管线能跑通，**已被本次真实数据结果取代**，不得写入公开报告或对外报告。",
        "- `simulated` 生成路径仅保留用于测试；默认配置走本地真实数据路径。",
        "",
        "## 边界声明",
        "",
        "- 纯软件、零硬件依赖；纯软件独立实现；不含 RAG / 工单 / 审批等无关业务功能。",
        "- 本任务不实现 FastAPI 服务、前端、L1 GBDT 及以后模型、torch。",
        "- 本次实际文件来源为 Kaggle 公开镜像（MIT），**非官方直发**；官方下载入口不可达，全程未访问外部网络。",
        "- 本机内存 8GB，按任务单第 6 条使用配置化子集（每文件前 N 行）+ 流式分批处理，",
        "  子集范围与逐文件校验和见 `artifacts/data/dataset_provenance.json`。",
        "",
    ]
    (art / "eval_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[diting][eval] {len(checks)} 项自检全部通过")
    return summary


STAGES = {
    "data": stage_data,
    "split": stage_split,
    "features": stage_features,
    "baseline": stage_baseline,
    "eval": stage_eval,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="diting", description="谛听 W1 数据管线")
    parser.add_argument("stage", choices=sorted(STAGES), help="要执行的阶段")
    parser.add_argument("--config", default="configs/default.yaml", help="YAML 配置路径")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    set_seed(int(cfg["project"]["seed"]))
    STAGES[args.stage](cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
