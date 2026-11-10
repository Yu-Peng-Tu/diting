"""谛听 · split-v3 划分生成（按攻击类型分层 + segment 分组防泄漏）。

设计目标：split-v2 按 segment 起始时间顺序切分，叠加
HCRL「按攻击类型分文件」的结构，导致 train 无 DoS、val 无 Fuzzy/Gear/RPM。
split-v3 只重新切分 split-v2 的 train∪val 池（59,831 + 19,702 = 79,533 窗口），
test 折（19,952 窗口）原样封存、不读取特征值/标签、不评估。

分层与防泄漏策略：
- split-v2/windows.parquet 含 segment_id 归属信息，故按设计约束采用
  **segment 分组分层切分**：抽样单元是 segment 而非单个窗口，同一注入片段
  的窗口绝不跨 train/val（硬约束 A1）。
- 每个 segment 内同时含 Normal 窗口与一种攻击窗口，segment 以前缀标注攻击
  类型（DoS / Fuzzy / Gear / RPM）。按前缀分层、每个层内按窗口数逼近 75:25
  分配 segment，则 train 与 val 都必然覆盖 Normal + 4 种攻击 = 全 5 类（A2）。
- 窗口级 (attack_type, label) 的 support 明细由分层结果统计并写入 manifest，
  用于证明覆盖与均衡；因 segment 不可拆分，(attack_type,label) 是分层结果的
  统计量而非逐窗口独立抽样目标，该权衡在 manifest 中如实声明。

test 封存纪律（本模块是唯一接触 split-v2 test 行的地方，且只取标识符）：
- pool 行用 pyarrow filters 下推只读窗口元数据列（不含 33 维特征）；
- test 行只读 window_id 一列，用于计算「集合指纹」证明与 split-v2 完全一致；
- v3 产物 windows.parquet 只含 train/val 行，不含任何 test 行；
- 后续 L1-v3 训练/评估用 window_id in-list 下推特征表，test 特征零载入。

运行：``python -m diting.split.make_split_v3``
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from ..common import PROJECT_ROOT, git_revision, sha256_file, sha256_obj, utc_now, write_json

VERSION = "split-v3"
SEED = 20261108
VAL_RATIO = 0.25
SPLITS = ("train", "val", "test")
CLASS_ORDER = ("Normal", "DoS", "Fuzzy", "Gear", "RPM")
CLASS_MAP = {
    "Normal": "Normal",
    "DoS": "DoS",
    "Fuzzy": "Fuzzy",
    "Spoofing_the_drive_gear": "Gear",
    "Spoofing_the_RPM_gauge": "RPM",
}

V2_DIR = PROJECT_ROOT / "data" / "processed" / "split" / "split-v2"
V3_DIR = PROJECT_ROOT / "data" / "processed" / "split" / "split-v3"
V2_WINDOWS = V2_DIR / "windows.parquet"
V2_MANIFEST = V2_DIR / "split_manifest.json"

# pool 行只读取窗口元数据（用于分层/断言/归因），不含 33 维特征列
POOL_META_COLS = [
    "window_id", "segment_id", "session_id", "source_tag",
    "frame_start", "frame_end", "t_start", "t_end",
    "n_frames", "n_injected", "attack_ratio", "label", "dominant_class",
]


def segment_prefix(segment_id: str) -> str:
    """从 segment_id（如 ``Fuzzy__seg0003``）取攻击类型前缀。"""
    return str(segment_id).split("__", 1)[0]


def _sorted_ids_sha256(ids: list[str]) -> str:
    return sha256_obj(sorted(str(x) for x in ids))


def allocate_segments(
    pool: pd.DataFrame, seed: int = SEED, val_ratio: float = VAL_RATIO
) -> tuple[dict[str, str], dict[str, Any]]:
    """按攻击类型前缀分层、以 segment 为单元做 train/val 分配。

    每层内用固定种子打乱 segment，按窗口数累加逼近 ``val_ratio``（用窗口数
    中点判据减少 overshoot），并强制 train 至少保留 1 个 segment。
    """
    seg = (
        pool.groupby("segment_id", sort=False)
        .agg(n_windows=("window_id", "count"))
        .reset_index()
    )
    seg["prefix"] = seg["segment_id"].map(segment_prefix)
    unknown = sorted(set(seg["prefix"]) - {"DoS", "Fuzzy", "Gear", "RPM"})
    if unknown:
        raise AssertionError(f"出现未知攻击类型前缀的 segment：{unknown[:10]}")

    rng = np.random.default_rng(seed)
    assignment: dict[str, str] = {}
    plan: dict[str, Any] = {}
    for prefix, grp in seg.groupby("prefix", sort=True):
        order = rng.permutation(len(grp))
        shuffled = grp.iloc[order]
        target_val = int(grp["n_windows"].sum() * val_ratio)
        val_ids: list[str] = []
        cum = 0
        for row in shuffled.itertuples(index=False):
            if len(val_ids) >= len(shuffled) - 1:  # train 至少留 1 个 segment
                break
            if cum + row.n_windows / 2.0 > target_val:
                break
            val_ids.append(row.segment_id)
            cum += row.n_windows
        val_set = set(val_ids)
        for sid in grp["segment_id"]:
            assignment[str(sid)] = "val" if sid in val_set else "train"
        n_train_seg = sum(1 for sid in grp["segment_id"] if assignment[str(sid)] == "train")
        plan[prefix] = {
            "n_segments": len(grp),
            "n_windows": int(grp["n_windows"].sum()),
            "n_segments_train": int(n_train_seg),
            "n_segments_val": int(len(grp) - n_train_seg),
            "target_val_windows": target_val,
            "selected_val_windows": int(cum),
        }
    return assignment, plan


def class_support(frame: pd.DataFrame, split: str) -> dict[str, int]:
    """统计某一折的 5 类 support（dominant_class 映射为固定 5 类名）。"""
    sub = frame.loc[frame["split"] == split, "dominant_class"].astype(str).map(CLASS_MAP)
    if sub.isna().any():
        raise AssertionError(f"dominant_class 出现未映射类别：{sorted(set(sub[sub.isna()].index))[:5]}")
    return {c: int((sub == c).sum()) for c in CLASS_ORDER}


def run_v3_assertions(
    v3: pd.DataFrame,
    test_ids: list[str],
    v2_manifest: dict[str, Any],
    plan: dict[str, Any],
) -> dict[str, Any]:
    """逐条跑 v3 防泄漏 / 覆盖 / 封存断言，全部结果写入 manifest。"""
    checks: dict[str, Any] = {}

    # A1 同一注入片段不跨 train/val
    cross = v3.groupby("segment_id")["split"].nunique()
    offenders = cross[cross > 1].index.tolist()
    assert not offenders, f"同一注入片段跨集合：{offenders[:5]}"
    checks["A1_no_segment_across_splits"] = {
        "passed": True, "n_segments": len(cross), "n_offenders": 0,
        "note": "抽样单元为 segment，窗口不跨折",
    }

    # A2 train 与 val 均覆盖全 5 类
    support = {s: class_support(v3, s) for s in ("train", "val")}
    missing = {s: [c for c, n in sp.items() if n <= 0] for s, sp in support.items()}
    assert not any(missing.values()), f"存在折未覆盖全部 5 类：{missing}"
    checks["A2_all_five_classes_in_train_and_val"] = {
        "passed": True, "class_support": support, "missing_classes": missing,
        "assertion": "每折类别覆盖 = 全 5 类",
    }

    # A3 窗口级分类比例接近 75:25
    n_train = int((v3["split"] == "train").sum())
    n_val = int((v3["split"] == "val").sum())
    ratio = n_val / (n_train + n_val)
    assert abs(ratio - VAL_RATIO) <= 0.03, f"val 占比 {ratio:.4f} 偏离目标 {VAL_RATIO}"
    checks["A3_train_val_ratio"] = {
        "passed": True, "n_train": n_train, "n_val": n_val,
        "train_ratio": round(n_train / (n_train + n_val), 6),
        "val_ratio": round(ratio, 6), "target_val_ratio": VAL_RATIO,
        "tolerance": 0.03,
        "note": "segment 不可拆分，实际比例相对 75:25 有小幅偏差",
    }

    # A4 三折 window_id 唯一且互不相交
    ids_train = set(v3.loc[v3["split"] == "train", "window_id"])
    ids_val = set(v3.loc[v3["split"] == "val", "window_id"])
    ids_test = set(test_ids)
    assert v3["window_id"].is_unique, "v3 存在重复 window_id"
    assert not (ids_train & ids_val), "train/val 有交集"
    assert not (ids_train & ids_test) and not (ids_val & ids_test), "train/val 与 test 有交集"
    checks["A4_disjoint_window_ids"] = {
        "passed": True,
        "n_unique_v3": len(ids_train) + len(ids_val),
        "n_test": len(ids_test),
        "n_pairwise_overlap": 0,
    }

    # A5 pool 与 split-v2 的 train∪val 完全一致
    v2_pool_expected = v2_manifest["split_windows"]["train"] + v2_manifest["split_windows"]["val"]
    assert len(v3) == v2_pool_expected, f"v3 pool 行数 {len(v3)} != split-v2 train∪val {v2_pool_expected}"
    checks["A5_pool_matches_v2_train_union_val"] = {
        "passed": True, "n_windows": len(v3),
        "v2_train": v2_manifest["split_windows"]["train"],
        "v2_val": v2_manifest["split_windows"]["val"],
        "expected_pool": v2_pool_expected,
    }

    # A6 test 集合与 split-v2 完全一致、且未进入 v3 产物
    assert len(ids_test) == v2_manifest["split_windows"]["test"], "test 窗口数与 split-v2 manifest 不一致"
    assert "test" not in set(v3["split"].tolist()), "v3 划分产物含 test 行，违反封存纪律"
    assert len(ids_train | ids_val | ids_test) == v2_manifest["total_windows"], "三折并集不等于全量窗口"
    checks["A6_test_set_identical_and_sealed"] = {
        "passed": True,
        "test_n_windows": len(ids_test),
        "test_window_ids_sorted_sha256": _sorted_ids_sha256(test_ids),
        "v2_test_n_windows": v2_manifest["split_windows"]["test"],
        "v2_split_segments_test_count": len(v2_manifest["split_segments"]["test"]),
        "test_rows_in_v3_windows_artifact": 0,
        "test_feature_columns_loaded": 0,
        "test_label_columns_loaded": 0,
        "policy": (
            "test 窗口集合沿用 split-v2，原样封存；本模块仅读取 test 行的 window_id "
            "标识符列用于集合一致性指纹校验，未载入任何特征值、标签或统计量，未做评估"
        ),
    }

    # A7 每折正负样本齐备（保证 Recall / FPR 可评估）
    binary = {}
    for s in ("train", "val"):
        sub = v3[v3["split"] == s]
        binary[s] = {
            "n_windows": len(sub),
            "n_attack": int((sub["label"] == 1).sum()),
            "n_normal": int((sub["label"] == 0).sum()),
        }
        assert binary[s]["n_attack"] > 0 and binary[s]["n_normal"] > 0, f"{s} 缺正负样本"
    checks["A7_binary_populated"] = {"passed": True, "binary_support": binary}

    return {"checks": checks, "support": support, "binary": binary, "plan": plan}


def build_split_v3() -> dict[str, Any]:
    """生成 split-v3 产物（windows.parquet + split_manifest.json），返回 manifest。"""
    v2_manifest = json.loads(V2_MANIFEST.read_text(encoding="utf-8"))

    # pool：只下推读取 train/val 行的窗口元数据（不含 33 维特征）
    pool = pq.read_table(
        V2_WINDOWS,
        columns=POOL_META_COLS + ["split"],
        filters=[("split", "in", ["train", "val"])],
    ).to_pandas()
    assert set(pool["split"].astype(str).unique()) == {"train", "val"}, "pool 混入非 train/val 行"

    # test：只读取 window_id 标识符列，用于「与 split-v2 完全一致」的指纹校验
    test_ids = (
        pq.read_table(V2_WINDOWS, columns=["window_id"], filters=[("split", "=", "test")])
        .to_pandas()["window_id"].astype(str).tolist()
    )

    assignment, plan = allocate_segments(pool)
    v3 = pool.drop(columns=["split"]).copy()
    v3["split"] = v3["segment_id"].map(lambda s: assignment[str(s)])
    if v3["split"].isna().any():
        raise AssertionError("存在未分配 v3 split 的 segment")
    assert set(v3["split"].unique()) == {"train", "val"}

    result = run_v3_assertions(v3, test_ids, v2_manifest, plan)
    checks = result["checks"]

    manifest: dict[str, Any] = {
        "version": VERSION,
        "created_at": utc_now(),
        "random_seed": SEED,
        "method": (
            "按 segment_id 分组分层切分：以攻击类型前缀（DoS/Fuzzy/Gear/RPM）为分层键，"
            "segment 为最小抽样单元，层内按窗口数逼近 75:25 分配到 train/val；"
            "同一注入片段不跨折；split-v2 的 test 折原样封存"
        ),
        "goal": (
            "修复 split-v2「train 无 DoS / val 无 Fuzzy、Gear、RPM」的类别错位缺陷，"
            "使 train 与 val 均覆盖全 5 类，从而让五分类评估与 Gear↔RPM 互混可真实观察"
        ),
        "source_split": "split-v2",
        "source_manifest_path": str(V2_MANIFEST.relative_to(PROJECT_ROOT)),
        "source_manifest_sha256": sha256_file(V2_MANIFEST),
        "source_windows_sha256": sha256_file(V2_WINDOWS),
        "source_manifest_version": v2_manifest.get("version"),
        "source_manifest_content_sha256": v2_manifest.get("content_sha256"),
        "code_version": git_revision(),
        "grouping_key": "segment_id",
        "stratification": {
            "sampling_unit": "segment_id",
            "stratification_key": "segment 攻击类型前缀（DoS/Fuzzy/Gear/RPM）",
            "window_level_key": "(attack_type, label)",
            "window_level_key_role": (
                "窗口级 (attack_type,label) 的 support 是分层结果的统计明细，"
                "不是逐窗口独立抽样目标：split-v2/windows.parquet 含 segment_id 归属信息，"
                "按设计约束改用 segment 分组分层切分以杜绝片段泄漏；"
                "每个 segment 内同时含 Normal 与一种攻击窗口，故按前缀分层即可让两折覆盖全 5 类"
            ),
            "target_train_val_ratio": [0.75, 0.25],
            "per_stratum_plan": plan,
            "seed": SEED,
            "shuffle_rng": "numpy.random.default_rng(seed)",
        },
        "pool_windows": len(v3),
        "pool_segments": int(v3["segment_id"].nunique()),
        "total_windows": v2_manifest["total_windows"],
        "total_segments": v2_manifest["total_segments"],
        "split_windows": {
            "train": checks["A3_train_val_ratio"]["n_train"],
            "val": checks["A3_train_val_ratio"]["n_val"],
            "test": len(test_ids),
        },
        "split_segments": {
            s: sorted(v3.loc[v3["split"] == s, "segment_id"].unique().tolist()) for s in ("train", "val")
        },
        "test_split_segments": list(v2_manifest["split_segments"]["test"]),
        "class_support_by_split": result["support"],
        "binary_support_by_split": result["binary"],
        "split_positive_windows": {s: result["binary"][s]["n_attack"] for s in ("train", "val")},
        "split_negative_windows": {s: result["binary"][s]["n_normal"] for s in ("train", "val")},
        "time_ranges": {
            s: [float(v3.loc[v3["split"] == s, "t_start"].min()),
                float(v3.loc[v3["split"] == s, "t_end"].max())]
            for s in ("train", "val")
        },
        "test_set": {
            "status": "sealed — 原样沿用 split-v2 的 test 窗口集合，不读取特征/标签、不评估、不引用",
            "n_windows": len(test_ids),
            "window_ids_sorted_sha256": _sorted_ids_sha256(test_ids),
            "source": str(V2_WINDOWS.relative_to(PROJECT_ROOT)),
            "rows_in_split_v3_artifact": 0,
            "feature_rows_loaded": 0,
            "label_rows_loaded": 0,
            "columns_read": ["window_id"],
            "note": (
                "为校验集合一致性，仅读取 split-v2 中 split=='test' 行的 window_id 标识符列；"
                "test 行的 33 维特征、label/dominant_class/attack_ratio 等数据列全程未载入内存"
            ),
        },
        "assertions": checks,
        "data_scope": "真实数据子集（每文件前 100 万行）",
        "level": "window",
        "v2_artifacts_policy": "split-v2 全部产物保留不动，作为跨攻击类型泛化口径的存档对照",
    }
    manifest["content_sha256"] = sha256_obj(
        {k: v for k, v in manifest.items() if k != "content_sha256"}
    )

    V3_DIR.mkdir(parents=True, exist_ok=True)
    v3.to_parquet(V3_DIR / "windows.parquet", index=False, engine="pyarrow")
    write_json(V3_DIR / "split_manifest.json", manifest)
    return manifest


if __name__ == "__main__":
    man = build_split_v3()
    print(json.dumps({
        "version": man["version"],
        "split_windows": man["split_windows"],
        "class_support_by_split": man["class_support_by_split"],
        "test_window_ids_sorted_sha256": man["test_set"]["window_ids_sorted_sha256"],
        "assertions": {k: v.get("passed") for k, v in man["assertions"].items()},
    }, ensure_ascii=False, indent=2))
