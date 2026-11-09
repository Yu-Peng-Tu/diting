"""W5 工作台 · 只读产物访问层。

职责：
1. 用白名单把工作台可读取的 artifacts JSON 固定下来（拒绝任意路径，防止路径穿越）；
2. 统一抽取每个产物的口径标注（evaluated_on / data_scope / level / ...），供前端展示；
3. 汇总 HCRL test 折封存的校验留痕，供 /api/health 与 /api/samples 暴露。

纪律：本模块只读，不写任何文件；不载入任何 Parquet，也不接触 HCRL test 折。
"""

from __future__ import annotations

from functools import cache
from typing import Any

from ..common import PROJECT_ROOT, read_json

ARTIFACT_ROOT = PROJECT_ROOT / "artifacts"

# 白名单：key -> 相对 artifacts/ 的路径。工作台只承认这些既有产物。
ARTIFACT_FILES: dict[str, str] = {
    "binary": "eval/l1_v3_binary_metrics.json",
    "multiclass": "eval/l1_v3_multiclass_metrics.json",
    "pr_curve": "eval/l1_v3_pr_curve.json",
    "threshold_curve": "eval/l1_v3_threshold_curve.json",
    "ablation": "eval/l1_v3_ablation_metrics.json",
    "errors": "eval/l1_v3_error_attribution.json",
    "per_attack": "eval/l1_v3_per_attack_type.json",
    "vs_l0": "eval/l1_v3_vs_l0.json",
    "threshold": "models/l1_v3/threshold.json",
    "features_meta": "models/l1_v3/features.json",
    "model_manifest": "models/l1_v3/manifest.json",
    "xdomain_binary": "eval/xdomain_binary_metrics.json",
    "xdomain_adaptation": "eval/xdomain_adaptation.json",
    "xdomain_per_attack": "eval/xdomain_per_attack_type.json",
    "xdomain_drift": "eval/xdomain_feature_drift.json",
    "xdomain_mapping": "eval/xdomain_five_class_mapping.json",
    "xdomain_id_ablation": "eval/xdomain_id_ablation.json",
    "xdomain_official": "eval/xdomain_official_label_sensitivity.json",
    "xdomain_metrics": "eval/xdomain_metrics.json",
}

# 所有对外数字必须随附的口径红线（前端在每个视图底部渲染，pytest 校验存在性）。
SCOPE_CAVEATS: list[str] = [
    "同分布指标为真实数据子集（每文件前 100 万行）上的 split-v3 val 窗口级结果，非全量、非逐帧。",
    "跨域零样本 = 冻结模型权重、阈值 0.54、归一化参数与 ID 频次表全部不改动，直接套用 OTIDS（Kia Soul）。",
    "适配上限仅重新标定阈值、使用了部分 OTIDS 数据，明确非零样本，不得与零样本数字混报。",
    "主口径是文件级标签，负类窗口全部来自独立 Attack-Free 文件；FPR=0 是口径乐观偏差，不是零误报。",
    "OTIDS 无 Gear/RPM 真值类、无 Impersonation 对应类别，跨域五分类 macro-F1 不可定义，禁止报出。",
    "去 ID 消融在跨域上 AP 相对下降 7.7179%（未崩塌），不能解读为「模型不依赖 ID」。",
    "HCRL test 折全程封存：工作台不读取、不评估、不引用任何 test 窗口。",
]

_SCOPE_KEYS = (
    "evaluated_on",
    "data_scope",
    "data_scope_short",
    "level",
    "split_version",
    "dataset",
    "test_set_status",
    "source_tag",
    "label_policy",
    "threshold",
    "threshold_used",
    "model_version",
    "threshold_version",
)


@cache
def load(key: str) -> dict[str, Any]:
    """按白名单读取一个 artifacts JSON（进程内缓存，避免 PR 曲线重复解析）。"""
    if key not in ARTIFACT_FILES:
        raise KeyError(f"未登记的产物 key：{key}")
    path = ARTIFACT_ROOT / ARTIFACT_FILES[key]
    doc = read_json(path)
    if not isinstance(doc, dict):
        raise TypeError(f"产物 {key} 顶层不是 JSON 对象")
    return doc


def source(*keys: str) -> dict[str, str]:
    """返回本次响应实际引用的产物相对路径（便于前端与验收溯源）。"""
    return {k: f"artifacts/{ARTIFACT_FILES[k]}" for k in keys}


def scope_of(doc: dict[str, Any], **overrides: Any) -> dict[str, Any]:
    """从产物顶层抽取口径字段；overrides 用于显式覆盖（例如标注本接口是组合口径）。"""
    out: dict[str, Any] = {k: doc[k] for k in _SCOPE_KEYS if k in doc}
    out.update({k: v for k, v in overrides.items() if v is not None})
    return out


def attach_scope(payload: dict[str, Any], doc: dict[str, Any], *used: str, **overrides: Any) -> dict[str, Any]:
    """给任意响应统一挂上 scope / sources / caveats / test 封存标记。"""
    payload["scope"] = scope_of(doc, **overrides)
    payload["sources"] = source(*used)
    payload["test_rows_loaded"] = 0
    payload["test_set_status"] = doc.get(
        "test_set_status", "sealed — 工作台不读取、不评估、不引用 HCRL test 窗口"
    )
    payload["caveats"] = list(SCOPE_CAVEATS)
    return payload


def sealing_report() -> dict[str, Any]:
    """汇总 test 折封存的可核查留痕（只读 JSON 声明 + 样本来源约束）。"""
    statuses: dict[str, str] = {}
    for key in ARTIFACT_FILES:
        doc = load(key)
        status = doc.get("test_set_status")
        if isinstance(status, str):
            statuses[key] = status
    non_sealed = [
        k for k, v in statuses.items() if ("封存" not in v) and ("sealed" not in v.lower())
    ]
    # 逐字检查：声明里必须出现「封存」或 sealed
    all_sealed = not non_sealed
    return {
        "n_artifacts_with_test_declaration": len(statuses),
        "all_declarations_sealed": all_sealed,
        "n_non_sealed_declarations": len(non_sealed),
        "test_rows_loaded": 0,
        "sample_sources_allowed": ["split-v3 val", "OTIDS (Kia Soul)"],
        "sample_sources_forbidden": ["HCRL test", "split-v3 train"],
        "split_v3_windows_file_only_has_train_val": True,
        "assertion": (
            "样本库只从 split-v3 val 与 OTIDS 窗口构建；split-v3/windows.parquet 断言 split 集合恰为 "
            "{train, val}；特征表按选中 window_id 做 pyarrow in-list 下推，test 特征行零载入"
        ),
    }
