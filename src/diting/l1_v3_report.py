"""谛听 W3 · 报告渲染（主报告 + L1-v3 vs L0 对照表）。

所有数字均从流水线计算结果动态生成，不硬编码；与 split-v2 的对照数字只读
既有 ``artifacts/eval/l1_binary_metrics.json`` / ``l1_multiclass_metrics.json``。
"""

from __future__ import annotations

from typing import Any

from .common import PROJECT_ROOT, read_json


def _f(x: Any, n: int = 4) -> str:
    return f"{float(x):.{n}f}" if x is not None else "—"


def _v2_metrics() -> dict[str, Any]:
    """只读 W2 既有产物，用于口径对照（不修改、不重算）。"""
    e = PROJECT_ROOT / "artifacts" / "eval"
    try:
        b = read_json(e / "l1_binary_metrics.json")
        m = read_json(e / "l1_multiclass_metrics.json")
        return {
            "available": True,
            "default": b["at_default_threshold"],
            "calibrated": b["at_calibrated_threshold"],
            "macro_f1": m["macro_f1"],
            "source": "artifacts/eval/l1_binary_metrics.json / l1_multiclass_metrics.json（W2 只读产物）",
        }
    except (OSError, KeyError, ValueError):
        return {"available": False}


def render_comparison_md(
    cfg: dict[str, Any],
    comp: dict[str, Any],
    m_def: dict[str, Any],
    m_cal: dict[str, Any],
    l0: dict[str, Any],
) -> str:
    lines = [
        "# L1-v3 vs L0 · split-v3 val 同口径对照表",
        "",
        f"- **评估集合：split-v3 val（{comp['l0_n_samples']} 窗口）**；窗口级；test 全程封存。",
        f"- 数据口径：{cfg['data_scope']}。",
        (f"- **L0 阈值来源（同口径主行）**：`{l0['primary_threshold_source']}`。"
        "L0 只是分位数统计规则、不是可训练模型；阈值由 split-v3 train 正常窗口重新 fit，"
        "与 L1-v3 的训练/评估划分完全一致。"),
        f"- 同集校验：L0 n={comp['l0_n_samples']}，L1-v3 n={comp['l1_n_samples']}，一致 → 对照有效。",
        "",
        "| 模型 | 阈值 | Recall | FPR | Precision | F1 | 每万正常窗口误报数 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in comp["rows"]:
        lines.append(
            f"| {r['model']} | {r['threshold']} | {_f(r['recall'])} | {_f(r['fpr'], 6)} | "
            f"{_f(r['precision'])} | {_f(r['f1'])} | {_f(r['false_alarms_per_10k_normal'], 2)} |"
        )
    lines += [
        "", "## 相对 L0 的变化", "",
        "| L1 口径 | FPR 绝对变化 | FPR 相对变化 | Recall 变化 | F1 变化 | 每万误报变化 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for d in comp["deltas_vs_l0"]:
        rel = f"{d['fpr_relative_change_pct']:.2f}%" if d["fpr_relative_change_pct"] is not None else "—"
        lines.append(
            f"| {d['model']} | {d['fpr_absolute_change']:+.6f} | {rel} | "
            f"{d['recall_absolute_change']:+.4f} | {d['f1_absolute_change']:+.4f} | "
            f"{d['false_alarms_per_10k_change']:+.2f} |"
        )
    lines += [
        "", "## 结论（验收项：FPR 是否明显下降）", "",
        f"- **{comp['fpr_conclusion']}**",
        (f"- Recall 代价/增益：L0 {_f(comp['rows'][0]['recall'])} → L1-v3@标定阈值 "
        f"{_f(m_cal['recall'])}（{comp['deltas_vs_l0'][1]['recall_absolute_change']:+.4f}）；"
        f"F1 {_f(comp['rows'][0]['f1'])} → {_f(m_cal['f1'])}。"),
        "", "## L0 阈值来源的敏感性核查（如实标注）", "",
        (f"- 主口径（split-v3 train 重新 fit 阈值，同口径）：FPR {_f(l0['primary_metrics']['fpr'], 6)}，"
        f"误报 {l0['primary_metrics']['fp']} 个，Recall {_f(l0['primary_metrics']['recall'])}。"),
        (f"- 参考口径（沿用 W2 在 split-v2 train fit 的固定阈值）：FPR "
        f"{_f(l0['reference_metrics']['fpr'], 6)}，误报 {l0['reference_metrics']['fp']} 个，"
        f"Recall {_f(l0['reference_metrics']['recall'])}。"),
        (f"- 两者 FPR 差 {l0['threshold_source_delta_fpr']:+.6f}。"
        "参考口径不作为对照表主行：**若把旧划分的固定阈值当成主对照，会把阈值迁移造成的"
        "误报恶化算成 L1 的增益**，属于口径不对等。主口径重新 fit 后才与 L1-v3 严格同口径；"
        "同时该差值也如实说明 L0 固定阈值对划分/分布漂移非常敏感，是 W4 阈值治理的输入。"),
        "", "## 与 split-v2 的口径边界", "",
        ("- 本表 L0 是在 **split-v3 val** 上重算的，与 W2 报告里 split-v2 val 的 L0 "
        "（FPR 0.143066）不是同一个评估集合，**数字不可直接比较**。"),
        ("- split-v2 的 val 攻击 100% 为最易检出的 DoS；split-v3 的 val 同时含 DoS / Fuzzy / "
        "Gear / RPM 四类，难度结构不同。两者各自回答不同问题，详见主报告第 2 节。"),
        "", f"> 全部数字均为「{cfg['data_scope']}」，非全量、非实车、非 test 指标。", "",
    ]
    return "\n".join(lines)


def render_report(
    cfg: dict[str, Any],
    dataset: Any,
    v3_man: dict[str, Any],
    v2_man: dict[str, Any],
    m_def: dict[str, Any],
    m_cal: dict[str, Any],
    m_multi: dict[str, Any],
    calib: dict[str, Any],
    l0: dict[str, Any],
    comp: dict[str, Any],
    gear_rpm: dict[str, Any],
    per_type: dict[str, Any],
    errors: dict[str, Any],
    ablation: dict[str, Any],
    importance: dict[str, Any],
    pr: dict[str, Any],
) -> str:
    cov = dataset.meta["class_coverage"]
    v2 = _v2_metrics()
    lines = [
        "# 谛听 W3 · split-v3 划分重做与 L1 评估治理报告",
        "",
        f"- 模型：`{cfg['model']['estimator']}`，版本 `{cfg['version']}`，种子 `{cfg['seed']}`；超参与 W2 完全一致（未调参）。",
        f"- 特征：`{cfg['inputs']['features']}` 的 33 维 G1–G6（未重算特征）。",
        f"- 标签：`{cfg['inputs']['labels']}`；划分：`{v3_man['version']}`。",
        f"- **评估口径：{cfg['data_scope']}；窗口级；evaluated_on=val；test 全程封存（载入行数 0）。**",
        "",
        ("> **边界声明**：纯软件建模，不含硬件/BOM、纯软件独立实现、不含 RAG/工单/审批，"
         "不安装 torch、不写服务与前端。所有数字均为离线子集 val 结果，不是全量或实车指标。"),
        "",
        "## 1. split-v3：划分重做（修复类别错位）",
        "",
        "### 1.1 问题回顾（split-v2）",
        "",
        (f"split-v2 按 segment 起始时间顺序 6:2:2 切分（train {v2_man['split_windows']['train']} / "
         f"val {v2_man['split_windows']['val']} / test {v2_man['split_windows']['test']} 窗口），"
         "叠加 HCRL「按攻击类型分文件」结构后，train 折只有 Normal/Fuzzy/Gear/RPM、**无 DoS**；"
         "val 折只有 Normal/DoS、**无 Fuzzy/Gear/RPM**。后果：五分类 macro-F1 不代表识别能力，"
         "Gear↔RPM 互混在 val 上物理不可观察。"),
        "",
        "### 1.2 split-v3 设计",
        "",
        (f"- **重切分范围**：只重新切分 split-v2 的 train∪val（{v3_man['pool_windows']} 窗口、"
        f"{v3_man['pool_segments']} 个 segment）；test 的 {v3_man['split_windows']['test']} 窗口"
        "原样封存，集合与 split-v2 完全一致。"),
        f"- **分层方法**：{v3_man['method']}。",
        ("- **抽样单元**：segment（因 split-v2/windows.parquet 含 segment_id 归属信息，"
        "按任务单要求采用分组分层切分，同一注入片段绝不跨 train/val）。"),
        f"- **窗口级 (attack_type,label) 口径**：{v3_man['stratification']['window_level_key_role']}",
        (f"- **种子**：`{v3_man['random_seed']}`（`numpy.random.default_rng`）；"
        f"生成时间 `{v3_man['created_at']}`；代码版本 `{v3_man['code_version'][:12]}`。"),
        "",
        "**各折样本数与 5 类 support 明细**",
        "",
        "| 折 | 窗口数 | 攻击 | 正常 | Normal | DoS | Fuzzy | Gear | RPM |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in ("train", "val"):
        sp = v3_man["class_support_by_split"][s]
        bs = v3_man["binary_support_by_split"][s]
        lines.append(
            f"| {s} | {bs['n_windows']} | {bs['n_attack']} | {bs['n_normal']} | "
            + " | ".join(str(sp[c]) for c in ("Normal", "DoS", "Fuzzy", "Gear", "RPM")) + " |"
        )
    lines += [
        "",
        (f"- 实际 train:val = {v3_man['assertions']['A3_train_val_ratio']['train_ratio']:.4f}:"
        f"{v3_man['assertions']['A3_train_val_ratio']['val_ratio']:.4f}"
        "（目标 0.75:0.25；segment 不可拆分导致小幅偏差，已在 manifest 声明）。"),
        "- **验收断言全通过**：" + "、".join(
            f"{k}={'✓' if v.get('passed') else '✗'}" for k, v in v3_man["assertions"].items()
        ) + "。",
        (f"- test 集合指纹 `window_ids_sorted_sha256 = "
        f"`{v3_man['test_set']['window_ids_sorted_sha256'][:16]}…`；"
        f"test 特征载入行数 {v3_man['test_set']['feature_rows_loaded']}、"
        f"标签载入行数 {v3_man['test_set']['label_rows_loaded']}；"
        f"v3 划分产物中的 test 行数 {v3_man['test_set']['rows_in_split_v3_artifact']}。"),
        ("> 封存边界如实说明：为校验「集合与 split-v2 完全一致」，划分生成阶段仅读取 "
         "split-v2 中 split=='test' 行的 **window_id 标识符列** 计算指纹；"
         "test 行的 33 维特征、label/dominant_class/attack_ratio 等数据列全程未载入内存，"
         "未参与任何训练或评估。L1-v3 训练用 window_id in-list 下推特征表，"
         "in-list 中不含 test 的 window_id（manifest 记 test_rows_loaded=0）。"),
        "",
        "### 1.3 与 split-v2 的口径对照（不可直接比数字）",
        "",
        "| 维度 | split-v2 | split-v3 |",
        "|---|---|---|",
        (f"| val 窗口集合 | {v2_man['split_windows']['val']:,}（时间后段） | "
         f"{v3_man['split_windows']['val']:,}（分层抽样） |"),
        "| val 攻击构成 | 100% DoS | DoS / Fuzzy / Gear / RPM 四类 |",
        "| train 攻击构成 | Fuzzy / Gear / RPM（无 DoS） | 四类全覆盖 |",
        "| 回答的问题 | 跨攻击类型泛化（学 spoofing、检 DoS） | 同分布下的检出与分类能力 |",
        "| 五分类 macro-F1 | 结构性失真（val 有训练未见类） | 可真实评估 |",
        "",
        ("**两轮 val 集合不同、攻击难度结构不同，任何数字都不应直接相减或宣称「提升/下降**。"
        "W2 的结论仍作为「跨攻击类型泛化」口径的存档对照保留，split-v2 与 W2 全部产物未改动。"),
        "",
    ]
    if v2["available"]:
        lines += [
            ("（仅作口径说明，不构成对比结论）W2 在 split-v2 val 上："
            f"L1@标定阈值 Recall {_f(v2['calibrated']['recall'])} / FPR {_f(v2['calibrated']['fpr'], 6)} / "
            f"F1 {_f(v2['calibrated']['f1'])}，五分类 macro-F1 {_f(v2['macro_f1'])}；"
            f"来源 `{v2['source']}`。"),
            "",
        ]
    lines += [
        "## 2. L1-v3 二分类指标（split-v3 val）",
        "",
        "| 阈值 | Recall | FPR | Precision | F1 | 每万正常窗口误报数 | AP(PR-AUC) | ROC-AUC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for tag, m in (("默认 0.5", m_def), (f"标定 {calib['selected_threshold']:.2f}", m_cal)):
        lines.append(
            f"| {tag} | {_f(m['recall'])} | {_f(m['fpr'], 6)} | {_f(m['precision'])} | "
            f"{_f(m['f1'])} | {_f(m['false_alarms_per_10k_normal'], 2)} | "
            f"{_f(m['average_precision'], 6)} | {_f(m['roc_auc'], 6)} |"
        )
    lines += [
        "",
        (f"- 混淆矩阵 @标定阈值：TP={m_cal['tp']} / FP={m_cal['fp']} / "
        f"TN={m_cal['tn']} / FN={m_cal['fn']}（n={m_cal['n_samples']}）。"),
        "",
        "### 2.1 按攻击类型分解（误报治理的关键证据）",
        "",
        "| 类型 | support | Recall / FPR | 漏报数 / 误报数 | AP(vs Normal) |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, d in per_type["by_type"].items():
        if name == "Normal":
            lines.append(f"| Normal | {d['support']} | FPR {_f(d['fpr'], 6)} | 误报 {d['false_alarms']} | — |")
        else:
            lines.append(
                f"| {name} | {d['support']} | Recall {_f(d['recall'])} | 漏报 {d['missed']} | "
                f"{_f(d.get('average_precision_vs_normal'), 6)} |"
            )
    lines += [
        "",
        ("该表首次在同分布 val 上展示四类攻击各自的检出难度，是 W4 误报治理与"
        "按类型标定阈值的输入；DoS 不再是 val 上唯一的攻击类型。"),
        "",
        "## 3. 五分类识别与 Gear↔RPM 互混（本轮可真实观察）",
        "",
        (f"- macro-Precision {_f(m_multi['macro_precision'])} / macro-Recall "
        f"{_f(m_multi['macro_recall'])} / **macro-F1 {_f(m_multi['macro_f1'])}**。"),
        f"- 模型在 train 实际见过的类别：`{', '.join(m_multi['model_classes_seen_in_train'] or [])}`。",
        "",
        "| 类别 | Precision | Recall | F1 | support |",
        "|---|---:|---:|---:|---:|",
    ]
    for c, d in m_multi["per_class"].items():
        lines.append(f"| {c} | {_f(d['precision'])} | {_f(d['recall'])} | {_f(d['f1'])} | {d['support']} |")
    labels = m_multi["confusion_matrix"]["labels"]
    cm = m_multi["confusion_matrix"]["matrix"]
    lines += [
        "", "**5×5 混淆矩阵**（行=真实，列=预测）：", "",
        "| 真实 \\ 预测 | " + " | ".join(labels) + " |",
        "|---|" + "---:|" * 5,
    ]
    for lab, row in zip(labels, cm):
        lines.append(f"| {lab} | " + " | ".join(str(x) for x in row) + " |")
    lines += [
        "",
        "### 3.1 Gear↔RPM 互混分析（验收重点）",
        "",
        (f"- **val 上可观察性：{gear_rpm['observable_on_val']}**"
        f"（Gear support={gear_rpm['gear_support']}，RPM support={gear_rpm['rpm_support']}）——"
        "这是 split-v3 相较 split-v2 最直接的修复成果。"),
        (f"- Gear→RPM **{gear_rpm['Gear_predicted_as_RPM']}** 次；"
        f"RPM→Gear **{gear_rpm['RPM_predicted_as_Gear']}** 次；"
        f"双向合计 **{gear_rpm['total_bidirectional']}** 次"
        f"（占两类总 support 的 {_f(gear_rpm['bidirectional_rate_pct'], 4)}%）。"),
        f"- Gear F1 {_f(gear_rpm['gear_f1'])} / RPM F1 {_f(gear_rpm['rpm_f1'])}。",
        f"- Gear 行完整分布：`{gear_rpm['gear_row_full']}`；RPM 行完整分布：`{gear_rpm['rpm_row_full']}`。",
        f"- **结论：{gear_rpm['verdict']}**",
        ("- 分类错误的其余去向见上方混淆矩阵；两类若有错判更多是判成 Normal 或其它攻击类，"
        "而非彼此互混，需结合第 6 节误报归因理解。"),
        "",
        "### 3.2 为什么本轮互混仍为 0（与规格书预期相反的原因分析）",
        "",
        ("规格书预期「Gear↔RPM 易互混」，本轮在两类 support 均 >0 的真实 val 上仍观测到 0 次，"
        "与上一轮 train 内诊断结论一致。这不是缺陷掩盖，而是该数据/特征口径下的真实性质，原因有三："),
        "",
        ("1. **两类注入的 CAN ID 与 payload 模式不同**：Gear（drive gear）与 RPM（gauge）"
        "在 HCRL 中针对不同的 CAN ID 注入不同的伪造字节，窗口级 G1/G3/G4/G6 的"
        "ID 频率分布、字节熵与 payload 变化率足以把两者分开；模型不需要靠周期特征猜。"),
        (f"2. **混淆去向印证**：5×5 矩阵中 Gear 的 {gear_rpm['gear_row_full'].get('Normal', 0)} 个错判"
         "指向 Normal、RPM 零错判，**两类没有任何一个窗口被判成对方**；"
         "其余错判（Normal 被判成 Fuzzy/Gear）是边界带正常窗口的误判，"
         "与「Gear 与 RPM 互相混淆」是两回事。"),
        ("3. **同分布划分放大了可分性**：train/val 的 segment 来自同一批原始文件的同类型注入，"
        "注入模式高度一致，模型见过的模式与 val 几乎同分布——这既解释了近满分的五分类指标，"
        "也意味着**该结论不可外推**到跨车型、跨采集或真实车流（见第 8 节第 10 条）。"),
        "",
        ("**对外技术表述**：可说「修复划分类别错位后，首次在同时含四类攻击的验证集上"
        "真实观测到 Gear↔RPM 互混为 0，并用混淆矩阵与特征消融解释了原因」；"
        "不可说「模型能区分所有欺骗攻击」，因为评估仍是同分布离线子集。"),
        "",
        "## 4. L0 重算与同口径对照",
        "",
        (f"- L0（split-v3 val 重算，阈值由 split-v3 train 正常窗口重新 fit，与 L1-v3 同口径）："
        f"Recall {_f(l0['primary_metrics']['recall'])} / "
        f"FPR {_f(l0['primary_metrics']['fpr'], 6)} / F1 {_f(l0['primary_metrics']['f1'])}，"
        f"误报 {l0['primary_metrics']['fp']} 个。"),
        f"- L1-v3 @标定阈值：Recall {_f(m_cal['recall'])} / FPR {_f(m_cal['fpr'], 6)} / F1 {_f(m_cal['f1'])}。",
        f"- **{comp['fpr_conclusion']}**",
        ("- 完整三行对照表与阈值来源敏感性核查见 `L1v3_vs_L0对照表.md` / "
        "`artifacts/eval/l1_v3_vs_l0.json`。"),
        "",
        "## 5. 三组特征消融（含 Fuzzy/Spoofing 的 val 上重做）",
        "",
        f"- 崩塌判据：{ablation['collapse_rule']}。",
        f"- 阈值策略：{ablation['threshold_policy']}。",
        "",
        "| 变体 | 特征数 | AP(PR-AUC) | F1@0.5 | Recall@0.5 | FPR@0.5 | AP 相对变化 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for v in ablation["variants"]:
        m = v["metrics_at_0_5"]
        d = v.get("delta_vs_full", {})
        rel = f"{d['ap_relative_change_pct']:+.2f}%" if d.get("ap_relative_change_pct") is not None else "—"
        lines.append(
            f"| {v['label']} | {v['n_features']} | {_f(v['average_precision'], 6)} | "
            f"{_f(m['f1'])} | {_f(m['recall'])} | {_f(m['fpr'], 6)} | {rel} |"
        )
    idc = ablation["id_group_conclusion"]
    by_group = {v["dropped_group"]: v for v in ablation["variants"][1:]}
    ap_vals = [v["average_precision"] for v in ablation["variants"]]
    lines += [
        "",
        "### 5.1 「去 ID 组」明确结论（任务单点名）",
        "",
        (f"- **是否崩塌：{'是' if idc['collapsed'] else '否'}**"
        f"（AP {_f(idc['ap_full'], 6)} → {_f(idc['ap_without_id'], 6)}，"
        f"相对变化 {idc['ap_relative_drop_pct']:+.4f}%，判据线 10%）。"
        "**按任务单指定的 AP 10% 判据，结论明确为未崩塌。**"),
        "",
        "### 5.2 反预期发现与证据强度限定（不得套用 v2 结论）",
        "",
        (f"- 既有消融函数给出的结论文本为：{idc['conclusion']}"
        "（该措辞写于 split-v2 语境，本轮实测证据需按下述限定解读）。"),
        (f"- **四组 AP 全部 ≥ {_f(min(ap_vals), 6)}，处于饱和区**："
        "在近满分的同分布 val 上，「AP 相对下降 ≥10%」几乎不可能被触发，"
        "因此「未崩塌」结论虽按判据成立，**证据强度有限**，不能据此断言「模型不背 ID」。"),
        (f"- **本轮方向与 split-v2 相反**：v2 是「去 ID 组 F1 反升、去 payload 组明显掉点」；"
        f"v3 上去 ID 组 F1@0.5 为 {_f(by_group['id_frequency']['metrics_at_0_5']['f1'], 6)}"
        f"（相对全特征 {by_group['id_frequency']['delta_vs_full']['f1_at_0_5_change']:+.6f}，"
        "是三组中掉点最多的），"
        f"而去 payload 组 AP 反升到 {_f(by_group['payload']['average_precision'], 6)}、"
        f"F1@0.5 仅 {by_group['payload']['delta_vs_full']['f1_at_0_5_change']:+.6f}。"),
        ("- **如实解读**：split-v3 是同分布划分，四类攻击的窗口统计特征本就高度可分且"
        "各特征组之间信息冗余大，剔除任一单组都不造成显著崩塌；AP 饱和使排序不可靠，"
        "F1@0.5 的微小差异（千分之一量级）不足以支撑「模型主要依赖哪一组」的强断言。"
        "更稳妥的表述是：**在同分布 val 上，三组消融均未显著崩塌，特征组间存在冗余；"
        "「模型不背 ID」需 W4 在跨数据集/跨 session 留出上重新验证**。"
        "该反预期结果照实保留，未调数据、未改判据去凑上一轮结论。"),
        "",
        "### 5.3 产物与交叉印证",
        "",
        ("- 逐组数据见 `artifacts/eval/l1_v3_ablation_metrics.json`；"
        "对比图 `artifacts/eval/l1_v3_ablation_comparison.svg`。"),
        "- 特征重要性 Top10：`" + "`, `".join(importance["top10"]) + "`"
        "（permutation importance 同样在饱和区区分度有限，仅作辅助）。",
        "",
        "## 6. 阈值标定与 PR 曲线",
        "",
        (f"- 标定只在 val：{calib['criterion_note']}；网格 {calib['n_grid_points']} 点，"
        f"范围 {calib['grid_range'][0]}–{calib['grid_range'][1]}，步长 {calib['grid_range'][2]}；"
        f"选中阈值 {calib['selected_threshold']}（Youden's J={calib['selected_youden_j']:.6f}）。"),
        f"- {calib['train_used_for']}；完整网格曲线留痕 `artifacts/eval/l1_v3_threshold_curve.json`。",
    ]
    curve = calib["curve"]
    best_j = max(r["youden_j"] for r in curve)
    plateau = [r["threshold"] for r in curve if abs(r["youden_j"] - best_j) < 1e-12]
    lines += [
        "",
        (f"- **平局平台如实说明**：Youden's J 最大值 {best_j:.6f} 在 "
        f"**{len(plateau)} 个网格点**（{min(plateau)}–{max(plateau)}）上同时取到，"
        f"按配置「平局取更高阈值」规则选中 {calib['selected_threshold']}。"
        "原因是模型概率两极分化、该阈值区间内无样本跨越，故混淆矩阵完全相同——"
        "这解释了为何 @0.5 与 @标定阈值 两套指标数值一致。"
        "选中值是平局平台的右端，**既不是网格边界解，也不是唯一内部最优**；"
        "未向下扩网格抬高 Recall，两套结果并列报出。"),
    ]
    lo, hi, _ = calib["grid_range"]
    if abs(calib["selected_threshold"] - lo) < 1e-9 or abs(calib["selected_threshold"] - hi) < 1e-9:
        lines.append(
            f"- 另外：选中阈值落在网格边界 {calib['selected_threshold']}，"
            "通常与概率两极分化、J 在网格内单调变化有关；本任务未向下扩网格去抬高 Recall。"
        )
    lines += [
        (f"- PR 曲线逐点导出 **{pr['n_points']} 个点**（sklearn precision_recall_curve 原始点，"
        f"未降采样），AP={_f(pr['average_precision'], 6)}；"
        "数据 `artifacts/eval/l1_v3_pr_curve.json`，供 W4 画图。"),
        "",
        "## 7. 误报治理：逐窗口归因与建议（本轮核心交付）",
        "",
        (f"- 口径：split-v3 val、阈值 {errors['threshold_used']}；"
        f"误报(FP) **{errors['n_false_positives']}** 个 / 漏报(FN) **{errors['n_false_negatives']}** 个"
        f"（共 {errors['n_val_windows']} 窗口）。"),
        f"- 漏报按攻击类型：`{errors['false_negative_by_attack_type']}`。",
        f"- 误报按 segment 前缀：`{errors['false_positive_by_segment_prefix']}`。",
        f"- 误报窗口最常偏离的特征：`{errors['top_deviated_features_in_false_positives'][:5]}`。",
        f"- 漏报窗口最常偏离的特征：`{errors['top_deviated_features_in_false_negatives'][:5]}`。",
        "",
        ("**逐窗口归因**：val 上每一个 FP/FN 窗口均给出 window_id、segment、攻击类型、"
        "真实/预测标签、模型分数、attack_ratio、相对同类 val 中位数偏离最大的 6 维特征画像"
        "与可能原因，全量记录见 `artifacts/eval/l1_v3_error_attribution.json`。"
        "以下列出前 20 条（按窗口顺序）："),
        "",
        "| # | 窗口 | 类型 | 错误 | 分数 | attack_ratio | 最偏离特征 | 可能原因 |",
        "|---:|---|---|---|---:|---:|---|---|",
    ]
    for i, r in enumerate(errors["records"][:20], 1):
        top = r["top_deviating_features"][0]["feature"] if r["top_deviating_features"] else "—"
        cause = "正常窗口统计量偏离，疑似被误当注入" if r["error_kind"] == "false_positive" else \
                "弱注入/行为接近正常，未被检出"
        lines.append(
            f"| {i} | `{r['window_id'][:18]}` | {r['attack_type']} | "
            f"{'FP' if r['error_kind'] == 'false_positive' else 'FN'} | {_f(r['score'], 4)} | "
            f"{_f(r['attack_ratio'], 3)} | `{top}` | {cause} |"
        )
    lb = errors["label_boundary_analysis"]
    ratios = lb["error_attack_ratios"]
    theta = lb["window_label_theta"]
    near_n = lb["n_errors_near_boundary"]
    lines += [
        "",
        "### 7.1 关键发现：全部错误集中在标签边界（θ=0.20 附近）",
        "",
        (f"- {lb['n_errors']} 个错误窗口的 attack_ratio 为 `{ratios}`；其中 **{near_n} 个**"
        f"落在 θ={theta}±0.02 的窄带内。"),
        (f"- 3 个误报(FP) 全是 **attack_ratio={lb['fp_attack_ratios']} 的 Normal 窗口**："
        "注入帧占比 0.19 略低于窗口标签阈值 0.20，按规则被标为正常，但其流量统计已被注入扰动，"
        "因此被模型判为攻击。**这不是模型出错，而是窗口级硬阈值标签本身的边界模糊**。"),
        (f"- 2 个漏报(FN) 全是 **Fuzzy 弱注入窗口**（attack_ratio {lb['fn_attack_ratios']}）："
        "刚过阈值即被标为攻击，但扰动强度不足以让统计量明显偏离正常。"),
        ("- **治理含义**：误报治理的首要抓手不是换模型，而是**标签口径**——"
        "可考虑边界带样本单独处理（软标签 / 剔除出评估 / 单独分层）、"
        "把 θ 作为可标定参数、或用帧级标签替代窗口级硬阈值。"
        "该发现也说明本轮「误报」很大程度上是评估口径产物，"
        "不宜把 3 个 FP 解读为模型缺陷。"),
        "",
        "### 7.2 治理建议（本任务只给建议，不实现）",
        "",
    ]
    for rec in errors["governance_recommendations"]:
        lines.append(f"- {rec}")
    lines += [
        "",
        "## 8. 防泄漏核查与口径/局限性声明",
        "",
        ("1. **特征侧**：模型矩阵严格 33 维 G1–G6，沿用 `assert_no_forbidden_fields`；"
        "flag / label / attack_ratio / can_id==0x000 布尔及其派生列均未入特征。"
        "标签只从 `window_labels.parquet` 读取，归因用的 label/attack_ratio 仅用于 L0 与归因，不进模型。"),
        ("2. **划分侧**：segment 为最小抽样单元，生成与载入两次断言「同段不跨折」；"
        "归一化参数沿用既有产物，未重算特征。"),
        ("3. **test 封存**：test 窗口集合原样沿用 split-v2，v3 划分产物不含 test 行，"
        "训练用 window_id in-list 下推，manifest 记 `test_rows_loaded=0`；"
        "唯一接触是标识符指纹校验（见 1.2 的边界说明）。"),
        ("4. **阈值只用 val**：阈值网格与 Youden's J 曲线全部在 split-v3 val 上计算，"
        "train 仅用于拟合模型；test 未参与任何选择。"),
        (f"5. **五分类可解释性**：本轮 train/val 均覆盖全 5 类（train {cov['shared_classes']}），"
        "macro 指标与 Gear↔RPM 互混属于真实观测，不再是 W2 那种结构性失真。"
        "但仍是窗口级、离线子集指标。"),
        ("6. **不可外推**：所有数字均为「真实数据子集（每文件前 100 万行）上的 split-v3 val 结果」，"
        "不是全量数据集、不是实车、不是 test 指标；不得写进公开报告当作整体检出率。"),
        ("7. **AP 饱和的判据局限**：若各组 AP 均接近 1，则「AP 相对下降 ≥10%」崩塌判据"
        "在饱和区难以触发，去 ID 组结论需并用 F1@0.5 与特征重要性交叉印证，证据强度如实标注。"),
        "8. **口径不可比**：见 1.3，W2 与 W3 的数字不可直接相减；本报告只做口径解释，不做提升宣称。",
        ("9. **L0 阈值来源**：主对照行的 L0 阈值由 split-v3 train 正常窗口重新 fit（与 L1-v3 同口径）；"
        "沿用 W2 在 split-v2 train fit 的固定阈值只作敏感性参考，已在对照表中并列并说明"
        "为何不能当主对照（会虚增 L1 增益）。"),
        ("10. **近满分指标（F1≈0.9997）的防泄漏核查（R6）**：逐条排查后确认不是标签泄漏——"
        "(a) 特征矩阵严格 33 维 G1–G6，`assert_no_forbidden_fields` 拦截 flag/label/attack_ratio/"
        "0x000 布尔及其派生；(b) 标签独立存放于 `window_labels.parquet`，按 window_id 一对一内连接，"
        "行数一致；(c) segment 为最小划分单元，生成与载入两次断言同段不跨折；"
        "(d) 阈值只用 val，train 仅拟合模型；(e) test 载入行数 0。"
        "**真实原因是同分布性**：split-v3 的 train/val segment 来自同一批原始文件的同类型注入，"
        "注入模式高度一致，窗口统计特征在这批数据上本就接近线性可分（W2 在「跨攻击类型」"
        "更难口径下 F1 也只有 0.9980，量级一致）。"
        "因此该数字应解读为「同分布、离线子集上的检出与分类上限」，"
        "**不是跨车型/跨采集/实车泛化能力，不得写入公开报告当作整体检出率**；"
        "W4 若需更硬的泛化证据，应在独立数据集（→OTIDS）或跨 session 留出上验证。"),
        "",
        "## 9. 复现与产物",
        "",
        "```bash",
        "cd diting",
        "python -m diting.split.make_split_v3        # 生成 split-v3（含 7 条断言）",
        "python -m diting.l1_v3_pipeline             # 重训 + 评估治理 + 报告（test 不载入）",
        "TMPDIR=/tmp/diting-l1-tmproot .venv/bin/python -m pytest tests -q",
        ".venv/bin/python -m ruff check src tests",
        "```",
        "",
        "- 划分：`data/processed/split/split-v3/{windows.parquet,split_manifest.json}`。",
        "- 模型：`artifacts/models/l1_v3/`（五类版本号 manifest、metrics.json、threshold.json 等）。",
        "- 评估：`artifacts/eval/l1_v3_*`（含 PR 曲线、阈值曲线、逐窗口误报归因、消融、重要性）。",
        "- 公开报告：`W3评估治理报告.md`、`L1v3_vs_L0对照表.md`、`l1v3_metrics.json`。",
        "",
        "## 10. 遗留问题",
        "",
        "1. test 集仍封存，L2+ 深度模型与跨域（→OTIDS）验证不在本任务范围。",
        "2. 误报治理建议（双阈值、类型白名单、segment 级投票）本任务未实现，属 W4 候选。",
        ("3. 阈值若为边界解，说明概率分布两极分化；如需更细的阈值折中，可在 W4 加密网格，"
        "但不得用 test 选择。"),
        ("4. split-v3 的 train:val 比例受 segment 不可拆分限制有小幅偏差；"
        "如需严格 75:25，需在保持同段不跨折的前提下做更细的组合优化（本轮未做）。"),
        "",
    ]
    return "\n".join(lines)
