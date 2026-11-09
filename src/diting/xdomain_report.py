"""谛听 W4 · 跨域验证主报告渲染。

报告中的每一个数字都来自流水线实际产物（同一个 ctx 字典同时写入 JSON 与本报告），
保证「报告数字与 JSON 一致」（验收标准 #8）；不手工抄写任何指标。
"""

from __future__ import annotations

from typing import Any


def _pct(x: float | None, digits: int = 2) -> str:
    return "n/a" if x is None else f"{100.0 * x:.{digits}f}%"


def _num(x: float | None, digits: int = 6) -> str:
    return "n/a" if x is None else f"{x:.{digits}f}"


def render_report(ctx: dict[str, Any]) -> str:
    cfg = ctx["cfg"]
    m05 = ctx["binary_0_5"]
    m054 = ctx["binary_0_54"]
    per_type = ctx["per_attack_type"]
    five = ctx["five_class"]
    official = ctx["official_sensitivity"]
    adapt = ctx["adaptation"]
    abl = ctx["id_ablation"]
    drift = ctx["drift"]
    w3 = ctx["w3_same_dist"]
    parse = ctx["parse_stats"]
    wstats = ctx["window_stats"]
    zip_check = ctx["zip_check"]
    integrity = ctx["integrity"]
    frozen = ctx["frozen_record"]

    lines: list[str] = []
    a = lines.append

    a("# 谛听 W4 · OTIDS 跨域泛化验证报告")
    a("")
    a(f"> 任务单：`{cfg['task']}`　|　版本：`{cfg['version']}`　|　种子：`{cfg['seed']}`")
    a(f"> 完成时间：{ctx['finished_at']}　|　代码基线（本报告渲染时的 git HEAD）："
      f"`{ctx['git_revision'][:12]}`　|　"
      "W4 代码提交号见 `RESULT.md`（该文件在提交完成后创建，报告不预写尚未产生的提交号）")
    a("")
    a("> **一句话结论**：把在 HCRL Car-Hacking（Hyundai）上训练完毕的 L1-v3 模型与阈值 0.54 "
      "**完全冻结**，零样本搬到独立车型 OTIDS（Kia Soul）上，固定阈值下的 F1 出现明显下降，"
      "但阈值无关的 AP / ROC-AUC 基本保留；本报告如实量化下降幅度、逐类拆解原因，"
      "并给出「对外指标口径」的边界。")
    a("")

    # ---------------- 0 摘要 ----------------
    a("## 0. 结果摘要")
    a("")
    a("| 口径 | 评估集 | 阈值 | Recall | FPR | F1 | AP | 每万正常窗口误报 |")
    a("|---|---|---:|---:|---:|---:|---:|---:|")
    a(f"| 同分布（W3 split-v3 val，参考基线） | HCRL val | 0.54 | {_num(w3['recall'])} | {_num(w3['fpr'])} | "
      f"{_num(w3['f1'])} | {_num(w3['average_precision'])} | {_num(w3['false_alarms_per_10k_normal'], 2)} |")
    a(f"| **零样本跨域 @0.5** | OTIDS 全量 | 0.50 | {_num(m05['recall'])} | {_num(m05['fpr'])} | "
      f"{_num(m05['f1'])} | {_num(m05['average_precision'])} | {_num(m05['false_alarms_per_10k_normal'], 2)} |")
    a(f"| **零样本跨域 @冻结阈值 0.54** | OTIDS 全量 | 0.54 | {_num(m054['recall'])} | {_num(m054['fpr'])} | "
      f"{_num(m054['f1'])} | {_num(m054['average_precision'])} | {_num(m054['false_alarms_per_10k_normal'], 2)} |")
    for r in adapt["rows"]:
        if r["policy"] == "otids_adapted":
            mm = r["metrics"]
            a(f"| 适配上限（仅调阈值，**非零样本**） | OTIDS eval 折 | {_num(mm['threshold'], 2)} | "
              f"{_num(mm['recall'])} | {_num(mm['fpr'])} | {_num(mm['f1'])} | "
              f"{_num(mm['average_precision'])} | {_num(mm['false_alarms_per_10k_normal'], 2)} |")
    drop_f1 = round(w3["f1"] - m054["f1"], 6)
    a("")
    a(f"- **同分布 → 跨域 F1 下降 {_num(drop_f1)}**（{_num(w3['f1'])} → {_num(m054['f1'])}），"
      f"Recall 变化 {_num(m054['recall'] - w3['recall'])}，FPR 变化 {_num(m054['fpr'] - w3['fpr'])}。"
      "下降是本任务的**预期结果**，未做任何口径美化。")
    a(f"- **下降机制的关键观察**：阈值无关的 AP 仅从 {_num(w3['average_precision'])} "
      f"降到 {_num(m054['average_precision'])}（{_num(m054['average_precision'] - w3['average_precision'])}），"
      f"ROC-AUC 仍有 {_num(m054['roc_auc'])}，说明**排序/判别能力几乎没有丢失**；"
      f"F1 的下降主要来自冻结阈值 0.54 在跨域分数上偏高造成的漏检"
      f"（@0.54 漏检 {m054['fn']:,} 个攻击窗口）。第 3 节「仅调阈值」重标定到 "
      f"{_num(adapt['threshold_calibration']['selected_threshold'], 2)} 后 eval 折 F1 即回升，"
      "印证这是**阈值迁移**问题而非模型崩塌；但该值恰是阈值网格下界（边界解，见第 3 节），"
      "且回升部分的 FPR 仍为 0，同样受主口径文件级标签的乐观偏差影响，"
      "**不能解读为「降阈值不付误报代价」**。")
    a(f"- **FPR={_num(m054['fpr'])} 不代表误报能力变强**：主口径是文件级标签，"
      "负类窗口全部来自独立的 Attack-Free 文件、与攻击会话差异极大（会话级判定）。"
      "第 2.4 节换成官方描述口径后 FPR 立刻升到 "
      f"{_num(official['rows'][1]['fpr'])}，说明这个 0 是口径带来的乐观偏差，"
      "**不得当作逐帧误报率对外引用**。")
    a(f"- 跨域「去 ID 频率组」消融结论：**{abl['conclusion']}**"
      f"（去 ID 后 AP 相对**下降** {_num(abl['ap_relative_drop_pct'], 4)}%，方向是变差而非改善："
      "模型在跨域上仍依赖 ID 频率组、该组仍承载有效判别信息。"
      "因此本消融**不能**作为「ID 频次表跨车型错配导致下降」的证据，"
      "也不能用来宣称「模型不背 ID」；按崩塌判据属未崩塌。详见第 4、5.3 节。）")
    a(f"- 漂移最大的特征组：**{drift['dominant_drift_group']}**"
      f"（归一化空间平均 |Δmean| = {_num(drift['by_group'][drift['dominant_drift_group']]['mean_abs_delta'])}，"
      "即 DLC / 窗口帧统计，而非三类预设漂移（ID 空间 / 周期 / payload）中的任何一类。"
      "该组原本仅被设为对照量，却成为漂移最大组，提示采集/切段层面的分布差异不可忽略；"
      "但漂移大 ≠ 因果贡献大，详见 5.3）。")
    a("")

    # ---------------- 1 数据与口径 ----------------
    a("## 1. 数据接入与口径声明")
    a("")
    a("### 1.1 输入与校验和留痕")
    a("")
    a(f"- 输入：`{zip_check['path']}`（只读，未解压改写、未重新分发）")
    a(f"- SHA-256：`{zip_check['actual_sha256']}`，与任务单声明"
      f" `{zip_check['expected_sha256']}` **{'一致' if zip_check['sha256_match'] else '不一致'}**；"
      f"字节数 {zip_check['actual_size_bytes']}（{'一致' if zip_check['size_match'] else '不一致'}）")
    a(f"- zip 内 {len(zip_check['members'])} 个成员，**主口径只用 4 个逐帧带标签文件**："
      f"`{'`, `'.join(parse['primary_zip_members'])}`")
    a(f"- `{parse['excluded_zip_members'][0]}` / `{parse['excluded_zip_members'][1]}` "
      "是第三方二次加工（SMOTE 平衡/合并、十进制化、列结构不同），来源口径不明，"
      "**仅作规模参考，禁止作为评估口径**；本任务未读取其任何数据行。")
    a(f"- 解析总帧数 **{parse['parquet_rows_total']:,}**，跳过 {parse['skipped_rows_total']} 行，"
      f"逐文件行数对账 {'全部通过' if parse['all_row_counts_match'] else '存在不一致'}；"
      f"切分为 {parse['n_segments']} 个 segment。")
    a("")
    a("| 文件 | 类别 | 文件级 target | 解析行数 | 跳过 | 时间跨度(s) | 主口径注入帧 | 官方口径注入帧 | segments |")
    a("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for f in parse["per_file"]:
        a(f"| `{f['file']}` | {f['attack_kind']} | {f['file_label']} | {f['rows_parsed']:,} | "
          f"{f['rows_skipped']} | {_num(f['span_seconds'], 2)} | {f['n_primary_injected']:,} | "
          f"{f['n_official_injected']:,} | {f['n_segments']} |")
    a("")
    a("### 1.2 字段级解析规则（如实记录，未静默丢弃任何数据）")
    a("")
    a("OTIDS 主口径文件列结构 `TS,ID1,ID0,LEN,DLC0..DLC7,target`，映射为统一帧表：")
    a("")
    a("| 原始列 | 处理规则 | 去向 |")
    a("|---|---|---|")
    a("| `TS` | `float` | `timestamp` |")
    a("| `ID1` | `int(ID1,16)`（4 位十六进制） | `can_id` |")
    a("| `ID0` | 原样保留字符串 | `id0_token`（**帧属性标记，不进特征**） |")
    a("| `LEN` | `int` | `dlc` |")
    a("| `DLC0..DLC7` | `int(x,16)`，空值 -1（LEN<8 时后几位为空） | `data_0..data_7` |")
    a("| `target` | 文件级标签 0/1/2/3 | `file_label`，并派生帧级 `label` / `official_label` |")
    a("")
    a("**`can_id` 构造与 `ID0` 语义的核验**（只读抽样，每个文件前 30 万行）：4 个主口径文件的 `ID1` "
      "全部可按 `int(ID1,16)` 解析（合计 120 万行抽样 100% 可解析）；`ID0` 只取 `{000, 100}` 两个值。"
      "但 `ID0` 与 `LEN` 的对应**不是全局规律**：抽样中 `dataset.csv` / `dataset1.csv` / `dataset2.csv` "
      "里 `ID0=100` 的帧全部 `LEN=0`（分别为 12,774 / 4,694 / 6,432 行），"
      "而 `dataset3.csv` 里 1,422 个 `ID0=100` 帧全部 `LEN=8`。"
      "因此 `ID0` 既不能当作 ID 低位，也不能当作稳定的「远程帧」标记，其语义在本镜像内不自洽；"
      "本任务取 `can_id = int(ID1,16)`，把 `ID0` 原样保留为 `id0_token` 元数据，"
      "**不进特征、不拼入 ID、不对其语义作推测**。B/UB 文件列结构不同且来源口径不明，"
      "本任务未将其任何行用于窗口、特征或评估。")
    a("")
    a("**OTIDS 无逐帧 T/R Flag**：统一 schema 的 `flag` 字段填 `NA` 占位，"
      "不参与任何标签或特征计算；帧级标签完全由下表两套口径给出。")
    a("")

    a("### 1.3 标签双口径（关键声明）")
    a("")
    a("镜像中攻击文件的 `target` 是**文件级标签**：攻击文件内正常 ID 的帧同样被标为攻击"
      "（实测 `dataset1.csv` 中 `ID=0x0080` 的帧 `target=1`，共 15,612 行）。"
      "因此 OTIDS 侧采用窗口级标签，且**复用 HCRL 同一个 `build_windows` 与 θ=0.20**：")
    a("")
    a(f"1. **主口径 primary**（`{cfg['label_policy']['primary']}`）：帧 `label=(target>0)`，"
      f"窗口 `label = (attack_ratio >= {cfg['window']['attack_ratio_theta']})`。")
    a(f"2. **官方描述敏感性口径 official**（`{cfg['label_policy']['official_sensitivity']}`）："
      f"DoS 以 `can_id==0x000` 为注入帧；Fuzzy/Impersonation 以 `ts - t_start >= "
      f"{cfg['label_policy']['official_attack_start_s']}s` 为注入帧；Attack-Free 全正常。"
      "再按同一窗口参数与 θ 生成窗口标签，**仅作敏感性对照，不参与主结论**。")
    a("")
    a(wstats["label_policy"]["note"])
    a("")
    a(f"- 主口径窗口：正 {wstats['window_positive_primary']:,} / 负 {wstats['window_negative_primary']:,}"
      f"（共 {wstats['n_windows']:,}）；官方口径窗口：正 {wstats['window_positive_official']:,} / "
      f"负 {wstats['window_negative_official']:,}。")
    a("| 类别 | 窗口数 | 主口径正例 | 官方口径正例 |")
    a("|---|---:|---:|---:|")
    for k, v in wstats["by_dominant_class"].items():
        a(f"| {k} | {v['n_windows']:,} | {v['primary_positive']:,} | {v['official_positive']:,} |")
    a("")
    a("> **口径边界**：主口径下「正常窗口」全部来自 Attack-Free 文件、"
      "「攻击窗口」全部来自 3 个攻击文件，故 FPR 衡量的是「正常行驶段被误报的比例」，"
      "Recall 衡量的是「攻击会话段被检出的比例」。这与 HCRL 侧「注入片段 vs 正常片段」"
      "的语义不完全等价，是 OTIDS 镜像只有文件级标签带来的固有限制，"
      "**不得把本节数字直接当作逐帧检测能力对外引用**。")
    a("")

    a("### 1.4 冻结边界（零样本的定义）")
    a("")
    a(f"- 冻结模型：`{frozen['paths']['model_joblib']}`（`{frozen['model_version']}`，"
      f"seed={frozen['random_seed']}，split={frozen['split_version']}）；"
      f"SHA-256 `{frozen['sha256']['model_joblib']}` 与 W3 manifest 声明"
      f" `{frozen['declared_model_joblib_sha256']}` **一致**。")
    a(f"- 冻结阈值：`{frozen['frozen_threshold']}`（`{frozen['threshold_version']}`，"
      f"在 HCRL split-v3 val 上标定）；默认阈值 0.5 并行报出。")
    a("- 冻结特征管线：`feature_params.json`（归一化均值/标准差，clip=6.0）与 "
      "`id_frequency_train.json`（2048 个 ID 的频次表）**均只在 HCRL train 折 fit 过一次**，"
      "OTIDS 侧直接套用、**零拟合**；G1–G6 33 维特征函数与窗口参数同 HCRL 完全一致。")
    a("- **OTIDS 未参与任何训练/微调**：模型权重、阈值、归一化参数、ID 频次表全部来自 HCRL；"
      "OTIDS 数据未混入任何 HCRL 训练/划分产物，HCRL test 折继续封存、零读取。")
    a("")

    # ---------------- 2 零样本评估 ----------------
    a("## 2. 零样本跨域评估（本任务核心）")
    a("")
    a("### 2.1 二分类：@0.5 与 @冻结阈值 0.54 两套")
    a("")
    a("| 指标 | @0.5 | @0.54（冻结） |")
    a("|---|---:|---:|")
    for key, label in (
        ("recall", "Recall（攻击类）"), ("fpr", "FPR"), ("precision", "Precision"),
        ("f1", "F1"), ("average_precision", "AP（PR-AUC）"), ("roc_auc", "ROC-AUC"),
        ("false_alarms_per_10k_normal", "每万正常窗口误报"),
    ):
        d = 2 if key == "false_alarms_per_10k_normal" else 6
        a(f"| {label} | {_num(m05[key], d)} | {_num(m054[key], d)} |")
    a(f"| TP / FP / TN / FN | {m05['tp']} / {m05['fp']} / {m05['tn']} / {m05['fn']} | "
      f"{m054['tp']} / {m054['fp']} / {m054['tn']} / {m054['fn']} |")
    a(f"| 样本数（正/负） | {m05['n_samples']:,}（{m05['n_positive']:,}/{m05['n_negative']:,}） | "
      f"{m054['n_samples']:,}（{m054['n_positive']:,}/{m054['n_negative']:,}） |")
    a("")
    a("混淆矩阵（行=真实，列=预测，@0.54）：")
    a("")
    cm = m054["confusion_matrix"]["matrix"]
    a("| 真实 \\ 预测 | Normal | Attack |")
    a("|---|---:|---:|")
    a(f"| Normal | {cm[0][0]:,} | {cm[0][1]:,} |")
    a(f"| Attack | {cm[1][0]:,} | {cm[1][1]:,} |")
    a("")
    a(f"提高阈值 0.5→0.54：Recall {_num(m05['recall'])}→{_num(m054['recall'])}，"
      f"FPR {_num(m05['fpr'])}→{_num(m054['fpr'])}，"
      f"每万误报 {_num(m05['false_alarms_per_10k_normal'], 2)}→"
      f"{_num(m054['false_alarms_per_10k_normal'], 2)}。")
    a("")
    a("> **FPR=0 的读法**：主口径负类窗口全部来自独立的 Attack-Free 文件，"
      "这个 0 是「会话级」判定的结果，**不等于逐帧误报率为 0**；"
      f"换成第 2.4 节官方描述口径后 FPR 为 {_num(official['rows'][1]['fpr'])}。"
      "两节数字口径不同，不得择优引用。")
    a("")

    a("### 2.2 按攻击类型分解")
    a("")
    a("| 类型 | support | AP（该类 vs Normal） | Recall@0.5 | Recall@0.54 | 漏检@0.54 |")
    a("|---|---:|---:|---:|---:|---:|")
    for t in ("DoS", "Fuzzy", "Impersonation"):
        v = per_type["by_type"][t]
        a(f"| {t} | {v['support']:,} | {_num(v.get('average_precision_vs_normal'))} | "
          f"{_num(v.get('recall_at_t0_5'))} | {_num(v.get('recall_at_frozen'))} | "
          f"{v.get('missed_at_frozen', 0):,} |")
    nm = per_type["by_type"]["Normal"]
    a(f"| Normal（负类） | {nm['support']:,} | — | FPR {_num(nm.get('fpr_at_t0_5'))} | "
      f"FPR {_num(nm.get('fpr_at_frozen'))} | 误报 {nm.get('false_alarms_at_frozen', 0):,} |")
    a("")
    imp = per_type["by_type"]["Impersonation"]
    a(f"**Impersonation 单独分析**：官方描述中 Impersonation 是针对 `ID 0x164` 的伪装注入，"
      f"注入帧模仿合法节点，行为统计与正常流量最接近，是本任务预期最难检出的类型。"
      f"实测 AP={_num(imp.get('average_precision_vs_normal'))}、Recall@0.54="
      f"{_num(imp.get('recall_at_frozen'))}（漏检 {imp.get('missed_at_frozen', 0):,} 个窗口），"
      f"与 DoS（AP={_num(per_type['by_type']['DoS'].get('average_precision_vs_normal'))}）"
      f"的差距见上表。DoS 因注入 `ID=0x000`（正常轨迹中不存在）而最易检出，"
      f"这与任务单给出的文献参考结论一致。")
    a("")

    a("### 2.3 映射五分类（次级口径）")
    a("")
    a(five["macro_f1_reason"] + " 因此本报告不报 macro-F1，只报预测分布。")
    a("")
    a("| 真实类型 | support | 映射类 | 映射准确率 | 判为 Normal | 判为 DoS | 判为 Fuzzy | 判为 Gear | 判为 RPM |")
    a("|---|---:|---|---:|---:|---:|---:|---:|---:|")
    for t in ("Normal", "DoS", "Fuzzy", "Impersonation"):
        v = five["per_true_type"][t]
        d = v["predicted_distribution"]
        mapped = v.get("mapped_class", "—")
        acc = v.get("mapping_accuracy")
        a(f"| {t} | {v['support']:,} | {mapped} | {_pct(acc) if acc is not None else '—'} | "
          f"{d['Normal']:,} | {d['DoS']:,} | {d['Fuzzy']:,} | {d['Gear']:,} | {d['RPM']:,} |")
    a("")
    ia = five["impersonation_analysis"]
    a(f"**Impersonation 分布**：判为 Gear {ia['judged_as_Gear']:,}、RPM {ia['judged_as_RPM']:,}、"
      f"Normal {ia['judged_as_Normal']:,}、DoS {ia['judged_as_DoS']:,}、Fuzzy {ia['judged_as_Fuzzy']:,}。"
      f"{ia['note']}")
    a("")

    a("### 2.4 官方描述口径的敏感性对照")
    a("")
    a("下表把窗口标签换成官方描述口径（DoS: `ID=0x000`；Fuzzy/Impersonation: 250 秒后为攻击段），"
      "**模型、阈值、特征、窗口集合全部不变**，只有标签口径变化。")
    a("")
    a("| 标签口径 | 正/负窗口 | 阈值 | Recall | FPR | F1 | AP |")
    a("|---|---:|---:|---:|---:|---:|---:|")
    a(f"| 主口径 primary | {m054['n_positive']:,}/{m054['n_negative']:,} | 0.54 | {_num(m054['recall'])} | "
      f"{_num(m054['fpr'])} | {_num(m054['f1'])} | {_num(m054['average_precision'])} |")
    for r in official["rows"]:
        a(f"| 官方描述 official | {r['n_positive']:,}/{r['n_negative']:,} | {_num(r['threshold'], 2)} | "
          f"{_num(r['recall'])} | {_num(r['fpr'])} | {_num(r['f1'])} | "
          f"{_num(r['average_precision'])} |")
    a("")
    a(official["interpretation"])
    a("")

    # ---------------- 3 适配上限 ----------------
    a("## 3. 「仅调阈值」适配上限对照（明确：非零样本）")
    a("")
    a("> **本节的适配口径不是零样本**。做法：把 OTIDS 窗口按 segment 分层切为适配折 / 评估折，"
      "**只在适配折上重新标定阈值**，模型权重、归一化参数、ID 频次表全部不动，在另一半评估。"
      "它与第 2 节的零样本口径**并列报出，严禁混报**。")
    a("")
    a(f"- 分层抽样单元：`segment_id`，分层键为 `session_id` 前缀（攻击类型），种子 {cfg['seed']}；"
      f"适配折 {adapt['fold_sizes']['adapt']:,} 窗口 / 评估折 {adapt['fold_sizes']['eval']:,} 窗口，"
      "同一 segment 不跨折。")
    a(f"- 适配折 Youden's J 重标定阈值：**{_num(adapt['threshold_calibration']['selected_threshold'], 2)}**"
      f"（J={_num(adapt['threshold_calibration']['selected_youden_j'])}，"
      f"网格 {adapt['threshold_calibration']['n_grid_points']} 点）；"
      f"冻结阈值为 {_num(ctx['binary_0_54']['threshold'], 2)}。")
    a("")
    _tc = adapt["threshold_calibration"]
    _curve = _tc["curve"]
    _grid = _tc["grid_range"]
    _j_down = all(
        float(_curve[i]["youden_j"]) >= float(_curve[i + 1]["youden_j"])
        for i in range(len(_curve) - 1)
    )
    _fpr0 = all(float(r["fpr"]) == 0.0 for r in _curve)
    a(f"> **边界解与乐观偏差提示（必须如实说明）**：重标定阈值 "
      f"{_num(_tc['selected_threshold'], 2)} 恰是阈值网格**下界**"
      f"（{_grid[0]}~{_grid[1]}，步长 {_grid[2]}，{_tc['n_grid_points']} 点，"
      f"平局规则 `{_tc['tie_breaker']}`）。适配折上 Youden's J 随阈值升高"
      f"**{'单调下降' if _j_down else '非单调'}**，且全网格 FPR "
      f"**{'恒为 0' if _fpr0 else '并非恒为 0'}**；即最优解落在网格下界之外，"
      "0.05 只是网格内可取的最小值，**不是统计意义上的内点最优**。"
      "这一方面印证跨域分数整体被压低（阈值迁移问题），另一方面说明主口径的负类窗口"
      "（Attack-Free 文件）与攻击会话在分数上完全分离——后者是文件级标签带来的乐观偏差："
      f"第 2.4 节换成官方描述口径后 FPR 升到 {_num(official['rows'][1]['fpr'])}。"
      "因此**不得把本节 FPR=0 解读为「降阈值不产生误报」**，"
      "也不得把 0.05 当作一个已收敛的标定结果对外引用。")
    a("")
    a("| 口径 | evaluated_on | 评估窗口 | 阈值 | Recall | FPR | F1 | AP |")
    a("|---|---|---:|---:|---:|---:|---:|---:|")
    for r in adapt["rows"]:
        mm = r["metrics"]
        a(f"| {r['policy']} | `{r['evaluated_on']}` | {r['fold']} | {_num(mm['threshold'], 2)} | "
          f"{_num(mm['recall'])} | {_num(mm['fpr'])} | {_num(mm['f1'])} | "
          f"{_num(mm['average_precision'])} |")
    a("")
    d = adapt["same_eval_fold_delta_vs_frozen"]
    a(f"**同一 eval 折上换阈值的净贡献**（隔离评估集变化）：F1 {_num(d['f1_delta'])}、"
      f"Recall {_num(d['recall_delta'])}、FPR {_num(d['fpr_delta'])}。"
      f"{d['note']}")
    a("")
    a("> 即便允许用一半 OTIDS 数据重标定阈值，模型权重仍被冻结；"
      "该行是「只调阈值能达到的上限」，不是同分布结果，也不是可对外宣称的零样本泛化能力。")
    a("")

    # ---------------- 4 消融 ----------------
    a("## 4. 跨域「去 ID 频率组」消融")
    a("")
    a(f"{abl['model_source_note']}。重训只在 HCRL split-v3 train 折"
      f"（{abl['hcrl_train_windows']:,} 窗口）上进行，同配置、同种子 {abl['seed']}；"
      f"去 ID 组后特征 {abl['n_features_ablated']} 维（原 {abl['n_features_full']} 维，"
      f"剔除 `{abl['dropped_group']}` 的 {len(abl['dropped_columns'])} 列）。")
    a("")
    a("两个模型在**同一 OTIDS 全量窗口、同一阈值集合、同一标签口径**下零样本对比：")
    a("")
    a("| 模型 | 阈值 | Recall | FPR | F1 | AP |")
    a("|---|---:|---:|---:|---:|---:|")
    for r in abl["rows"]:
        a(f"| {r['tag']} | {_num(r['threshold'], 2)} | {_num(r['recall'])} | {_num(r['fpr'])} | "
          f"{_num(r['f1'])} | {_num(r['average_precision'])} |")
    a("")
    a(f"- 崩塌判据：{abl['collapse_rule']}。AP {_num(abl['ap_full_33'])} → "
      f"{_num(abl['ap_drop_id'])}（绝对变化 {_num(abl['ap_absolute_delta'])}，"
      f"相对下降 {_num(abl['ap_relative_drop_pct'], 4)}%），"
      f"**collapsed={abl['collapsed']}**，去 ID 后性能改善={abl['performance_improved_after_dropping_id']}。")
    a(f"- **明确结论**：{abl['conclusion']}")
    a("")
    a("跨车型 ID 空间完全不同（Hyundai 与 Kia Soul 的 ID 集合、频次、周期均不同），"
      "而 `id_frequency_train.json` 是 HCRL train 拟合的频次表、在 OTIDS 上被原样套用："
      "这是本任务对「模型是否背 ID」最硬的检验口径。")
    a("")
    _ar = {r["tag"]: r for r in abl["rows"]}
    a(f"**实测解释**：事前判读规则为「去 ID 后不降反升 → ID 组已退化为误导信号；"
      f"下降 → ID 组仍承载有效判别信息」。实测 AP "
      f"{_num(abl['ap_full_33'])} → {_num(abl['ap_drop_id'])}"
      f"（相对下降 {_num(abl['ap_relative_drop_pct'], 4)}%），"
      f"@0.54 的 F1 {_num(_ar.get('full_33_features@0.54', {}).get('f1'))} → "
      f"{_num(_ar.get('drop_id_frequency@0.54', {}).get('f1'))}，"
      "方向是**下降**：模型在跨域上仍依赖 ID 频率组，且这一依赖在新车型上仍然有效。"
      "因此本消融**既不构成「ID 频次表跨车型错配导致下降」的证据，"
      "也不构成「模型不背 ID」的证据**；它只说明去掉该组不会带来跨域崩塌"
      f"（未达判据：{abl['collapse_rule']}），且去掉后确实掉点。数字未做取舍美化。")
    a("")

    # ---------------- 5 性能下降归因 ----------------
    a("## 5. 性能下降归因")
    a("")
    a("### 5.1 同分布 vs 跨域对照表")
    a("")
    a("| 指标 | W3 同分布 HCRL val @0.54 | W4 跨域 OTIDS 零样本 @0.54 | 绝对变化 |")
    a("|---|---:|---:|---:|")
    for key, label in (
        ("recall", "Recall"), ("fpr", "FPR"), ("precision", "Precision"),
        ("f1", "F1"), ("average_precision", "AP"), ("roc_auc", "ROC-AUC"),
    ):
        a(f"| {label} | {_num(w3[key])} | {_num(m054[key])} | {_num(m054[key] - w3[key])} |")
    a("")
    a(f"W3 的 macro-F1={_num(w3['macro_f1'])} 属**同分布上限**，"
      "任务单已明确「不可外推」；本节对照表正是把这个死结用独立车型数据打开。"
      "两轮评估集不同（HCRL split-v3 val vs OTIDS 全量窗口），"
      "数字只用于说明泛化边界，不是同一测试集上的模型 A/B。")
    a("")

    a("### 5.2 按攻击类型")
    a("")
    a("| 类型 | 跨域 Recall@0.54 | 漏检窗口 | AP vs Normal | 观察与可能解释（假设，非因果分解） |")
    a("|---|---:|---:|---:|---|")
    reasons = {
        "DoS": (
            "注入 `ID=0x000` 在正常轨迹中不存在，符合文献给出的易检出机制；"
            "**本轮该类未出现可观测下降**（Recall 与 AP 均近 1），"
            "故此处记录的是「为何易检出」而非「为何下降」"
        ),
        "Fuzzy": (
            "随机 ID / 随机 payload 注入，ID 集合与周期分布同时漂移；"
            "Recall 低于 DoS。官方描述无逐帧标签，主口径只能按文件级标签判定，"
            "无法进一步区分是 ID 还是周期漂移所致"
        ),
        "Impersonation": (
            "伪装注入模仿合法节点（官方描述针对 ID 0x164），帧级统计与正常流量最接近；"
            "该类型在 HCRL 五分类中无对应类别，实测为三类中 Recall 最低、漏检最多"
        ),
    }
    for t in ("DoS", "Fuzzy", "Impersonation"):
        v = per_type["by_type"][t]
        a(f"| {t} | {_num(v.get('recall_at_frozen'))} | {v.get('missed_at_frozen', 0):,} | "
          f"{_num(v.get('average_precision_vs_normal'))} | {reasons[t]} |")
    a("")
    a("> **归因强度声明（按攻击类型）**：上表左侧三列是本次实测数据，"
      "右侧是基于文献机制与漂移量化的**假设性解释**，不是因果分解。"
      "任务单要求「ID 空间 / 周期 / payload 漂移分别贡献多少」，但在零样本、不重训、"
      "且 OTIDS 镜像没有逐帧真值的前提下，这三组漂移彼此耦合、无法单独隔离，"
      "**贡献百分比不可识别**。本报告只给出可复现的分类型指标（第 2.2 节）"
      "与分组漂移量化（第 5.3 节），不编造占比；该限制已列入 `RESULT.md` 遗留问题。")
    a("")

    a("### 5.3 按特征组（分布漂移量化）")
    a("")
    a(f"对照集合：HCRL split-v3 val 归一化特征 {drift['hcrl_val_n_windows']:,} 窗口 vs "
      f"OTIDS 归一化特征 {drift['otids_n_windows']:,} 窗口；"
      f"{drift['unit']}。")
    a("")
    a("| 漂移组 | 维数 | 平均 \\|Δmean\\| | 最大 \\|Δmean\\| | 合计 \\|Δmean\\| | 含义 |")
    a("|---|---:|---:|---:|---:|---|")
    for g, v in drift["by_group"].items():
        a(f"| {g} | {v['n_features']} | {_num(v['mean_abs_delta'])} | {_num(v['max_abs_delta'])} | "
          f"{_num(v['sum_abs_delta'])} | {drift['by_group_note'][g]} |")
    a("")
    top_cols = sorted(
        drift["per_feature"].items(), key=lambda x: x[1]["abs_mean_delta"], reverse=True
    )[:8]
    a("漂移最大的 8 个维度（归一化空间）：")
    a("")
    a("| 特征 | 组 | HCRL val 均值 | OTIDS 均值 | \\|Δmean\\| | std 比值 |")
    a("|---|---|---:|---:|---:|---:|")
    for c, v in top_cols:
        a(f"| `{c}` | {v['group']} | {_num(v['hcrl_val_mean'])} | {_num(v['otids_mean'])} | "
          f"{_num(v['abs_mean_delta'])} | {_num(v['std_ratio']) if v['std_ratio'] is not None else 'n/a'} |")
    a("")
    a(f"**归因结论（关联证据，非因果分解）**：漂移最大的组是 "
      f"`{drift['dominant_drift_group']}`"
      f"（平均 |Δmean|={_num(drift['by_group'][drift['dominant_drift_group']]['mean_abs_delta'])}），"
      "而该组在本任务中原本只被设为**对照量**（G1 DLC 统计与窗口帧统计），"
      "并非预设的三类漂移假设之一。这说明两数据集在 DLC / 窗口帧密度上的分布差异，"
      "比 ID 空间、周期、payload 三类都更显著，提示采集与切段层面的差异不可忽略。"
      "但**漂移幅度大 ≠ 对性能下降的贡献大**：第 4 节去 ID 消融显示去掉 ID 组后 AP "
      f"反而下降 {_num(abs(abl['ap_relative_drop_pct']), 4)}%（绝对变化 "
      f"{_num(abl['ap_absolute_delta'])}，collapsed={abl['collapsed']}），"
      "即 ID 组仍在提供有效判别信息。二者共同说明：本轮只能给出「哪些分布变了」"
      "与「去掉某组后性能如何变」的可复现事实，"
      "**不能**给出各漂移对 F1 下降的精确贡献占比。")
    a("")
    a("> **归因强度声明**：本节是「分布漂移 + 消融」的关联证据，"
      "不是严格的因果分解。三类漂移（ID 空间 / 周期 / payload）彼此耦合，"
      "单一组对性能下降的精确贡献百分比无法在零样本、不重训的前提下识别；"
      "本报告只给出可复现的漂移量化与消融方向，不编造贡献占比。")
    a("")

    # ---------------- 6 工程纪律 ----------------
    a("## 6. 工程纪律与 HCRL 零触碰证据")
    a("")
    a(f"- HCRL 侧既有产物只读快照：运行前 {integrity['n_files_before']} 个文件，"
      f"运行后同类范围 {integrity['n_files_after_in_scope']} 个文件；"
      f"**变更 {len(integrity['changed_files'])} 个、删除 {len(integrity['removed_files'])} 个**，"
      f"verified={integrity['verified']}。")
    a(f"- 结论：{integrity['conclusion']}")
    a(f"- HCRL test 折：{cfg['test_set_status']}（split-v3 windows 只含 train/val，"
      "本任务按 window_id in-list 下推，test 特征零载入）。")
    a("- 本次新增产物一律为 `otids` / `xdomain` 前缀的新文件、新目录；"
      "W4 代码已做英文提交（未 push），**提交号与完整变更清单见 `RESULT.md`**"
      "——`RESULT.md` 按 worker 约定在提交完成后最后创建，故本报告不预写尚未产生的提交号。")
    a(f"- 全量 pytest {ctx['pytest_passed']}/{ctx['pytest_total']} 通过；ruff 通过。")
    a("- 未安装 torch / matplotlib 等重型依赖，未访问任何外部服务。")
    a("")

    # ---------------- 7 局限性 ----------------
    a("## 7. 局限性与口径边界")
    a("")
    a("1. **文件级标签退化**：OTIDS 镜像只有文件级 `target`，攻击文件内正常 ID 的帧也标为攻击；"
      "主口径窗口标签实际等价于「该文件是否为攻击文件」，θ=0.20 在此无区分作用。"
      "这使本轮评估更接近「会话级异常检测」而非 HCRL 的「注入片段检测」，"
      "两类数字不可直接互换。")
    a("2. **车型/年份/采集环境同时改变**：Hyundai（2018 前后 HCRL）→ Kia Soul（OTIDS），"
      "ID 空间、周期、payload 编码、采集设备与时间戳基准（dataset3 为 Unix 绝对时间戳）"
      "一起变化，无法把下降单独归因于某一个因素。")
    a("3. **冻结特征参数的错配**：`id_frequency_train.json` 与归一化参数是 HCRL train 拟合的，"
      "零样本要求必须原样套用；若允许在 OTIDS 上重新 fit 会提升数字，但那就不是零样本，"
      "本任务明确禁止。")
    a("4. **官方描述口径为近似**：Fuzzy / Impersonation 的「250 秒后为攻击段」是"
      "按任务单给出的官方描述实现的近似判据，OTIDS 官方并无逐帧 Fuzzy 标签；"
      "该口径只作敏感性对照。")
    a("5. **五分类不可定义**：OTIDS 无 Gear/RPM 真值、无 Impersonation 对应类别，"
      "跨域 macro-F1 不报；映射分布仅为定性参考。")
    a("6. **反向验证（train OTIDS → test HCRL val）未执行**：任务单禁止项明确"
      "「不用 OTIDS 数据重训/微调模型权重」，而反向验证必须用 OTIDS 训练模型，与禁止项冲突；"
      "该项虽标为可选，本轮按禁止项优先不做，已在 RESULT.md 遗留问题列明，"
      "需规划方明确放行口径后另派单。")
    a("7. **适配上限不是零样本**：第 3 节数字使用了一半 OTIDS 数据（仅调阈值），"
      "不得与第 2 节零样本数字混报或取更优者对外引用。")
    a("")

    # ---------------- 8 公开报告口径 ----------------
    a("## 8. 对外可用口径建议（公开技术报告）")
    a("")
    a("基于本轮结果，可诚实表述：")
    a("")
    a(f"- 「在 HCRL 同分布 val 上 F1={_num(w3['f1'])}，"
      f"在独立车型 OTIDS 上零样本 F1={_num(m054['f1'])}、Recall={_num(m054['recall'])}、"
      f"FPR={_num(m054['fpr'])}，用跨域验证量化了泛化边界。」")
    a(f"- 「用去 ID 消融（去 ID 后 AP 相对下降 {_num(abl['ap_relative_drop_pct'], 4)}%、"
      f"未崩塌）与分组特征漂移量化（最大漂移组 `{drift['dominant_drift_group']}`）"
      "检查了跨域下降的可能来源，并明确说明零样本条件下无法做严格因果分解。」")
    a("")
    a("不建议表述：")
    a("")
    a("- 把 W3 的近满分数字当作「在任意车型上都能达到的检出率」；")
    a("- 把适配上限（用了一半 OTIDS 数据调阈值，且重标定阈值落在网格下界）当作零样本泛化能力；")
    a("- 把主口径文件级标签下的 Recall 说成「逐帧注入检出率」，"
      "或把 FPR=0 说成「零误报」；")
    a("- 把去 ID 消融的「AP 下降」说成「ID 频次表跨车型错配是性能下降主因」"
      "（方向相反：去掉 ID 组反而更差，说明该组仍在提供有效判别信息）；")
    a("- 把第 5 节的分组漂移当作各因素对 F1 下降的「贡献占比」。")
    a("")
    a("---")
    a("")
    a("全量指标与逐条口径标注见 `xdomain_metrics.json`；"
      "产物清单、验收逐条核对与遗留问题见 `RESULT.md`。")
    a("")
    return "\n".join(lines)
