"""L1 · permutation importance（val 上、固定种子）与零依赖 SVG 出图。

出图不引入 matplotlib（本机未安装，且任务单禁止新增重型依赖/联网），
直接用标准库拼 SVG 文本；SVG 是任务单允许的产物格式之一。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from sklearn.inspection import permutation_importance

from ..features.build import FEATURE_GROUPS


def compute_permutation_importance(
    model: Any,
    X_val: np.ndarray,
    y_val: np.ndarray,
    columns: list[str],
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """在 val 上计算 permutation importance；固定随机种子，返回排序表。"""
    p = cfg["importance"]
    result = permutation_importance(
        model,
        X_val,
        y_val,
        scoring=p["scoring"],
        n_repeats=int(p["n_repeats"]),
        random_state=int(p["random_state"]),
        n_jobs=1,
    )
    group_of = {c: g for g, cols in FEATURE_GROUPS.items() for c in cols}
    rows = [
        {
            "rank": 0,
            "feature": c,
            "group": group_of.get(c, "UNKNOWN"),
            "importance_mean": round(float(result.importances_mean[i]), 8),
            "importance_std": round(float(result.importances_std[i]), 8),
        }
        for i, c in enumerate(columns)
    ]
    rows.sort(key=lambda r: r["importance_mean"], reverse=True)
    for i, r in enumerate(rows, 1):
        r["rank"] = i

    # 按 G1–G6 聚合，便于回答「排前列的是不是周期/频率特征」（§5.2）
    by_group: dict[str, dict[str, Any]] = {}
    for r in rows:
        g = r["group"]
        rec = by_group.setdefault(g, {"group": g, "n_features": 0, "sum_importance": 0.0, "mean_importance": 0.0})
        rec["n_features"] += 1
        rec["sum_importance"] += r["importance_mean"]
    for rec in by_group.values():
        rec["mean_importance"] = round(rec["sum_importance"] / rec["n_features"], 8)
        rec["sum_importance"] = round(rec["sum_importance"], 8)
    group_rank = sorted(by_group.values(), key=lambda r: r["sum_importance"], reverse=True)

    return {
        "method": "sklearn.inspection.permutation_importance",
        "evaluated_on": cfg["evaluated_on"],
        "data_scope": cfg["data_scope"],
        "scoring": p["scoring"],
        "n_repeats": int(p["n_repeats"]),
        "random_state": int(p["random_state"]),
        "n_features": len(columns),
        "ranked": rows,
        "by_group": group_rank,
        "top10": [r["feature"] for r in rows[:10]],
    }


def render_importance_svg(importance: dict[str, Any], cfg: dict[str, Any], top_k: int | None = None) -> str:
    """手写横向条形图 SVG（浅色主题，深色文字），不依赖任何绘图库。"""
    k = int(top_k or cfg["importance"]["top_k_plot"])
    rows = importance["ranked"][:k]
    palette = {
        "G1": "#4C78A8", "G2": "#F58518", "G3": "#54A24B",
        "G4": "#E45756", "G5": "#72B7B2", "G6": "#B279A2",
    }
    row_h, top, left, chart_w = 26, 74, 250, 520
    width, height = 900, top + len(rows) * row_h + 56
    max_v = max([abs(r["importance_mean"]) for r in rows] + [1e-9])
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="24" y="30" font-family="-apple-system,Helvetica,Arial,sans-serif" font-size="18" font-weight="700" fill="#1f2933">L1 特征重要性 · permutation importance（val, f1）</text>',
        f'<text x="24" y="52" font-family="-apple-system,Helvetica,Arial,sans-serif" font-size="12" fill="#52606d">{_esc(cfg["data_scope"])}；颜色=G1–G6 特征组</text>',
    ]
    # x 轴刻度
    for i in range(5):
        x = left + chart_w * i / 4
        parts.append(f'<line x1="{x:.1f}" y1="{top-8}" x2="{x:.1f}" y2="{top+len(rows)*row_h}" stroke="#e4e7eb" stroke-width="1"/>')
        parts.append(f'<text x="{x:.1f}" y="{top-14}" font-family="Helvetica,Arial,sans-serif" font-size="11" fill="#7b8794" text-anchor="middle">{max_v*i/4:.4f}</text>')
    for i, r in enumerate(rows):
        y = top + i * row_h
        bar = chart_w * max(r["importance_mean"], 0.0) / max_v
        color = palette.get(r["group"], "#8895a7")
        parts.append(f'<text x="{left-10}" y="{y+13}" font-family="Helvetica,Arial,sans-serif" font-size="12" fill="#323f4b" text-anchor="end">{_esc(r["feature"])}</text>')
        parts.append(f'<rect x="{left}" y="{y+2}" width="{bar:.2f}" height="16" rx="2" fill="{color}"/>')
        parts.append(f'<text x="{left+bar+6:.2f}" y="{y+14}" font-family="Helvetica,Arial,sans-serif" font-size="11" fill="#3e4c59">{r["importance_mean"]:.5f}</text>')
        parts.append(f'<text x="{left-10-0}" y="{y+13}" font-family="Helvetica,Arial,sans-serif" font-size="11" fill="#9aa5b1" text-anchor="start" opacity="0"> </text>')
    # 图例
    ly = height - 26
    lx = 24
    for g in ("G1", "G2", "G3", "G4", "G5", "G6"):
        parts.append(f'<rect x="{lx}" y="{ly-10}" width="11" height="11" rx="2" fill="{palette[g]}"/>')
        parts.append(f'<text x="{lx+16}" y="{ly}" font-family="Helvetica,Arial,sans-serif" font-size="11" fill="#52606d">{g}</text>')
        lx += 62
    parts.append('</svg>')
    return "\n".join(parts)


def _esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def write_importance_table(importance: dict[str, Any], out_dir: Path) -> dict[str, str]:
    """输出 JSON（由统一写盘负责）+ Markdown 排序表 CSV。"""
    import csv

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "l1_feature_importance.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["rank", "feature", "group", "importance_mean", "importance_std"])
        for r in importance["ranked"]:
            w.writerow([r["rank"], r["feature"], r["group"], r["importance_mean"], r["importance_std"]])
    lines = [
        "# L1 特征重要性（permutation importance）",
        "",
        f"- 评估集合：val；评分：{importance['scoring']}；重复次数：{importance['n_repeats']}；随机种子：{importance['random_state']}",
        f"- 数据口径：{importance['data_scope']}",
        "",
        "| 排名 | 特征 | 组 | importance 均值 | 标准差 |",
        "|---:|---|---|---:|---:|",
    ]
    for r in importance["ranked"]:
        lines.append(f"| {r['rank']} | `{r['feature']}` | {r['group']} | {r['importance_mean']:.6f} | {r['importance_std']:.6f} |")
    lines += ["", "## 按 G1–G6 聚合", "", "| 组 | 特征数 | 重要性之和 | 组内均值 |", "|---|---:|---:|---:|"]
    for g in importance["by_group"]:
        lines.append(f"| {g['group']} | {g['n_features']} | {g['sum_importance']:.6f} | {g['mean_importance']:.6f} |")
    lines.append("")
    md_path = out_dir / "l1_feature_importance.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return {"csv": str(csv_path), "md": str(md_path)}
