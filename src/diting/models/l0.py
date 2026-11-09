"""L0 · 统计规则 baseline（不训练、可解释、给后续模型一条共同下限）。

对应策划方案 §5.1 L0：基于「窗口帧密度 / ID 频率 / 周期间隔」的固定阈值规则。
阈值只用 **train 折的正常窗口** 统计得到（不接触 val 标签、不接触 test），
在留出验证集 val 上给出 Recall / FPR / 混淆矩阵。

刻意不使用 g4_min_id_share：该特征与 DoS 注入的 0x000 ID 相关，
虽为窗口级相对统计量，但作为规则条件容易退化为标签代理，
故 L0 只用 ID 频率集中度、帧密度与周期间隔三类统计量；
该特征保留在特征矩阵中并列入 W2 消融清单（§4.6 L2 / §5.2）。
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

# L0 使用的规则特征（均为 G2/G4/G5 的统计量，非标签等价字段）
RULE_FEATURES = ("g2_dt_mean", "g4_top1_id_ratio", "g2_same_id_dt_cv", "g5_same_id_dt_std")


def fit_rule_thresholds(
    train_features: pd.DataFrame,
    id_frequency_percentile: float = 95.0,
    interval_cv_percentile: float = 95.0,
) -> dict[str, Any]:
    """只用 train 折正常窗口 fit 规则阈值。"""
    normal = train_features[(train_features["split"] == "train") & (train_features["label"] == 0)]
    if normal.empty:
        raise ValueError("train 折无正常窗口，无法 fit L0 阈值")

    thresholds = {
        # 帧密度：dt 越小越密；取 train 正常窗口的 5% 分位作为「异常密集」上界
        "g2_dt_mean_max": float(np.percentile(normal["g2_dt_mean"], 100 - id_frequency_percentile)),
        # ID 频率集中度：top1 占比过高视为异常
        "g4_top1_id_ratio_min": float(np.percentile(normal["g4_top1_id_ratio"], id_frequency_percentile)),
        # 同 ID 周期间隔变异：过大视为注入扰动
        "g2_same_id_dt_cv_min": float(np.percentile(normal["g2_same_id_dt_cv"], interval_cv_percentile)),
        "g5_same_id_dt_std_min": float(np.percentile(normal["g5_same_id_dt_std"], interval_cv_percentile)),
    }
    return {
        "version": "l0-thresholds-v1",
        "fit_on": "train_normal_windows",
        "n_train_normal_windows": len(normal),
        "id_frequency_percentile": id_frequency_percentile,
        "interval_cv_percentile": interval_cv_percentile,
        "thresholds": thresholds,
        "rule_features": list(RULE_FEATURES),
        "excluded_from_rules": ["g4_min_id_share"],
        "exclusion_reason": "与 0x000 注入 ID 相关，避免规则退化为标签代理；列入 W2 消融",
    }


def predict_l0(features: pd.DataFrame, rule: dict[str, Any], min_vote: int = 1) -> pd.DataFrame:
    """对任意分折（含服务侧）输出规则投票与判定；不接触标签。"""
    th = rule["thresholds"]
    votes = pd.DataFrame(index=features.index)
    votes["vote_frame_density"] = (features["g2_dt_mean"] < th["g2_dt_mean_max"]).astype(int)
    votes["vote_id_frequency"] = (features["g4_top1_id_ratio"] > th["g4_top1_id_ratio_min"]).astype(int)
    votes["vote_interval_cv"] = (features["g2_same_id_dt_cv"] > th["g2_same_id_dt_cv_min"]).astype(int)
    votes["vote_interval_std"] = (features["g5_same_id_dt_std"] > th["g5_same_id_dt_std_min"]).astype(int)
    total = votes.sum(axis=1)
    out = features[["window_id", "split", "segment_id", "source_tag", "label", "attack_ratio"]].copy()
    for c in votes.columns:
        out[c] = votes[c].to_numpy()
    out["rule_votes"] = total.to_numpy()
    out["pred"] = (total >= min_vote).astype(int).to_numpy()
    return out
