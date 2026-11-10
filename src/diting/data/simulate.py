"""D1 替代路径 · 自生成 simulated 小样本（明确标注非真实数据）。

允许的替代路径：「若数据集下载失败/需登录/许可受限：不得伪造数据……可用自生成的
source_tag=simulated 小样本来验证管线代码正确性，但明确标注非真实数据」。

本模块**只用于验证管线代码正确性**：写出与官方 §1.1 字段一致的 CSV 原始文件，
让 D1 归一化 → 划分 → 特征 → L0 评估全链路被真实执行一遍。
生成数据绝不冒充 HCRL Car-Hacking 数据集，不用于产出任何真实指标，
且在所有产物中以 source_tag=simulated / is_real_data=false 显式标记。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..common import set_seed

# 官方 §1.1 数据属性对应的表头（HEX CAN ID / DLC / DATA[0..7] / Flag T|R）
RAW_HEADER = "Timestamp,CAN ID,DLC,DATA[0],DATA[1],DATA[2],DATA[3],DATA[4],DATA[5],DATA[6],DATA[7],Flag"

# 正常流量里出现的常规 CAN ID（十六进制），均为普通报文，不含标签语义。
NORMAL_IDS = ["0316", "043f", "018f", "02b0", "0350", "0440", "05a0", "00a8"]

# 四类注入攻击的注入 ID 语义（仅用于构造时序形态，不作为特征布尔使用）
ATTACK_PROFILES = {
    "DoS": {"injected_id": "0000", "id_pool": None},
    "Fuzzy": {"injected_id": None, "id_pool": True},
    "Gear": {"injected_id": "02b0", "id_pool": False},
    "RPM": {"injected_id": "0316", "id_pool": False},
}


def _fmt_bytes(rng: np.random.Generator, dlc: int) -> list[str]:
    vals = [f"{int(v):02X}" for v in rng.integers(0, 256, size=dlc)]
    # 官方文件 DLC<8 时后置 DATA 列留空，归一化层按 -1 填充
    return vals + [""] * (8 - dlc)


def _emit_segment(
    rng: np.random.Generator,
    rows: list[str],
    t: float,
    n_frames: int,
    is_attack: bool,
    attack_kind: str | None,
    session_id: str,
    segment_no: int,
) -> tuple[float, dict[str, Any]]:
    """生成一段连续帧（一个注入片段或一段正常片段）。"""
    injected_count = 0
    for i in range(n_frames):
        if is_attack:
            # 注入帧：短周期高频（官方描述 0.3ms/0.5ms/1ms 注入），帧密度显著抬升
            t += float(rng.uniform(0.0003, 0.0008))
            profile = ATTACK_PROFILES[attack_kind or "DoS"]
            if profile["id_pool"] is True:
                can_id = f"{int(rng.integers(0, 0x800)):04X}"
            elif profile["injected_id"]:
                can_id = profile["injected_id"]
            else:
                can_id = NORMAL_IDS[int(rng.integers(0, len(NORMAL_IDS)))]
            dlc = int(rng.integers(0, 9))
            flag = "T"
            injected_count += 1
        else:
            t += float(rng.uniform(0.004, 0.012))
            can_id = NORMAL_IDS[i % len(NORMAL_IDS)]
            dlc = 8 if i % 3 else int(rng.integers(4, 9))
            flag = "R"
        # 攻击片段内混入少量正常帧，更贴近「注入与正常混合」的真实记录
        if is_attack and i % 7 == 6:
            flag = "R"
            can_id = NORMAL_IDS[i % len(NORMAL_IDS)]
            injected_count -= 1
        data = _fmt_bytes(rng, dlc)
        rows.append(
            ",".join([f"{t:.6f}", can_id, str(dlc), *data, flag])
        )
    seg = {
        "session_id": session_id,
        "segment_no": segment_no,
        "is_attack": is_attack,
        "attack_kind": attack_kind if is_attack else "Normal",
        "n_frames": n_frames,
        "n_injected": max(injected_count, 0),
    }
    return t, seg


def generate_simulated_raw(cfg: dict[str, Any], raw_dir: Path) -> dict[str, Any]:
    """生成 simulated 原始 CSV（每个 session 一个文件），返回生成说明。

    文件命名带 SIMULATED_NOT_REAL_DATA 前缀，内容只有官方字段，不伪造出处。
    """
    set_seed(int(cfg["project"]["seed"]))
    rng = np.random.default_rng(int(cfg["project"]["seed"]))
    raw_dir.mkdir(parents=True, exist_ok=True)
    sim_dir = raw_dir / "simulated"
    sim_dir.mkdir(parents=True, exist_ok=True)

    sessions = [
        ("sim-normal", None, 10, 200),
        ("sim-dos", "DoS", 8, 200),
        ("sim-fuzzy", "Fuzzy", 8, 200),
        ("sim-gear", "Gear", 8, 200),
        ("sim-rpm", "RPM", 8, 200),
    ]
    written: list[dict[str, Any]] = []
    total_rows = 0
    for session_no, (session_id, attack_kind, n_segments, seg_len) in enumerate(sessions):
        rows: list[str] = []
        # 每个 session 使用不重叠的时间起点，保证全局时间单调递增（时间顺序切分才有意义）
        t = float(session_no * 10000.0)
        seg_meta = []
        for s in range(n_segments):
            # 每个片段交替为「正常」或「注入」，保证每个 session 内两类都有
            is_attack = (attack_kind is not None) and (s % 2 == 1)
            t, seg = _emit_segment(
                rng, rows, t, seg_len, is_attack, attack_kind, session_id, s
            )
            seg_meta.append(seg)
            t += 0.05  # 片段之间的时间间隙（远大于段内帧间隔，用于片段切分）
        path = sim_dir / f"SIMULATED_NOT_REAL_DATA__{session_id}.csv"
        path.write_text(RAW_HEADER + "\n" + "\n".join(rows) + "\n", encoding="utf-8")
        total_rows += len(rows)
        written.append(
            {
                "path": str(path),
                "session_id": session_id,
                "rows": len(rows),
                "segments": seg_meta,
            }
        )

    return {
        "is_real_data": False,
        "source_tag": "simulated",
        "warning": "本批文件为自生成 simulated 样本，仅用于验证管线代码正确性，不是 HCRL Car-Hacking 数据集，不得用于任何真实指标或结论。",
        "official_page": cfg["dataset"]["official_page"],
        "raw_header": RAW_HEADER,
        "seed": int(cfg["project"]["seed"]),
        "n_files": len(written),
        "total_rows": total_rows,
        "files": written,
    }
