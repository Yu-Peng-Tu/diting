"""D1 · schema 归一测试（官方 §1.1 字段 / HEX 解析 / DLC 缺位 / 行数对齐）。"""

from __future__ import annotations

import numpy as np

from diting.data.schema import FRAME_COLUMNS, parse_raw_file

OFFICIAL_HEADER = (
    "Timestamp,CAN ID,DLC,DATA[0],DATA[1],DATA[2],DATA[3],"
    "DATA[4],DATA[5],DATA[6],DATA[7],Flag"
)


def _write(tmp_path, lines, name="subset.csv"):
    p = tmp_path / name
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


def test_official_header_hex_parsing_and_dlc_padding(tmp_path):
    """官方表头 + HEX CAN ID/DATA + DLC<8 时缺失字节填 -1。"""
    p = _write(
        tmp_path,
        [
            OFFICIAL_HEADER,
            "1.000000,043f,8,00,01,02,03,04,05,06,07,R",
            "1.001000,0000,2,AA,BB,,,,,,T",
            "1.002000,02b0,4,10,20,30,40,,,,,R",
        ],
    )
    df, stats = parse_raw_file(p, "public:car-hacking", "DoS")
    assert list(df.columns) == FRAME_COLUMNS
    assert len(df) == 3
    assert stats["row_count_match"] is True
    assert stats["n_injected"] == 1

    assert int(df.loc[0, "can_id"]) == 0x043F
    assert int(df.loc[0, "data_0"]) == 0x00
    assert int(df.loc[0, "data_7"]) == 0x07

    r1 = df[df["can_id"] == 0x0000].iloc[0]
    assert int(r1["dlc"]) == 2
    assert int(r1["data_0"]) == 0xAA and int(r1["data_1"]) == 0xBB
    assert all(int(r1[f"data_{i}"]) == -1 for i in range(2, 8))
    assert r1["flag"] == "T" and int(r1["label"]) == 1


def test_headerless_whitespace_format(tmp_path):
    """无表头、空白分隔排版也能归一。"""
    p = _write(
        tmp_path,
        [
            "1.0 0316 8 05 28 84 66 00 20 00 0F R",
            "1.01 043f 8 00 00 00 00 00 00 00 00 T",
        ],
        name="space.txt",
    )
    df, stats = parse_raw_file(p, "public:car-hacking", "Fuzzy")
    assert len(df) == 2
    assert stats["row_count_match"] is True
    assert int(df.loc[0, "can_id"]) == 0x0316
    assert int(df.loc[1, "label"]) == 1


def test_flag_mapping_and_segment_grouping(tmp_path):
    """Flag T/R → 帧级标签；时间间隔切分片段，片段内不跨。"""
    lines = [OFFICIAL_HEADER]
    for i in range(6):
        lines.append(f"{i * 0.01:.6f},0316,8,00,00,00,00,00,00,00,00,{'T' if i >= 3 else 'R'}")
    lines.append("10.000000,0316,8,00,00,00,00,00,00,00,00,R")
    p = _write(tmp_path, lines)
    df, stats = parse_raw_file(p, "public:car-hacking", "Gear", gap_threshold=0.02)
    assert df["label"].tolist()[:6] == [0, 0, 0, 1, 1, 1]
    assert stats["n_segments"] == 2, "时间间隙应把最后一帧切为独立片段"
    # 同一帧只属于一个片段
    assert df.groupby("frame_index")["segment_id"].nunique().max() == 1


def test_frame_index_sorted_by_timestamp(tmp_path):
    p = _write(
        tmp_path,
        [
            OFFICIAL_HEADER,
            "3.000000,0316,8,00,00,00,00,00,00,00,00,R",
            "1.000000,0316,8,00,00,00,00,00,00,00,00,R",
            "2.000000,0316,8,00,00,00,00,00,00,00,00,R",
        ],
    )
    df, _ = parse_raw_file(p, "public:car-hacking", "RPM")
    assert df["timestamp"].is_monotonic_increasing
    assert df["frame_index"].tolist() == [0, 1, 2]
    assert np.all(np.diff(df["timestamp"].to_numpy()) > 0)
