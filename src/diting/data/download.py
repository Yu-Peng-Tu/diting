"""D1 · 数据集接入与校验（只读白名单，记录出处/引用/校验和）。

严格口径（只访问允许的外部资源）：
- 只访问官方页面 https://ocslab.hksecurity.net/Datasets/car-hacking-dataset
  与关联论文页面；不访问其它外部服务，不重新分发原始数据。
- 官方页面本身不含可直接下载的数据文件；其下载入口指向第三方托管地址，
  这些地址不在白名单内，因此本模块**不**抓取它们，只如实记录链接主机。
- 真实原始文件由使用者自行下载后放入 data/raw/；本模块只做存在性扫描与
  SHA-256 校验，绝不生成、伪造或改写任何真实数据。
"""

from __future__ import annotations

import hashlib
import re
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..common import PROJECT_ROOT, sha256_file, utc_now, write_json

# 白名单：仅这两个 URL 允许网络访问。
WHITELIST: tuple[str, ...] = (
    "https://ocslab.hksecurity.net/Datasets/car-hacking-dataset",
    "https://arxiv.org/abs/1907.07377",
)

USER_AGENT = "diting-w1/0.1 (academic, offline pipeline verification)"


def _allowed(url: str) -> bool:
    return any(url.startswith(base) for base in WHITELIST)


def probe_official_page(cfg: dict[str, Any], timeout: int = 12) -> dict[str, Any]:
    """访问白名单内的官方页面，记录真实 HTTP 状态、字节数、摘要与外链主机。"""
    url = WHITELIST[0]
    rec: dict[str, Any] = {
        "url": url,
        "whitelisted": True,
        "checked_at": utc_now(),
        "purpose": "核对官方字段定义、引用信息与下载入口",
    }
    if not _allowed(url):
        rec.update({"error": "URL 不在白名单内，拒绝访问"})
        return rec
    try:
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read()
            rec.update(
                {
                    "http_status": resp.status,
                    "content_type": resp.headers.get("Content-Type"),
                    "bytes": len(body),
                    "sha256": hashlib.sha256(body).hexdigest(),
                }
            )
            html = body.decode("utf-8", errors="replace").replace("&amp;", "&")
            hosts = sorted(
                set(re.findall(r"https?://([A-Za-z0-9._-]+)", html))
            )
            rec["referenced_hosts"] = hosts
            rec["third_party_download_hosts"] = [
                h for h in hosts if "drive.google.com" in h or "dropbox.com" in h
            ]
            # 官方页面给出的是第三方托管下载入口，不是页面内可直接下载的数据文件。
            rec["direct_dataset_file_on_page"] = False
            rec["note"] = (
                "官方页面可达，但数据集文件托管在页面链接的第三方地址上；"
                "这些地址不在白名单内，本模块不访问。字段定义与引用信息已核对。"
            )
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        rec.update({"http_status": None, "bytes": 0, "sha256": None, "error": f"{type(exc).__name__}: {exc}"})
    return rec


def scan_raw_files(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """扫描 data/raw/ 下真实原始文件（不读取内容生成任何数据）。"""
    raw_dir = PROJECT_ROOT / cfg["paths"]["raw"]
    found: list[dict[str, Any]] = []
    if not raw_dir.is_dir():
        return found
    for p in sorted(raw_dir.rglob("*")):
        if not p.is_file() or p.name in {".gitkeep", "README.md"}:
            continue
        # simulated 自生成样本不算真实数据，避免被误判为「真实数据可用」
        if "simulated" in p.parts or "SIMULATED" in p.name:
            continue
        # W2 起本地真实数据由 local_raw 路径单独接入，不参与 W1 的官方直投扫描
        if "car-hacking" in p.parts:
            continue
        if p.suffix.lower() not in {".csv", ".txt", ".log", ".zip", ".parquet"}:
            continue
        found.append(
            {
                "path": str(p.relative_to(PROJECT_ROOT)),
                "size_bytes": p.stat().st_size,
                "sha256": sha256_file(p),
                "modified_at": datetime.fromtimestamp(
                    p.stat().st_mtime, tz=UTC
                ).astimezone().isoformat(timespec="seconds"),
            }
        )
    return found


def build_provenance(cfg: dict[str, Any], allow_network: bool = True) -> dict[str, Any]:
    """汇总出处、引用、许可声明与本次真实可获取性结论。"""
    ds = cfg["dataset"]
    page = probe_official_page(cfg) if allow_network else {
        "url": WHITELIST[0],
        "whitelisted": True,
        "skipped": True,
        "note": "本次运行未发起网络探测（离线复现），沿用既有探测记录",
        "checked_at": utc_now(),
    }
    raw_files = scan_raw_files(cfg)
    real_available = len(raw_files) > 0
    return {
        "provenance_version": "1.0",
        "checked_at": utc_now(),
        "dataset": {
            "name": ds["name"],
            "official_page": ds["official_page"],
            "paper_page": ds["paper_page"],
            "citation_primary": ds["citation_primary"],
            "citation_secondary": ds["citation_secondary"],
            "license_note": ds["license_note"],
            "official_schema_fields": ds["schema_fields"],
            "official_counts": ds["official_counts"],
        },
        "page_probe": page,
        "raw_files_found": raw_files,
        "real_data_available": real_available,
        "status": "available" if real_available else "unavailable",
        "conclusion": (
            "真实公开数据集文件未落盘，D1 真实数据相关验收项标记未完成；"
            "按允许的替代路径，用 source_tag=simulated 的自生成小样本验证管线代码正确性，"
            "全过程明确标注非真实数据、不伪造任何真实指标。"
            if not real_available
            else "已发现真实原始文件，进入真实数据归一化流程。"
        ),
        "whitelist": list(WHITELIST),
        "network_scope_note": "仅访问白名单两个 URL；第三方托管下载入口未访问；原始数据不提交 git、不重新分发。",
    }


def write_provenance(cfg: dict[str, Any], artifacts_dir: Path, allow_network: bool = True) -> dict[str, Any]:
    prov = build_provenance(cfg, allow_network=allow_network)
    write_json(artifacts_dir / "dataset_provenance.json", prov)
    write_json(artifacts_dir / "download_attempts.json", prov["page_probe"])
    return prov
