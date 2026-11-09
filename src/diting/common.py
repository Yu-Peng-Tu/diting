"""配置加载、路径解析、manifest 与通用 IO。

设计要点（对应任务单「依赖分层 / 配置与复现」）：
- 所有路径相对项目根目录，由 YAML 驱动，不散落在各模块里。
- 每个产物目录带 manifest.json：输入校验和、时间戳、随机种子、代码版本。
- 固定随机种子，保证 golden vector 与划分清单可复现。
"""

from __future__ import annotations

import hashlib
import json
import random
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from . import __version__

# 项目根目录：src/diting/common.py -> 上溯 3 层
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """加载 YAML 配置并注入解析后的绝对路径。"""
    cfg_path = Path(path) if path else PROJECT_ROOT / "configs" / "default.yaml"
    if not cfg_path.is_absolute():
        cfg_path = (PROJECT_ROOT / cfg_path).resolve()
    with cfg_path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg["_config_path"] = str(cfg_path)
    cfg["_config_sha256"] = sha256_file(cfg_path)
    cfg["_root"] = str(PROJECT_ROOT)
    return cfg


def subdir(cfg: dict[str, Any], key: str, *parts: str, create: bool = True) -> Path:
    """取得配置中的子目录（可拼接子路径），必要时创建。"""
    p = PROJECT_ROOT / cfg["paths"][key]
    for part in parts:
        p = p / part
    if create:
        p.mkdir(parents=True, exist_ok=True)
    return p


def sha256_file(path: str | Path) -> str:
    """计算文件 SHA-256，流式读取以适配大文件。"""
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_obj(obj: Any) -> str:
    """对任意可 JSON 序列化对象计算稳定摘要。"""
    payload = json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def set_seed(seed: int) -> None:
    """固定 Python / NumPy 随机种子。"""
    random.seed(seed)
    np.random.seed(seed)


def utc_now() -> str:
    return datetime.now(UTC).astimezone().isoformat(timespec="seconds")


def git_revision() -> str:
    """读取 git HEAD（若仓库尚未初始化则返回时间戳占位，不造假）。"""
    import subprocess

    try:
        out = subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return f"uncommitted@{utc_now()}"


def write_manifest(
    out_dir: str | Path,
    stage: str,
    cfg: dict[str, Any],
    inputs: list[str | Path],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """在产物目录写 manifest.json（带输入校验和、时间、种子、代码版本）。

    输入文件可能尚不存在（例如真实数据集未下载），此时如实记录 missing，
    不用伪造校验和。
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    input_records = []
    for raw in inputs:
        p = Path(raw)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        rec: dict[str, Any] = {"path": str(p.relative_to(PROJECT_ROOT)) if _is_relative(p) else str(p)}
        if p.is_file():
            rec.update({"exists": True, "size_bytes": p.stat().st_size, "sha256": sha256_file(p)})
        else:
            rec.update({"exists": False, "sha256": None})
        input_records.append(rec)

    manifest: dict[str, Any] = {
        "manifest_version": "1.0",
        "stage": stage,
        "project": cfg["project"]["name"],
        "project_version": cfg["project"]["version"],
        "code_version": __version__,
        "git_revision": git_revision(),
        "config_path": cfg["_config_path"],
        "config_sha256": cfg["_config_sha256"],
        "random_seed": cfg["project"]["seed"],
        "created_at": utc_now(),
        "inputs": input_records,
    }
    if extra:
        manifest.update(extra)
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def _is_relative(p: Path) -> bool:
    try:
        p.relative_to(PROJECT_ROOT)
        return True
    except ValueError:
        return False


def write_json(path: str | Path, obj: Any) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    return p


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def assert_no_forbidden_fields(columns: list[str] | Any, context: str = "特征矩阵") -> None:
    """代码级泄漏断言：flag / label / can_id==0x000 布尔等不得出现在特征中。

    同时拦截以这些名字为前缀/后缀派生的列（例如 flag_T、is_0x000_bool）。
    """
    from . import FORBIDDEN_FEATURE_FIELDS

    cols = [str(c).lower() for c in columns]
    hits: list[str] = []
    for col in cols:
        for forbidden in FORBIDDEN_FEATURE_FIELDS:
            if col == forbidden or col.startswith(forbidden + "_") or col.endswith("_" + forbidden):
                hits.append(f"{col} (匹配禁用字段 {forbidden})")
    # 0x000 布尔的其它常见写法
    zero_id_aliases = ("id000", "id_000", "zero_id", "is_zero", "can_id_eq_0", "can_id_0_bool")
    for col in cols:
        if any(alias in col for alias in zero_id_aliases):
            hits.append(f"{col} (疑似 can_id==0x000 布尔)")
    if hits:
        raise AssertionError(f"泄漏防线失败：{context} 含标签等价字段 -> {sorted(set(hits))}")
