"""W5 工作台 · FastAPI API 契约 / test 封存 / 口径标注 / 真实推理测试。

不依赖 httpx：用零外部依赖的 ASGI 调用辅助直接驱动 app，避免新增依赖。
既有 84 项测试不受影响；全部断言只读，不写任何文件。
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import joblib
import numpy as np
import pytest

from diting.common import PROJECT_ROOT
from diting.server import assets, inference
from diting.server.app import app

GET_ENDPOINTS = [
    "/api/health",
    "/api/metrics/summary",
    "/api/curves/pr",
    "/api/curves/threshold",
    "/api/ablation",
    "/api/errors",
    "/api/samples",
]


def _call(method: str, path: str, body: dict[str, Any] | None = None) -> tuple[int, dict[str, Any]]:
    """最小 ASGI 客户端：构造 scope，收集 response.start / response.body。"""
    payload = json.dumps(body, ensure_ascii=False).encode("utf-8") if body is not None else b""
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": method,
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "root_path": "",
        "headers": [
            (b"host", b"testserver"),
            (b"content-type", b"application/json"),
            (b"content-length", str(len(payload)).encode()),
        ],
        "client": ("127.0.0.1", 1234),
        "server": ("testserver", 80),
    }
    sent: list[dict[str, Any]] = []
    requested = [False]

    async def receive() -> dict[str, Any]:
        if not requested[0]:
            requested[0] = True
            return {"type": "http.request", "body": payload, "more_body": False}
        return {"type": "http.disconnect"}

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    async def run() -> None:
        await app(scope, receive, send)

    asyncio.run(run())
    status = next(m["status"] for m in sent if m["type"] == "http.response.start")
    raw = b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body")
    return status, json.loads(raw)


@pytest.mark.parametrize("path", GET_ENDPOINTS)
def test_get_endpoints_return_200_with_scope(path: str) -> None:
    status, doc = _call("GET", path)
    assert status == 200, f"{path} -> {status}: {doc}"
    assert isinstance(doc, dict)


def test_all_responses_carry_scope_annotation() -> None:
    """每个 API 响应都必须带口径标注、红线提示、产物溯源与 test 零载入留痕。"""
    responses = [_call("GET", p)[1] for p in GET_ENDPOINTS]
    responses.append(_call("POST", "/api/detect", {"sample_id": inference.sample_catalog()["samples"][0]["sample_id"]})[1])
    for doc in responses:
        assert "scope" in doc and isinstance(doc["scope"], dict)
        scope = doc["scope"]
        for key in ("evaluated_on", "data_scope", "level"):
            assert scope.get(key), f"口径字段缺失：{key}"
        assert doc["test_rows_loaded"] == 0
        assert "封存" in str(doc["test_set_status"]) or "sealed" in str(doc["test_set_status"]).lower()
        assert doc["caveats"] and any("零误报" in c for c in doc["caveats"])
        assert any("跨域五分类" in c for c in doc["caveats"])
        assert doc["sources"] and all(p.startswith("artifacts/") for p in doc["sources"].values())


def test_health_and_sealing_report() -> None:
    status, doc = _call("GET", "/api/health")
    assert status == 200
    assert doc["status"] == "ok"
    assert doc["model_version"] == "l1-gbdt-v3"
    assert doc["frozen_threshold"] == inference.frozen_threshold()
    sealing = doc["sealing"]
    assert sealing["all_declarations_sealed"] is True
    assert sealing["n_non_sealed_declarations"] == 0
    assert sealing["test_rows_loaded"] == 0
    assert "HCRL test" in sealing["sample_sources_forbidden"]


def test_metrics_summary_matches_artifacts() -> None:
    _, doc = _call("GET", "/api/metrics/summary")
    binary = assets.load("binary")["at_calibrated_threshold"]
    xb = assets.load("xdomain_binary")["at_frozen_0_54"]
    assert doc["same_distribution"]["binary_at_calibrated_threshold"]["f1"] == binary["f1"]
    assert doc["cross_domain"]["zeroshot_at_frozen_0_54"]["f1"] == xb["f1"]
    assert doc["frozen_threshold"] == assets.load("threshold")["selected_threshold"]
    mc = doc["same_distribution"]["multiclass"]["confusion_matrix"]
    assert len(mc["labels"]) == 5 and len(mc["matrix"]) == 5
    assert doc["cross_domain"]["five_class_mapping"]["macro_f1_reported"] is False
    policies = {r["policy"] for r in doc["cross_domain"]["adaptation_rows"]}
    assert {"otids_zeroshot", "otids_adapted", "otids_zeroshot_on_eval_fold"} <= policies


def test_pr_curve_full_pointwise() -> None:
    _, doc = _call("GET", "/api/curves/pr")
    assert doc["n_points"] == 6917
    assert len(doc["points"]) == 6917
    assert set(doc["points"][0]) == {"index", "precision", "recall", "threshold"}
    assert doc["average_precision"] == assets.load("pr_curve")["average_precision"]


def test_threshold_curve_supports_slider_recompute() -> None:
    _, doc = _call("GET", "/api/curves/threshold")
    assert doc["n_grid_points"] == 91 and len(doc["curve"]) == 91
    assert doc["selected_threshold"] == 0.54
    for p in doc["curve"]:
        for key in ("threshold", "recall", "fpr", "youden_j", "tp", "fp", "tn", "fn"):
            assert key in p
        recall = p["tp"] / (p["tp"] + p["fn"])
        fpr = p["fp"] / (p["fp"] + p["tn"])
        assert abs(recall - p["recall"]) < 1e-5
        assert abs(fpr - p["fpr"]) < 1e-5


def test_ablation_and_errors_contract() -> None:
    _, abl = _call("GET", "/api/ablation")
    assert len(abl["same_distribution"]["variants"]) == 4
    assert abl["cross_domain"]["collapsed"] is False
    assert "模型不依赖 ID" in abl["direction_note"]

    _, err = _call("GET", "/api/errors")
    assert err["n_false_positives"] == 3 and err["n_false_negatives"] == 2
    assert len(err["records"]) == 5
    assert err["label_boundary_analysis"]["window_label_theta"] == 0.2
    assert err["governance_recommendations"]


def test_samples_only_val_and_otids() -> None:
    _, doc = _call("GET", "/api/samples")
    assert doc["n_samples"] >= 10
    for s in doc["samples"]:
        assert s["source"] in ("hcrl_val", "otids")
        assert s["split"] in ("val", "otids")
        assert not s["window_id"].startswith("TEST")
    sealing = doc["sealing"]
    assert sealing["split_v3_splits"] == ["train", "val"]
    assert "test" not in sealing["split_v3_splits"]
    assert sealing["test_rows_loaded"] == 0
    assert len(sealing["assertions"]) >= 5


def test_test_fold_is_rejected() -> None:
    """任意未登记 window_id（含潜在 test 窗口）必须被拒绝，不允许下推读取。"""
    with pytest.raises(inference.DetectInputError):
        inference.find_sample("TEST__seg0001__w000000")
    with pytest.raises(inference.DetectInputError):
        inference.feature_row_for("not-in-catalog__w000000")
    status, _doc = _call("POST", "/api/detect", {"sample_id": "TEST__seg0001__w000000"})
    assert status == 404


def test_detect_calls_frozen_model_for_real() -> None:
    """手工加载冻结模型复算，证明 /api/detect 是真实推理而非 mock。"""
    sample = inference.sample_catalog()["samples"][0]
    status, doc = _call("POST", "/api/detect", {"sample_id": sample["sample_id"]})
    assert status == 200
    assert doc["input"]["mode"] == "sample"
    assert doc["model_version"] == "l1-gbdt-v3"
    assert doc["n_features"] == 33
    assert doc["threshold_is_frozen"] is True

    bundle = joblib.load(inference.MODEL_PATH)
    cols = bundle["binary_feature_columns"]
    vector = inference.feature_row_for(sample["sample_id"])
    x = np.asarray([[vector[c] for c in cols]], dtype=np.float64)
    expected = float(bundle["binary_model"].predict_proba(x)[0, 1])
    assert abs(doc["attack_probability"] - expected) < 1e-12
    assert doc["decision"] == ("attack" if expected >= doc["threshold"] else "normal")
    assert len(doc["top_features"]) == 8
    assert "置零" in doc["contribution_method"]
    assert abs(sum(doc["multiclass"]["probabilities"].values()) - 1.0) < 1e-9


def test_detect_custom_threshold_and_candump() -> None:
    sample = inference.sample_catalog()["samples"][0]
    status, doc = _call("POST", "/api/detect", {"sample_id": sample["sample_id"], "threshold": 0.7})
    assert status == 200 and doc["threshold"] == 0.7
    assert doc["threshold_is_frozen"] is False

    lines = [
        f"({1479121434.850202 + i * 0.0005:.6f}) can0 {0x316:03X}#"
        + "".join(f"{(i + j) % 256:02X}" for j in range(8))
        for i in range(100)
    ]
    status, doc = _call("POST", "/api/detect", {"candump": "\n".join(lines)})
    assert status == 200
    assert doc["input"]["mode"] == "candump"
    assert doc["input"]["n_frames"] == 100
    assert "caveat" in doc["input"]
    assert 0.0 <= doc["attack_probability"] <= 1.0


def test_detect_input_validation() -> None:
    assert _call("POST", "/api/detect", {})[0] == 400
    assert _call("POST", "/api/detect", {"sample_id": "x", "candump": "y"})[0] == 400
    assert _call("POST", "/api/detect", {"candump": "garbage"})[0] == 400
    assert _call("POST", "/api/detect", {"threshold": 1.5})[0] == 422


def test_no_write_api_routes() -> None:
    """后端全部只读推理 + /api/detect，不得出现 PUT / PATCH / DELETE。"""
    methods = {m for route in app.routes for m in getattr(route, "methods", set()) or set()}
    assert methods <= {"GET", "POST", "HEAD", "OPTIONS"}
    paths = [getattr(r, "path", "") for r in app.routes]
    api_paths = [p for p in paths if p.startswith("/api/")]
    assert len(api_paths) == 8


def test_frontend_has_no_hardcoded_metrics() -> None:
    """前端不得写死任何评估指标数字，全部数字必须来自后端 JSON。"""
    banned = [
        "0.999736",
        "0.914101",
        "0.999838",
        "0.998172",
        "0.841792",
        "0.112413",
        "19277",
        "92013",
        "7.7179",
    ]
    web_src = PROJECT_ROOT / "web" / "src"
    tsx_files = sorted(web_src.rglob("*.tsx")) + sorted(web_src.rglob("*.ts"))
    assert tsx_files, "前端源码不存在"
    for path in tsx_files:
        text = path.read_text(encoding="utf-8")
        for token in banned:
            assert token not in text, f"{path.name} 写死了指标数字 {token}"
