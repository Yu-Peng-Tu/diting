"""谛听 W5 · Web 检测工作台 FastAPI 后端。

7 个只读 API（无写操作、无数据库、无用户系统）：
  GET  /api/health            健康检查 + test 折封存留痕
  GET  /api/metrics/summary   同分布 val + OTIDS 跨域核心指标（含混淆矩阵 / 漂移 / 适配）
  GET  /api/curves/pr         PR 曲线逐点数据
  GET  /api/curves/threshold  阈值曲线逐点数据（供前端滑块实时重算）
  GET  /api/ablation          同分布 + 跨域去 ID 消融对照
  GET  /api/errors            误报/漏报逐窗归因 + 标签边界带 + 治理建议
  GET  /api/samples           演示样本窗口列表（只来自 split-v3 val / OTIDS）
  POST /api/detect            冻结模型真实推理（样本 id 或粘贴 candump）

所有响应统一挂 scope / sources / caveats / test_rows_loaded，前端据此渲染口径声明。
数字全部来自既有 artifacts JSON，后端不生成、不改写、不写死任何指标。
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from . import assets, inference


class DetectRequest(BaseModel):
    """实时检测演示输入：sample_id 与 candump 恰好二选一。"""

    sample_id: str | None = None
    candump: str | None = None
    threshold: float | None = Field(default=None, ge=0.0, le=1.0)


def _pick_doc(key: str) -> dict[str, Any]:
    return assets.load(key)


def create_app() -> FastAPI:
    app = FastAPI(
        title="谛听 · CAN 入侵检测工作台 API",
        version="w5-1.0",
        description="只读推理与评估产物浏览；HCRL test 折全程封存",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        catalog = inference.sample_catalog()
        payload = {
            "status": "ok",
            "service": "diting-w5-workbench",
            "model_version": "l1-gbdt-v3",
            "split_version": "split-v3",
            "frozen_threshold": inference.frozen_threshold(),
            "n_features": 33,
            "n_demo_samples": len(catalog["samples"]),
            "sealing": assets.sealing_report(),
        }
        return assets.attach_scope(
            payload,
            _pick_doc("features_meta"),
            "features_meta",
            "threshold",
            "model_manifest",
        )

    @app.get("/api/metrics/summary")
    def metrics_summary() -> dict[str, Any]:
        binary = _pick_doc("binary")
        multiclass = _pick_doc("multiclass")
        per_attack = _pick_doc("per_attack")
        vs_l0 = _pick_doc("vs_l0")
        xb = _pick_doc("xdomain_binary")
        xa = _pick_doc("xdomain_adaptation")
        xp = _pick_doc("xdomain_per_attack")
        xd = _pick_doc("xdomain_drift")
        xm = _pick_doc("xdomain_mapping")
        xo = _pick_doc("xdomain_official")

        drift_top = sorted(
            (
                {"feature": k, **v}
                for k, v in xd["per_feature"].items()
            ),
            key=lambda d: d["abs_mean_delta"],
            reverse=True,
        )[:8]

        payload = {
            "frozen_threshold": inference.frozen_threshold(),
            "same_distribution": {
                "binary_at_calibrated_threshold": binary["at_calibrated_threshold"],
                "binary_at_default_threshold": binary["at_default_threshold"],
                "multiclass": {
                    "macro_precision": multiclass["macro_precision"],
                    "macro_recall": multiclass["macro_recall"],
                    "macro_f1": multiclass["macro_f1"],
                    "per_class": multiclass["per_class"],
                    "confusion_matrix": multiclass["confusion_matrix"],
                    "gear_rpm_confusion": multiclass["gear_rpm_confusion"],
                },
                "per_attack_type": per_attack["by_type"],
                "vs_l0_rows": vs_l0["rows"],
                "vs_l0_deltas": vs_l0["deltas_vs_l0"],
            },
            "cross_domain": {
                "zeroshot_at_frozen_0_54": xb["at_frozen_0_54"],
                "zeroshot_at_0_5": xb["at_0_5"],
                "official_label_sensitivity": xo,
                "adaptation_rows": xa["rows"],
                "same_eval_fold_delta_vs_frozen": xa["same_eval_fold_delta_vs_frozen"],
                "fold_sizes": xa["fold_sizes"],
                "per_attack_type": xp["by_type"],
                "feature_drift_by_group": xd["by_group"],
                "feature_drift_top": drift_top,
                "dominant_drift_group": xd["dominant_drift_group"],
                "five_class_mapping": {
                    "per_true_type": xm["per_true_type"],
                    "macro_f1_reported": xm["macro_f1_reported"],
                    "macro_f1_reason": xm["macro_f1_reason"],
                },
            },
            "sealing": assets.sealing_report(),
        }
        return assets.attach_scope(
            payload,
            binary,
            "binary",
            "multiclass",
            "per_attack",
            "vs_l0",
            "xdomain_binary",
            "xdomain_adaptation",
            "xdomain_per_attack",
            "xdomain_drift",
            "xdomain_mapping",
            "xdomain_official",
            evaluated_on="组合口径：HCRL split-v3 val + OTIDS 零样本 / 阈值适配",
            data_scope=(
                "同分布：真实数据子集（每文件前 100 万行）split-v3 val 窗口级；"
                "跨域：OTIDS（Kia Soul）全量 4 个逐帧标签文件 / 窗口级 / 冻结模型零样本"
            ),
        )

    @app.get("/api/curves/pr")
    def curves_pr() -> dict[str, Any]:
        doc = _pick_doc("pr_curve")
        payload = {
            "average_precision": doc["average_precision"],
            "n_points": doc["n_points"],
            "n_thresholds": doc["n_thresholds"],
            "source": doc["source"],
            "threshold_note": doc["threshold_note"],
            "points": doc["points"],
        }
        return assets.attach_scope(payload, doc, "pr_curve")

    @app.get("/api/curves/threshold")
    def curves_threshold() -> dict[str, Any]:
        doc = _pick_doc("threshold_curve")
        payload = {
            "version": doc["version"],
            "criterion": doc["criterion"],
            "criterion_note": doc["criterion_note"],
            "calibrated_on": doc["calibrated_on"],
            "default_threshold": doc["default_threshold"],
            "selected_threshold": doc["selected_threshold"],
            "selected_youden_j": doc["selected_youden_j"],
            "n_grid_points": doc["n_grid_points"],
            "grid_range": doc["grid_range"],
            "curve": doc["curve"],
            "recompute_note": (
                "每个网格点自带 tp/fp/tn/fn，前端滑块可实时重算 "
                "Recall=tp/(tp+fn)、FPR=fp/(fp+tn)、F1=2PR/(P+R)，不依赖后端二次请求"
            ),
        }
        return assets.attach_scope(
            payload,
            doc,
            "threshold_curve",
            evaluated_on="split-v3 val（阈值网格扫描）",
            level="window",
            threshold=doc["selected_threshold"],
        )

    @app.get("/api/ablation")
    def ablation() -> dict[str, Any]:
        same = _pick_doc("ablation")
        cross = _pick_doc("xdomain_id_ablation")
        payload = {
            "same_distribution": {
                "collapse_rule": same["collapse_rule"],
                "baseline_calibrated_threshold": same["baseline_calibrated_threshold"],
                "variants": same["variants"],
                "id_group_conclusion": same["id_group_conclusion"],
            },
            "cross_domain": {
                "evaluated_on": cross["evaluated_on"],
                "collapse_rule": cross["collapse_rule"],
                "ap_full_33": cross["ap_full_33"],
                "ap_drop_id": cross["ap_drop_id"],
                "ap_absolute_delta": cross["ap_absolute_delta"],
                "ap_relative_drop_pct": cross["ap_relative_drop_pct"],
                "collapsed": cross["collapsed"],
                "performance_improved_after_dropping_id": cross[
                    "performance_improved_after_dropping_id"
                ],
                "conclusion": cross["conclusion"],
                "rows": cross["rows"],
                "trained_on": cross["trained_on"],
                "model_source_note": cross["model_source_note"],
            },
            "direction_note": (
                "同分布与跨域的消融方向不同，不可互相套用；跨域去 ID 后 AP 下降 "
                "7.7179% 说明 ID 频率组仍在提供有效判别信息，不能解读为「模型不依赖 ID」"
            ),
        }
        return assets.attach_scope(
            payload,
            same,
            "ablation",
            "xdomain_id_ablation",
            evaluated_on="组合口径：HCRL split-v3 val 消融 + OTIDS 零样本去 ID 消融",
            level="window",
        )

    @app.get("/api/errors")
    def errors() -> dict[str, Any]:
        doc = _pick_doc("errors")
        payload = {
            "threshold_used": doc["threshold_used"],
            "n_val_windows": doc["n_val_windows"],
            "n_false_positives": doc["n_false_positives"],
            "n_false_negatives": doc["n_false_negatives"],
            "label_boundary_analysis": doc["label_boundary_analysis"],
            "false_positive_by_segment_prefix": doc["false_positive_by_segment_prefix"],
            "false_negative_by_attack_type": doc["false_negative_by_attack_type"],
            "top_deviated_features_in_false_positives": doc[
                "top_deviated_features_in_false_positives"
            ],
            "top_deviated_features_in_false_negatives": doc[
                "top_deviated_features_in_false_negatives"
            ],
            "governance_recommendations": doc["governance_recommendations"],
            "records": doc["records"],
            "note": doc["note"],
        }
        return assets.attach_scope(payload, doc, "errors")

    @app.get("/api/samples")
    def samples() -> dict[str, Any]:
        catalog = inference.sample_catalog()
        payload = {
            "samples": catalog["samples"],
            "n_samples": len(catalog["samples"]),
            "sealing": catalog["sealing"],
            "usage": (
                "把 sample_id 传给 POST /api/detect 即可用冻结模型对该窗口做真实推理；"
                "样本只来自 split-v3 val 与 OTIDS，HCRL test 折与 train 折均不可演示"
            ),
        }
        return assets.attach_scope(payload, _pick_doc("features_meta"), "features_meta")

    @app.post("/api/detect")
    def detect(req: DetectRequest) -> dict[str, Any]:
        if bool(req.sample_id) == bool(req.candump):
            raise HTTPException(
                status_code=400,
                detail="sample_id 与 candump 必须恰好提供一个（二选一）",
            )
        input_note: dict[str, Any]
        if req.sample_id:
            try:
                entry = inference.find_sample(req.sample_id)
                vector = inference.feature_row_for(req.sample_id)
            except inference.DetectInputError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            input_note = {
                "mode": "sample",
                "sample_id": req.sample_id,
                "window_id": entry["window_id"],
                "source": entry["source"],
                "split": entry["split"],
                "dominant_class": entry["dominant_class"],
                "label": entry["label"],
                "attack_ratio": entry["attack_ratio"],
                "caveat": (
                    "样本特征来自既有 window_features.parquet（已按 HCRL train 参数归一化），"
                    "推理结果仅作演示，不进入任何评估统计"
                ),
            }
        else:
            try:
                vector, parsed = inference.parse_candump(req.candump or "")
            except inference.DetectInputError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            input_note = {"mode": "candump", **parsed}

        result = inference.detect(vector, threshold=req.threshold)
        payload = {**result, "input": input_note, "sealing": assets.sealing_report()}
        return assets.attach_scope(
            payload,
            _pick_doc("threshold"),
            "threshold",
            "features_meta",
            "model_manifest",
            evaluated_on="single-window inference (demo only)",
            data_scope=(
                "演示样本：split-v3 val / OTIDS 窗口；粘贴文本：用户即时输入的 CAN 窗口。"
                "均不构成评估结论，不与 val / 跨域指标混用"
            ),
            level="window",
        )

    # 单进程本地预览：若 web/dist 已构建，直接把前端静态产物挂在 / 下，
    # 浏览器只需访问 uvicorn 一个端口（API 路由先注册，不受静态挂载影响）。
    from fastapi.staticfiles import StaticFiles

    from ..common import PROJECT_ROOT

    dist = PROJECT_ROOT / "web" / "dist"
    if dist.is_dir():
        app.mount("/", StaticFiles(directory=str(dist), html=True), name="web")

    return app


app = create_app()
