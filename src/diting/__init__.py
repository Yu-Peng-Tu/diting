"""谛听（diting）：车载 CAN 总线入侵检测 AI 平台。

W1 范围：数据接入 → 统一 schema → 划分清单 → 特征工程 → L0 规则基线。
纯软件、零硬件依赖；纯软件独立实现；不含 RAG / 工单 / 审批等能力。
本包不提供推理服务与前端（W4 范围），也不安装 torch（W2 范围）。
"""

__version__ = "0.2.0"

# 泄漏防线：以下字段（及其任何布尔派生）严禁进入特征矩阵。
# 见策划方案 §4.6 L2 与 W1 验收口径。
FORBIDDEN_FEATURE_FIELDS = (
    "flag",
    "label",
    "is_attack",
    "attack_ratio",
    "can_id_zero",
    "is_can_id_zero",
    "is_0x000",
    "source_tag",
    "session_id",
    "segment_id",
)
