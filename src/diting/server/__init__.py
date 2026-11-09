"""谛听 W5 · Web 检测工作台后端包（FastAPI）。

8GB 内存约束：在导入任何 numpy / sklearn 之前把 BLAS / OpenMP 线程数限制为 1。
该设置与 W3/W4 训练脚本一致，避免 HistGradientBoosting 的多线程内存放大。
"""

from __future__ import annotations

import os

for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_var] = "1"

__all__: list[str] = []
