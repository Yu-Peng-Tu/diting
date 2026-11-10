# 谛听 W1/W2 · 管线：data → split → features → baseline → eval
# 每个目标可独立运行；依赖链由前置目标的文件产物保证，不隐式重跑。
# 用法：make install && make reproduce   （或分别 make data / split / features / baseline / eval）

PYTHON ?= .venv/bin/python
PIP    ?= uv pip install --python $(PYTHON)
RUN    := $(PYTHON) -m diting.cli
CFG    := configs/default.yaml

.DEFAULT_GOAL := help
.PHONY: help install data split features baseline eval test lint reproduce verify

help:
	@echo "谛听 W1/W2 可用目标："
	@echo "  install     创建 uv 虚拟环境并以可编辑模式安装 diting（含 dev 依赖）"
	@echo "  data        D1/D2 数据接入、校验、schema 归一（Parquet）与标签生成"
	@echo "  split       D3 生成带版本号的 train/val/test 划分清单并跑防泄漏断言"
	@echo "  features    D4 G1–G6 特征、训练折归一化参数、golden vector、泄漏断言"
	@echo "  baseline    L0 规则基线检测与留出验证集评估"
	@echo "  eval        汇总评估报告与产物 manifest 校验"
	@echo "  test        pytest 全量测试"
	@echo "  lint        ruff 静态检查（可选，不阻断）"
	@echo "  reproduce   一条命令复现整条链路 = data split features baseline eval test"

install:
	uv venv .venv
	$(PIP) -e ".[dev]"

data:
	$(RUN) data --config $(CFG)

split:
	$(RUN) split --config $(CFG)

features:
	$(RUN) features --config $(CFG)

baseline:
	$(RUN) baseline --config $(CFG)

eval:
	$(RUN) eval --config $(CFG)

test:
	$(PYTHON) -m pytest tests -q

lint:
	-$(PYTHON) -m ruff check src tests || true

reproduce: data split features baseline eval test
	@echo "[diting] W1/W2 全链路复现完成"

verify: eval test
	@echo "[diting] 验收项自检完成"
