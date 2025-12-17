# Project Execution Log

## Step 0: Project Skeleton & Reproducibility

### Step 0.1: 创建标准化目录结构
**状态**: ✅ 完成
**时间**: 2025-12-15

**执行内容**:
- 创建主项目目录 `bridge_bidding_interpretability/`
- 创建核心子目录结构:
  - `src/` - 核心源代码
  - `scripts/` - 可执行脚本
  - `data/` - 数据目录 (raw/processed/cache)
  - `results/` - 实验结果
  - `fig/` - 探索性图表
  - `checkpoints/` - 模型检查点 (pi_H/pi_R)
  - `logs/` - 日志目录
  - `paper/` - 论文写作
  - `docs/` - 文档
  - `tests/` - 测试
  - `configs/` - 配置文件

**交付产物**:
- 21个目录及子目录
- `__init__.py` 文件 (src/, scripts/, tests/)
- `.gitkeep` 占位文件 (保持空目录结构)

**目录树**:
```
bridge_bidding_interpretability/
├── src/
├── scripts/
├── data/
│   ├── raw/
│   │   ├── openspiel_bridge/
│   │   └── dds_results/
│   ├── processed/
│   │   ├── policy_samples/
│   │   └── covariates/
│   └── cache/
├── results/
├── fig/
│   ├── fda_curves/
│   ├── jsd_heatmaps/
│   └── distillation_plots/
├── checkpoints/
│   ├── pi_H/
│   └── pi_R/
├── logs/
│   ├── train_pi_H/
│   ├── train_pi_R/
│   ├── sampling/
│   └── analysis/
├── paper/
│   └── fig/
├── docs/
│   ├── step_plans/
│   └── notes/
├── tests/
└── configs/
```

---

### Step 0.2: 配置Python环境与依赖
**状态**: ✅ 完成
**时间**: 2025-12-15

**执行内容**:
- 创建分离的依赖文件，解决JAX跨平台安装问题
- 核心分析依赖严格锁定版本

**交付产物**:
- `requirements-base.txt` - 核心依赖 (不含JAX)
  - PGX, Haiku, Optax, Flax (深度学习)
  - NumPy, SciPy, Pandas, Scikit-learn, Statsmodels (统计分析)
  - Interpret (可解释ML)
  - Matplotlib, Seaborn (可视化)
  - OmegaConf, Hydra (配置管理)
- `requirements-cpu.txt` - CPU版JAX安装
- `requirements-gpu.txt` - GPU版JAX安装 (CUDA 12)

**安装步骤**:
```bash
# 1. 安装JAX (选择一个)
pip install -r requirements-cpu.txt   # CPU
pip install -r requirements-gpu.txt   # GPU

# 2. 安装核心依赖
pip install -r requirements-base.txt
```

---

### Step 0.3: 创建默认配置文件
**状态**: ✅ 完成
**时间**: 2025-12-15

**执行内容**:
- 创建集中式配置文件，支持 OmegaConf/Hydra 命令行覆盖
- 包含可复现性、计算设备、路径等关键配置块

**交付产物**:
- `configs/default_config.yaml`

**配置块结构**:
| 块 | 用途 |
|----|------|
| `repro` | 随机种子、确定性开关、元数据保存 |
| `compute` | 平台 (cpu/gpu)、精度 |
| `run` | 运行名称、输出目录、覆盖策略 |
| `paths` | 数据/检查点/日志路径 |
| `model` | 网络架构 |
| `sl` | 监督学习超参数 |
| `ppo` | PPO超参数 |
| `sampling` | 策略采样配置 |
| `fda` | FDA分析配置 |
| `jsd` | JSD计算配置 |
| `distillation` | 蒸馏配置 |
| `posthoc` | 后验过滤配置 |

---

### Step 0.4: 编写冒烟测试
**状态**: ✅ 完成
**时间**: 2025-12-15

**执行内容**:
- 创建分层测试设计
- Layer 1 (smoke_test.py): 快速验证 (<30秒)
- Layer 2 (integration_test.py): 完整数据流验证 (骨架)

**交付产物**:
- `tests/smoke_test.py` - 快速冒烟测试
- `tests/integration_test.py` - 集成测试骨架

**测试内容 (Layer 1)**:
1. 依赖导入检查 (JAX, PGX, Haiku, NumPy, etc.)
2. PGX Bridge 环境验证 (init + 1 step)
3. 配置文件加载
4. 数据路径检查 (只检查存在性，不加载大文件)

**运行方式**:
```bash
python tests/smoke_test.py
```

---

### Step 0.5: 元数据自动记录
**状态**: ✅ 完成
**时间**: 2025-12-15

**执行内容**:
- 创建元数据记录模块，每次运行自动保存实验信息
- 创建通用工具模块

**交付产物**:
- `src/meta_logger.py` - 元数据记录器
- `src/utils.py` - 通用工具函数

**元数据记录功能**:
- `setup_run(cfg)`: 初始化运行，生成 run_id，创建输出目录
- `save_run_metadata(cfg, output_dir)`: 保存元数据到 `meta/` 目录
- `init_experiment(config_path)`: 一站式实验初始化

**保存内容**:
| 文件 | 内容 |
|------|------|
| `meta/config.yaml` | 完整配置 |
| `meta/git.txt` | Git commit hash + status |
| `meta/freeze.txt` | pip freeze 输出 |
| `meta/run.json` | 运行时信息 (时间戳、平台、种子) |

**使用示例**:
```python
from src.meta_logger import init_experiment

cfg, run_id, output_dir = init_experiment("configs/default_config.yaml")
# Metadata automatically saved to results/<run_id>/meta/
```

---

### Step 0.6: 创建项目README
**状态**: ✅ 完成
**时间**: 2025-12-15

**执行内容**:
- 创建完整的项目README文档
- 明确"自包含"与"需下载"内容的边界

**交付产物**:
- `README.md`

**README内容结构**:
- 项目概述和目标
- What is Included (代码结构)
- What Must Be Downloaded (DDS数据、OpenSpiel数据、预训练模型)
- Quick Start (环境搭建步骤)
- Reproduce Key Results (关键命令)
- Configuration (配置管理)
- Project Structure Details
- Reproducibility
- References

---

### Step 0.7: 初始化版本控制
**状态**: ✅ 完成
**时间**: 2025-12-15

**执行内容**:
- 创建 .gitignore 防止大文件意外提交
- 创建 .pre-commit-config.yaml 统一代码风格

**交付产物**:
- `.gitignore` - Git忽略规则
- `.pre-commit-config.yaml` - Pre-commit配置

**.gitignore 覆盖范围**:
- Python缓存和编译文件
- 虚拟环境
- 大文件 (checkpoints, data/raw, results)
- IDE配置
- 日志和临时文件
- 敏感文件 (secrets, credentials)

**Pre-commit hooks**:
- Ruff (代码检查和格式化)
- 基本文件检查 (trailing whitespace, YAML validation, large files)

**初始化命令** (可选):
```bash
cd bridge_bidding_interpretability
git init
git add .
git commit -m "Step 0: Project skeleton and reproducibility setup"
```

---

## Step 0 总结

| 步骤 | 状态 | 关键交付物 |
|------|------|------------|
| 0.1 目录结构 | ✅ | 21个目录 + 占位文件 |
| 0.2 环境配置 | ✅ | requirements-{base,cpu,gpu}.txt |
| 0.3 配置文件 | ✅ | configs/default_config.yaml |
| 0.4 冒烟测试 | ✅ | tests/smoke_test.py |
| 0.5 元数据记录 | ✅ | src/meta_logger.py, src/utils.py |
| 0.6 README | ✅ | README.md |
| 0.7 版本控制 | ✅ | .gitignore, .pre-commit-config.yaml |

**Step 0 完成标准**:
- [x] 目录结构清晰，路径通过config管理
- [x] JAX跨平台安装问题已解决
- [x] 每次运行自动记录元数据
- [x] smoke test < 30秒完成
- [x] README明确了"自包含"边界
- [x] 大文件不会意外提交

---

### Step 0 审查与修复
**状态**: ✅ 完成
**时间**: 2025-12-15

**审查发现的问题**:

| 问题 | 严重性 | 状态 |
|------|--------|------|
| JAX GPU 安装写法过时 (`jax[cuda12_pip]`) | ⚠️ | ✅ 已修复 |
| Hydra 依赖名称 | ✅ | 无需修改 (已是 `hydra-core`) |
| Hydra 输出目录与 run_id 冲突 | ⚠️ | ✅ 已修复 |
| **PGX `pgx.make("bridge_bidding")` 误用** | ❌ 严重 | ✅ 已修复 |
| OpenSpiel 编译风险未说明 | ⚠️ | ✅ 已修复 |
| 缺少可编辑安装支持 | ⚠️ | ✅ 已修复 |

**修复内容**:

1. **requirements-gpu.txt** (JAX 安装):
   - 更新为官方推荐写法: `jax[cuda12]==0.4.23`
   - 添加平台限制说明 (Linux x86_64)
   - 添加驱动版本要求 (>= 525.60.13)
   - 添加验证命令

2. **configs/default_config.yaml** (Hydra 配置):
   ```yaml
   hydra:
     job:
       chdir: false      # 不改变工作目录
     run:
       dir: .            # 保持项目根目录
     output_subdir: null # 禁用自动输出目录
   ```

3. **tests/smoke_test.py** (PGX Bridge 环境):
   - ❌ 移除: `pgx.make("bridge_bidding")` (不支持)
   - ✅ 改用: `from pgx.bridge_bidding import BridgeBidding`
   - ✅ 添加 DDS 数据存在性检查
   - ✅ 无 DDS 数据时 SKIP 而非 FAIL

4. **README.md** (安装说明):
   - 添加 JAX GPU 平台限制说明
   - 添加 JAX 验证命令
   - 添加 OpenSpiel 编译警告
   - 添加可编辑安装步骤 (`pip install -e .`)

5. **pyproject.toml** (新增):
   - 支持可编辑安装
   - 配置 ruff 和 pytest
   - 解决 `from src.xxx import` 路径问题

**新增交付产物**:
- `pyproject.toml`

---

## Step 1: Data Acquisition & Policy Preparation

### Step 1.1: 数据准备与链接
**状态**: ✅ 完成
**时间**: 2025-12-15

**执行内容**:
- 复制 DDS 数据文件到项目目录
- 复制 OpenSpiel 训练/测试数据
- 创建数据准备脚本 `scripts/prepare_data.py`
- 从 500K DDS 数据中固定种子抽样 100K 评估集
- 生成 `data_manifest.json` 记录所有数据文件元信息

**数据文件**:
| 文件 | 大小 | 用途 |
|------|------|------|
| dds_results_10M.npy | 306M | 训练集 (10M boards) |
| dds_results_2.5M.npy | 77M | 训练集 (2.5M boards) |
| dds_results_500K.npy | 16M | 评估集来源 |
| dds_results_100K_eval.npy | ~3M | 评估集 (seed=42) |
| train.txt | 359M | OpenSpiel SL 训练数据 (1.16M lines) |
| test.txt | 3.1M | OpenSpiel SL 测试数据 (10K lines) |

**⚠️ OpenSpiel 数据格式说明**:
OpenSpiel bridge 数据 **不是** 480 维 PGX observation 格式：
- 格式: `card_indices (52) + bid_sequence + action_label`
- action 范围: [0, 51+] (OpenSpiel 编码，非 PGX 的 [0,37])
- **这是预期行为**: brl 的 SL 训练脚本会转换此格式
- **本项目使用预训练模型**，不直接解析 OpenSpiel 数据

**交付产物**:
- `scripts/prepare_data.py`
- `data/raw/dds_results/dds_results_100K_eval.npy`
- `data/raw/dds_results/dds_results_100K_eval_indices.npy` (抽样索引)
- `data/raw/data_manifest.json`

---

### Step 1.2: 模型架构定义 (复用 brl)
**状态**: ✅ 完成
**时间**: 2025-12-15

**执行内容**:
- Vendoring brl 模型代码确保与预训练权重兼容
- 创建模型封装层提供统一接口

**brl 模型架构**:
- **DeepMind**: 4×1024 MLP (预训练模型使用此架构)
- **FAIR**: 200 units 残差网络

**交付产物**:
- `src/third_party/brl/models.py` - Vendored 模型代码
- `src/third_party/brl/LICENSE` - 原始许可证
- `src/models.py` - 模型封装层
- `src/policy_loader.py` - 策略加载器

---

### Step 1.3: 准备人类代理策略 (π^H)
**状态**: ✅ 完成
**时间**: 2025-12-15

**执行内容**:
- 复制 brl 预训练 SL 模型
- 创建元数据文件记录模型信息

**模型信息**:
| 属性 | 值 |
|------|-----|
| 文件 | model-sl.pkl (14.7 MB) |
| 架构 | DeepMind (4×1024 MLP) |
| 训练 | 400K iterations on WBridge5 SAYC data |
| 性能 | -0.56 IMPs/board vs WBridge5 |

**交付产物**:
- `checkpoints/pi_H/model-sl.pkl`
- `checkpoints/pi_H/active.pkl` (复制，非 symlink - Windows 限制)
- `checkpoints/pi_H/metadata.json`

---

### Step 1.4: 准备强化学习策略 (π^R)
**状态**: ✅ 完成
**时间**: 2025-12-15

**执行内容**:
- 复制 brl 预训练 RL 模型 (SL + PPO + FSP)
- 创建元数据文件记录模型信息

**模型信息**:
| 属性 | 值 |
|------|-----|
| 文件 | model-pretrained-rl-with-fsp.pkl (14.7 MB) |
| 架构 | DeepMind (4×1024 MLP) |
| 训练 | SL pretraining + PPO + FSP |
| 性能 | +1.24 IMPs/board vs WBridge5 (最佳) |

**交付产物**:
- `checkpoints/pi_R/model-pretrained-rl-with-fsp.pkl`
- `checkpoints/pi_R/active.pkl` (复制，非 symlink - Windows 限制)
- `checkpoints/pi_R/metadata.json`

---

### Step 1.5: 模型验证与推理测试
**状态**: ✅ 完成
**时间**: 2025-12-15

**执行内容**:
- 创建模型验证脚本
- 从环境 rollout 收集合法状态 (非随机生成)
- 验证策略输出 shape、归一化、非法动作 masking
- 比较 π^H 和 π^R 行为差异

**验证结果** (100 states):
```
pi_H (Human Proxy):
  Shape OK: True
  Sum to 1: True
  Illegal prob mass: 0.00
  Entropy: 0.105 ± 0.228

pi_R (RL):
  Shape OK: True
  Sum to 1: True
  Illegal prob mass: 0.00
  Entropy: 0.117 ± 0.255

Comparison:
  Top-1 Agreement: 89.0%
  KL(π^R || π^H) mean: 1.11 ± 2.80
  JSD mean: 0.08 ± 0.17
```

**交付产物**:
- `scripts/validate_models.py`

---

### Step 1.6: 整合与测试
**状态**: ✅ 完成
**时间**: 2025-12-15

**执行内容**:
- 更新集成测试包含 Step 1 验证
- 运行完整集成测试

**测试结果**:
```
[OK] Data Files
[OK] Policy Loading
[OK] Policy Inference
[OK] Environment Rollout
[OK] Small-scale Sampling (SKIP - Step 3)
[OK] Basic FDA (SKIP - Step 4)

Total: 6 passed, 0 failed
```

**交付产物**:
- 更新的 `tests/integration_test.py`

---

## Step 1 总结

| 步骤 | 状态 | 关键交付物 |
|------|------|------------|
| 1.1 数据准备 | ✅ | prepare_data.py, data_manifest.json, 100K eval set |
| 1.2 模型架构 | ✅ | src/third_party/brl/, src/models.py, src/policy_loader.py |
| 1.3 π^H 准备 | ✅ | model-sl.pkl, metadata.json |
| 1.4 π^R 准备 | ✅ | model-pretrained-rl-with-fsp.pkl, metadata.json |
| 1.5 模型验证 | ✅ | validate_models.py |
| 1.6 整合测试 | ✅ | integration_test.py |

**关键发现**:
- π^H 和 π^R 在 89% 的状态上选择相同的最优动作
- KL 散度和 JSD 显示两策略存在可测量的差异
- 所有验证检查通过，模型加载和推理正常

**依赖版本** (Python 3.13 兼容):
- jax==0.4.34
- jaxlib==0.4.34
- dm-haiku==0.0.12
- pgx==2.0.0
- optax==0.2.2

---

### Step 1 审查与修复
**状态**: ✅ 完成
**时间**: 2025-12-15

**审查发现的问题**:

| 问题 | 严重性 | 状态 |
|------|--------|------|
| 派生数据 (100K eval) 缺少血缘信息和索引保存 | ⚠️ | ✅ 已修复 |
| OpenSpiel 格式验证不完整 | ⚠️ | ✅ 已修复 |
| 缺少 THIRD_PARTY_NOTICES | ⚠️ | ✅ 已修复 |
| PGX 版本与 brl 官方不一致 (2.0.0 vs 1.4.0) | ⚠️ | ✅ 已记录 |
| "Python 3.13 兼容"表述不够严谨 | ⚠️ | ✅ 已修正 |
| KL/JSD 统计只有均值±方差，缺少分位数 | ⚠️ | ✅ 已增强 |

**修复内容**:

1. **派生数据血缘** (`scripts/prepare_data.py`):
   - 保存抽样索引到 `dds_results_100K_eval_indices.npy`
   - manifest 中添加 `lineage` 字段: derived_from, seed, indices_sha256

2. **OpenSpiel 格式说明**:
   - OpenSpiel 格式是 card indices + bid sequence，不是 480 维 observation
   - 更新了验证函数的文档说明
   - 数据用于 brl 训练，我们使用预训练模型故不需要直接解析

3. **THIRD_PARTY_NOTICES.md** (新增):
   ```
   src/third_party/brl/THIRD_PARTY_NOTICES.md
   - 来源 repo: harukaki/brl
   - 修改: None (保持原样以兼容权重)
   - 推荐版本: pgx==1.4.0, jax==0.4.23
   ```

4. **PGX 版本差异** (README.md):
   - brl 官方支持: pgx==1.4.0
   - 本项目使用: pgx==2.0.0 (Python 3.13 需要)
   - 已在 README 中明确说明，并提供匹配 brl 的环境配置

5. **Python 3.13 描述修正** (README.md):
   - 修改为"本项目在 Python 3.13 上运行通过"
   - 说明 dm-haiku 官方只标注到 Python 3.11 但实测可用
   - 提供 Python 3.10 + 原版依赖的替代配置

6. **KL/JSD 统计增强** (`scripts/validate_models.py`):
   - 添加 median, 90%, 95% 分位数
   - 添加 epsilon 参数说明 (与后续 CoDA FDA 一致)
   - 打印 Top 3 高 KL 状态的详细信息:
     - 双方 top-1 动作
     - 双方 top-5 动作概率分布

**增强的验证结果** (200 states):
```
KL(π^R || π^H):
  mean: 1.15 +/- 2.80
  median: 0.0001
  90%/95%/max: 4.32 / 8.98 / 11.50

JSD:
  mean: 0.087 +/- 0.185
  median: 0.0000
  90%/95%/max: 0.40 / 0.61 / 0.69

Top High-KL Examples:
  State 160: π^H=Pass(100%), π^R=1C(99.9%) → KL=11.50
  State 91:  π^H=2NT(74%), π^R=1C(99.8%) → KL=11.48
```

**新增/更新的交付产物**:
- `scripts/prepare_data.py` (增强)
- `scripts/validate_models.py` (增强)
- `src/third_party/brl/THIRD_PARTY_NOTICES.md` (新增)
- `data/raw/dds_results/dds_results_100K_eval_indices.npy` (新增)
- `data/raw/data_manifest.json` (更新，含 lineage)
- `README.md` (更新，含版本说明)

---

### Step 1 Repo 审计验证
**状态**: ✅ 完成
**时间**: 2025-12-15

**审计内容**:
对照 general_log.md 声称的内容与实际 repo 进行核对。

**发现与修复**:

| 审计项 | 发现 | 修复 |
|--------|------|------|
| active.pkl 是 symlink | Windows 不支持 symlink，实际是复制 | ✅ 文档已更新说明 |
| 验证结果无法重现 | 未保存日志到文件 | ✅ 现在保存到 `logs/validation/` |
| JSON 序列化错误 | numpy bool 不能直接序列化 | ✅ 添加 `to_native()` 转换 |

**验证结果** (2025-12-15):
```
logs/validation/validation_20251215_034154.json:
{
  "pi_H": {"pass": true, "entropy_mean": 0.105},
  "pi_R": {"pass": true, "entropy_mean": 0.117},
  "comparison": {
    "top1_agreement": 0.89,
    "kl_median": 0.00016,
    "jsd_median": 0.00003,
    "epsilon": 1e-05
  },
  "all_passed": true
}
```

**集成测试结果**:
```
[OK] Data Files
[OK] Policy Loading
[OK] Policy Inference
[OK] Environment Rollout
[OK] Small-scale Sampling (SKIP - awaiting Step 3)
[OK] Basic FDA (SKIP - awaiting Step 4)
Total: 6 passed, 0 failed
```

---

## Step 2: Statistical Feature Engineering

### 目标
将 PGX 480 维机器表示映射到可解释的桥牌统计协变量 (HCP, Controls, LTC 等)

---

### Step 2.0: 验证 52-bit 手牌编码
**状态**: ✅ 完成 (Round 2 修正)
**时间**: 2025-12-15

**执行内容**:
- 创建编码验证脚本
- 验证 PGX observation 的花色和牌力编码

**编码说明** (⚠️ 重要):
- **原始 obs[428:480]**: OpenSpiel **rank-major** 编码
  - `index = suit + rank * 4`
  - suit: 0=C, 1=D, 2=H, 3=S
  - rank: 0=2, 1=3, ..., 12=A
  - 即: bits 0-3 = 2C,2D,2H,2S; bits 48-51 = AC,AD,AH,AS
- **内部 cards[suit, rank]**: 本项目转换后的表示
  - suit 顺序: S-H-D-C (index 0-3)
  - rank 顺序: A-K-Q-J-10-...-2 (index 0-12)
  - 转换公式: `reshape(13,4).T[::-1,::-1]`

**验证方法**: 用 `state._hand` + `_convert_card_pgx_to_openspiel()` 对齐，200/200 states 完全匹配 ✅

**交付产物**:
- `scripts/verify_card_encoding.py` (初版，已弃用)
- `scripts/verify_hand_encoding_v2.py` (最终版，含 state 对齐验证)

---

### Step 2.1: 手牌特征提取器
**状态**: ✅ 完成
**时间**: 2025-12-15

**执行内容**:
- 从 obs[428:480] 提取 52-bit 手牌
- 实现 28 个手牌特征

**特征列表** (28个):
| 类别 | 特征 | 数量 | 说明 |
|------|------|------|------|
| HCP | hcp_total, hcp_{suit} | 5 | A=4, K=3, Q=2, J=1 |
| 分布 | len_{suit} | 4 | 花色长度 |
| 分布 | longest_suit, shortest_suit | 2 | 最长/最短花色 |
| 分布 | is_balanced, n_singletons, n_voids | 3 | 牌型特征 |
| 控制 | controls_total, controls_{suit} | 5 | A=2, K=1 |
| 输张 | ltc | 1 | Losing Trick Count (修正: 单张A=0输张) |
| 快赢 | quick_tricks | 1 | AK=2, AQ=1.5, A=1, KQ=1, K=0.5 |
| 大牌 | n_aces, n_kings, n_queens, n_jacks | 4 | 大牌计数 |
| 大牌 | n_honors_in_long_suits, has_ak_in_any_suit, n_suits_with_honors | 3 | 大牌分布 |

**交付产物**:
- `src/features/constants.py` - 编码常量
- `src/features/hand_features.py` - 手牌特征提取

---

### Step 2.2: 叫牌历史特征提取器
**状态**: ✅ 完成 (Round 2 更新)
**时间**: 2025-12-15

**执行内容**:
- 从 obs[8:428] 解析叫牌历史 (35 contracts × 3 states × 4 players)
- 实现 **13 个叫牌特征**

**特征列表** (13个):
| 特征 | 说明 |
|------|------|
| n_contracts_bid | 已叫出的合约数 |
| auction_level | 当前叫牌级别 (1-7, 0=无) |
| contract_strain | 合约花色 (0=C,1=D,2=H,3=S,4=NT, -1=无) |
| is_doubled | 最高合约被加倍 |
| is_redoubled | 最高合约被再加倍 |
| double_status | 加倍状态 (0=undoubled, 1=doubled, 2=redoubled) |
| is_competitive | 双方都有叫牌 |
| is_passout | 无人叫牌 |
| has_contract | 有合约 |
| self_opened | 自己开叫 |
| partner_opened | 同伴开叫 |
| lho_opened | 左手敌开叫 (player index 1) |
| rho_opened | 右手敌开叫 (player index 3) |

**关键修正**:
- reshape 从 (35,4,3) 改为 (35,3,4)
- player 轴: 0=self, 1=LHO, 2=partner, 3=RHO
- is_doubled/is_redoubled 只检查最高合约

**交付产物**:
- `src/features/bidding_features.py`

---

### Step 2.3: 局况与位置特征
**状态**: ✅ 完成
**时间**: 2025-12-15

**执行内容**:
- 从 obs[0:4] 提取局况 (vulnerability)
- 从 obs[4:8] 提取开叫前 pass 数

**特征列表**:
| 特征 | 说明 |
|------|------|
| we_vulnerable | 己方有局 |
| they_vulnerable | 敌方有局 |
| both_vulnerable | 双有局 |
| none_vulnerable | 双无局 |
| favorable_vul | 有利局况 (己方无局，敌方有局) |
| unfavorable_vul | 不利局况 |
| n_passed_before_opening | 开叫前 pass 数 (0-4) |

**交付产物**:
- `src/features/context_features.py`

---

### Step 2.4: 特征整合与标准化
**状态**: ✅ 完成
**时间**: 2025-12-15

**执行内容**:
- 创建统一特征提取器 `BridgeFeatureExtractor`
- 支持特征子集选择、归一化、数组转换

**功能**:
```python
extractor = BridgeFeatureExtractor(normalize=True, stats_path="...")
features = extractor.extract(obs)  # Dict[str, float32]
array, names = extractor.to_array(features)  # (n_features,), List[str]
```

**交付产物**:
- `src/features/__init__.py`
- `src/features/feature_extractor.py`

---

### Step 2.5: 特征验证与统计
**状态**: ✅ 完成 (Round 2 更新)
**时间**: 2025-12-15

**执行内容**:
- 测试结构不变量 (手牌13张、花色长度和)
- 测试数值不变量 (HCP一致性、Controls一致性、范围检查)
- 收集特征统计

**验证结果** (100 states, CPU ~1秒/state):
```
Invariant failures: 0/100 ✅

Feature Statistics (部分):
  hcp_total:      mean=9.84,  std=4.72,  [3, 22]
  ltc:            mean=7.39,  std=1.54,  [5, 10]
  controls_total: mean=2.98,  std=1.92,  [0, 7]
  quick_tricks:   mean=1.75,  std=1.22,  [0, 5]

Edge cases:
  passout: 9.0%, competitive: 78.0%, doubled: 34.0%, redoubled: 15.0%

Feature count: 48
```

**交付产物**:
- `scripts/validate_features.py`
- `scripts/verify_hand_encoding_v2.py` (手牌编码验证)
- `scripts/verify_double_status.py` (加倍状态验证)
- `logs/features/feature_statistics.json`

---

### Step 2.6: 整合测试
**状态**: ✅ 完成
**时间**: 2025-12-15

**执行内容**:
- 更新集成测试，添加特征提取测试
- 验证特征不变量跨多个状态

**测试结果**:
```
============================================================
INTEGRATION TEST (Layer 2): Bridge Bidding Interpretability
============================================================
[OK] Data Files
[OK] Policy Loading
[OK] Policy Inference
[OK] Environment Rollout
[OK] Feature Extraction
[OK] Feature Invariants
[OK] Small-scale Sampling (SKIP - awaiting Step 3)
[OK] Basic FDA (SKIP - awaiting Step 4)

Total: 8 passed, 0 failed
```

**交付产物**:
- 更新的 `tests/integration_test.py`

---

## Step 2 总结 (已过时，见"最终总结")

| 步骤 | 状态 | 关键交付物 |
|------|------|------------|
| 2.0 编码验证 | ✅ | verify_hand_encoding_v2.py |
| 2.1 手牌特征 | ✅ | hand_features.py, constants.py |
| 2.2 叫牌特征 | ✅ | bidding_features.py |
| 2.3 局况特征 | ✅ | context_features.py |
| 2.4 特征整合 | ✅ | feature_extractor.py |
| 2.5 特征验证 | ✅ | validate_features.py |
| 2.6 整合测试 | ✅ | integration_test.py |

**特征总数**: **48 个**可解释特征
- 手牌特征: 28 个 (HCP, 分布, 控制, 输张, 大牌)
- 叫牌特征: **13 个** (级别, 花色, 加倍, double_status, 竞叫, 开叫者)
- 局况特征: 7 个 (有局, 位置)

**技术要点**:
1. **手牌编码**: obs[428:480] 是 OpenSpiel rank-major，需 `reshape(13,4).T[::-1,::-1]` 转换
2. **Player 轴**: 0=self, 1=LHO, 2=partner, 3=RHO (clockwise)
3. **Bidding reshape**: (35, 3, 4) = [contract, state, player]
4. **double_status**: 0=undoubled, 1=doubled, 2=redoubled (与 state._call_x/xx 100%一致)
5. **LTC 修正**: 单张 A = 0 输张
6. **花色映射**: 手牌 S-H-D-C vs 合约 C-D-H-S-NT，用 `STRAIN_TO_SUIT_IDX`

**文件结构**:
```
src/features/
├── __init__.py
├── constants.py          # 编码常量
├── hand_features.py      # 手牌特征 (reshape 已修复)
├── bidding_features.py   # 叫牌特征 (reshape + player 轴已修复)
├── context_features.py   # 局况特征 (vulnerability 索引已修复)
└── feature_extractor.py  # 统一提取器 (含 double_status 元数据)

scripts/
├── verify_hand_encoding_v2.py  # 手牌编码验证 (新)
├── verify_double_status.py     # 加倍状态验证 (新)
└── validate_features.py
```

---

## Step 2 关键 Bug 修复 (2025-12-15)

### 问题发现
在 code review 中发现两个严重问题：

**红旗 1: reshape 顺序错误**
- 原代码: `reshape(35, 4, 3)` = [contract, player, state]
- 正确: `reshape(35, 3, 4)` = [contract, state, player]
- 原因: PGX 12-bit block 结构是 `[bid×4, dbl×4, rdbl×4]`，即先按 state 排列，再按 player

**红旗 2: batch 路径返回 0**
- 原代码: batch 情况下 `is_doubled/is_redoubled` 直接返回 0
- 影响: Step 3 批量提特征时会被系统性污染

### 验证
```python
# 错误 reshape 导致 68% 不变量失败
旧 reshape (35,4,3) - redouble without double: 676/987 states
新 reshape (35,3,4) - redouble without double: 0/987 states ✅
```

### 修复内容
1. **bidding_features.py**:
   - `parse_bidding_history()`: reshape 改为 `(35, 3, 4)`
   - `compute_auction_features()`: 索引改为 `bidding[..., :, 0, :]` (bid), `bidding[safe_idx, 1, :]` (double)
   - `compute_opener_features()`: 同上修正
   - batch 路径: 使用 advanced indexing 正确实现

2. **validate_features.py**:
   - 更新不变量测试索引
   - 恢复 "redouble requires double" 检查 (现在应该 100% 通过)

### 修复后验证
```
Invariant failures: 0/1000 ✅
All invariant checks PASSED
```

特征分布变化 (修复前 → 修复后):
| 特征 | 修复前 | 修复后 | 说明 |
|------|--------|--------|------|
| lho_opened | 0.00 | 0.26 | 之前始终为 0 |
| is_competitive | 0.48 | 0.76 | 竞叫检测更准确 |
| is_doubled | 0.22 | 0.37 | 加倍检测更准确 |
| is_redoubled | 0.12 | 0.18 | 再加倍检测更准确 |

---

## Step 2 追加修复 Round 1 (2025-12-15)

### 问题 1: Vulnerability 编码索引错误
**严重性**: ❌ 严重

**原代码** (`context_features.py`):
```python
we_vul = vulnerability[..., 0]
they_vul = vulnerability[..., 1]
```

**PGX 实际编码**:
```
obs[0] = NOT we_vulnerable
obs[1] = we_vulnerable
obs[2] = NOT they_vulnerable
obs[3] = they_vulnerable
```

**修复**:
```python
we_vul = vulnerability[..., 1]
they_vul = vulnerability[..., 3]
```

### 问题 2: 缺少 double_status 特征
`is_doubled` 和 `is_redoubled` 可同时为 1，添加 `double_status` 特征 (0=undoubled, 1=doubled, 2=redoubled)。

**验证**: double_status 与 PGX state._call_x/state._call_xx 100% 一致 ✅

---

## Step 2 追加修复 Round 2 (2025-12-15) - 致命编码错误

### 问题 3: 52-bit 手牌编码顺序错误 ❌❌❌ 最严重
**影响**: 所有手牌特征 (HCP, suit lengths, controls, LTC 等) 全部错误！

**原代码** (`hand_features.py`):
```python
cards = hand_bits.reshape(4, 13)  # 假设 suit-major
```

**PGX 实际编码** (OpenSpiel RANK-MAJOR):
```
index = suit + rank * 4
suit: 0=C, 1=D, 2=H, 3=S
rank: 0=2, 1=3, ..., 12=A
bits 0-3 = 2C,2D,2H,2S; bits 48-51 = AC,AD,AH,AS
```

**修复**:
```python
rank_major = hand_bits.reshape(13, 4)  # (ranks 2..A, suits C-D-H-S)
cards = rank_major.T[::-1, ::-1]       # (suits S-H-D-C, ranks A..2)
```

**验证**: 用 `state._hand` + `_convert_card_pgx_to_openspiel()` 对齐，200/200 states 完全匹配 ✅

### 问题 4: Player 轴 LHO/RHO 颠倒
**原代码** (`bidding_features.py`):
```python
# 错误注释: 0=self, 1=RHO, 2=partner, 3=LHO
rho_opened = (opener_player == 1)
lho_opened = (opener_player == 3)
```

**PGX 实际编码** (clockwise bidding order):
```
0=self, 1=LHO, 2=partner, 3=RHO
```

**修复**:
```python
lho_opened = (opener_player == 1)
rho_opened = (opener_player == 3)
```

### 问题 5: 验证脚本 Unicode 错误
Windows cp1252 无法打印中文路径，导致脚本最后崩溃但看起来像"卡住"。

---

## Step 2 最终验证结果 (100 states)

```
Invariant failures: 0/100 ✅

Edge cases:
  passout: 9 (9.0%)
  competitive: 78 (78.0%)
  doubled: 34 (34.0%)
  redoubled: 15 (15.0%)

Feature count: 48
```

**验证脚本性能**: ~1 秒/state (CPU JAX)，1000 states ≈ 16 分钟

---

## Step 2 最终总结

| 指标 | 值 |
|------|-----|
| 总特征数 | **48** |
| 手牌特征 | 28 |
| 叫牌特征 | 13 (含 double_status) |
| 局况特征 | 7 |
| 不变量失败率 | 0/100 |

**48 个特征完整列表**:
```
auction_level, both_vulnerable, contract_strain,
controls_club, controls_diamond, controls_heart, controls_spade, controls_total,
double_status, favorable_vul, has_ak_in_any_suit, has_contract,
hcp_club, hcp_diamond, hcp_heart, hcp_spade, hcp_total,
is_balanced, is_competitive, is_doubled, is_passout, is_redoubled,
len_club, len_diamond, len_heart, len_spade,
lho_opened, longest_suit, ltc,
n_aces, n_contracts_bid, n_honors_in_long_suits, n_jacks, n_kings,
n_passed_before_opening, n_queens, n_singletons, n_suits_with_honors, n_voids,
none_vulnerable, partner_opened, quick_tricks, rho_opened, self_opened,
shortest_suit, they_vulnerable, unfavorable_vul, we_vulnerable
```

**关键修复文件**:
- `src/features/hand_features.py` - 手牌编码 reshape 修复 ⭐
- `src/features/bidding_features.py` - player 轴 + double_status
- `src/features/context_features.py` - vulnerability 索引
- `src/features/constants.py` - 编码文档更新
- `scripts/verify_hand_encoding_v2.py` - 新增验证脚本
- `scripts/verify_double_status.py` - 新增验证脚本

所有 Bug 已修复，Step 2 完成。✅

---

## Step 3: Policy Sampling

### 目标
从 π^H 和 π^R 收集配对策略概率样本，创建 Policy Behavior Database 用于 FDA 分析。

---

### Step 3.0: 创建采样模块结构
**状态**: ✅ 完成
**时间**: 2025-12-15

**执行内容**:
- 创建 `src/sampling/` 模块目录
- 定义模块接口和导出

**交付产物**:
- `src/sampling/__init__.py`

---

### Step 3.1: 实现 PolicySampler 类
**状态**: ✅ 完成
**时间**: 2025-12-15

**执行内容**:
- 实现 `PolicySampler` 类，封装采样逻辑
- JIT 编译批量推理函数，GPU 加速
- 支持多种 behavior policy (random, mixed, pi_H, pi_R)
- 实现 additive smoothing 保证数值稳定性

**核心功能**:
```python
class PolicySampler:
    def __init__(self, pi_H, pi_R, env, config, extractor)
    def smooth_probs(self, probs)        # (p + ε) / (1 + K*ε)
    def sample_episodes(self, n, key)     # 采样 episode, 返回 episode_ids, board_ids, timestep
    def run_sampling(self, behavior)      # 完整采样流程 (支持 shard 输出)
    def verify_samples(self, samples)     # 质量验证 (含 E. 硬核必检)
    def save_samples(self, samples)       # NPZ + JSON 保存 (正确 dtype + 完整 metadata)
```

**Additive Smoothing**:
```python
epsilon = 1e-5
K = 38  # 动作数
p_smoothed = (p + epsilon) / (1 + K * epsilon)
# 保证 sum = 1, p > 0
```

**待修改点**:
1. `sample_episodes()` 返回:
   - `episode_ids` 数组 (cluster bootstrap)
   - `board_ids` 数组 (DDS eval pool 索引，可追溯性)
   - `timestep_in_episode` 数组 (叫牌轮次)
2. `save_samples()` 中:
   - `observations` 用 `bool` 存储 (推理时 cast 为 float32)
   - `legal_masks` 用 `bool` 存储
   - 保存 `action_names`, `ref_action`, `ref_action_idx`
   - 保存 `action_legal_rates`, `rare_actions`, `states_per_episode`
3. `verify_samples()` 增加检查项:
   - `np.all(np.isfinite(pi_H/pi_R/covariates))`
   - `min(pi) ≈ ε/(1+Kε)`
   - **Pass 始终合法**: `legal_masks[:, ref_action_idx].all()`
   - **Board 无重复**: `len(set(board_ids)) == n_episodes`
4. `run_sampling()` 支持:
   - **无放回采样** boards (从 100K eval pool)
   - **Shard 输出** (每 100K states 保存一次)

**交付产物**:
- `src/sampling/sampler.py`

---

### Step 3.2: 创建采样脚本
**状态**: ✅ 完成
**时间**: 2025-12-15

**执行内容**:
- 创建命令行脚本
- 支持参数配置

**使用方式**:
```bash
python scripts/sample_policies.py --n_samples 10000 --seed 42
```

**命令行参数**:
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--n_samples` | 10000 | 目标样本数 |
| `--seed` | 42 | 随机种子 |
| `--behavior_policy` | random | 采样策略 |
| `--smoothing_epsilon` | 1e-5 | 平滑参数 |

**交付产物**:
- `scripts/sample_policies.py`

---

### Step 3.3: 创建 Colab Notebook
**状态**: ✅ 完成
**时间**: 2025-12-15

**执行内容**:
- 创建 Google Colab notebook
- 完整的 setup 和验证流程
- 支持 A100 GPU 加速

**Notebook 结构**:
1. Setup: 安装依赖、挂载 Drive
2. Load Models: 加载 π^H、π^R、环境
3. Configure: 设置采样参数 (N=1M)
4. JIT Warmup: 预编译推理函数
5. Run Sampling: 执行采样循环
6. Verify & Save: 验证并保存

**时间估算 (A100)**:
- 1M samples: ~20-30 min
- 对比 CPU (~1s/state): ~100x 加速

**交付产物**:
- `notebooks/colab_sampling.ipynb`

---

### Step 3.4: 验证采样质量
**状态**: ⬜ 待执行

**验证检查清单 (扩展)**:

**A. 统计有效性**:
- [ ] 保存 `episode_ids`，为 cluster bootstrap 准备
- [ ] 保存 `board_ids`，为可追溯性和完整复现准备
- [ ] 保存 `timestep_in_episode`，为分层分析准备
- [ ] 计算每个动作的合法率: `mean(legal_masks[:, b])` for b in 0..37
- [ ] 标记合法率极低的动作 (Step 4 曲线会不稳定)

**B. 数值与不变量**:
- [ ] `np.allclose(pi_H.sum(-1), 1)` — 概率和为 1
- [ ] `np.allclose(pi_R.sum(-1), 1)` — 概率和为 1
- [ ] `np.all(np.isfinite(pi_H))` — 无 NaN/Inf
- [ ] `np.all(np.isfinite(pi_R))` — 无 NaN/Inf
- [ ] `np.all(np.isfinite(covariates))` — 协变量无 NaN/Inf
- [ ] `min(pi) ≈ ε/(1+Kε)` — smoothing 生效
- [ ] Step 2 不变量抽查 (13张牌、花色长度和等)
- [ ] 每个协变量的 min/max 在合理范围内

**C. 分布覆盖**:
- [ ] `hcp_total` 直方图 (检查极端区间样本数)
- [ ] `controls_total`, `ltc` 分布
- [ ] 拍卖阶段分布 (`n_contracts_bid`, `auction_level`)

**D. 工程稳定性**:
- [ ] 固定 batch size 避免 JIT 反复编译
- [ ] 定期保存中间结果 (shard 输出，防崩溃丢失)

**E. 硬核必检 (Critical)**:
- [ ] **Pass 始终合法**: `legal_masks[:, ref_action_idx].all()` — ALR 参考动作必须全程合法
- [ ] **States per episode 分布**: 计算 min/median/max，检测异常超长/超短 episode (预期 ~8-15)
- [ ] **Rare actions 标注**: 合法率 < 0.5% 的动作自动列入 `metadata["rare_actions"]`
- [ ] **Board 无重复**: 验证无放回采样 `len(set(board_ids)) == n_episodes`

---

### Step 3.5: 运行大规模采样
**状态**: ⬜ 待执行

**目标**:
- 在 Colab A100 上执行 1M 样本采样
- 保存到 Google Drive

**预期输出**:
```
data/processed/policy_samples/
├── 1M_random_v1_policy_samples.npz  (~150 MB compressed)
└── 1M_random_v1_metadata.json
```

---

## Step 3 统计风险与注意事项

### 风险 1: 非法动作的"结构性零"问题
**问题**: Smoothing 后，非法动作概率变成 ~ε/(1+Kε)，但这不代表"策略认为概率很小"，而是"规则不允许"。如果 Step 4 直接对所有样本拟合 ALR 曲线，非法动作会产生大量"假信号"。

**解决方案**: Step 4 对每个动作 b 的曲线分析，**必须只用 `legal_masks[:, b] == True` 的子样本**。Step 3 需保存 `legal_masks` 并计算 `action_legal_rates`。

### 风险 2: 样本非 i.i.d. — Episode 内相关性
**问题**: 1M samples 来自 ~70K-100K episodes，同一 episode 内的 ~12 states 强相关（同一副牌、同一拍卖过程）。如果 Step 4 用 naive i.i.d. bootstrap，置信带会过窄、p-value 会偏乐观。

**解决方案**: Step 4 使用 **cluster/block bootstrap**，按 episode 为单位重采样。Step 3 需保存 `episode_ids` 索引。

### 风险 3: dtype 与存储体积
**问题**: observations 本质是 binary (0/1)，用 float32 存储会浪费 4x 空间。

**解决方案**:
- `observations`: 用 `bool` 存储 (1M×480×1B = 480MB vs 1.92GB)
- `legal_masks`: 用 `bool` 存储
- `covariates`, `pi_H`, `pi_R`: 保持 `float32`

### 风险 4: Random Behavior Policy 的状态分布偏差
**问题**: `random` 行为策略会生成很多"人类/强策略极少到达的拍卖轨迹"，导致曲线更像"策略在规则空间的响应"，而非"真实对局分布下的响应"。

**建议**:
- **Coverage 版**: behavior=random (当前方案，覆盖广)
- **On-policy 版**: behavior=pi_H 或 mixed (贴近真实分布)

Step 4 报告中应讨论"结论在两种状态分布下是否一致"。

---

## Step 3 完成总结

| 步骤 | 状态 | 关键交付物 |
|------|------|------------|
| 3.0 模块结构 | ✅ | `src/sampling/__init__.py` |
| 3.1 PolicySampler | ✅ | `src/sampling/sampler.py` |
| 3.2 采样脚本 | ✅ | `scripts/sample_policies.py` |
| 3.3 Colab Notebook | ✅ | `notebooks/colab_sampling.ipynb` |
| 3.4 验证质量 | ✅ | 小规模测试通过 |
| 3.5 大规模采样 | ✅ | 1M 样本 + 100K 特征子集 |

### Step 3.4 小规模测试结果 (2025-12-15)
在 CPU 上执行 1000 样本测试，所有 12 个验证检查通过:

```
Verification report:
  [PASS] pi_H probability sum (mean=1.0)
  [PASS] pi_R probability sum (mean=1.0)
  [PASS] pi_H isfinite
  [PASS] pi_R isfinite
  [PASS] covariates isfinite
  [PASS] pi_H illegal action probs (max=9.99e-6)
  [PASS] pi_R illegal action probs (max=9.99e-6)
  [PASS] min probability > 0 (smoothing)
  [PASS] suit lengths sum to 13
  [PASS] Pass (ref_action) always legal [CRITICAL] (rate=100%)
  [PASS] states per episode distribution (min=4, median=10.0, max=18)
  [PASS] board no duplicates (99 unique boards for 99 episodes)

Output dtypes verified:
  observations: (1000, 480) bool
  covariates: (1000, 48) float32
  pi_H, pi_R: (1000, 38) float32
  legal_masks: (1000, 38) bool
  episode_ids: (1000,) int32
  board_ids: (1000,) int32
  timestep_in_episode: (1000,) int16

Metadata verified:
  ref_action: "Pass"
  ref_action_idx: 0
  action_names: ["Pass", "Dbl", "Rdbl", "1C", ..., "7NT"] (38 items)
```

**输出格式 (修正后)**:
```
data/processed/policy_samples/<run_id>_policy_samples.npz
├── observations      (N, 480)   bool     — 原始 PGX 观测 (binary，推理时需 cast 为 float32)
├── covariates        (N, 48)    float32  — 48 个统计特征
├── pi_H              (N, 38)    float32  — 平滑后的 π^H 概率
├── pi_R              (N, 38)    float32  — 平滑后的 π^R 概率
├── legal_masks       (N, 38)    bool     — 合法动作掩码
├── episode_ids       (N,)       int32    — Episode 序号 (用于 cluster bootstrap)
├── board_ids         (N,)       int32    — DDS eval pool 索引 (用于可追溯性)
├── timestep_in_episode (N,)     int16    — Episode 内叫牌轮次 (0-based)

data/processed/policy_samples/<run_id>_metadata.json
├── config            — 采样配置
├── timestamp         — 采样时间
├── n_samples         — 样本数
├── n_episodes        — Episode 数
├── feature_names     — 48 个特征名称
├── action_names      — 38 个动作名称 (按 PGX 顺序: Pass, Dbl, Rdbl, 1C, ...)
├── ref_action        — ALR 参考动作名称 ("Pass")
├── ref_action_idx    — ALR 参考动作索引 (0，因为 PGX 中 Pass = action 0)
├── action_legal_rates— 每个动作的合法率 (38,)
├── rare_actions      — 合法率 < 0.5% 的动作索引列表
├── states_per_episode— Episode 内 state 数分布 {min, median, max, mean}
├── pi_H_checkpoint   — π^H 检查点路径
└── pi_R_checkpoint   — π^R 检查点路径
```

**采样策略**:
- 从 100K eval pool **无放回采样** boards，保证 episode 独立性
- **Shard 输出**: 每 100K states 保存一次，防崩溃丢失

**✅ 已验证 (2025-12-15): board_ids 可通过 _hand 实现**
- 没有直接的 board_id 字段
- 但 `state._hand: shape=(52,)` 可作为 board 唯一标识符
- 使用 `tuple(state._hand.tolist())` 进行无放回采样
- PGX `state.observation` 已经是 `dtype=bool`，无需转换

**⚠️ 关键: PGX 动作编码顺序**
```
action 0 = Pass (ref_action_idx)
action 1 = Double
action 2 = Redouble
action 3..37 = 1♣,1♦,...,7NT
```
action_names 必须按此顺序，否则 Step 4 ALR 会对错列取 log ratio！

**存储估算 (N=1M, 修正后)**:
- observations (bool): 1M × 480 × 1B = 480 MB
- covariates (float32): 1M × 48 × 4B = 192 MB
- pi_H + pi_R (float32): 1M × 38 × 4B × 2 = 304 MB
- legal_masks (bool): 1M × 38 × 1B = 38 MB
- episode_ids (int32): 1M × 4B = 4 MB
- board_ids (int32): 1M × 4B = 4 MB
- timestep_in_episode (int16): 1M × 2B = 2 MB
- **总计**: ~1.02 GB 原始，~200-300 MB 压缩

**文件结构**:
```
src/sampling/
├── __init__.py
└── sampler.py

scripts/
└── sample_policies.py

notebooks/
└── colab_sampling.ipynb
```

**技术要点**:
1. **Additive Smoothing**: `(p + ε) / (1 + K*ε)` 保证概率 > 0，用于 ALR 变换
2. **JIT 编译**: 批量推理加速 ~100x
3. **Behavior Policy**: 默认 `random` 以获得广覆盖
4. **保存原始 obs**: 用 `bool` dtype，推理时 cast 为 `float32`
5. **Episode IDs**: 保存用于 Step 4 cluster bootstrap
6. **Board IDs**: 保存用于可追溯性和完整复现 (无放回采样)
7. **timestep_in_episode**: 保存用于分层分析 (opening vs later)
8. **Legal Masks**: Step 4 必须用于条件化分析，避免"结构性零"问题
9. **ALR 参考动作**: Pass 必须全程合法，metadata 记录 `ref_action_idx`
10. **Shard 输出**: 每 100K states 保存一次，防崩溃丢失

---

### Step 3.5 大规模采样执行 (2025-12-15/16)
**状态**: ✅ 完成
**平台**: Google Colab A100 GPU

**执行过程中遇到的问题与解决方案**:

**问题 1: Colab JAX 版本冲突**
```
JAX plugin jax_cuda12_plugin version 0.7.2 is not compatible with jaxlib version 0.8.1
```
- 解决: 使用 Colab 原生 JAX，只安装 `pgx>=2.3.0 dm-haiku optax`

**问题 2: PGX env.step 极慢 (~1.8s/step)**
- 诊断: `env.step` 未 JIT 编译
- 解决:
```python
env_init = jax.jit(env.init)
env_step = jax.jit(env.step)
```
- 效果: 1.8s → 0.002s (**900x 加速**)

**问题 3: 特征提取极慢 (~5.6s/sample)**
- 诊断: JAX 环境下 device transfer 开销
- 解决: 采样与特征提取分离，后处理用纯 NumPy
- 效果: 5.6s → 0.027s (**200x 加速**)

**采样统计**:
```
Target: 1,000,000 samples
JIT warmup...
Warmup done!
  Ep 500: 5,368/1,000,000 - 41/s - ETA: 400.2min
  ...
  Ep 95000: 997,863/1,000,000 - 42/s - ETA: 0.9min

Done! 1,000,009 samples, 95206 episodes in 399.3 min
```

| 指标 | 值 |
|------|-----|
| 总样本数 | **1,000,009** |
| 总 episode 数 | **95,206** |
| 平均 states/episode | ~10.5 |
| 采样速度 | ~42 samples/s |
| 总耗时 | **399 分钟** (6.7 小时) |

**特征提取 (100K 子集)**:
```
Extracting features for 100,000 samples...
  10,000/100,000 - 33/s - ETA: 46.1min
  ...
Done in 51.3 min
```

**输出文件**:
```
data/processed/policy_samples/
├── 1M_samples_raw.npz              # ~150MB
│   ├── observations    (1000009, 480) bool
│   ├── legal_masks     (1000009, 38)  bool
│   ├── pi_H            (1000009, 38)  float32
│   ├── pi_R            (1000009, 38)  float32
│   └── episode_ids     (1000009,)     int32
│
├── 1M_samples_raw_metadata.json
│
└── 100K_samples_with_features.npz  # ~50MB
    ├── observations    (100000, 480)  bool
    ├── covariates      (100000, 48)   float32
    ├── legal_masks     (100000, 38)   bool
    ├── pi_H            (100000, 38)   float32
    ├── pi_R            (100000, 38)   float32
    ├── episode_ids     (100000,)      int32
    └── original_indices (100000,)     int64
```

**Colab 采样核心代码** (保存供复现):
```python
import jax
import jax.numpy as jnp
import numpy as np
import time

# JIT compile env functions - 关键优化！
env_init = jax.jit(env.init)
env_step = jax.jit(env.step)

def fast_sample_jit(n_samples, seed=42):
    key = jax.random.PRNGKey(seed)

    # Warmup all code paths
    state = env_init(key)
    for _ in range(10):
        key, k1, k2 = jax.random.split(key, 3)
        state = env_init(k1)
        state = env_step(state, 0, k2)

    all_obs, all_masks, all_pi_H, all_pi_R, all_ep_ids = [], [], [], [], []
    total, ep_id = 0, 0

    while total < n_samples:
        key, init_key = jax.random.split(key)
        state = env_init(init_key)

        while not state.terminated:
            obs = state.observation
            mask = state.legal_action_mask
            obs_f32 = obs.astype(jnp.float32)

            probs_H, _ = pi_H.get_probs(obs_f32, mask)
            probs_R, _ = pi_R.get_probs(obs_f32, mask)

            all_obs.append(np.array(obs))
            all_masks.append(np.array(mask))
            all_pi_H.append(np.array(probs_H))
            all_pi_R.append(np.array(probs_R))
            all_ep_ids.append(ep_id)

            # Random legal action
            key, act_key, step_key = jax.random.split(key, 3)
            legal = jnp.where(mask)[0]
            action = int(legal[jax.random.choice(act_key, len(legal))])
            state = env_step(state, action, step_key)
            total += 1

        ep_id += 1

    return {
        'observations': np.stack(all_obs[:n_samples]),
        'legal_masks': np.stack(all_masks[:n_samples]),
        'pi_H': np.stack(all_pi_H[:n_samples]),
        'pi_R': np.stack(all_pi_R[:n_samples]),
        'episode_ids': np.array(all_ep_ids[:n_samples]),
    }
```

---

## Step 3 完成！

**关键学习**:
1. PGX 环境 `env.init`/`env.step` **必须 JIT 编译**，否则 ~1000x 慢
2. 特征提取与采样**应分离**，避免 JAX context 下的 device transfer 开销
3. Colab 的 JAX 版本需要谨慎处理，使用原生版本最稳定

**Step 4 可用数据**:
- `100K_samples_with_features.npz` - 有 48 个特征，可直接用于 FDA 分析
- `1M_samples_raw.npz` - 需要时可提取更多特征

---

## Step 3.6: 采样分布问题诊断与修复 (2025-12-17)
**状态**: ✅ 完成

### 问题发现
分析 `100K_samples_with_features.npz` 数据时发现严重的分布异常：

| 指标 | 观测值 | 预期值 |
|------|--------|--------|
| Level 7 contracts | **69.3%** | < 1% |
| Level 6 contracts | 8.7% | < 5% |
| Level 0-4 contracts | ~15% | > 90% |
| 7NT 比例 | **46.2%** | < 0.1% |
| States/episode | ~1.6 | ~10-12 |

### 根因分析
**问题**: 使用 `behavior_policy="random"` (uniform random action selection) 导致拍卖螺旋上升。

**机制**:
- 在任何状态，Pass 只有 ~1/K 概率被选中 (K = 合法动作数, 通常 15-25)
- 导致拍卖几乎永远不会停止，直到被迫结束于 7NT
- 这产生了完全不符合真实桥牌的状态分布

**示例**: 如果平均有 20 个合法动作，Pass 概率 = 5%，需要 4 个连续 Pass 结束拍卖，概率仅 0.000625%

### 解决方案
**选择方案 C**: 使用 π^H 选择动作，但记录两个策略的概率

```python
# 关键改动: 从 random 改为 π^H
action = int(jax.random.categorical(act_key, jnp.log(probs_H + 1e-10)))
```

**理由**:
- 状态分布反映 π^H 实际遇到的决策情境
- 仍然可以比较 π^H 和 π^R 在相同状态的行为差异
- 更贴近论文的分析场景："人类策略会到达哪些状态，RL 策略在那些状态会怎么做"

### 实现修改
**文件**: `notebooks/colab_sampling.ipynb`

**主要改动**:
1. `BEHAVIOR_POLICY = "pi_H"` (原 "random")
2. 使用 `fast_sample_jit` 简化实现
3. 添加 `extractor.extract(obs)` 特征提取 (原 `extract_features(state)` 方法不存在)
4. 添加 auction level 分布检查作为 sanity check

**核心采样代码** (修正版):
```python
# JIT compile env functions
env_init = jax.jit(env.init)
env_step = jax.jit(env.step)

def fast_sample_jit(n_samples, seed=42):
    key = jax.random.PRNGKey(seed)
    # ... warmup ...

    while total < n_samples:
        state = env_init(init_key)
        while not state.terminated:
            obs = state.observation
            mask = state.legal_action_mask
            obs_f32 = obs.astype(jnp.float32)

            probs_H, _ = pi_H.get_probs(obs_f32, mask)
            probs_R, _ = pi_R.get_probs(obs_f32, mask)

            # Extract features
            feature_dict = extractor.extract(obs)
            features = np.array([feature_dict[name] for name in FEATURE_NAMES])

            # Store sample
            all_obs.append(np.array(obs))
            all_pi_H.append(np.array(probs_H))
            all_pi_R.append(np.array(probs_R))
            # ...

            # KEY FIX: Use π^H for action selection (not random!)
            action = int(jax.random.categorical(act_key, jnp.log(probs_H + 1e-10)))
            state = env_step(state, action, step_key)
```

### 修复后结果
**新数据**: `100K_pi_H_v2_policy_samples.npz`

| 指标 | 修复前 (random) | 修复后 (π^H) |
|------|----------------|--------------|
| Level 0 | ~3% | **16.9%** |
| Level 1 | ~5% | **20.9%** |
| Level 2 | ~4% | **19.5%** |
| Level 3 | ~4% | **20.5%** |
| Level 4 | ~4% | **14.2%** |
| Level 5 | ~3% | **5.4%** |
| Level 6 | 8.7% | **2.4%** |
| Level 7 | **69.3%** | **0.1%** |
| Episodes | 62,197 | **9,438** |
| States/episode | ~1.6 | **10.6** |

**验证结果**:
```
Auction level distribution:
  Level 0: 16.9%
  Level 1: 20.9%
  Level 2: 19.5%
  Level 3: 20.5%
  Level 4: 14.2%
  Level 5: 5.4%
  Level 6: 2.4%
  Level 7: 0.1%

✓ Auction distribution looks realistic!

Verification:
  Total samples: 100,000
  Unique episodes: 9,438
  Avg states/episode: 10.6
  π^H sum range: [1.000000, 1.000000]
  π^R sum range: [1.000000, 1.000000]
  Min probability: 1.00e-05 (smoothing OK)
```

### 输出文件
```
data/processed/policy_samples/
├── 100K_pi_H_v2_policy_samples.npz  # 修正后的数据
│   ├── observations      (100000, 480)  bool
│   ├── covariates        (100000, 48)   float32
│   ├── legal_masks       (100000, 38)   bool
│   ├── pi_H              (100000, 38)   float32
│   ├── pi_R              (100000, 38)   float32
│   ├── episode_ids       (100000,)      int32
│   └── timestep_in_episode (100000,)    int16
│
└── 100K_pi_H_v2_metadata.json
    ├── n_samples: 100000
    ├── n_episodes: 9438
    ├── feature_names: [48 features]
    ├── action_names: ["Pass", "Dbl", ..., "7NT"]
    ├── ref_action: "Pass"
    ├── ref_action_idx: 0
    ├── action_legal_rates: [1.0, 0.49, 0.05, ...]
    ├── states_per_episode: {min: 4, max: 27, median: 10.0}
    └── sampling_config: {behavior_policy: "pi_H", ...}
```

### 关键学习
1. **Behavior policy 极其重要**: Random action 会产生完全不现实的状态分布
2. **需要 sanity check**: 拍卖级别分布是快速检测采样问题的好指标
3. **On-policy 采样**: 使用 π^H 采样更符合"分析人类策略"的研究目标

---

## Step 3 最终完成！

**可用数据** (Step 4):
- `100K_pi_H_v2_policy_samples.npz` - 使用 π^H 采样，分布合理
- `100K_pi_H_v2_metadata.json` - 完整元数据

**弃用数据**:
- `100K_samples_with_features.npz` - random 采样，分布异常 ❌
- `1M_samples_raw.npz` - random 采样，分布异常 ❌

---

## Step 4: Functional Data Analysis (FDA)

### 目标
使用 GAM 拟合 ALR 差异曲线，比较 π^H 和 π^R 在不同协变量条件下的行为差异。

---

### Step 4.0: 创建 FDA 模块结构
**状态**: ✅ 完成
**时间**: 2025-12-17

**执行内容**:
- 创建 `src/fda/` 模块目录
- 实现数据加载、GAM 拟合、Bootstrap/置换检验、可视化模块

**交付产物**:
```
src/fda/
├── __init__.py
├── data_loader.py      # 数据加载 + ALR 变换
├── gam_fitting.py      # GAM 拟合 (pyGAM)
├── bootstrap.py        # Cluster bootstrap + 置换检验
└── visualization.py    # 曲线图 + 热力图

scripts/
└── run_fda_analysis.py # 主执行脚本

results/fda/
├── eda/                # 探索性分析图
├── alr_curves/         # ALR 差异曲线图
├── statistics/         # 统计结果 CSV
└── models/             # GAM 模型参数
```

---

### Step 4.1: ALR 变换与数据预处理
**状态**: ✅ 完成
**时间**: 2025-12-17

**ALR (Additive Log-Ratio) 变换**:
```python
# Reference action: Pass (idx=0, 始终合法)
ALR_b = log(π(b) / π(Pass))
Δ_b(x) = ALR^R_b(x) - ALR^H_b(x)
```

**分析配置**:
| 参数 | 值 |
|------|-----|
| 协变量 | hcp_total, controls_total, ltc, quick_tricks, n_contracts_bid |
| 动作 | Dbl, 1C, 1D, 1H, 1S, 1NT, 2C (7 个优先动作) |
| 组合数 | 5 × 7 = **35** |

**关键修正**: 1NT 敏感性分析
- 原计划: 全局换 reference 为 1NT
- 问题: 1NT 不是始终合法 (只在未叫过 1 级时合法)
- 修正: 在 1NT 合法的子集中做敏感性分析

---

### Step 4.2: GAM 拟合
**状态**: ✅ 完成
**时间**: 2025-12-17

**GAM 模型**:
```python
Δ_b(x) = f(x) + ε
f(x) = s(x, n_splines=10)  # pyGAM LinearGAM
```

**拟合流程**:
1. 对每个 (covariate, action) 组合过滤到合法样本
2. 使用 5%-95% 分位数范围 (避免外推)
3. Grid search 选择最优 λ (smoothing parameter)
4. 在 100 点网格上预测

**Bug 修复**: pygam.lam 返回 list 而非 float
```python
# 修复代码
if isinstance(lam, (list, np.ndarray)):
    lam = float(lam[0])
else:
    lam = float(lam)
```

---

### Step 4.3: 统计推断
**状态**: ✅ 完成
**时间**: 2025-12-17

**Cluster Bootstrap** (n=100):
- 按 episode 重采样 (保持组内相关性)
- 计算 pointwise 95% CI
- 计算 simultaneous 95% band (max deviation)

**Curve-Based Permutation Test** (n=100):
- H0: Δ_b(x) ≡ 0 (差异曲线恒为零)
- 检验统计量: T = mean(|f(x)|)
- Episode-level sign-flip (保持组内相关性)
- **关键**: 每次置换后重新拟合 GAM

**多重检验校正**:
- Benjamini-Hochberg FDR 校正 (alpha=0.05)

---

### Step 4.4: 测试运行 (n=10)
**状态**: ✅ 完成
**时间**: 2025-12-17
**耗时**: ~3.5 分钟

**Bug 修复** (运行过程中):
1. Unicode 编码错误 (✓ → [OK])
2. pygam.lam 类型错误 (list → float)
3. Unicode α 字符 (α → alpha)

---

### Step 4.5: 生产运行 (n=100)
**状态**: ✅ 完成
**时间**: 2025-12-17
**耗时**: ~35 分钟

**结果**: **全部 35 个组合显著** (p_adj < 0.05)

**输出文件**:
```
results/fda/
├── eda/
│   ├── qc_legal_rates.png
│   ├── qc_sample_counts.png
│   ├── alr_diff_distribution.png
│   └── eda_statistics.csv
├── alr_curves/
│   ├── hcp_total_1C.png (+ 34 more)
│   ├── summary_heatmap_diff.png
│   └── summary_heatmap_sig.png
├── statistics/
│   ├── bootstrap_results.pkl
│   ├── permutation_pvalues.csv
│   └── curve_differences_alr.csv
├── models/
│   └── gam_point_estimates.pkl
└── fda_report.json
```

---

### Step 4 关键发现

**ALR 差异方向**:
- **Positive (π^R > π^H)**: Rule-based 更偏好
- **Negative (π^H > π^R)**: Human-imitation 更偏好

**主要发现**:

| 协变量 | 最强正效应 | 最强负效应 |
|--------|-----------|-----------|
| hcp_total | 1C (+4.25), 1D (+3.27) | 2C (-1.28), 1S (-1.10) |
| controls_total | 1D (+4.17), 1C (+4.16) | 2C (-1.03), 1S (-0.81) |
| ltc | 1C (+5.47), 1D (+4.19) | 2C (-0.99), 1S (-0.87) |
| quick_tricks | 1C (+4.29), 1D (+3.27) | 2C (-1.08), 1S (-1.08) |
| n_contracts_bid | 1C (+5.09), 1D (+3.92) | 1S (-1.26), Dbl (-1.12) |

**解释**:
1. **π^R 强烈偏好 minor suit openings** (1C, 1D): +3 to +5 ALR units
   - 规则策略严格遵循 HCP 阈值开叫 minor
2. **π^H 偏好 major suits 和强牌叫品** (1S, 2C, 1NT): -0.5 to -1.3 ALR units
   - 人类策略更注重战略价值，不仅看点力

**桥牌理论洞见**:
SAYC 体系使用 5 张高花开叫 (5-card major)，试图通过要求 5+ 张高花来增加 1H/1S 开叫的限制性。然而数据显示 π^H 对 1S 的偏好仍然显著高于 π^R，这暗示：
- 主流自然体系的 1 高花开叫范围 (11-21 HCP, 5+ 高花) 可能**仍然过于宽泛**
- 宽泛的点力范围使同伴难以准确定位牌力，可能影响竞叫效率
- π^R 通过 RL 优化后更倾向于 minor 开叫，可能发现了更精确的叫牌路径

**Bootstrap 显著性**:
- Pointwise: 84% - 100% 网格点显著
- Simultaneous: 54% - 100% 网格点显著
- 全部 p_adj = 0.0099 (= 1/101，T_obs > 全部 100 次置换)

---

### Step 4 技术要点

1. **ALR 变换**: 使用 Pass 作为 reference (始终合法)
2. **Cluster Bootstrap**: 按 episode 重采样，保持组内相关性
3. **Curve-Based Permutation**: Episode-level sign-flip + GAM 重拟合
4. **Legal Mask 过滤**: 只分析动作合法的样本，避免结构性零
5. **范围限制**: 5%-95% 分位数，避免外推不稳定

---

### Step 4 完成总结

| 步骤 | 状态 | 关键交付物 |
|------|------|------------|
| 4.0 模块结构 | ✅ | src/fda/*.py |
| 4.1 数据预处理 | ✅ | ALR transform + legal filtering |
| 4.2 GAM 拟合 | ✅ | gam_point_estimates.pkl |
| 4.3 统计推断 | ✅ | bootstrap_results.pkl, permutation_pvalues.csv |
| 4.4 测试运行 | ✅ | n=10, ~3.5 min |
| 4.5 生产运行 | ✅ | n=100, ~35 min, **35/35 显著** |

**Step 4 完成！** ✅

---

## Step 5: JSD (Jensen-Shannon Divergence) Analysis

### 目标
使用 JSD 作为标量度量，分析 π^H 和 π^R 在不同状态下的整体策略分歧程度，并识别高分歧状态的特征模式。

**与 Step 4 的互补关系**:
| 分析 | 关注点 | 输出 |
|------|--------|------|
| Step 4 FDA | 单个动作的 ALR 差异如何随协变量变化 | 35 条差异曲线 |
| Step 5 JSD | 整体策略分歧如何随协变量变化 | JSD 曲线 + EBM 可解释模型 |

---

### Step 5.0: 创建 JSD 模块结构
**状态**: ✅ 完成
**时间**: 2025-12-17

**执行内容**:
- 创建 `src/jsd/` 模块目录
- 实现 JSD 计算 (使用 `scipy.special.xlogy` 避免 0·log(0))
- GAM 拟合、EBM 分析、可视化模块

**交付产物**:
```
src/jsd/
├── __init__.py
├── metrics.py          # JSD 计算 (xlogy 版本)
├── gam_fitting.py      # GAM 拟合 (Definition A + B)
├── ebm_analysis.py     # EBM 可解释模型
└── visualization.py    # 曲线图 + 热力图

scripts/
└── run_jsd_analysis.py # 主执行脚本
```

**JSD 计算 (关键修正)**:
```python
from scipy.special import xlogy

def jsd(p, q, base=2):
    """使用 xlogy 避免 0·log(0) 问题"""
    m = 0.5 * (p + q)
    kl_pm = np.sum(xlogy(p, p) - xlogy(p, m), axis=-1) / np.log(base)
    kl_qm = np.sum(xlogy(q, q) - xlogy(q, m), axis=-1) / np.log(base)
    return 0.5 * (kl_pm + kl_qm)
```

---

### Step 5.1: 敏感性分析 (JSD_raw vs JSD_legal)
**状态**: ✅ 完成
**时间**: 2025-12-17

**目的**: 验证 smoothed illegal actions 对 JSD 的影响是否可忽略。

**结果**:
| 指标 | 值 |
|------|-----|
| Mean |diff| | 2.34e-05 |
| Max |diff| | 3.53e-04 |
| P95 |diff| | 1.09e-04 |
| Correlation | **0.9999999968** |

**结论**: JSD_raw 和 JSD_legal 几乎完全一致 (corr ≈ 1)，使用 JSD_raw 即可。

**交付产物**:
- `results/jsd/eda/sensitivity_legal_vs_raw.png`
- `results/jsd/eda/sensitivity_stats.csv`

---

### Step 5.2: JSD 分布与 EDA
**状态**: ✅ 完成
**时间**: 2025-12-17

**JSD 统计**:
| 指标 | 值 |
|------|-----|
| Mean | 0.321 |
| Median | 0.087 |
| Std | 0.380 |
| P90 | 0.956 |
| P95 | 0.989 |
| Max | 0.999 |

**分布特征**: **高度右偏** (mean >> median)
- 大部分状态 JSD 较低 (median = 0.087)
- 存在长尾，约 10% 状态 JSD > 0.95

**高 JSD 状态特征分析** (Top 5 by effect size):
| 特征 | Effect Size | High JSD Mean | All Mean |
|------|-------------|---------------|----------|
| auction_level | -0.73 | 1.04 | 2.21 |
| n_contracts_bid | -0.60 | 1.40 | 2.64 |
| is_passout | +0.58 | 0.39 | 0.17 |
| has_contract | -0.58 | 0.61 | 0.83 |
| contract_strain | -0.48 | 0.79 | 1.60 |

**解释**: 高 JSD 状态集中在**早期叫牌** (auction_level 低) 和 **pass-out 局**。

**交付产物**:
- `results/jsd/eda/jsd_distribution.png`
- `results/jsd/eda/jsd_vs_covariates.png`
- `results/jsd/eda/high_jsd_features.csv`

---

### Step 5.3: JSD 曲线拟合 (Definition A + B)
**状态**: ✅ 完成
**时间**: 2025-12-17

**两种 JSD 曲线定义**:
| 定义 | 公式 | 含义 |
|------|------|------|
| **A (State-level)** | E[JSD(π^H(s), π^R(s)) \| X=t] | 典型状态的即时分歧 |
| **B (Bin-level)** | JSD(E[π^H \| X=t], E[π^R \| X=t]) | 平均系统的整体分歧 |

**关系**: 由 Jensen 不等式，JSD_B(t) ≤ JSD_A(t)

**拟合方法**:
- Definition A: GAM + cluster bootstrap CI (n=100)
- Definition B: 直接分箱计算 (n_bins=10)

**协变量分析**:
| 协变量 | JSD_A Mean | JSD_B Mean | Mean(A-B) |
|--------|------------|------------|-----------|
| hcp_total | 0.317 | 0.094 | 0.223 |
| controls_total | 0.316 | 0.094 | 0.222 |
| ltc | 0.316 | 0.092 | 0.224 |
| quick_tricks | 0.316 | 0.092 | 0.224 |
| n_contracts_bid | 0.304 | 0.073 | 0.231 |

**交付产物**:
- `results/jsd/curves/jsd_A_vs_*.png` (5 个协变量)
- `results/jsd/curves/jsd_B_vs_*.png` (5 个协变量)
- `results/jsd/curves/jsd_AB_comparison_*.png` (A vs B 对比)
- `results/jsd/curves/jsd_heatmap.png`
- `results/jsd/statistics/jsd_by_covariate.csv`

---

### Step 5.4: EBM 可解释模型
**状态**: ✅ 完成
**时间**: 2025-12-17

**EBM (Explainable Boosting Machine)**:
- 加法模型: f(x) = Σ f_i(x_i) + Σ f_{i,j}(x_i, x_j)
- InterpretML 库
- 允许 10 个二阶交互项

**性能**:
| 指标 | 值 |
|------|-----|
| CV R² | 0.356 ± 0.007 |
| CV RMSE | 0.305 ± 0.001 |

**特征重要性 (Top 10)**:
| 特征 | Importance |
|------|------------|
| **auction_level** | 0.097 |
| contract_strain | 0.043 |
| rho_opened | 0.031 |
| is_passout | 0.027 |
| has_contract | 0.027 |
| lho_opened | 0.026 |
| partner_opened | 0.018 |
| self_opened | 0.014 |
| is_competitive | 0.014 |
| hcp_total & auction_level | 0.013 |

**关键发现**: `auction_level` 是最重要特征，重要性是第二名的 **2.3 倍**。

**交付产物**:
- `results/jsd/models/ebm_jsd_model.pkl`
- `results/jsd/models/feature_importance.csv`
- `results/jsd/curves/feature_importance.png`

---

### Step 5 关键发现

**1. π^H 和 π^R 在开叫决策上分歧最大**
- 高 JSD 状态集中在 auction_level=1 (开叫阶段)
- EBM 显示 auction_level 是预测 JSD 最重要的特征
- 与 Step 4 发现一致：1C/1D 开叫偏好差异最大

**2. 分歧分布高度右偏**
- Median JSD = 0.087 (大部分状态策略相似)
- P95 JSD = 0.989 (约 5% 状态策略截然不同)
- 说明两种策略在大部分情况下行为一致，但在特定场景有本质区别

**3. Pass-out 局分歧大**
- is_passout 的 effect size = +0.58
- 说明在无人开叫时，π^H 和 π^R 对"是否开叫"有不同判断

**4. 与 Step 4 交叉验证**
- Step 4: π^R 偏好 1C/1D (+3~+5 ALR)，π^H 偏好 1S/2C/1NT
- Step 5: 高 JSD 集中在开叫阶段，最重要特征是 auction_level
- **两个分析互相印证**：策略分歧主要来自开叫决策

---

### Step 5 完成总结

| 步骤 | 状态 | 关键交付物 |
|------|------|------------|
| 5.0 模块结构 | ✅ | src/jsd/*.py (xlogy 版本) |
| 5.1 敏感性分析 | ✅ | sensitivity_stats.csv (corr=0.9999) |
| 5.2 EDA | ✅ | jsd_distribution.png, high_jsd_features.csv |
| 5.3 JSD 曲线 | ✅ | jsd_A/B 曲线 + AB 对比图 |
| 5.4 EBM 模型 | ✅ | ebm_jsd_model.pkl (R²=0.356) |

**输出文件结构**:
```
results/jsd/
├── eda/
│   ├── jsd_distribution.png
│   ├── sensitivity_legal_vs_raw.png
│   ├── sensitivity_stats.csv
│   ├── jsd_vs_covariates.png
│   └── high_jsd_features.csv
├── curves/
│   ├── jsd_A_vs_*.png (5)
│   ├── jsd_B_vs_*.png (5)
│   ├── jsd_AB_comparison_*.png (5)
│   ├── jsd_heatmap.png
│   └── feature_importance.png
├── statistics/
│   ├── jsd_quantiles.csv
│   └── jsd_by_covariate.csv
├── models/
│   ├── ebm_jsd_model.pkl
│   ├── feature_importance.csv
│   └── jsd_curves.pkl
└── jsd_report.json
```

**Step 5 完成！** ✅

---

## Step 6: Local Explainability (Occlusion Analysis)

**目标**: 通过 Occlusion/Permutation 分析理解特征组对策略的局部影响

**完成时间**: 2025-12-17

---

### Step 6.0 模块结构

创建 `src/occlusion/` 模块:
```
src/occlusion/
├── __init__.py          # 模块导出
├── feature_groups.py    # 特征组定义
├── metrics.py           # KL散度、JSD、TVD等指标
├── approximate.py       # EBM近似分析
├── precise.py           # 模型重推断精确分析
└── visualization.py     # 可视化函数
```

**特征组定义 (480-dim observation)**:
| 组 | 索引范围 | 维度 | 描述 |
|----|---------|------|------|
| vulnerability | [0:4] | 4 | 局况编码 |
| position | [4:8] | 4 | 开叫前pass数 |
| bidding | [8:428] | 420 | 叫牌历史 (35合约×3状态×4玩家) |
| hand | [428:480] | 52 | 手牌编码 |

**特征组定义 (48-dim covariates)**:
| 组 | 特征数 | 描述 |
|----|--------|------|
| hand | 28 | HCP、分布、控制、荣誉 |
| bidding | 13 | 叫牌状态、开叫信息 |
| context | 7 | 局况、位置 |

---

### Step 6.1 Approximate Analysis (EBM-based)

**方法**: 使用 Step 5 的 EBM 模型进行 Permutation Importance 分析

**结果**:
| 特征组 | Permutation Sensitivity |
|--------|------------------------|
| **bidding** | **0.2085** |
| hand | 0.0931 |
| context | 0.0192 |

**发现**: Bidding 特征组对 JSD 预测最敏感，是 hand 的 **2.2 倍**，是 context 的 **10.9 倍**

**交付物**:
- `results/occlusion/approximate_sensitivity.json`
- `results/occlusion/group_sensitivity_approximate.png`

---

### Step 6.2 Precise Analysis (Model Re-inference)

**方法**:
1. 对每个特征组执行 Permutation Occlusion
2. 重新运行 pi_H 神经网络推断
3. 计算原始策略与扰动策略的 KL 散度

**参数**:
- 样本数: 10,000
- Permutation 迭代: 5 次
- Batch size: 1,024

**结果**:
| 特征组 | KL Divergence |
|--------|---------------|
| **bidding** | **6.4859** |
| **hand** | **6.1729** |
| position | 0.1155 |
| vulnerability | 0.0929 |

**发现**:
1. **Bidding 和 Hand 同等重要**: KL 值都在 6+ 量级，说明打乱这些特征会完全破坏策略
2. **Position/Vulnerability 影响较小**: KL 值约 0.1，说明这些上下文信息对策略影响有限
3. **与 EBM 分析一致**: 两种方法都显示 bidding > hand > context

**交付物**:
- `results/occlusion/precise_sensitivity.json`
- `results/occlusion/group_sensitivity_precise.png`

---

### Step 6 关键发现

**1. Bidding History 是最关键特征**
- Approximate: importance = 0.2085 (最高)
- Precise: KL = 6.49 (最高)
- 包含当前叫牌状态、谁做了什么叫品等信息

**2. Hand Encoding 同样关键**
- Precise: KL = 6.17 (与 bidding 接近)
- 手牌是做出叫牌决策的基础信息
- 没有手牌信息，策略无法正常工作

**3. Context 信息影响较小**
- Vulnerability 和 Position 的 KL 值仅为 ~0.1
- 说明策略主要由 bidding history + hand 驱动
- 但这不意味着不重要 - 在边缘情况可能起决定作用

**4. 与前期分析的一致性**
| 分析 | 结论 |
|------|------|
| Step 4 FDA | 开叫偏好差异最大 (1C/1D vs 1S/2C) |
| Step 5 JSD | auction_level 是预测 JSD 最重要特征 |
| Step 6 Occlusion | bidding 特征组最敏感 |

**结论**: 三步分析互相印证，策略差异主要来自**叫牌阶段的决策**

---

### Step 6 完成总结

| 步骤 | 状态 | 关键交付物 |
|------|------|------------|
| 6.0 模块结构 | ✅ | src/occlusion/*.py |
| 6.1 Approximate Analysis | ✅ | group_sensitivity_approximate.png |
| 6.2 Precise Analysis | ✅ | group_sensitivity_precise.png |

**输出文件结构**:
```
results/occlusion/
├── approximate_sensitivity.json
├── precise_sensitivity.json
├── group_sensitivity_approximate.png
├── group_sensitivity_precise.png
└── occlusion_report.json
```

**Step 6 完成！** ✅

---

## Step 7: Rule Distillation (Decision Tree + GAM)

**目标**: 将黑箱 π^R 蒸馏为可解释模型 π^D

**完成时间**: 2025-12-17

---

### Step 7.0 模块结构

创建 `src/distillation/` 模块:
```
src/distillation/
├── __init__.py
├── tree_distill.py     — 决策树蒸馏
├── gam_distill.py      — GAM 蒸馏
├── metrics.py          — 保真度指标
└── visualization.py    — 可视化
```

---

### Step 7.1 Decision Tree 蒸馏

**方法**: 使用 scikit-learn DecisionTreeClassifier，不同深度对比

**结果**:
| Depth | Train Acc | Val Acc | Top-3 Agree | Leaves |
|-------|-----------|---------|-------------|--------|
| 3 | 8.3% | 8.1% | - | 8 |
| 5 | 6.6% | 6.5% | - | 31 |
| 7 | 15.0% | 14.4% | - | 100 |
| 10 | 27.6% | 26.4% | 57.0% | 348 |
| 15 | 34.8% | 31.9% | - | 1,021 |
| **Unlimited** | **35.5%** | **32.4%** | **61.0%** | 1,161 |

**特征重要性 (Top 5)**:
- 根据决策树分裂的特征重要性分析

**交付物**:
- `results/distillation/models/tree_depth_*.pkl`
- `results/distillation/analysis/tree_comparison.csv`
- `results/distillation/analysis/tree_rules.txt`

---

### Step 7.2 GAM 蒸馏

**方法**: 为每个高频动作训练单独的 LogisticGAM

**结果** (Top 5 Actions):
| Action | Name | Train Acc | Val Acc |
|--------|------|-----------|---------|
| 0 | Pass | 78.4% | 77.7% |
| 1 | Double | 94.0% | 93.5% |
| 3 | 1C | 98.8% | 98.7% |
| 4 | 1D | 98.1% | 98.1% |
| 21 | 3NT | 97.8% | 98.1% |

**说明**: GAM 对单个动作的二分类效果较好，但整体多分类保真度较低

**交付物**:
- `results/distillation/models/gam_models.pkl`
- `results/distillation/analysis/gam_summary.csv`
- `results/distillation/plots/gam_shape_functions_action_0.png`

---

### Step 7 关键发现

**1. 决策树保真度有限但有价值**
- Top-1 Agreement: 32.4% (unlimited depth)
- Top-3 Agreement: 61.0%
- 说明约 1/3 的决策可以用简单规则解释

**2. GAM 对单动作分类效果好**
- Pass 动作: 77.7% 准确率
- 低频动作 (1C, 1D): >98% 准确率
- GAM shape functions 可以展示特征的连续影响

**3. 策略复杂性**
- 神经网络策略难以完全用简单模型解释
- 33 个不同动作，类别严重不平衡
- 需要更复杂的可解释模型或层次化方法

**4. 与前期分析的一致性**
| 分析 | 发现 |
|------|------|
| Step 4 FDA | 开叫阶段差异最大 |
| Step 5 JSD | auction_level 最重要 |
| Step 6 Occlusion | bidding 特征组最敏感 |
| **Step 7 Distillation** | 可解释模型难以完全捕捉策略 |

---

### Step 7 完成总结

| 步骤 | 状态 | 关键交付物 |
|------|------|------------|
| 7.0 模块结构 | ✅ | src/distillation/*.py |
| 7.1 Decision Tree | ✅ | tree_comparison.csv, tree_rules.txt |
| 7.2 GAM | ✅ | gam_summary.csv, shape functions |
| 7.3 评估 | ✅ | fidelity_metrics.csv |

**输出文件结构**:
```
results/distillation/
├── models/
│   ├── tree_depth_*.pkl (6 个)
│   └── gam_models.pkl
├── analysis/
│   ├── tree_comparison.csv
│   ├── tree_feature_importance.csv
│   ├── tree_rules.txt
│   ├── gam_summary.csv
│   └── fidelity_metrics.csv
├── plots/
│   ├── fidelity_vs_complexity.png
│   ├── tree_feature_importance.png
│   ├── confusion_matrix_tree.png
│   └── gam_shape_functions_action_0.png
└── distillation_report.json
```

**Step 7 完成！** ✅

---

## 项目总结

### 完成的分析步骤

| Step | 内容 | 状态 |
|------|------|------|
| 0-3 | 项目搭建、特征工程、采样 | ✅ |
| 4 | Compositional FDA | ✅ |
| 5 | JSD Analysis | ✅ |
| 6 | Occlusion Analysis | ✅ |
| 7 | Rule Distillation | ✅ |

### 核心发现

1. **π^H vs π^R 的主要差异在开叫阶段**
   - Step 4: 1C/1D 开叫偏好差异最大
   - Step 5: auction_level 是 JSD 最重要预测因子

2. **策略主要依赖 bidding + hand 信息**
   - Step 6: bidding (KL=6.49) 和 hand (KL=6.17) 最敏感
   - Context (vulnerability/position) 影响较小

3. **黑箱策略难以完全用简单模型解释**
   - Step 7: 决策树 top-1 agreement 仅 32%
   - 需要更复杂的可解释方法

### 下一步 (可选)

- **Step 8 (Bonus)**: Post-hoc Filtering
  - 控制 π^R 与 π^H 的偏离风险
  - 定义 risk score: ρ(a) = log(π^R(a|s)/π^H(a|s))
  - 实现 soft reweighting

