# ProjectX · 基于共享 MLP 与位置分类头的车牌字符识别与跨域泛化研究

本项目依据《ProjectX · AI 开发提示词》规格实现：输入**已裁剪、已对齐**的 32×128 灰度车牌图像，
输出汉字之后**六个字符位置**各自的类别预测，并系统研究损失函数、优化策略、正则化与数据增强
对**同分布 / 合成域 / 强扰动**三类测试集表现的影响。

> **本项目使用 NumPy 手写前向与反向传播**，不使用 `torch.autograd`、TensorFlow、Keras
> 等任何自动求导机制；可选 CuPy 作为 GPU 后端，仅加速矩阵运算（§8.1 路线 A）。

---

## 1. 环境准备

需要 Python ≥ 3.10（本项目在 Python 3.12.14 上开发验证）。

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Linux / macOS
pip install -r requirements.txt
```

国内网络可加速：

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 可选：GPU 后端（CuPy）

训练默认使用 GPU（若可用）。请按本机 CUDA 版本安装对应 CuPy 包，例如：

```bash
pip install cupy-cuda12x
```

随后在 `configs/default.yaml` 中设置：

```yaml
optim:
  backend: "cupy"      # numpy | cupy
```

**无 GPU 环境必须能回退纯 CPU 运行**：把 `backend` 设为 `"numpy"` 即可，
程序启动时也会自动探测 CuPy，不可用时打印告警并回退，不会中断实验。

---

## 2. 工程结构

```text
projectX/
├─ README.md                      # 本文件：环境、运行方式、复现步骤
├─ requirements.txt
├─ .gitignore
├─ configs/
│  └─ default.yaml                # ★ 全部超参数的唯一来源
├─ docs/
│  └─ CCPD_README.md              # 官方 README 存档（字符映射核对依据）
├─ data/
│  ├─ ccpd/                       # CCPD 原图（不入库）
│  ├─ synth_test/                 # 合成测试图（不入库）
│  ├─ processed/                  # 预处理缓存 .npz（不入库）
│  └─ manifest.csv                # 来源文件、六标签、子集名、裁剪参数、随机种子
├─ models/
│  ├─ config.py                   # 配置加载 / 点分路径补丁 / Git 溯源
│  ├─ charset.py                  # 34 类字符集与 CCPD 索引映射表
│  ├─ ccpd_parse.py               # 文件名解析、透视矫正、裁剪、标准化
│  ├─ dataset.py                  # 批加载器（支持 batch=1 与全批量）
│  ├─ model.py                    # 共享模型与独立模型（前向 + 反向手写）
│  ├─ losses.py                   # 交叉熵 / MSE / L2 及梯度
│  └─ optim.py                    # SGD、动量、批量策略
├─ train/
│  ├─ train.py                    # 训练循环、早停、日志落盘
│  ├─ phase1_prepare.py           # P1：CCPD 解析 + 裁剪 + 过滤统计
│  ├─ phase15_split.py            # P1.5：号码去重划分 + 合成测试集生成
│  ├─ synth_plates.py             # 合成域测试集生成器（PIL 确定性渲染）
│  └─ grad_check.py               # 数值梯度检查
├─ evaluate/
│  ├─ evaluate.py                 # 各评价指标、混淆矩阵、分位置准确率
│  └─ visualize.py                # 曲线、样本网格、错误样本可视化
├─ reports/
│  ├─ run_all.py                  # 一键跑完全部启用的对照实验
│  ├─ logs/                       # 每实验逐 epoch 指标（入库，是结论的直接证据）
│  ├─ figs/                       # 训练曲线、混淆矩阵、错误样本（不入库）
│  ├─ tables/                     # 汇总 CSV / Markdown 表
│  ├─ checkpoints/                # 最佳验证权重（不入库）
│  └─ 实验报告.md                  # 结论、图表、误差分析、局限
└─ .venv/                         # 虚拟环境（不入库）
```

---

## 3. 数据准备

### 3.1 数据来源

| 角色 | 来源 |
| --- | --- |
| 训练 / 验证 / 同分布测试 / 强扰动测试 | **CCPD**（<https://github.com/detectRecog/CCPD>，MIT） |
| 合成域测试集 | 程序生成的仿真中国车牌（**只测试、不训练**） |

真实车牌含隐私信息，**原图永不入库**（见 `.gitignore`）。CCPD 官方下载入口为
Google Drive / 百度网盘；本项目在开发阶段使用了保留原始文件名的公开镜像
（<https://huggingface.co/datasets/zenitsu09/ccpd-subset-30k>，MIT），
因为 CCPD 的标注**内嵌在文件名中**，镜像必须保留原始文件名才有价值。

把解压后的 CCPD 图片按子集放入 `data/ccpd/`：

```text
data/ccpd/
├─ ccpd_base/        # 训练 / 验证 / 同分布测试
├─ ccpd_blur/        # 强扰动测试
├─ ccpd_challenge/   # 强扰动测试
├─ ccpd_rotate/      # 强扰动测试
├─ ccpd_tilt/        # 强扰动测试
└─ ccpd_weather/     # 强扰动测试
```

### 3.2 一键数据准备

```bash
# P1：解析 + 透视矫正裁剪 + 过滤统计
python train/phase1_prepare.py

# ★ 人工核对（必做，通过前不得进入训练）
#   产出 20 张「裁剪图 + 标签字符串」网格图
python evaluate/visualize.py check-grid

# P1.5：号码去重划分 + 生成合成域测试集
python train/phase15_split.py

# 缓存为 .npz（训练脚本也会自动按需生成）
python models/dataset.py build
```

---

## 4. 训练与评估

```bash
# 梯度检查（P3 验收：相对误差 < 1e-5）
python train/grad_check.py

# 100 张小样本过拟合自检（P2/P3 验收）
python train/train.py overfit

# 单次基线训练
python train/train.py --config configs/default.yaml

# 评估（含 CPU 单张/批量推理时间）
python evaluate/evaluate.py --run <run_id>
```

### 一键跑完对照实验

```bash
python reports/run_all.py                       # 跑配置中 enabled 的实验
python reports/run_all.py --experiments E1 E3 E4 E7
python reports/run_all.py --seeds 42 43 44
python reports/run_all.py --list
```

每个实验至少 **3 个随机种子**重复，报告**均值 ± 标准差**。

---

## 5. 实验与代码的绑定（复现性）

每次实验的日志与汇总 CSV 都写入四个溯源字段：

| 字段 | 含义 |
| --- | --- |
| `commit` | 短 7 位 Git hash（`git rev-parse --short HEAD`） |
| `config` | 配置名 / 内容指纹（sha1 前 12 位） |
| `seed` | 随机种子 |
| `dirty` | 工作区是否有未提交改动（`git status --porcelain`） |

> ⚠️ **`dirty=True` 的结果不可作为最终结论。** 做实验前请先提交代码。

`reports/logs/` 建议入库：逐 epoch 指标体积小、价值高，是结论的直接证据。
`reports/figs/` 与 `reports/checkpoints/` 不入库，可由日志与脚本重绘 / 重训得到。

---

## 6. 复现步骤（端到端）

```bash
# 0) 环境
python -m venv .venv && .venv\Scripts\activate && pip install -r requirements.txt

# 1) 数据（先解压 CCPD 到 data/ccpd/）
python train/phase1_prepare.py
python evaluate/visualize.py check-grid      # 人工核对 20 张
python train/phase15_split.py

# 2) 正确性验证
python train/grad_check.py
python train/train.py overfit

# 3) 基线训练 + 对照实验
python reports/run_all.py

# 4) 汇总表与图
python reports/run_all.py --summarize-only

# 5) 报告见 reports/实验报告.md
```

---

## 7. 口径要点（易错处）

* **类别索引顺序遵循 CCPD 官方 `ads` 表**：索引 `0–23` 为字母 `A-Z`（去掉 `I`、`O`），
  `24–33` 为数字 `0–9`。因此 `A→0`、`Z→23`、`0→24`、`9→33`。
  官方表来源已存档于 `docs/CCPD_README.md`，并与官方源码 `rpnet/demo.py` 逐项核对。
* **划分按车牌号码去重**（§2.5）：同一车牌号码的全部图片必须归入同一集合，
  否则模型只需记住号码纹理即可拿高分，测试结论不成立。
* **偏置项不参与 L2 惩罚**（§4.1）。
* **必须同时报告字符准确率与整牌准确率**：字符准确率 95% 时整牌准确率上界仅约 73.5%。
* **合成域与强扰动测试集在调参阶段不得参与任何选择决策**（§2.5 要求 3）。

---

## 8. 许可与致谢

* CCPD 数据集：MIT License，论文 *Towards End-to-End License Plate Detection and
  Recognition: A Large Dataset and Baseline* (ECCV 2018)。
* 本项目不进行车牌检测，仅使用 CCPD 文件名中标注的四角顶点做透视矫正与裁剪。