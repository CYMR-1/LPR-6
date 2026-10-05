# ProjectX（交付版）· 共享 MLP 与位置分类头的车牌后六位字符识别

输入**已裁剪、已对齐**的 32×128 灰度车牌图，输出汉字之后**六个字符位置**各自的类别预测（每位 34 类：数字 0–9 + 大写字母 A–Z 去掉易混的 I、O）。

> **本项目使用 NumPy 手写前向与反向传播**，不使用 `torch.autograd`、TensorFlow、Keras 等任何自动求导机制；可选 CuPy 作为 GPU 后端，仅加速矩阵运算。

本仓库只保留**最终训练好的模型**与产生它的最小可复现管线（数据准备 → 训练 → 评测 → 推理）。开发过程中的全部对照实验代码、运行日志、汇总表、图表与实验报告均已从本分支移除，`configs/default.yaml` 是超参数的唯一来源。

---

## 1. 最终模型

最终模型 = **共享六头 MLP + sigmoid + 交叉熵 + SGD(μ=0.9, lr=0.05, bs=64) + 关闭数据增强**。关闭增强是本架构的关键：MLP 没有平移不变性，几何增强会把字符挪到训练时没见过的像素位置，等于注入噪声（`configs/default.yaml` 里默认档位仍是`weak`，因此**复现最终模型必须用 `reports/configs/final_s*.yaml`**）。

三个随机种子的检查点与实测指标（数字取自 `reports/logs/final_s*_run.json`）：

| 检查点 | val 字符 | test 字符 | test 整牌 | hard 字符 | hard 整牌 | synth 字符 | 最佳轮 / 实跑轮 |
|---|---|---|---|---|---|---|---|
| `final_s42_best.npz` | 98.42% | **98.40%** | 91.90% | 73.17% | 34.40% | 45.52% | 57 / 65 |
| `final_s43_best.npz` | 98.56% | **98.33%** | 91.85% | 73.03% | 34.85% | 43.65% | 78 / 80 |
| `final_s44_best.npz` | 98.55% | **98.42%** | 91.90% | 72.77% | 34.65% | 45.64% | 80 / 80 |

* `test` = CCPD-Base 同分布测试集（2000 张）；`hard` = 强扰动测试集（2000 张）；`synth` = 外部生成器合成域测试集（2000 张）。
* 模型结构：`shared` / `sigmoid` / `hidden_dim=256` / 头维度 `[34]*6`，**参数量 1,101,260**，输出节点 204。
* 独立评测（`python evaluate/main.py --run final_s42`，载入检查点、**不重训**）复现出上表**逐位一致**的指标（val 98.42% / 91.95%，test 98.40% / 91.90%，hard 73.17% / 34.40%，synth 45.52% / 0.80%）；CPU 单张前向 **0.20–0.28 ms**（50 张中位数，含标准化与六头 softmax，实测随机器负载波动）。
* 训练环境：Python 3.13 + numpy + cupy-cuda12x（RTX 4060 Laptop，8 GB）；单次训练约 146–183 秒（早停或跑满 80 轮），三个种子合计约 9 分钟。
* 复现最终模型必须使用 `reports/configs/final_s*.yaml`：它与 `configs/default.yaml` 的唯一实质差别是 `augmentation.baseline_level: none`（默认配置是 `weak`），用默认配置训练会得到完全不同的结果。

> ⚠️ **域差距提醒**：模型训练于 CCPD 风格的真实车牌特写，对风格差异大的输入会显著退化——同分布整牌 91.9%，而同分布之外的强扰动集整牌 34.4%、合成域字符45.5%（34 类随机猜测为 2.9%）。手机远距离拍摄再放大裁切、字体差异大的图片都属于"域外"输入，此时预测结果只能当参考。

---

## 2. 工程结构

```text
projectX/
├─ README.md                      # 本文件：环境、数据、训练、评测、推理
├─ predict.py                     # ★ 用训练好的检查点识别自己的车牌图（§6）
├─ requirements.txt
├─ .gitignore
├─ configs/
│  └─ default.yaml                # ★ 全部超参数的唯一来源（代码默认口径）
├─ docs/
│  └─ CCPD_README.md              # 官方 README 存档（字符映射核对依据）
├─ data/                          # 全部不入库（见 .gitignore）
│  ├─ ccpd/                       # CCPD 原图：扁平放 30000 张，子集名在文件名 token（§4.1）
│  ├─ raw_dl/                     # 下载压缩包（如 ccpd_subset_30k.zip）
│  ├─ external/                   # 外部合成域生成器仓库（固定 commit 43bac43）
│  ├─ synth_test/                 # 合成测试图输出目录
│  ├─ synth_test_preview/         # 合成过程逐张预览（人工核对用）
│  ├─ processed/                  # 预处理缓存与划分 .npz
│  └─ manifest.csv                # 来源文件、六标签、子集名、裁剪参数、随机种子
├─ models/
│  ├─ config.py                   # 配置加载 / 点分路径补丁 / Git 溯源
│  ├─ charset.py                  # 34 类字符集与 CCPD 索引映射表
│  ├─ ccpd_parse.py               # 文件名解析、透视矫正、裁剪、标准化
│  ├─ dataset.py                  # 批加载器（支持 batch=1 与全批量）
│  ├─ model.py                    # 共享/独立模型（前向 + 手写反向）与损失
│  ├─ augment.py                  # 数据增强（无 / 弱 / 强 三档）
│  ├─ backend.py                  # NumPy / CuPy 统一后端入口
│  ├─ metrics.py                  # 评价指标（字符 / 整牌准确率等）
│  └─ optim.py                    # SGD、动量、批量策略
├─ train/
│  ├─ prepare_data.py             # 数据准备：CCPD 解析 + 裁剪 + 过滤统计（生成 .npz 缓存）
│  ├─ split_dataset.py            # 数据划分：号码去重划分 + 合成测试集生成（按 synth.backend 分发）
│  ├─ synth_from_generator.py     # ★ 合成域后端 generator_repo：外部生成器牌面级输出 + 同口径几何
│  ├─ synth_plates.py             # 合成域后端 pil_renderer（内置 PIL 渲染器，保留可切回）
│  ├─ train.py                    # 训练循环、早停、日志落盘
│  ├─ grad_check.py               # 数值梯度检查（相对误差 < 1e-5）
│  ├─ overfit_check.py            # 小样本过拟合自检（loss < 1e-3 且字符/整牌 100%）
│  └─ parity_check.py             # NumPy / CuPy 前后端数值一致性校验
├─ evaluate/
│  ├─ main.py                     # 独立评测入口（载入检查点，不重训）
│  ├─ model_eval.py               # 各评价指标、混淆矩阵、分位置准确率、CPU 计时
│  └─ visualize.py                # 人工核对网格、曲线、混淆矩阵、错误样本
└─ reports/
   ├─ configs/                    # ★ 最终模型的部署配置（入库）
   ├─ checkpoints/                # ★ 最终模型权重 final_s{42,43,44}_best.npz（入库）
   └─ logs/                       # 最终模型的逐 epoch 指标与独立评测产物（入库）
```

---

## 3. 环境准备

需要 Python ≥ 3.10（本项目本轮在 Python 3.13.12 上训练与评测验证）。

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

**无 GPU 环境必须能回退纯 CPU 运行**：把 `configs/default.yaml` 的`optim.backend` 设为 `"numpy"` 即可，程序启动时也会自动探测 CuPy，不可用时打印告警并回退，不会中断运行。合成域生成器额外需要 OpenCV（**仅数据生成环节**，训练 / 评测 / 预处理不依赖）。

---

## 4. 数据准备

### 4.1 数据来源

| 角色 | 来源 |
| --- | --- |
| 训练 / 验证 / 同分布测试 / 强扰动测试 | **CCPD**（<https://github.com/detectRecog/CCPD>，MIT） |
| 合成域测试集 | 外部开源生成器 **[Nenger/chinese_licence_plate_generator](https://github.com/Nenger/chinese_licence_plate_generator)**（固定 commit `43bac43`）的**牌面级**输出，**只测试、不训练** |

合成域用其牌面级接口（`FakePlateGenerator.generate_one_plate()` + 上游`jittering_color/add_noise/jittering_blur/jittering_scale` 扰动链），再套用与 CCPD **完全相同**的几何裁剪（裁左 1/7 → 128×32 → 灰度）；上游字符素材含字母 I/O，生成时按本项目 34 类字符集**拒绝重采**。该仓库的主打产物是"车牌贴进街景图"的**检测**数据集，本项目不做检测，故不使用其场景整图。

```bash
# 克隆生成器（不入库，见 .gitignore 的 data/external/）
git clone https://github.com/Nenger/chinese_licence_plate_generator data/external/chinese_licence_plate_generator
```

真实车牌含隐私信息，**原图永不入库**（见 `.gitignore`）。CCPD 官方下载入口为 Google Drive / 百度网盘；本项目开发阶段使用了保留原始文件名的公开镜像（<https://huggingface.co/datasets/zenitsu09/ccpd-subset-30k>，MIT）。

#### 目录布局：`data/ccpd/` 是**扁平**的，子集名在**文件名**里

本仓库的 `data/ccpd/` 下**没有** `ccpd_base/`、`ccpd_blur/` 这类子目录——30000 张 jpg 直接平铺，子集名以 `_ccpd_<subset>_<序号>.jpg` 的 token **追加在文件名尾部**（该镜像的命名约定）：

```text
data/ccpd/
├─ 00292624521073-90_83-334,464_452,506-440,501_341,503_339,470_438,468-0_16_15_29_24_33_27-118-10_ccpd_base_012856.jpg
├─ 0023-2_2-286,531_360,558-360,558_286,555_286,531_360,534-0_0_4_32_4_33_33-101-5_ccpd_blur_009176.jpg
├─ 0021-1_0-302,471_372,497-372,495_303,497_302,473_371,471-0_0_30_16_29_32_32-75-21_ccpd_challenge_020298.jpg
├─ ...                                                     ..._ccpd_fn_027078.jpg
├─ ...                                                     ..._ccpd_rotate_023117.jpg
├─ ...                                                     ..._ccpd_tilt_025833.jpg
└─ ...                                                     ..._ccpd_weather_006602.jpg
```

* 子集**不由目录名判定**，而是由 `models/ccpd_parse.py::extract_subset`从文件名解析（兼容 `ccpd_base` 连写与 `ccpd` + `base` 拆开两种形式）。
* 图片发现用 `models/dataset.py::discover_images` 的 **`rglob` 递归扫描**，因此**也允许**用子目录组织；但无论放哪，**文件名里必须带`_ccpd_<subset>_` token**，否则该图会被判为"无子集"而在划分时被忽略（`split_dataset.py` 会因找不到 `ccpd_base` 样本而报错退出）。

`zenitsu09/ccpd-subset-30k` 实测子集分布（扫描 30000 张所得）：

| 子集 token | 张数 | 用途 |
| --- | --- | --- |
| `ccpd_base` | 14987 | **训练 / 验证 / 同分布测试** |
| `ccpd_blur` | 1845 | 强扰动测试 |
| `ccpd_challenge` | 4077 | 强扰动测试 |
| `ccpd_rotate` | 1050 | 强扰动测试 |
| `ccpd_tilt` | 2504 | 强扰动测试 |
| `ccpd_weather` | 1021 | 强扰动测试 |
| `ccpd_fn` | 1822 | 强扰动测试 |
| `ccpd_green` | 1179 | **丢弃**：新能源 8 字符牌，与本任务 6 位定义不符 |
| `ccpd_np` | 480 | **丢弃**：文件名无角点标注，无法定位车牌 |
| `ccpd_db` | 1035 | **当前未使用**：既非 base 也非 6 类强扰动，缓存后不参与任何划分 |

因此 30000 − 1179 − 480 = **28341** 张进入缓存。

> **若你使用官方 CCPD 原始文件名**（不带 `_ccpd_<subset>_` token，子集由目录名表示），需要先用脚本把目录名补进文件名，或修改`models/ccpd_parse.py::extract_subset` 传入所在目录名；否则划分会失败。

把下载到的压缩包放 `data/raw_dl/`，解压出的 jpg 直接摊进 `data/ccpd/` 即可。

### 4.2 一键数据准备

```bash
# 数据准备：解析 + 透视矫正裁剪 + 过滤统计
#   写出 data/processed/ccpd_<W>x<H>.npz（缓存）
#   同时把裁剪后的图片逐张写成 PNG 到 data/crops/<tag>/（默认开启，可肉眼核对）
python train/prepare_data.py

# 只想重建缓存、不要 2.8 万张 PNG 时：
python train/prepare_data.py --no-save-crops

# 剪裁图改写到别的目录，或只存前 500 张：
python train/prepare_data.py --save-crops D:/crops_dump
python train/prepare_data.py --save-crops-n 500

# ★ 人工核对（必做，通过前不得进入训练）
#   产出 20 张「裁剪图 + 标签字符串」网格图，确认裁剪区域与标签逐位对应
python evaluate/visualize.py check-grid

# 数据划分：号码去重划分 + 生成合成域测试集（按 configs 的 synth.backend 选择后端）
python train/split_dataset.py
```

* 剪裁图文件名为 `<缓存行号>_<车牌文本>_<来源文件名>.png`，行号与 `data/processed/ccpd_<W>x<H>.npz` 的行、以及 `splits.npz` 里的下标**严格一一对应**，可逐张对照标签；剪裁图本质是缓存内容的可视化副本，因此不入库（在 `data/crops/` 下，已被 `.gitignore` 排除）。
* 划分严格按**车牌号码去重**：同一号码的全部图片归入同一集合，且train/val/test/hard 两两交集为 0、合成域与真实域无重叠（`split_summary.json`记录全部交集计数）。
* 标准化统计量**只在训练集上拟合**（mean=0.421204、std=0.221177、n=9000），验证 / 测试 / 强扰动 / 合成域一律复用，不得各自重新拟合。
* 合成域生成器可单独预览：`python train/synth_from_generator.py --n 40`，会打印生成器 commit / 种子 / 扰动链并输出人工核对网格。
* 预处理缓存由 `prepare_data.py` 写出、划分文件由 `split_dataset.py` 写出；`train/train.py` 只读取这两个文件，缺失时报错并提示先跑对应步骤，**不会**自动重建（避免口径被无意改变）。

---

## 5. 训练、正确性自检与评测

```bash
# 梯度检查（数值梯度 vs 手写反向；相对误差 < 1e-5）
python train/grad_check.py

# 小样本过拟合自检（100 样本，要求 loss < 1e-3 且字符/整牌 100%）
python train/overfit_check.py

# NumPy / CuPy 前后端数值一致性校验（可选）
python train/parity_check.py

# ★ 训练最终交付口径（最终模型使用关闭增强的配置）
python train/train.py --config reports/configs/final_s42.yaml --name final_s42 --seed 42

# ★ 独立评测（载入检查点，不重新训练；含 CPU 单张/批量推理时间）
#   注意入口是 evaluate/main.py —— 命名为 evaluate.py 会遮蔽 evaluate 包
python evaluate/main.py --run final_s42
```

* `train/train.py` 会在 `reports/logs/` 写出 `<name>_history.csv`（逐 epoch）与 `<name>_run.json`（含 commit / 配置指纹 / 种子 / 三测试集指标），并把最佳验证权重存为 `reports/checkpoints/<name>_best.npz`。
* `evaluate/main.py --run <name>` 会读取 `reports/configs/<name>.yaml`（缺失时回退 `reports/configs/final.yaml`），并写出 `<name>_eval.json`、 `<name>_eval_arrays.npz`（逐位置 34×34 混淆矩阵）、`<name>_errors.npz`（错误样本图像与真实/预测标签）。
* 训练、评测与推理的输入缓存、划分文件与 `ccpd.input_size` **必须配套**：换输入尺寸就要重建对应缓存与划分（`paths.splits_file` 可切换）。

---

## 6. 识别自己准备的车牌（推理）

最终权重在 `reports/checkpoints/final_s{42,43,44}_best.npz`。用根目录的`predict.py` 即可识别任意一张车牌图（本项目**不做车牌检测**：车牌区域需要你自己裁，或用 `--corners` 给出四角顶点）：

```bash
# 图片是完整 7 位车牌正视图（含首位省份汉字）——脚本自动裁掉汉字区域
python predict.py my_plate.jpg

# 指定另外两个种子的检查点
python predict.py my_plate.jpg --run final_s43

# 图片已经是后 6 位字符区域（已裁掉汉字）
python predict.py my_plate_6chars.jpg --mode cropped

# 车牌在照片里带倾斜/透视：给出四角顶点，先矫正再识别
# （顺序与 CCPD 标注一致：右下、左下、左上、右上）
python predict.py car_photo.jpg --corners "433,341;120,315;128,272;445,295"

# 多张图片 + JSON 输出 + 导出实际送入模型的 32×128 灰度图（核对预处理）
python predict.py a.jpg b.jpg --json --save-debug reports/figs/_debug
```

* `--run` 选择检查点（默认 `final_s42`，同分布字符 98.40% / 整牌 91.90%）；`--ckpt PATH` 可直接指定 `*_best.npz` 文件。
* 预处理与训练**严格同口径**：透视矫正（可选）→ 裁掉首位汉字 → 缩放 128×32 →灰度 → [0,1] → 用**训练集拟合的**标准化统计量做零均值单位方差（绝不用你自己的图片现算均值/方差，否则输入分布就变了）。
* 输出：六位字符串、逐位置置信度、整牌联合置信度（六位概率乘积）。**整牌置信度极低（如 < 0.05）通常意味着预处理就错了**（`--mode` 选错、裁剪区域不对），先用 `--save-debug` 看一眼实际送入模型的图；反过来，置信度高也不保证逐位全对——字形相近对（8/B、5/S、2/Z、1/7）仍是主要错误来源。
* 验证方式：把 `reports/logs/final_s42_errors.npz` 里的错误样本（`error_images`是 **uint8 原始像素**）存成 PNG，再用 `--mode cropped` 识别，结果与`final_s42_eval.json` 的预测**逐位一致**，可用于确认推理链路未被改坏。

---

## 7. 复现性与口径要点

每次运行的日志都写入四个溯源字段：

| 字段 | 含义 |
| --- | --- |
| `commit` | 短 7 位 Git hash（`git rev-parse --short HEAD`） |
| `config` | 配置名 / 内容指纹（sha1 前 12 位） |
| `seed` | 随机种子 |
| `dirty` | 工作区是否有未提交改动（`git status --porcelain`） |

> ⚠️ **`dirty=True` 的结果不可作为最终结论。** 重跑前请先提交代码。

最终模型的三个检查点记录在 `reports/logs/final_s*_run.json`：训练时`commit=eee708e`、`dirty=true`、配置指纹 `a5cbaf953280`、后端 `cupy`。（`run.json` 内的 `meta.run_name` 保留训练时的原始运行名，权重与数值未做任何改动；`final_s*` 即为该口径的三次运行。）

口径要点（易错处）：

* **类别索引顺序遵循 CCPD 官方 `ads` 表**：索引 `0–23` 为字母 `A–Z`（去掉 `I`、`O`），`24–33` 为数字 `0–9`。因此 `A→0`、`Z→23`、`0→24`、`9→33`。官方表来源已存档于`docs/CCPD_README.md`。
* **偏置项不参与 L2 惩罚**，L2 直接写进损失（`loss.l2_lambda`）。
* **必须同时报告字符准确率与整牌准确率**：字符准确率 95% 时整牌准确率上界仅约 73.5%。
* **`evaluate/main.py` 不能命名为 `evaluate/evaluate.py`**：直接运行会把它注册为顶层模块 `evaluate`，遮蔽同名包，导致 `ModuleNotFoundError: No module named 'evaluate.model_eval'`。
* **评测入口必须载入检查点**：`Params.load` 是 classmethod、返回 `(Params, extra)`；写成 `params.load(ckpt)` 会丢弃返回值、静默使用随机权重（本项目踩过这个坑，表现为独立评测 3.10% 而训练脚本 86.73%）。
* **`configs/default.yaml` 的增强档位（`weak`）是针对本架构调过的默认值**，但**不是最终模型口径**：MLP 无平移不变性，几何增强（旋转/缩放/平移）在本架构上产生负效果，最终模型使用 `reports/configs/final_s*.yaml`（`baseline_level: none`）。
* **`reports/checkpoints/` 只入库最终权重** `final_s{42,43,44}_best.npz`（3 个文件共约 12 MB，是"最终训练好的模型"本体，不可由脚本重建）；其余权重（冒烟 / 临时 / 中间轮次）仍不入库。`reports/figs/` 不入库（可由日志重绘），`reports/logs/` 与 `reports/configs/` 入库（体积小、是结论的直接证据）。

---

## 8. 许可与致谢

* CCPD 数据集：MIT License，论文 *Towards End-to-End License Plate Detection and Recognition: A Large Dataset and Baseline* (ECCV 2018)。
* 合成域生成器：[Nenger/chinese_licence_plate_generator](https://github.com/Nenger/chinese_licence_plate_generator)（固定 commit `43bac43`）。
* 本项目不进行车牌检测，仅使用 CCPD 文件名中标注的四角顶点做透视矫正与裁剪。