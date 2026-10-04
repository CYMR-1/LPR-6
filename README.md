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
├─ predict.py                     # ★ 用训练好的检查点识别自己的车牌图（见 §5）
├─ requirements.txt
├─ .gitignore
├─ configs/
│  ├─ default.yaml                # ★ 全部超参数的唯一来源
│  ├─ _e8_24x96.yaml              # E8 变体配置（24×96 输入口径）
│  └─ _e9_check.yaml              # E9 结构校验配置
├─ docs/
│  └─ CCPD_README.md              # 官方 README 存档（字符映射核对依据）
├─ data/
│  ├─ ccpd/                       # CCPD 原图（不入库）
│  ├─ external/                   # 外部合成域生成器仓库（不入库，见 §3.1；固定 commit 43bac43）
│  ├─ synth_test/                 # 合成测试图（不入库；图像已并入 splits.npz）
│  ├─ processed/                  # 预处理缓存 .npz（不入库）
│  └─ manifest.csv                # 来源文件、六标签、子集名、裁剪参数、随机种子
├─ models/
│  ├─ config.py                   # 配置加载 / 点分路径补丁 / Git 溯源
│  ├─ charset.py                  # 34 类字符集与 CCPD 索引映射表
│  ├─ ccpd_parse.py               # 文件名解析、透视矫正、裁剪、标准化
│  ├─ dataset.py                  # 批加载器（支持 batch=1 与全批量）
│  ├─ model.py                    # 共享/独立模型（前向 + 手写反向）与损失（交叉熵 / MSE / L2）
│  ├─ augment.py                  # 数据增强（无 / 弱 / 强三档）
│  ├─ backend.py                  # NumPy / CuPy 统一后端入口
│  ├─ metrics.py                  # 评价指标（字符 / 整牌准确率等）
│  └─ optim.py                    # SGD、动量、批量策略
├─ train/
│  ├─ train.py                    # 训练循环、早停、日志落盘
│  ├─ phase1_prepare.py           # P1：CCPD 解析 + 裁剪 + 过滤统计（生成 .npz 缓存）
│  ├─ phase15_split.py            # P1.5：号码去重划分 + 合成测试集生成（按 synth.backend 分发）
│  ├─ synth_from_generator.py     # ★ 合成域后端 generator_repo：外部生成器牌面级输出 + 同口径几何
│  ├─ synth_plates.py             # 合成域后端 pil_renderer（内置 PIL 渲染器，保留可切回）
│  ├─ swap_synth_to_generator.py  # 一次性迁移：只替换划分里的 synth 数组，历史 hard 集不变
│  ├─ grad_check.py               # 数值梯度检查
│  ├─ overfit_check.py            # 小样本过拟合自检
│  ├─ parity_check.py             # NumPy / CuPy 前后端数值一致性校验
│  ├─ make_e8_cache.py            # E8 的 24×96 缓存与划分生成
│  ├─ diag_augment.py             # 诊断：逐算子量化增强的破坏程度
│  ├─ diag_domain_gap.py          # 诊断：真实域 vs 合成域的差异定位
│  ├─ diag_relu_death.py          # 诊断：E2 ReLU 隐层死亡机制
│  ├─ diag_synth_learnability.py  # 诊断：合成域可学性对照
│  ├─ diag_bs1_timing.py          # 诊断：bs=1 单轮耗时实测
│  ├─ repair_splits_remap.py      # 一次性修复：划分重映射到规范化缓存
│  └─ restore_split_summary.py    # 一次性修复：恢复主口径 split_summary.json
├─ evaluate/
│  ├─ main.py                     # 独立评测入口（载入检查点，不重训）
│  ├─ model_eval.py               # 各评价指标、混淆矩阵、分位置准确率、CPU 计时
│  └─ visualize.py                # 曲线、样本网格、错误样本可视化
├─ reports/
│  ├─ run_all.py                  # 一键跑完全部启用的对照实验
│  ├─ make_figs.py                # 由数值产物重画全部图片
│  ├─ make_split_bar_figs.py      # 由汇总表重画各测试集变体对比柱状图（不训练）
│  ├─ build_report_tables.py      # 由数值产物生成报告用 Markdown 表格
│  ├─ rebuild_summary_tables.py   # 由 logs/*_run.json 重建汇总表（不训练）
│  ├─ refresh_synth_eval.py       # 合成域换后端后：用现有检查点重测（不重训）
│  ├─ configs/                    # 每次运行实际生效的变体配置（入库）
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
| 合成域测试集 | 外部开源生成器 **[Nenger/chinese_licence_plate_generator](https://github.com/Nenger/chinese_licence_plate_generator)**（固定 commit `43bac43`，2018-04-09）的**牌面级**输出，**只测试、不训练** |

合成域用其牌面级接口（`FakePlateGenerator.generate_one_plate()` + 上游
`jittering_color/add_noise/jittering_blur/jittering_scale` 扰动链），再套用与
CCPD **完全相同**的几何裁剪（裁左 1/7 → 128×32 → 灰度）；上游字符素材含
字母 I/O，生成时按本项目 34 类字符集**拒绝重采**。该仓库的主打产物是
"车牌贴进街景图"的**检测**数据集，本项目不做检测（§0.3 规范），故不使用其
场景整图。生成器依赖 OpenCV（**仅数据生成环节**，训练/评测/预处理不依赖）。

```bash
# 克隆生成器（不入库，见 .gitignore 的 data/external/）
git clone https://github.com/Nenger/chinese_licence_plate_generator data/external/chinese_licence_plate_generator
# 生成器主分支即 43bac43；若上游有更新，可用 --branch 指定 tag/commit
```

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

# P1.5：号码去重划分 + 生成合成域测试集（按 configs 的 synth.backend 选择后端）
python train/phase15_split.py
```

> 合成域生成器的克隆在 §3.1；`train/synth_from_generator.py --n 40` 可单独
> 生成预览并打印溯源信息（生成器 commit / 种子 / 扰动链）。
> **合成域只测试不训练**；若在已有划分上只更换合成域后端（保留历史 hard 集），
> 用 `python train/swap_synth_to_generator.py`（内置"其余内容逐字节不变"校验），
> 再用 `python reports/refresh_synth_eval.py` 以现有检查点**重测**（不重训）。

> 说明：预处理缓存（`data/processed/ccpd_<W>x<H>.npz`）由 P1 写出，
> 划分文件（`data/processed/splits.npz`）由 P1.5 写出；
> `train/train.py` 只读取这两个文件，缺失时会报错并提示先跑对应步骤，
> **不会**自动重建（避免口径在无意中被改变）。

---

## 4. 训练与评估

```bash
# 梯度检查（P3 验收：相对误差 < 1e-5）
python train/grad_check.py

# 小样本过拟合自检（P2/P3 验收：loss < 1e-3 且字符/整牌 100%）
python train/overfit_check.py

# 单次基线训练（脚本名与运行短名；训练、日志落盘、最佳模型保存一体）
python train/train.py --name baseline_s42 --seed 42

# ★ 独立评测（载入检查点，不重新训练；含 CPU 单张/批量推理时间）
#   注意入口是 evaluate/main.py —— 命名为 evaluate.py 会遮蔽 evaluate 包
python evaluate/main.py --run baseline_s42

# 出图（由数值产物重画曲线 / 混淆矩阵 / 错误样本）
python reports/make_figs.py --runs baseline_s42
```

### 诊断脚本（负结果的可复核证据）

```bash
# 逐算子量化数据增强的破坏程度 -> reports/logs/augment_diag.json
python train/diag_augment.py

# 定位真实域与合成域的差异（标签一致性、字符位置、输入尺度）
python train/diag_domain_gap.py
```

### 一键跑完对照实验

```bash
python reports/run_all.py                       # 跑配置中 enabled 的实验
python reports/run_all.py --only E1 E3 E4 E7
python reports/run_all.py --seeds 42 43 44
python reports/run_all.py --skip-existing       # 断点续跑，复用已有产物
python reports/run_all.py --only E4 --limit 2000 --prefix e4n_ \
       --skip-variant batch_1                   # 统一小样本口径 + 显式跳过某变体
python reports/run_all.py --run-timeout 1500    # 单次运行时间预算（秒）
```

每个实验至少 **3 个随机种子**重复，报告**均值 ± 标准差**。
`--run-timeout` 会在超预算时提前结束训练并写入 `budget_exhausted: true`，
**不会把"没跑满"伪装成跑满的结果**。

### 生成报告表格

```bash
python reports/build_report_tables.py --write --inject
```

报告里的表格由数值产物自动生成并注入 `reports/实验报告.md` 的
`<!-- TABLE:XXX -->` 占位符，避免手工抄写数字出错。

---

## 5. 识别自己准备的车牌（推理）

训练好的权重在 `reports/checkpoints/<run>_best.npz`。用根目录的
`predict.py` 即可识别任意一张**已裁出车牌区域**的图片（本项目不做车牌检测：车牌区域需要你自己裁，或用 `--corners` 给出四角顶点）：

```bash
# 图片是完整 7 位车牌正视图（含首位省份汉字）——脚本自动裁掉汉字区域
python predict.py my_plate.jpg

# 图片已经是后 6 位字符区域（已裁掉汉字）
python predict.py my_plate_6chars.jpg --mode cropped

# 车牌在照片里带倾斜/透视：给出四角顶点，先矫正再识别
# （顺序与 CCPD 标注一致：右下、左下、左上、右上）
python predict.py car_photo.jpg --corners "433,341;120,315;128,272;445,295"

# 多张图片 + JSON 输出 + 导出实际送入模型的 32×128 灰度图（核对预处理）
python predict.py a.jpg b.jpg --json --save-debug reports/figs/_debug
```

* `--run` 选择检查点，默认 `E7_aug_none_s42`（同分布字符 98.4% / 整牌
  91.8%，本项目最优，见实验报告 §1）；全部可用运行名见 `reports/checkpoints/`。
* 预处理与训练**严格同口径**：透视矫正（可选）→ 裁掉首位汉字 → 缩放
  128×32 → 灰度 → [0,1] → 用**训练集拟合的**标准化统计量做零均值单位
  方差（绝不用你自己的图片现算均值/方差，否则输入分布就变了）。
* 输出：六位字符串、逐位置置信度、整牌联合置信度（六位概率乘积）。
  **整牌置信度极低（如 < 0.05）通常意味着预处理就错了**（`--mode` 选错、
  裁剪区域不对），先用 `--save-debug` 看一眼实际送入模型的图；
  反过来，置信度高也不保证逐位全对——字形相近对（8/B、5/S、2/Z、1/7）
  仍是主要错误来源（实验报告 §6.2）。

> ⚠️ **域差距提醒**：模型训练于 CCPD 风格的真实车牌特写（同分布整牌
> 91.8%），对风格差异大的输入会显著退化——实验报告 §8 实测：外部生成器
> 合成域（`generator_repo`）字符准确率 **43.3%**（基线）/ **66.9%**
> （E1 独立架构），整牌几乎为 0；旧版内置渲染器合成域则只有约 5%。
> 手机远距离拍摄后再放大裁切、字体差异大的图片都属于"域外"输入，
> 此时预测结果只能当参考。

---

## 6. 实验与代码的绑定（复现性）

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

## 7. 复现步骤（端到端）

```bash
# 0) 环境
python -m venv .venv && .venv\Scripts\activate && pip install -r requirements.txt

# 1) 数据（先解压 CCPD 到 data/ccpd/）
python train/phase1_prepare.py
python evaluate/visualize.py check-grid      # 人工核对 20 张
git clone https://github.com/Nenger/chinese_licence_plate_generator data/external/chinese_licence_plate_generator
python train/synth_from_generator.py --n 40  # 合成域自检 + 人工核对网格
python train/phase15_split.py

# 2) 正确性验证
python train/grad_check.py
python train/overfit_check.py

# 3) 基线训练 + 对照实验（必做集 E1/E3/E4/E7）
python reports/run_all.py --baseline --only E1 E3 E4 E7 --skip-existing

# 4) 独立评测 + 出图 + 报告表格
python evaluate/main.py --run baseline_s42
python reports/make_figs.py --runs baseline_s42
python reports/rebuild_summary_tables.py     # 由 logs 重建汇总表（不训练）
python reports/make_split_bar_figs.py        # 由汇总表重画变体对比图（不训练）
python reports/build_report_tables.py --write --inject

# 5) 报告见 reports/实验报告.md

# 6)（可选）用训练好的模型识别自己的车牌，见 §5
python predict.py my_plate.jpg
```

---

## 8. 口径要点（易错处）

* **类别索引顺序遵循 CCPD 官方 `ads` 表**：索引 `0–23` 为字母 `A-Z`（去掉 `I`、`O`），
  `24–33` 为数字 `0–9`。因此 `A→0`、`Z→23`、`0→24`、`9→33`。
  官方表来源已存档于 `docs/CCPD_README.md`，并与官方源码 `rpnet/demo.py` 逐项核对。
* **划分按车牌号码去重**（§2.5）：同一车牌号码的全部图片必须归入同一集合，
  否则模型只需记住号码纹理即可拿高分，测试结论不成立。
* **偏置项不参与 L2 惩罚**（§4.1）。
* **必须同时报告字符准确率与整牌准确率**：字符准确率 95% 时整牌准确率上界仅约 73.5%。
* **合成域与强扰动测试集在调参阶段不得参与任何选择决策**（§2.5 要求 3）。
* **合成域后端由 `synth.backend` 决定**：默认 `generator_repo`（外部生成器
  牌面级输出，需先按 §3.1 克隆，固定 commit `43bac43`，OpenCV 只在生成环节
  使用）；`pil_renderer` 为内置确定性 PIL 渲染器，代码保留可切回，但
  **不再是报告口径**——同一模型在两种合成集上测出的域差差别极大
  （5.6% vs 41.6%），报告数字必须注明后端与 commit。
* **合成域生成器上游含字母 I/O**，与本项目 34 类字符集不符，生成时按
  `check_label_legal` 拒绝重采（2000 张共拒绝 912 张）；改生成器/种子后
  必须重跑 `train/diag_domain_gap.py` 复核标签一致性。
* **标准化统计量只在训练集上拟合**（mean=0.421204、std=0.221177、n=9000），
  验证 / 测试 / 强扰动 / 合成域一律复用，不得各自重新拟合。
* **`evaluate/main.py` 不能命名为 `evaluate/evaluate.py`**：直接运行会把它注册为
  顶层模块 `evaluate`，遮蔽同名包，导致
  `ModuleNotFoundError: No module named 'evaluate.model_eval'`。
* **评测入口必须载入检查点**：`Params.load` 是 classmethod、返回 `(Params, extra)`；
  写成 `params.load(ckpt)` 会丢弃返回值、静默使用随机权重（本项目踩过这个坑，
  表现为独立评测 3.10% 而训练脚本 86.73%）。
* **`configs/default.yaml` 的增强强度是针对本架构调过的**：MLP 无平移不变性，
  几何增强（旋转/缩放/平移）会产生负效果，详见实验报告 §7.4。

---

## 9. 许可与致谢

* CCPD 数据集：MIT License，论文 *Towards End-to-End License Plate Detection and
  Recognition: A Large Dataset and Baseline* (ECCV 2018)。
* 本项目不进行车牌检测，仅使用 CCPD 文件名中标注的四角顶点做透视矫正与裁剪。