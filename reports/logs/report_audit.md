# 实验报告数值审计（独立核验，第 2 版）

**被审计快照**：`reports/实验报告.md`，LastWriteTime = 2026-10-04 16:08:56，52423 字节 / 1050 行
（含 E2/E5 章节、46 行 RANKING、12 行 CPU 表。本文件上一版针对的是 945 行旧快照，已作废并被本版整体覆盖。）
**产物快照**：`reports/logs/` 下 48 份完整 `*_run.json`（无残缺）；其中 `E5_momentum_09_s42_run.json`
写于 16:11:28，**晚于**报告快照，相应差异按"竞态"单独标注。
**审计方式**：只从产物读数再与报告比对，不以报告自证。聚合用样本标准差（ddof=1，与
`reports/build_report_tables.py` 的 `statistics.stdev` 一致）；临时脚本放在 `%TEMP%\px_audit.py`，
未在仓库内留下中间文件；未修改报告及任何被测产物，只重写本文件。
**环境**：`.\.venv\Scripts\python.exe`（Python 3.12.14），`$env:PYTHONIOENCODING="utf-8"`。

## 0. 结论概览

| 项 | 数量 |
|---|---|
| 逐条核验的数值断言（11 张注入表 + 正文散落数字，按可判定断言归并） | **74 条** |
| 一致 | **47 条** |
| 不一致 | **17 条**（高 3 / 中 5 / 低 9） |
| 无法核验（产物中无对应记录） | **10 条** |

**一句话结论**：11 张注入表与当前产物**逐值一致**（BASELINE/E1/E2/E3/E4/E5/E7/MIDTRAIN/POSITION/
CPU/RANKING 全部吻合，旧版审计发现的 4 张过期表已在 16:08 的重注入中修复）；其余问题集中在
**正文散落数字**：§5.1 independent 梯度校验两个数值与产物不符、§7.1 "95.8%" 口径混用、
§8.2 "真实图 101.3" 与 §8.3/产物矛盾、§7.5 batch_1 三区间小表 4 格偏差、§10.5 仍称 E2/E5 "未做"。

### 0.1 复现命令（项目根目录 PowerShell）

```powershell
$env:PYTHONIOENCODING="utf-8"
# ① 用当前产物重生成全部表格（只打印不写文件），与报告注入块逐行比对
.\.venv\Scripts\python.exe reports\build_report_tables.py
# ② §5.1 independent 梯度校验真实数值（报告写 4.65e-05 / 4.11e-09）
.\.venv\Scripts\python.exe -c "import json;d=json.load(open('reports/logs/grad_check_independent.json',encoding='utf-8'));print(d['max_rel_err'],d['max_abs_err'],d['n_failed'],d['passed'],d['meta'])"
# ③ §7.1 "训练字符 95.8%" 的 3 种子均值
.\.venv\Scripts\python.exe -c "import json,statistics as s;print(s.fmean([json.load(open('reports/logs/E1_independent_s%d_run.json'%i,encoding='utf-8'))['train']['char_acc'] for i in (42,43,44)]))"
# ④ §8.2 "真实图 101.3" 的对照（domain_gap.json 实测 108.25）
.\.venv\Scripts\python.exe -c "import json;print(json.load(open('reports/logs/domain_gap.json',encoding='utf-8'))['scale_real'])"
# ⑤ §7.5 batch_1 三区间真实范围
.\.venv\Scripts\python.exe -c "import csv;R=list(csv.DictReader(open('reports/logs/e4n_E4_batch_1_s42_history.csv',encoding='utf-8-sig')));import itertools;[print(lo,hi,min(float(r['val_loss']) for r in R if lo<=int(r['epoch'])<=hi),max(float(r['val_loss']) for r in R if lo<=int(r['epoch'])<=hi),min(float(r['val_char_acc']) for r in R if lo<=int(r['epoch'])<=hi),max(float(r['val_char_acc']) for r in R if lo<=int(r['epoch'])<=hi)) for lo,hi in ((1,5),(10,20),(25,35))]"
# ⑥ §10.5 "E2/E5 未做" 的反例
Get-Item reports\logs\E2_relu_s42_run.json, reports\logs\E5_momentum_0_s44_run.json
# ⑦ E2 死亡 ReLU 解剖（6.2% / −110.7 / 99.5% / 0.0% / +0.04 / 49.0%）
.\.venv\Scripts\python.exe _diag_relu.py
```

---

## 1. 任务点名 11 条断言的逐条核验

| # | 断言原文（位置） | 产物实测 | 结论 | 证据路径与字段 |
|---|---|---|---|---|
| 1 | baseline 3 种子：测试字符 86.82±1.76%、整牌 43.43±6.49%、强扰动 62.25±1.87%、合成域 5.84±0.28%（§6 表 baseline 与 shared 行） | `baseline_s42/s43/s44`：test.char [0.867333,0.851083,0.886333]→86.82±1.76；plate [0.426,0.374,0.503]→43.43±6.49；hard.char 62.25±1.87；synth.char 5.84±0.28；平均轮数 58.3。与 `E1_shared_*` 逐字节等价（同种子各项指标完全相同），与 `E2_sigmoid_*`/`E3_cross_entropy_*`/`E7_aug_weak_*` 亦逐字节等价 | ✅ 一致（当前快照 baseline 行已是 3 种子，与 shared 行完全相同） | `reports/logs/baseline_s4{2,3,4}_run.json`、`E1_shared_s4{2,3,4}_run.json` 的 `test/hard_test/synth_test` |
| 2 | E1：independent 94.30±0.48 / 72.55±1.75；shared 同基线；参数量 6345420 vs 1101260；CPU 1.934 ms vs 0.23–0.39 ms（§7.1 表与结论表） | independent 3 种子：94.30±0.48 / 72.55±1.75 / 69.75±1.12 / 6.46±0.02，平均 35.3 轮，6345420；shared 全部同上一条；`E1_independent_s42_eval.json::cpu_inference`=1.9344 ms/516.9 张·s⁻¹；共享族 12 份 eval 在 0.2277–0.3878 ms | ✅ 一致（**正文另有 1 处口径混用**，见不一致 #2） | `E1_*_run.json::meta.model.num_parameters`、`E1_*_eval.json::cpu_inference` |
| 3 | E3：交叉熵首次达 80% 验证字符为第 16 轮、MSE 第 37 轮；第 20 轮 81.62% vs 75.47%；最终整牌 43.43% vs 24.42%（§7.3/§7.4） | 每种子首次≥80%：CE=[13,18,16] 均值 15.7→"16 轮"；MSE=[36,36,40] 均值 37.3→"37 轮"。第 1/5/10/20 轮均值 46.41/72.05/77.92/81.62 vs 41.80/65.78/70.58/75.47；最终（最佳权重）86.59 vs 81.52；val_loss 13.0038/4.3458/3.6252 vs 2.2704/1.2719/1.0757；最终测试整牌 43.43 vs 24.42 | ✅ 一致（全部 15 个数值 + 2 个轮次） | `E3_*_run.json::history.epochs[*].val_char_acc/val_loss`（与同名 `_history.csv` 交叉一致） |
| 4 | E4（e4n_ 统一 2000 张口径）：128 最优（整牌 27.95±1.11%）；batch_1 训练 35 轮、验证字符约 20%、整牌恒 0.00%、val_loss 43–81 震荡；batch_32 整牌 7.30±2.26%（§7.5） | E4 表 5 个变体全部 `e4n_` 前缀、`meta` 训练集 2000 张：batch_128 27.95±1.11 ✓、batch_32 7.30±2.26 ✓、batch_64 19.52±1.94 ✓、batch_full 1.58±0.34 ✓、batch_1（仅 s42）21.20/0.00/19.20/4.02、35 轮早停（best=27，27+8=35 ✓）、35 轮 val_loss 全程 43.56–81.23、val_char 13.22–21.43%、val_plate 恒 0 | ✅ 表与主断言一致（**三区间小表 4 格不符**，见不一致 #4） | `e4n_E4_*_run.json`、`e4n_E4_batch_1_s42_history.csv` |
| 5 | E7：none 91.77±0.40 > weak 43.43±6.49 > strong 2.35±0.44（§7.8） | aug_none 0.917667±0.004041、aug_weak 0.434333±0.064902、aug_strong 0.023500±0.004444，严格单调；整表（98.36±0.08 / 72.64±0.09 / 5.68±0.18 等）逐值吻合 | ✅ 一致 | `E7_aug_{none,weak,strong}_s4{2,3,4}_run.json::test.plate_acc` |
| 6 | 合成域 ≈5%：各运行 synth char_acc 落在约 4–7%（§1.6/§8） | 48 份运行 min=4.02%（e4n_E4_batch_1_s42）、max=6.47%（E1_independent_s42/s43），全部落区间 | ✅ 一致 | `reports/logs/*_run.json::synth_test.char_acc` |
| 7 | 梯度校验：shared max_rel 6.09e-07、independent max_rel 4.65e-05、n_failed=0（§5.1） | shared：max_rel=6.0902e-07、max_abs=3.7453e-09、n_failed=0、passed ✓；independent：**max_rel=5.3364e-06、max_abs=4.9062e-09**、n_failed=0、passed ✓ | ⚠️ shared 一致；**independent 两个数值均不符**（PASS 结论不变，5.34e-06<1e-5） | `reports/logs/grad_check_{shared,independent}.json::max_rel_err/max_abs_err/n_failed` |
| 8 | 过拟合自检：loss 9.594e-04、epoch 18000、字符/整牌 100%（§5.2） | `final_loss`=9.59376e-04、`epochs_run`=18000/20000、char/plate=1.0/1.0、passed=true；meta：n=100、lr=0.5、μ=0、l2=0 ✓ | ✅ 一致 | `reports/logs/overfit_check.json` |
| 9 | 划分 9000/2000/2000/2000/2000，唯一号码 8684/1921/1924/1994，两两交集全 0（§2.3） | 从 `splits.npz` 索引 + `ccpd_128x32.npz::labels` 独立重算：长度 9000/2000/2000/2000、synth 2000；唯一 6 位号码 **8684/1921/1924/1994**、synth 2000 全唯一；6 个真实集两两交集=0、与 synth 交集=0；`synth_texts` 与标签解码 2000/2000 一致 | ✅ 一致（旁证 `split_summary.json`；其 `unique_plates` 未含 hard，1994 由本次独立重算确认） | `data/processed/splits.npz`、`data/processed/ccpd_128x32.npz`、`reports/logs/split_summary.json` |
| 10 | diag_augment 逐算子：几何 0.1128 vs 其余 0.021–0.038（§7.8 两表 24 个数值） | 逐值核对 24/24 吻合（四舍五入后）：几何 0.11280/0.15483/0.02982，其余六算子 0.0213–0.0376；档位 none 0/0.13675/0.03429、weak 0.12221/0.17064/0.03290、strong 0.15555/0.16373/0.02964。"比其它项大 3–5 倍"实测比值 3.00–5.30（上限略超 5，视为约数成立） | ✅ 一致 | `reports/logs/augment_diag.json::by_operator/by_level/reference` |
| 11 | domain_gap：标签一致性 2000/2000、质心 6 格、尺度统计（§8.1/§8.3） | `labels`=2000/2000/0 ✓；质心 12 个数值与 §8.1 表逐值吻合（9.03±0.62/10.04±0.62 … 114.55±1.50/115.05±0.74）；尺度 108.2495/136.5941、+0.0149σ/+0.5175σ、0.982/1.176 → §8.3 表全部吻合 | ✅ 一致（**唯一例外**：§8.2 另写 "真实图 101.3"，见不一致 #3） | `reports/logs/domain_gap.json` |

---

## 2. 11 张注入表格 vs 当前产物（`build_report_tables.py` 只打印模式重生成 + 独立重算）

| 表 | 结论 | 说明 |
|---|---|---|
| BASELINE | ✅ | baseline/shared 两行 3 种子、数值完全相同，与重算一致（旧版审计的"baseline 行只有 1 种子"已在 16:08 修复） |
| E1 | ✅ | 2 行 12 值全合 |
| E2 | ✅ | relu 22.07±0.13/0.00/20.77±0.44/4.46±0.07、28.3 轮；sigmoid 行=基线行，与产物一致 |
| E3 | ✅ | 2 行全合 |
| E4 | ✅ | 5 个 `e4n_` 变体（batch_1 仅 1 种子，与产物一致） |
| E5 | ⚠️ 竞态 | 表中 momentum_0 行（89.06±0.17/45.50±0.63/68.41±0.27/6.11±0.10、80.0 轮）与产物一致；但 `E5_momentum_09_s42`（16:11:28 写入，晚于报告快照 3 分钟）未入表，重生成会多出 momentum_09 行（1 种子：86.73/42.60/62.80/5.62、51 轮） |
| E7 | ✅ | 3 行全合 |
| MIDTRAIN | ✅ | 三张小表全部吻合（另见编辑性问题 E-1：锚点块外重复了一份） |
| POSITION | ✅ | 现表 = baseline×3 + E1_shared×3 共 6 份运行的均值（93.20±0.19 … 整牌 43.43±5.81、字符 86.82±1.58），本次按 6 运行与 3 运行两种口径独立重算确认 6 运行口径逐值吻合 |
| CPU | ✅ | 12 行与 12 份 `*_eval.json::cpu_inference` 逐值吻合 |
| RANKING | ⚠️ 竞态 | 报告 46 行逐值全部吻合；当前产物重生成 47 行，仅多 `E5_momentum_09_s42`（86.73/42.60，应插在第 18 名附近）——报告快照后新到的运行，非抄错 |

---

## 3. 发现的不一致清单（17 条，按严重度）

### 高（3 条：与产物直接冲突的正文数字）

**#1 §5.1 梯度校验 independent 行两个数值与产物不符。**
- 报告（`实验报告.md:241`）："independent｜相对误差最大 **4.65e-05**，绝对误差最大 **4.11e-09**，失败 0 项"，
  且注释据此写"某些被抽查的分量梯度接近 0（解析值 ≈ −1.2e-05）"。
- 产物 `reports/logs/grad_check_independent.json`：`max_rel_err`=**5.3364e-06**、`max_abs_err`=**4.9062e-09**、
  `n_failed`=0、`passed`=true；`records` 中没有任何 ≈−1.2e-05 的解析值（|analytic| 最小为 **5.79e-05**，`b1i_2`）。
- 附带披露缺口：independent 校验用的是**缩量模型**（`meta.hidden_dim`=64、n_samples=4、每参数抽 3 分量 →
  n_checked=72），shared 校验是全尺寸（hidden_dim=256、n_samples=8、每参数 15 分量 → n_checked=210），
  报告未披露这一口径差。
- 影响：PASS 结论仍成立（5.34e-06 < 1e-5），但报告陈述的数值与其解释依托的产物不存在。
- 复现：§0.1 ②。

**#2 §7.1 "独立架构训练字符 95.8%" 口径混用。**
- 报告（`实验报告.md:433`）："共享架构欠拟合（训练字符 88.8%），而独立架构训练字符 **95.8%**"。
- 产物：shared `train.char_acc` 3 种子均值 88.84%（88.8% ✓ 用的是均值）；independent 3 种子为
  95.79/94.64/94.74 → 均值 **95.06%**；95.8% 只是 s42 单次值。项目自己的 `reports/tables/exp_E1_summary.csv`
  也记 `train_char_acc_mean`=0.950562。
- 复现：§0.1 ③。

**#3 §8.2 "真实图 101.3" 与产物及 §8.3 自相矛盾。**
- 报告（`实验报告.md:845`）："合成图原始像素均值 136.6，真实图 **101.3**"；同报告 §8.3 表（:857）写 108.2。
- 产物 `domain_gap.json::scale_real.raw_mean_0_255`=**108.2495**（test 前 600 张口径）。
  其它可复算子集也不是 101.3：train[:600]=113.1、hard[:600]=96.5。
- 复现：§0.1 ④。

### 中（5 条）

**#4 §7.5 batch_1 三区间小表 4 格与 `_history.csv` 不符。**
- 报告（:618-622）：第 1–5 轮 `71.5–81.2 / 18.8–21.2%`；第 10–20 轮 `52.6–61.4 / 18.2–21.4%`；第 25–35 轮 `43.6–53.7 / 19.7–21.3%`。
- 实测（`e4n_E4_batch_1_s42_history.csv`，与 run.json 内嵌 history 一致）：
  第 1–5 轮 **72.00**–81.23 / **19.73**–21.16%；第 10–20 轮 52.59–**61.32** / 18.19–21.36%；
  第 25–35 轮 43.56–53.70 / **13.22**–21.17%（ep30 附近深谷 13.2% 被抹平）。
- §1/§7.5 的全程口径 "43–81" ✓（实测 43.56–81.23）；但 §6.2 的 "45–80" ✗（见 #10）。
- 复现：§0.1 ⑤。

**#5 §10.5 仍称 "未做的实验：E2（ReLU）、E5（动量）…"，与 §7.2/§7.6 及产物冲突。**
- §7.2/§7.6 已给出 E2/E5 的 3 种子结果；产物中 `E2_relu_*`、`E2_sigmoid_*`、`E5_momentum_0_*`（各 3 种子，
  完整 80 轮口径）均存在。真正未做的是 E6/E8/E9 正式口径。报告内部自相矛盾。
- 复现：§0.1 ⑥。

**#6 E5 表与 RANKING 表落后产物一拍（竞态）。**
- `E5_momentum_09_s42_run.json` 写于 16:11:28，晚于报告快照 16:08:56：E5 表缺 momentum_09 臂
  （与 E5 标题 "(μ=0, lr=0.01) vs (μ=0.9, lr=0.05)" 的双臂结构不符），RANKING 46 行 → 应为 47 行。
- 处置建议：E5/E6 跑完后重新 `--write --inject` 再定稿；不算抄写错误。

**#7 §9 "批量评测的耗时（0.1–0.4 s/整个测试集）" 上限高估约 3 倍。**
- 12 份 `*_eval.json::datasets.{test,hard_test,synth_test}.eval_seconds` 实测区间 **0.06–0.13 s**。

**#8 §8 引言 "『总出某一个字符』的平凡基线约 5.6%" 无可复算口径。**
- 由 `splits.npz::synth_labels` 计算：每位置众数频率均值 **5.95%**；固定 train 众数预测 synth **5.32%**；
  每位置 test 众数 4.81%。没有任何常规定义给出 5.6%，产物中也无记录。（随机猜测 1/34=2.94% ✓。）

### 低（9 条）

**#9** §1.2 "测试字符 94.4% vs 86.8%，整牌 72.9% vs 43.4%"：均值为 94.30/72.55/86.82/43.43，
一位小数应为 **94.3 / 72.6** / 86.8 / 43.4；94.4 与 72.9 均无法由产物得到。
**#10** §6.2 "验证损失在 45–80 之间持续震荡"：实测 43.56–81.23，且 §1/§7.5 同一现象写 43–81 → 报告内部不一致。
**#11** §1.1/§6.2/§6.3 混用单种子 baseline_s42（86.7%/42.6%）与 §6 自定的 "以 3 种子 shared 为准"（86.82%/43.43%）；
§6.1 关系段的 "基线的实测整牌 **43.2%**" 任何现行口径都得不到（3 种子 43.43、s42 42.60、POSITION 表 43.43）。
**#12** §7.8 "`aug_strong` 训练字符仅 60.2%"：60.2 是 s42 单次值，3 种子均值为 **59.65%**（59.11–60.16）。
**#13** §8.3 "合成域准确率仍然只有 5.5%"：aug_none 3 种子均值 **5.68±0.18%**（E7 表口径），5.5% 是 s42 单次（5.52%）。
**#14** §8.1 "两域质心相差都在约 1 像素以内"：第 2 格实测差 **1.82 px**（32.82 vs 30.99），其余 5 格 ≤1.01。
**#15** §8.1 表 "标签与图像不一致" 行的证据路径写 `reports/logs/split_summary.json`，该文件无标签一致性字段；
实际数据在 `reports/logs/domain_gap.json::labels`（2000/2000/0）。
**#16** §7.2 "训练损失 ≈ 16.7（6 路边缘熵之和）"：平台值 16.60–16.70 ✓，但按 train 标签实测的 6 路边缘熵之和为
**15.84**，等式只是近似成立（差约 0.8，含 L2 与偏置未完全拟合边缘分布的因素）。
**#17** §7.2 "ReLU 版梯度校验 max_rel ≈ 1e-7"：`grad_check_shared_relu.json`=5.67e-07、
`grad_check_independent_relu.json`=**1.21e-06**，写 "≈1e-7" 偏小近一个量级（"≤1.3e-06" 才准）。

---

## 4. 无法核验的断言清单（10 条：产物中没有对应记录）

| # | 断言（位置） | 缺失证据 | 说明 |
|---|---|---|---|
| U1 | §5.3 前后端一致性：前向 5.2e-08、反向 2.2e-08、预测索引一致 | 无任何 parity JSON/日志 | 全仓 grep 无该两数（仅报告与备份） |
| U2 | §8.2 域迁移表：合成域(300 张) lr=0.5→33.3%（loss 27→51 发散）、lr=0.1/0.05→100%、真实域(300 张) lr=0.1→100% | 无脚本输出/JSON/日志 | `_job*.log` 中 "33.3" 均为 val_plate=33.30% 之类巧合 |
| U3 | §8.1 "标准化统计量不匹配：三种均为 ~5%" 与 "逐图归一化 4.78% vs 4.94%" | 无产物 | `train/diag_domain_gap.py` docstring 声称有第 4 项"标准化方案敏感性"，但 `main()` 只写 labels/centroids/scale，`domain_gap.json` 无任何准确率字段 |
| U4 | §7.5 "bs=1 在 9000 张上每轮 141.6 s（实测）" 及据此的 "3.1 小时/种子" | 无对应运行 | 产物只有 2000 张口径（26.17 s/轮实测）；26.17×4.5≈117.8 s 的线性外推与 141.6 不符。9000 张 bs=32（~4.7 s）/128（~4.5 s）两行同样无对应运行 |
| U5 | §8.4 修复前独立评测 "只有 3.10%" | 历史数字，产物已被覆盖 | 修复后断言可核验：12 份 `*_eval.json` 与同名 `*_run.json` 的 test/hard/synth `char_acc` 比对 **0 条差异**，"完全一致（86.73%/42.60%）" ✓ |
| U6 | §5.2 注释 "μ=0.9 极限环把损失卡在 ~2e-2" | 无 μ=0.9 版过拟合记录 | `overfit_check.json` 是 μ=0 口径 |
| U7 | §7.8 注释 "第一次误传 0–255，差异 86.02/255" | 无产物 | 作为轶事保留，无数值证据 |
| U8 | §2.1 "driverGetVersion 13030" | 无产物 | 旁证：本机 `nvidia-smi` 驱动 610.74、显存 8188 MiB，与报告一致；`*_run.json::gpu_memory.total_mb`=8187.5 ✓ |
| U9 | §9 "每次运行计时 50 张取中位数" | 部分 | `*_eval.json::cpu_inference` 只记 `n_samples`=50、`repeats`=20，未记 mean/median 口径 |
| U10 | §2.1 关键依赖之外的来源声明（hf-mirror.com、MIT 协议） | 无下载日志 | 仅 `data/raw_dl/ccpd_subset_30k.zip` 文件名佐证 |

> 说明：旧版审计列入"无法核验"的 §2.3 "`ccpd_base` 子集 14987 张 / 14442 个唯一号码"，本次已通过
> 直接统计 `data/ccpd/` 原始文件名**核验为真**（14987 个 `_ccpd_base_` 文件、14442 个唯一 6 位号码），
> 同时佐证 §2.2 "扫描总数 30000"（目录恰含 30000 个 jpg）。

---

## 5. 产物链与编辑性问题（不直接改动数值结论，但影响"每个数字都可复核"的承诺）

- **E-1** §7.4 MIDTRAIN 的"首次达到目标轮次"与"val_loss 绝对值"两小表在 `TABLE:MIDTRAIN:END` 之后
  **重复出现一次**（:533-545 与 :517-531 完全重复），是注入/手工合并残留。
- **E-2** §9 末段 "（约 5–8→系列）" 语句残缺（:999）。
- **E-3** §7.6 E5、§7.7 E6、§7.9 E8、§7.10 E9 的 "**结论**：" 均为空（E5 已有数据可写；E6/E8/E9 无数据）。
- **E-4** `reports/tables/report_tables.md`（15:18）与报告（16:08）不同步：其 BASELINE 表仍是旧的
  "baseline 1 种子" 行，且无 E2/E5 表；§11 称它为"本报告表格的自动生成版"，需重跑 `--write`。
- **E-5** `reports/tables/exp_E4_runs.csv` / `exp_E4_summary.csv` 缺 `batch_1`（只 4 变体 12 行），
  而报告 E4 表、§7.5、RANKING 都含 `e4n_E4_batch_1_s42`；`all_experiments_runs.csv` /
  `all_experiments_summary.csv` 只剩 1 行 verify_E8（2 轮 limit=900 的探针），§11 所称"总汇总（跨实验）"
  已名不副实；且全部 `exp_*_runs.csv` 的 `seed` 列为空、summary 的 `seeds` 为 `[0,0,0]`，
  与 §4 "日志/CSV/JSON 都带 seed" 的自设要求不符（run.json 的 seed 正常）。
- **E-6** `reports/logs/prepare_stats.json` 已被 96×24 预处理覆盖（`tag`=`ccpd_96x24`、`input_size`=[96,24]），
  不再对应 §2.2 的 128×32 流程；其 `filter_summary`（30000/28341/1659/480/1179/5.53%）数值仍与报告一致。
- **E-7** 全部 48 份 `*_run.json` 的 `meta.dirty` 均为 `true`（commit 分两批：eee708e / e4c15e1），
  而 §4 自设 "`dirty`=true 的结果不作为最终结论" —— 按报告自己的规则，所有结论都不满足其证据门槛，
  建议要么改规则表述，要么提交后重跑关键运行。
- **E-8** §7 引言 "全部实验都用 3 个随机种子 [42,43,44] 重复"：E4 `batch_1` 只有 1 个种子（表中已披露）、
  `e2lr` 探针 1 个种子（文中已标注非正式口径）——引言宜加"除已注明者外"。

---

## 6. 复核记录（本次审计实际执行的独立重算）

| 重算内容 | 数据源 | 结果 |
|---|---|---|
| 48 份 run.json 全指标聚合（15 个变体组） | `reports/logs/*_run.json` | 与 BASELINE/E1/E2/E3/E4/E5/E7 表逐值一致 |
| 等价配置同种子逐字节等价性 | baseline_/E1_shared_/E3_ce_/E7_weak_/E2_sigmoid_ × s42/43/44 | 每组 6 位小数完全一致 |
| POSITION 表（6 运行口径） | 6 份等价基线 run.json 的 `per_position_*` | 8 行 24 值全合（3 运行口径数值不同，确认现表为 6 运行口径） |
| E3 中期曲线与首次达标轮次 | run.json 内嵌 history × `_history.csv` | 15 值 + 4 轮次全合 |
| batch_1 逐轮区间 | `e4n_E4_batch_1_s42_history.csv` | 报告小表 4 格不符（#4） |
| e4n 吞吐 | `*_run.json::train_seconds/epochs_run` | bs1=26.17s ✓(≈26.5)；bs32=1.56–1.60s（报告 ~1.3 偏低）；bs64=1.18–1.23 ✓(~1.2)；bs128=1.02–1.04（报告 ~1.2 偏高）；9000 张 bs64=4.96–5.51（报告 ~4.5 偏低）；9000 张其余无运行（U4） |
| 每轮更新次数 2000/63/32/16/1 等 | `train/train.py:458`（`drop_last=False`） | 向上取整口径成立 ✓ |
| CPU 表 12 行 + eval↔run 一致性 | 12 份 `*_eval.json` | 全合；eval 与 run 指标 0 条差异 ✓（§8.4 "完全一致" 成立） |
| 梯度校验 4 份 | `grad_check_{shared,independent}{,_relu}.json` | shared ✓；independent 数值不符（#1）；relu 两份 passed ✓ 但报告 "≈1e-7" 不准（#17） |
| 过拟合自检 ×2 | `overfit_check{,_relu}.json` | sigmoid ✓（9.59376e-04/18000/100%）；relu `final_loss`=NaN、passed=false ✓（§7.2 "发散到 NaN" 成立） |
| E2 死亡 ReLU 解剖（重跑 `_diag_relu.py`） | `E2_relu_s42_best.npz` + train[:512] | 实测 0.0%→6.2% 死亡单元、z1 均值 +0.04→−110.70、h==0 49.0%→99.5%，与 §7.2 表逐值一致 ✓ |
| E2 平台 ≈ 边缘众数 | test/train 标签逐位众数 | test 22.68% / train 22.28% ≈ relu 实测 22.07% ✓；6 路边缘熵之和=15.84（≠16.7，见 #16） |
| e2lr 探针 | `e2lr_relu_lr001_s42_run.json`（lr=0.01/μ=0.9/relu） | 78.12% / 21.05% ✓（§7.2） |
| 划分与泄漏 | `splits.npz` + `ccpd_128x32.npz::labels` | 9000/2000/2000/2000/2000；唯一 8684/1921/1924/1994；交集全 0 ✓ |
| ccpd_base 规模 | `data/ccpd/` 文件名解析 | 14987 张 / 14442 唯一 ✓；全目录恰 30000 jpg ✓ |
| 缓存与清单 | `ccpd_128x32.npz`、`manifest.csv` | 28341×32×128 uint8、93.1 MB ✓；manifest 17001 行=表头+17000 数据行 ✓ |
| 标准化统计量 | `splits.npz::standardizer`、`split_summary.json` | mean 0.421204 / std 0.221177 / n=9000 ✓ |
| 域差异产物 | `domain_gap.json` | §8.1/§8.3 全合；§8.2 的 101.3 不符（#3） |
| 增强诊断 | `augment_diag.json` | 24 值全合（比值 3.00–5.30） |
| 环境 | `pip` 实测 + `nvidia-smi` + `gpu_memory` | Python 3.12.14、numpy 2.5.3、Pillow 12.3.0、matplotlib 3.11.2、pandas 3.0.6、PyYAML 6.0.3、cupy 14.2.0、驱动 610.74、8188 MiB 全合（driverGetVersion 13030 无产物，U8） |
| 参数量公式 | 4096×256+256+6×(256×34+34)、独立 ×6 | 1101260 / 6345420 ✓（比值 5.76≈"5.8 倍" ✓） |
| 平凡基线口径 | `splits.npz::synth_labels` | 5.95%/5.32%/4.81%，无 5.6%（#8） |
| §8.5 差值 | E7_aug_none_s42 run.json | 98.28/91.35/72.55/32.05/5.52/0.00 ✓；−25.7/−59.3/−92.8/−91.4 ✓ |
| §6.2/§7.8 细节 | run.json | "早停第 51 轮" ✓（best=43+patience8）；"0.76 vs 3.58" ✓（0.7649/3.5775）；aug_none s42 val 98.27/test 98.28/train 100 ✓ |

**审计边界**：未修改 `reports/实验报告.md` 或任何被测产物；本文件为唯一交付物（覆盖旧版）。
审计脚本在 `%TEMP%\px_audit.py`（工作区外）。若报告在 16:08:56 之后再次变动，本审计需以新快照重核。

---

## 附录 A：合成域后端切换后的复核（v3）

**背景**：报告口径的合成域测试集已从内置 PIL 渲染器（`pil_renderer`）切换为
外部生成器 `Nenger/chinese_licence_plate_generator`@`43bac43` 的牌面级输出
（`generator_repo`）。切换**只替换划分文件里的 `synth_images/synth_labels/
synth_texts`**，`train/val/test/hard` 索引与标准化统计量逐字节不变
（`train/swap_synth_to_generator.py` 内置断言）；合成域从不参与训练或调参，
故按"只重测、不重训"处理：`reports/refresh_synth_eval.py` 用各运行的**现有
最佳检查点**重测并刷新 `*_run.json::synth_test` + 全部 `*_eval.json`。

**已完成的重测覆盖**：75 份 `*_run.json` 的 `synth_test` 字段全部刷新；
48 份 `*_eval.json` 全量重评（新增字段 `synth_backend="generator_repo@43bac43"`）。

| 复核内容 | 数据源 | 结果 |
|---|---|---|
| 非合成域数值是否被重测改动 | 5 份 `*_eval.json` vs `git show HEAD:<同名>` | **逐位一致**（<1e-12）：train/val/test/hard 的 char_acc 全部相同 → 重测链路无副作用 |
| 合成域数值变化 | 同上 | 变化只出现在 `synth_test`（如 `baseline_s42` 5.62% → 41.62%） |
| 两分辨率合成集是否同标签 | `splits.npz` / `splits_24x96.npz` | 标签序列完全相同（脚本内断言） |
| 生成器确定性（同种子重跑） | `generate_dataset_from_repo` 2000 张 vs `splits.npz::synth_*` | 文本/标签/图像**逐位一致**，拒绝数 912 相同 → 可复现 |
| 标签合法性 | `splits.npz::synth_labels` + `check_label_legal` | 2000/2000 合法；共拒绝 I/O 标签 912 张 |
| 标签-图像一致性 | `domain_gap.json::labels` | 2000/2000 一致 |
| 六格质心对齐 | `domain_gap.json` | 最大偏差 1.94 px（pos2）；合成域 σ≈1.1–1.3 px |
| 标准化敏感性 | `domain_gap.json` | global_train 41.62% / global_synth 42.85% / per_image 43.38% |
| 平凡基线 | `splits.npz::synth_labels` + train 众数 | 逐位众数 4.05%；train 众数迁移 3.12%；随机 2.94% |
| 域差（E7_aug_none_s42） | run.json / eval.json | 98.28/91.35（test）、72.55/32.05（hard）、43.67/0.80（synth） |
| 工程一致性 | `exp_E*_summary.csv` / `report_tables.md` / 报告注入表 | 三者同源（`reports/rebuild_summary_tables.py` + `build_report_tables.py`）复核一致 |

**本附录取代的旧条目**（上一版审计中涉及合成域的数字，均以本文档为准）：

* §8.1 的"合成域更稳定 / σ≈0.3–0.7 px"、标准化敏感性 5.62/5.45/5.79%；
* §8.2 的可学性对照表（本轮**未重跑**，按"只重测不重训"约束，报告中已改为
  仅保留不依赖训练的自洽性证据）；
* §8.3 的两域统计（原始像素均值 136.6 → **82.1**，标准化后 +0.518σ → **−0.449σ**，
  std 1.176 → 1.034）；
* §8.5 差值 −92.8/−91.4 pt → **−54.6/−90.6 pt**（s42 口径）；
* §7.1/§7.7 中"合成域两架构都 5–6%、L2 无效"等基于旧合成集的结论；
* 审计 `#3`（101.3 vs 108.2 的像素均值口径不一致）在重写后已统一为
  `domain_gap.json` 的实测值（真实 108.2 / 合成 82.1）。

**未做的检查**（如实声明）：合成图的人工视觉核对仍待人工完成
（`reports/figs/synth_check_grid.png`）；本模型无法读图，故只做了数值几何核对
（墨迹质心/簇数）与标签核对。此外重测发生在工作区含产物变动的状态下，
刷新后的 `*_eval.json::git_dirty` 记为 `true`（与历史产物同性质，见报告 §4 说明）。
