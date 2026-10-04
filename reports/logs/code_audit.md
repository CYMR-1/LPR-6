# ProjectX 训练/评测代码路径静默缺陷审计报告

- 审计人：code-auditor-k3（task-3）
- 审计时间：2026-10-04
- 审计基线：工作区脏树（HEAD=e4c15e1 + 未提交改动）。审计窗口内 `train/phase15_split.py`、`train/train.py`、`evaluate/main.py`、`configs/default.yaml` 被并行加入了 `paths.splits_file` 参数化（E8 24×96 口径用）；所有实测均在不设置该键的配置上完成，默认行为与审计对象一致，结论不受影响。`train/grad_check.py` 的 `--activation` 参数在本次审计的版本中存在且实测可用。
- 方法：静态阅读 + **全部实测**（临时脚本 `reports/_audit_tmp/t1…t6*.py`，输出同目录 `t*_out.txt`；全部 numpy 后端、小样本短跑，未触碰 GPU）。未修改任何源文件。

## 逐风险点结论表

| # | 风险点 | 代码位置 | 判定 | 触发条件 | 实测证据 |
|---|--------|----------|------|----------|----------|
| 1a | 六头与共享层梯度错配（顺序/维度/六路累加） | `models/model.py` `backward_shared` / `backward_independent` | **无缺陷** | — | 基线 34×6 数值梯度检查 CLI 全过：shared 42 分量 max_rel=5.3e-7，independent 72 分量 max_rel=5.5e-6（`t7` 基线运行，exit=0） |
| 1b | E9 head_mask 是否真生效、194 节点结构 | `train/train.py:387-424`、`models/model.py:927-966` | **结构正确；掩码链路存在真缺陷（见 1c）** | — | E9 短训练（limit=300, epochs=2, numpy）实跑：`output_nodes=194`、`num_parameters=1,098,690`（与 34×6 的 1,101,260 精确差 2570=256×10+10）、`W2_0=(256,24)`、ckpt 内 `head_dims=[24,34,…]`；数据侧：缓存全量首位标签 max=22，train/val/test/hard/synth 首位 ≥24 样本数均为 **0**（CCPD 解析强制首位为字母，合成器首位仅从 LETTERS 采样）→ `head_mask` 在真实数据上恒为全 1 |
| 1c | **head_mask 只进损失、不进反向** | `models/model.py`：`compute_loss`（L1022-1024 应用 mask）vs `backward_shared`（L753-767 无 mask 参数）；`train/train.py:460-466`（mask 只传 compute_loss）；`train/overfit_check.py:198-211`（同样模式） | **已确认真缺陷（代码层）；训练侧当前不触发，校验工具侧必然触发** | 任一位置标签 ≥ 该头维度（E9 首位 ≥24）时出现：损失侧贡献为 0，反向侧 δ=(y−0)/B=y/B≠0 | ① E9 配置下跑 `train/grad_check.py`：**shared 12/42 FAIL（max_rel=4.9e-1）、independent 12/72 FAIL（max_rel=1.0）**，失败精确集中在 W2_0/b2_0/W1/b1（独立结构为 W1i_0/b1i_0），其余头全 OK（~1e-7）；② 根因隔离：同配置把随机标签首位限制到 0..23 后**全过**（max_rel=2.0e-7）；③ 损失侧加 mask、反向不加 → 仍 FAIL（与现状一致）；④ `--activation relu` 同样 FAIL（与激活无关）；⑤ E9 真实短训练不受影响（数据无越界样本，见 1b） |
| 2 | 越界标签是否静默 | 入口：`models/ccpd_parse.py:240-261`、`models/charset.py:165-235`、`models/dataset.py:137-157`；链路内：`models/model.py:959-965` | **入口全部响亮报错；缓存→训练链路内静默（真缺陷，当前管线保证不触发）** | 越界标签绕过解析/过滤直接进入缓存 labels 数组 | 哨兵/越界 ads 索引→ValueError；非法字符→ValueError；decode 索引 34→KeyError；未拟合 transform→RuntimeError。但 `build_onehot` 对含 30/34 的标签**静默清零 one-hot**（`valid=labels<c` 掩码），`compute_loss`+`backward` 全程无异常，且该批梯度与合法批最大差 5.0e-1（静默改变优化方向） |
| 3 | 标准化统计量只来自训练集 | `train/phase15_split.py:398-401`、`models/dataset.py:88-157`、`train/train.py:329` | **无缺陷** | — | splits.npz 内 standardizer 与"仅用 train 索引重算"逐位一致（mean/std 差 <1e-8），n=9000=len(train)；val/test/hard 经 train 统计量标准化后均值分别为 −0.0105/−0.0325/−0.2056（≠0，证明**未**各自重新拟合）；全仓库对真实数据的 `GlobalStandardizer.fit` 仅 phase15_split 一处 |
| 4a | 早停监控 val_loss、保存最佳、评测用回滚权重 | `train/train.py:488-507, 509-542, 607-655`；`evaluate/main.py:128` | **无缺陷（默认 monitor=val_loss）** | — | lr=2.0 震荡训练实测：best_epoch=1，末轮 val_loss=126.78≠最佳 96.56；run.json 的 val/test 指标与用落盘 best ckpt 复评一致（\|Δ\|≤3.3e-7）→ 最终评测确实用回滚后的最佳权重；`Params.load` 返回值被正确接收（历史缺陷 a 已修复） |
| 4b | monitor 方向硬编码 | `train/train.py:501`（`score < best_score - min_delta`）、`train/train.py:608` | **已确认真缺陷（需改配置触发）** | `train.monitor` 改为"越大越好"指标（val_char_acc / val_plate_acc） | monitor=val_char_acc 实测 3 轮 acc 0.2225/0.2328/0.2372 单调上升，`is_best` 却标在第 1 轮（**最低**），best_epoch=1 并回滚到最差权重。默认配置不触发，但没有任何报错提示方向接反 |
| 5 | 预算截断如实记录 | `train/train.py:574-581, 638, 850-853`；`reports/run_all.py:341-348`；`reports/build_report_tables.py:205-207` | **无缺陷（如实记录）；附 2 个边界提示** | — | budget=2.0s 实测：第 5 轮前停止，run.json `budget_exhausted=true, epochs_run=4, planned_epochs=10`；聚合 CSV 有 `n_budget_exhausted`，报告表格有"（含预算截断）"。真实 48 个 run.json：0 个 budget_exhausted，epochs_run<80 均如实记录。边界提示：① budget≈0（0 轮）时无 ckpt 产出（`paths.checkpoint` 悬空），val/test 指标来自随机初始权重（val_char=2.1%），`best_score` 序列化为 `Infinity`（非严格 JSON）——flag 齐全不算静默，但下游若误读指标会看错；② run_all 的**逐次** runs CSV 无 budget_exhausted 列（聚合级才有）；③ 旧产物 baseline_s42 的 planned_epochs=None |
| 6 | 数据泄漏（号码跨集 / hard / synth 参训） | `train/phase15_split.py:104-203, 371-384`；`train/train.py:340-352` | **无缺陷** | — | 产物实测：train/val/test/hard 行级索引两两交集=0；6 位号码级两两交集=0；synth∩base=0、synth∩hard=0；synth_labels 解码与 synth_texts 100% 一致；manifest.csv 17000 行与 split 计数一致；hard/synth 从不参与训练（train 仅引用 train 索引；synth 为独立数组） |
| 7 | 随机性覆盖与同 seed 可复现 | `models/config.py:468-513`；`models/model.py:500`；`models/dataset.py:503, 544` | **无缺陷（numpy 后端）** | — | 同 seed 两次短训（limit=200, epochs=2，含 weak 增强）：val_loss 10 位小数一致（17.1971770000）、val_char 一致、ckpt 全部 19 个权重/结构数组**逐位一致**（文件 md5 差异仅来自 extra 里的 run_name/started_at）；异 seed 结果确实不同；无增强档位同样可复现。机制：初始化 `default_rng(seed)`、批顺序 `default_rng(seed+epoch)`、增强 `default_rng(seed+idx.sum())`，均不依赖全局状态。cupy 后端未在本次实测（GPU 被占用） |

## 确认的真缺陷（按严重度排序）

1. **损失/反向在 head_mask 上不一致，且使 E9 配置下的梯度校验工具必然 FAIL。**
   `compute_loss` 支持 `head_mask`（E9 首位越界样本损失置 0），`backward_shared`/`backward_independent` 根本没有 mask 参数，`Trainer` 与 `overfit_check` 都只把 mask 传给损失侧。后果分两层：
   - **校验工具层（当前即触发）**：E9 头维度（[24,34,…]，194 节点）下 `train/grad_check.py` 用 0..33 随机标签，首位 ≥24 必然出现（B=8 时概率 ≈94%），shared/independent、sigmoid/relu 全部 FAIL，失败集中在头 0 相关参数——P3 验收手段对 E9 失效，且 FAIL 信息无法区分"实现错"与"mask 不一致"。
   - **训练层（当前不触发，属潜伏）**：真实数据首位恒为字母（实测 5 个集合 0 个越界样本），mask 恒全 1，梯度正确；但若未来数据源允许首位为数字，训练将静默地对这些样本朝"压低全部头 0 logits"的方向优化，无任何报错。
2. **monitor 指标方向硬编码"越小越好"。** 把 `train.monitor` 改成 `val_char_acc`/`val_plate_acc` 后，best 选择、早停、权重回滚全部反向（实测确认把 acc 最低轮当最佳）。默认 val_loss 不触发，但属于一改配置就静默产出错误结论的典型。
3. **缓存标签越界在训练链路内全程静默。** `build_onehot` 用 `labels[:, i] < c` 掩码把越界标签的 one-hot 静默清零，不报错；该样本的 δ=(y−0)/B 仍参与梯度（与缺陷 1 同根因）。当前解析/过滤入口（ValueError/KeyError/RuntimeError 俱全）保证不触发，属纵深防护缺口。

## 理论风险（已实测/核实，实际不会触发）

- **E9 训练梯度被越界样本污染**：真实 CCPD+合成数据首位标签全部 ≤23（实测 max=22/23），mask 恒全 1，不触发。
- **`_copy_params` 静默跳过名字/形状不匹配项**（`train/train.py:670-672`）：ckpt 为同一次运行自存自取，arch/形状恒匹配，不触发。
- **0 轮预算截断产物**：`budget_exhausted=true`、`epochs_run=0` 如实写入；随机权重指标 + `Infinity` best_score + 悬空 ckpt 路径属边界健壮性问题，flag 完整、不算静默错误。
- **E3 mse + Softmax 的对角雅可比近似**：反向注释明确声明（`models/model.py:709-713`），是 E3 要观察的设计；实测 mse 损失下数值梯度检查必然全 FAIL（max_rel=1.0）——**意味着 grad_check 不能用于校验 mse 配置**，属工具适用边界而非隐藏错误。
- **cupy 后端可复现性**：本次未实测（GPU 被其它实验占用）；numpy 后端已确认逐位可复现。

## 附：E9 高价值检查的直接回答（lead 提问）

- 194 输出节点：确认。E9 短训练 meta：`head_dims=[24,34,34,34,34,34]`、`output_nodes=194`、`num_parameters=1,098,690`，与 34×6 基线（204 节点 / 1,101,260 参数）不同且差值精确符合公式。
- 首位标签上界：确认只用到 0..23（全部 5 个集合 0 个 ≥24 样本）。
- `train/grad_check.py --activation`：存在且工作正常（正确接收 `apply_patch` 返回值）。
- **194 节点结构下数值梯度校验跑不通**：shared 与 independent 均 FAIL（exit=1），根因是 `backward()` 不应用 head_mask 而 `compute_loss` 应用，对随机标签中首位 ≥24 的样本产生系统性梯度差；将首位标签限制到 0..23 后同配置全过（max_rel≈2e-7），证明除该 mask 缺口外反向实现本身正确。

## 审计产物

临时脚本与输出位于 `reports/_audit_tmp/`（`t1…t6*.py`、`t*_out.txt`、`t_cfg_*.yaml`、`ckpt/`、`logs/`），按要求将在提交本报告后删除。
