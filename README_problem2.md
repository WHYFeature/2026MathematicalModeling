# 问题二：局部模态缺失下的鲁棒多模态预测

问题二脚本读取附件二 `aligned_50.pkl` 的 train/valid/test 划分，训练一个三模态时序融合模型，同时完成分类和情感强度回归。附件三不参与训练、验证、阈值选择或伪标签，只在训练完成后进行推理。

模型特点：

- 每个时间 bin 都有 text/audio/vision 的局部有效掩码；
- 默认使用正则化的 masked mean/std summary fusion，并保留每种模态的有效比例；
- 训练时随机遮挡连续局部时间段，模拟附件三的缺失模式；
- 不做整模态 dropout；
- 使用训练集统计量归一化，避免验证集和专项测试集信息泄漏；
- 输出三分类、回归强度、分类概率以及附件三的局部有效率。

运行环境沿用问题一的虚拟环境。首次运行前确认 `matplotlib` 无关，问题二只需要已经安装的 NumPy、PyTorch、TorchAudio 和 Transformers：

```powershell
Set-Location D:\E_math
$Py = "D:\E_math\.venv_problem1\Scripts\python.exe"
& $Py D:\E_math\problem2_train.py --architecture summary --hidden 256 --epochs 40 --batch-size 64 --local-dropout 0.10 --reg-weight 0.15
```

默认使用 RTX 4080 的 CUDA。训练完成后，结果位于 `D:\E_math\problem2_outputs`：

```text
problem2_checkpoint.pt
normalization.json
problem2_metrics.json
problem2_test_predictions.csv
problem2_missing_aligned_predictions.csv
problem2_missing_unaligned_predictions.csv
```

正式报告中应使用 `problem2_metrics.json` 的 test 指标，并报告 Negative/Neutral/Positive 的逐类 F1、macro-F1、Accuracy、回归 MAE、RMSE 和 Pearson 相关系数。附件三的预测文件没有真实标签，只能作为专项约束预测结果。

## 当前实验结论

在不使用测试标签、不混用划分、训练集与测试集 ID 完全不重叠的条件下，原始 gated 模型约为 65%，本地 BERT 与音频/视觉摘要联合微调模型达到约 71.8% 测试准确率，Macro-F1 约 0.671。BERT 单独微调约为 66.4%。验证集在 2--5 轮达到峰值，继续增加 epoch 会过拟合，因此不能靠把训练轮数提高到 35 或 100 得到 90%--95%。

联合微调对照命令：

```powershell
$Py = "D:\E_math\.venv_problem1\Scripts\python.exe"
& $Py D:\E_math\problem2_bert_fusion_train.py `
  --data-root D:\E_math\DATA `
  --text-model D:\E_math\models\bert-base-uncased `
  --output D:\E_math\problem2_bert_fusion_outputs `
  --epochs 5 --batch-size 20 --lr 0.000015
```

上述准确率是已有实验结果，不能保证新的运行重复达到相同数值；模型选择使用验证集，测试集用于最终评价。

## BERT 音视频融合：补导出已有模型的结果

已训练的目录中有 `bert_av_fusion.pt` 时，用 `--export-only` 跳过训练：

```powershell
Set-Location 'D:\E_math'
$Py = 'D:\E_math\.venv_problem1\Scripts\python.exe'
& $Py 'D:\E_math\problem2_bert_fusion_train.py' `
  --data-root 'D:\E_math\DATA' `
  --text-model 'D:\E_math\models\bert-base-uncased' `
  --output 'D:\E_math\problem2_outputs_bert_av_reproduce' `
  --batch-size 20 `
  --export-only
```

`--output` 必须指向已有模型所在目录。模型权重不会重新训练或覆盖。运行后该目录包含：

| 文件 | 内容 |
| --- | --- |
| `bert_av_fusion.pt` | 原有模型权重 |
| `metrics.json` | 验证/测试指标、混淆矩阵、附件三预测分布和导出说明 |
| `normalization.json` | 仅由训练集拟合的归一化统计量 |
| `problem2_valid_predictions.csv` | 附件二验证集 728 条，含真实标签 |
| `problem2_test_predictions.csv` | 附件二测试集 727 条，含真实标签 |
| `problem2_missing_aligned_predictions.csv` | 附件三 aligned 30 条，无真实标签 |
| `problem2_missing_unaligned_predictions.csv` | 附件三 unaligned 30 条，无真实标签 |

CSV 包含样本编号、预测类别/名称、情感强度和三类概率；类别顺序为 Negative、Neutral、Positive。

旧模型没有保存归一化参数时，脚本从原附件二的训练集恢复相同统计量，因此应保留原始数据。旧脚本没有记录的最佳轮次和训练历史会标记为未知，无法仅凭权重恢复。新版训练会自动导出这些文件，并在每轮结束后保存训练集与验证集指标、最佳轮次和训练配置。

附件三 aligned 的文本掩码保持原值；unaligned 文本使用本地分词器截取至 50 个位置，音视频将有效行平均到 50 个区间后归一化。两种格式分别输出，各 30 条，不能当作 60 条独立带标签测试样本。附件三没有真实标签，CSV 是模型预测；导出功能不等于完成了缺失率/缺失类型消融实验，也不证明模型的鲁棒性。

## 已确定模型的阶段分析与可视化

当前采用 `problem2_outputs_bert_av_reproduce` 中的 BERT 音视频融合模型，测试 Accuracy 为 71.80%，Macro-F1 为 0.6708。生成分析报告时仅读取已导出的四份预测 CSV 和 `metrics.json`，不重新训练：

```powershell
Set-Location 'D:\E_math'
.\run_problem2_report.ps1
```

输出位于 `D:\E_math\problem2_outputs_bert_av_reproduce\stage_report`：

- `stage_report.html`：可直接用浏览器打开的中文分析报告，六组图片已内嵌，单独复制此文件即可阅读。
- `stage_report.md`：可编辑的文字分析及指标表格，引用同目录下的 `figures`。
- `figures`：整体概览、混淆矩阵、逐类与误判分析、回归诊断、强度与置信度分析、附件三预测对比，共六组 300 dpi PNG 和对应矢量 PDF。有逐轮历史时，额外生成第七组训练曲线。所有图片使用英文标注和 Times New Roman，不含总标题及长段说明，解释保留在报告正文中。
- `errors_valid.csv`、`errors_test.csv`：按置信度排序的全部分类错误记录，便于回查原始样本。
- `analysis.json`：重算的指标、诊断统计、输入文件 SHA-256 和一致性检查结果。

脚本核对 CSV 与原指标，检查样本 ID 和预测概率；缺失训练历史时不生成训练曲线。附件三的两种格式一致率仅表示预测一致程度，不是准确率，也不能替代模态缺失实验。

报告依赖 NumPy 与 Matplotlib；如缺少依赖，安装命令为：

```powershell
& 'D:\E_math\.venv_problem1\Scripts\python.exe' -m pip install 'numpy>=1.26' 'matplotlib>=3.8'
```

可通过 `-InputRoot '其他结果目录'`、`-OutputRoot '报告保存目录'`、`-Dpi 600` 修改输入、输出或图片分辨率。输入仍需符合该 BERT 融合模型的导出格式。

## 给已确定的 reproduce 模型补充经校验的复现曲线

`problem2_outputs_bert_av_reproduce` 对应最初 5 轮、batch size 20、学习率 1.5e-5、seed 42、不使用类别加权的实验。原始逐轮日志没有保存，但目标权重与原实验留存权重的 SHA-256 相同。因此可按原配置重放，并以逐张量完全一致为门槛确认是否复现到同一个模型。

```powershell
Set-Location 'D:\E_math'
.\run_problem2_recover_history.ps1 -Epochs 30
```

脚本默认从第 1 轮训练到第 30 轮，每轮保存记录并与参考权重逐项比较。原实验的选模范围仍固定为前 5 轮，要求该范围内按验证 Macro-F1 选中的权重与参考权重完全一致，以及归一化统计量一致；第 6—30 轮为延长训练分析，后续的全程最佳轮次单独记录，不替换 71.80% 的参考模型。通过后，原目录增加：

```text
problem2_outputs_bert_av_reproduce/
  history_recovery/training_history.json
  history_recovery/training_history.csv
  history_recovery/replay_*/checkpoint_checks.json
  history_recovery/replay_*/verification.json
  training_curves/training_curves.png
  training_curves/training_curves.pdf
  stage_report/figures/07_training_curves.png
  stage_report/figures/07_training_curves.pdf
```

原 `bert_av_fusion.pt`、`metrics.json`、归一化文件和四份预测 CSV 都不覆盖；执行前后核验哈希。这里新增的是“经权重校验的复现及延长训练实测记录”，不是声称找回了当时的原始日志。图中用英文图例标记参考模型轮次，浅色背景表示延长训练，记录来源在 JSON 和中文报告中说明。

若已有 5 轮记录，再执行 `-Epochs 30` 会从第 1 轮重新跑满 30 轮，因为旧记录没有保存用于精确续训的优化器及随机数状态。旧记录备份为本次 `replay_*` 下的 `previous_verified_history.json`，并要求新记录的前 5 轮与它一致，校验通过后才更新正式曲线。已有不少于所需轮数的记录时只重新绘图。整个过程由用户在终端运行。

若输出 `unverified`，复现未满足完全一致门槛，尝试日志留在 `history_recovery/replay_*`，不会给原模型挂接曲线。不能仅凭验证指标接近或准确率相同判定权重一致，也不能直接使用另一份 20 轮实验的曲线。

## 新的独立训练实验：记录训练曲线

已采用的 71.80% 模型没有逐轮历史，无法从权重或最终预测恢复。以下命令在新目录开始一次独立训练，保留旧模型与结果；新实验的指标可能不同，不把它的曲线冒充旧实验历史：

```powershell
Set-Location 'D:\E_math'
$Py = 'D:\E_math\.venv_problem1\Scripts\python.exe'
& $Py '.\problem2_bert_fusion_train.py' `
  --data-root 'D:\E_math\DATA' `
  --text-model 'D:\E_math\models\bert-base-uncased' `
  --output 'D:\E_math\problem2_outputs_bert_av_history' `
  --epochs 20 `
  --batch-size 20 `
  --lr 0.000015 `
  --seed 42
```

每轮结束后对完整训练集和验证集使用固定权重、eval 模式重新评价，保存 Loss、Accuracy、Macro-F1、MAE 等指标，关闭 Dropout。联合损失仍为交叉熵加 0.15 倍 SmoothL1，没有改变优化目标、模型结构或权重选择规则。每轮会增加一次训练集前向评价，用于可比的训练/验证曲线。测试集不参与逐轮评价或选择。

输出目录包含 `training_history.json`、`training_history.csv`，每轮更新，包括没有改善的轮次。训练完成会自动生成 `training_curves/training_curves.png` 和 `.pdf`，虚线标记按验证 Macro-F1 选择的最佳轮次。若训练中断，已经完成的轮次可独立绘制：

```powershell
& $Py '.\problem2_training_curves.py' --input-root 'D:\E_math\problem2_outputs_bert_av_history'
```

在完整导出后生成带训练曲线的报告：

```powershell
.\run_problem2_report.ps1 -InputRoot 'D:\E_math\problem2_outputs_bert_av_history'
```

旧日志若只记录验证指标，只绘制实际记录的验证曲线；缺失的训练指标或损失不会补造。训练命令会拒绝覆盖已有权重或历史，请为独立实验使用新的输出目录。
