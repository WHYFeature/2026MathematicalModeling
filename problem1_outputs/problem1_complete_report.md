# 问题一补齐材料与核验报告

本报告由 `problem1_report.py` 从已生成的 100 条特征文件读取并生成。它不把覆盖率当作时间戳准确率，也不把未覆盖文本 bin 自动解释成静音。

## 已交付文件

- `problem1_feature_summary.csv`：100 条样本 × 3 个模态的 300 行逐模态汇总表。
- `problem1_feature_schema.json`：三类特征的维度、时间轴、计算定义和限制。
- `problem1_environment.json`：报告生成时的 Python/包/FFmpeg 版本、数据哈希和历史运行参数。
- `problem1_validation.json`：样本 ID、形状、有限值、时间边界、掩码和覆盖率检查。
- `problem1_text_truncation.csv/json`：同一 BERT tokenizer 对固定 50-token 接口的截断审计。
- `typical_sample_words.csv`、`typical_sample_bins.csv`：典型样本的词级时间段与 50 个对齐 bin 的可复核明细。
- `figures/problem1_typical_sample.png` 和 `.pdf`：真实波形、CTC 词段、三模态特征曲线、覆盖率和真实视频帧。

## 数据规模与测量结果

- 样本数：100；特征维度：text=768，audio=74，vision=35。
- 视频时长：2.260–29.290 s；均值 7.875 s；中位数 6.725 s。
- 标签：Negative 18、Neutral 25、Positive 57。
- 每个模态都以 50 个等时长 bin 输出；有效 bin 和覆盖率分别保存在 `valid_masks` 与 `coverage`。
- 文本 50-token 审计：6/100 条样本超过上限，累计省略 70 个 token。
- text：样本平均 bin 覆盖率 0.5212；有效 bin 比例 0.7340。
- audio：样本平均 bin 覆盖率 0.9827；有效 bin 比例 0.9932。
- vision：样本平均 bin 覆盖率 1.0000；有效 bin 比例 1.0000。

## 数学处理说明

令样本时长为 T，等时长边界为 e_k=kT/50。源特征 x_i 在时间区间 [a_i,b_i) 上定义，和第 k 个 bin 的重叠长度为 w_{ik}=max(0,min(b_i,e_{k+1})-max(a_i,e_k))。输出为 y_k=Σ_i w_{ik}x_i/Σ_i w_{ik}；coverage_k 是这些区间并集长度除以 bin 长度，避免重叠音频窗把覆盖率重复计数。Σ_i w_{ik}=0 时输出零向量且 valid_mask_k=false。
- 文本：本地 BERT 的 768 维 hidden state；CTC 强制对齐产生词级区间，再将词元映射到这些区间。CTC 置信度是声学后验诊断量，不是人工标注的时间误差。
- 音频：16 kHz 单声道；25 ms 窗、10 ms hop；10 个标量谱/波形描述、32 个线性谱带均值和 32 个差分，共 74 维。
- 视觉：5 fps、64×64 RGB；RGB/灰度统计、百分位、横纵差分和 16-bin 灰度直方图，共 35 维。视觉特征是全帧外观描述，不等同于面部关键点或面部情绪识别器。

## 复现命令

```powershell
Set-Location D:\E_math
.\run_problem1_report.ps1
```

如需重做完整特征提取，先运行 `.\run_problem1.ps1 -Overwrite`，再运行上述报告命令。正式提取必须使用本地 BERT 和默认 CTC；hash fallback 与 uniform alignment 只能用于接口检查。

## 当前结果的边界

- 报告中的覆盖率表示源区间在目标 bin 中的时间占用，不等于 CTC 时间戳已经通过人工标注验证。
- 旧 PKL 如果没有 `source_intervals` 字段，汇总表会明确标空源区间的起止时间，并使用已有 `source_indices` 做可追溯计数；修改后的提取器会把文本、音频和视觉源区间直接写入新 PKL。
- 视频帧时间来自 FFmpeg 的 5 fps 重采样时间轴；原始视频若没有逐帧 PTS 导出，报告不会把它写成更精确的原始帧时间。
- 50 个文本位置是题目对齐文件的固定接口，因此不改变输出形状；若有截断样本，明细中列出省略 token 数，不能把截断后的表示写成完整转写表示。
- 当前 PKL 只代表问题一特征结果，不包含原始视频、模型权重或虚拟环境；最终附件是否小于 50 MB 仍需按提交清单整体核算。
