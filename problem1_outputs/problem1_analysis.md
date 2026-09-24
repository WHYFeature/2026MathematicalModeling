# 问题一实验结果分析

- 样本数：100；输出格式：`problem1-v2`。
- 三模态维度：text=768，audio=74，vision=35。
- 视频时长：2.260s–29.290s，中位数 6.725s。
- 标签分布：Negative 18，Neutral 25，Positive 57。

## 时间对齐质量

文本使用 CTC 强制对齐后再进行 50 段等时长区间的重叠加权池化；音频和视觉特征使用各自源区间直接池化。

- Text：平均覆盖率 0.5212，有效 bin 比例 0.7340，每 bin 源区间中位数 1.0。
- Audio：平均覆盖率 0.9827，有效 bin 比例 0.9932，每 bin 源区间中位数 16.0。
- Vision：平均覆盖率 1.0000，有效 bin 比例 1.0000，每 bin 源区间中位数 2.0。

## 文件

- `figures/problem1_overview.png`：数据规模、标签和时间覆盖概览。
- `figures/problem1_alignment_example.png`：中位时长样本的逐 bin 对齐诊断。
- `figures/problem1_feature_statistics.png`：三模态特征范数和源区间数量。
- `figures/problem1_label_duration.png`：情感强度标签与视频时长关系。
- `problem1_analysis.json`：可复用的统计数值。
