
## 图21 新增鲁棒性实验的覆盖范围

![图21 新增鲁棒性实验的覆盖范围](figures/21_experiment_coverage.svg)

**数据来源与统计口径：**condition_metrics.csv，1044行；3模型×3种子×2划分×58条件。

每个模型与种子都有58个条件：完整输入1个、缺失率28个、位置12个、时长9个、全模态消融8个。验证和测试分别使用728、727条带标签样本。图中每格116行是条件数乘以两个划分，不是116次独立训练。

消融中还含一个完整输入对照，与baseline组重复，核对指标一致；汇总所有条件时将其去重，得到57个独特条件，其中56个为缺失条件。缺失率、位置和时长采用分组设计，并非全部因素的完整笛卡尔积，不能据此估计任意高阶交互。

相较上次结果，当前已具备分析局部缺失规律和训练随机性的数据依据。误差线统一使用3个种子的样本标准差（ddof=1），不称为95%置信区间；同一种子下不同条件也不能当作独立样本扩大显著性。



## 图22 完整输入下的多种子性能与稳定性

![图22 完整输入下的多种子性能与稳定性](figures/22_complete_seed_stability.svg)

**数据来源与统计口径：**完整输入，3个种子；点为单次训练，误差线为种子样本标准差。

本图将单次最优结果与跨种子表现分开。旧参考模型对应baseline的seed 42，不能把其71.80%测试准确率当作baseline跨种子均值。下表同时列出两个划分的四项赛题指标。

比较Robust与No aug.可以观察加入训练缺失增强后的变化，比较No aug.与Baseline反映可用性感知架构及相关配置的整体差异。完整输入性能的提升或下降只说明完整输入代价；是否获得鲁棒性，需要结合后续缺失条件。三种子足以提供初步波动范围，但不足以证明普遍稳定。

| split | model | accuracy | accuracy_sd | macro_f1 | macro_f1_sd | mae | mae_sd | pearson | pearson_sd |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| valid | baseline | 0.6287 | 0.0112 | 0.6096 | 0.0022 | 0.5780 | 0.0082 | 0.6826 | 0.0034 |
| valid | robust_noaug | 0.6337 | 0.0080 | 0.6152 | 0.0061 | 0.5747 | 0.0121 | 0.6804 | 0.0083 |
| valid | robust | 0.6447 | 0.0056 | 0.6138 | 0.0134 | 0.5675 | 0.0111 | 0.6863 | 0.0079 |
| test | baseline | 0.6923 | 0.0248 | 0.6558 | 0.0164 | 0.6137 | 0.0067 | 0.7022 | 0.0025 |
| test | robust_noaug | 0.6896 | 0.0103 | 0.6525 | 0.0036 | 0.6093 | 0.0066 | 0.7020 | 0.0133 |
| test | robust | 0.6933 | 0.0107 | 0.6437 | 0.0208 | 0.6076 | 0.0072 | 0.6978 | 0.0023 |



## 图23 验证集局部缺失率对Macro-F1的影响

![图23 验证集局部缺失率对Macro-F1的影响](figures/23_rate_valid_macro_f1.svg)

**数据来源与统计口径：**rate组，7种模态组合×4个非零缺失率×3模型×3种子；零点取完整输入。

曲线从完整输入延伸至请求缺失率0.4，图中T/A/V分别表示文本/语音/视觉。误差线为跨种子标准差，保留非单调结果；没有对曲线作平滑或强制下降。基线端点变化幅度最大的组合是T，Macro-F1由0.6096变为0.5371，差值-0.0725。该组合0.4端点表现最好的模型为robust，值为0.5500。

应在同一模态组合、同一请求缺失率下比较模型，不能用不同面板的横向位置替代相同实际信息损失。全局缺失率的分母含三个模态全部有效bin，因此文本缺失30%并不等于总信息缺失30%。下表报告全部端点，帮助区分对文本损伤敏感与对音视频损伤敏感的情况。

验证集用于建模选择，测试集用于描述最终泛化；本报告不据测试曲线重新挑选训练轮次或阈值。分类与回归曲线需共同判断，F1提升不自动意味着强度MAE下降。

| modalities | model | at_0 | at_0.4 | change | sd_at_0.4 |
| --- | --- | --- | --- | --- | --- |
| T | baseline | 0.6096 | 0.5371 | -0.0725 | 0.0153 |
| T | robust_noaug | 0.6152 | 0.5423 | -0.0729 | 0.0061 |
| T | robust | 0.6138 | 0.5500 | -0.0638 | 0.0107 |
| A | baseline | 0.6096 | 0.6126 | 0.0030 | 0.0052 |
| A | robust_noaug | 0.6152 | 0.6144 | -0.0007 | 0.0034 |
| A | robust | 0.6138 | 0.6147 | 0.0009 | 0.0108 |
| V | baseline | 0.6096 | 0.6089 | -0.0007 | 0.0066 |
| V | robust_noaug | 0.6152 | 0.6146 | -0.0006 | 0.0056 |
| V | robust | 0.6138 | 0.6149 | 0.0011 | 0.0148 |
| T+A | baseline | 0.6096 | 0.5386 | -0.0710 | 0.0159 |
| T+A | robust_noaug | 0.6152 | 0.5388 | -0.0764 | 0.0078 |
| T+A | robust | 0.6138 | 0.5473 | -0.0665 | 0.0098 |
| T+V | baseline | 0.6096 | 0.5469 | -0.0627 | 0.0132 |
| T+V | robust_noaug | 0.6152 | 0.5561 | -0.0590 | 0.0068 |
| T+V | robust | 0.6138 | 0.5569 | -0.0569 | 0.0180 |
| A+V | baseline | 0.6096 | 0.6057 | -0.0039 | 0.0023 |
| A+V | robust_noaug | 0.6152 | 0.6147 | -0.0005 | 0.0080 |
| A+V | robust | 0.6138 | 0.6167 | 0.0029 | 0.0106 |
| T+A+V | baseline | 0.6096 | 0.5414 | -0.0682 | 0.0114 |
| T+A+V | robust_noaug | 0.6152 | 0.5297 | -0.0855 | 0.0113 |
| T+A+V | robust | 0.6138 | 0.5581 | -0.0557 | 0.0128 |



## 图24 测试集局部缺失率对Macro-F1的影响

![图24 测试集局部缺失率对Macro-F1的影响](figures/24_rate_test_macro_f1.svg)

**数据来源与统计口径：**rate组，7种模态组合×4个非零缺失率×3模型×3种子；零点取完整输入。

曲线从完整输入延伸至请求缺失率0.4，图中T/A/V分别表示文本/语音/视觉。误差线为跨种子标准差，保留非单调结果；没有对曲线作平滑或强制下降。基线端点变化幅度最大的组合是T+V，Macro-F1由0.6558变为0.5522，差值-0.1036。该组合0.4端点表现最好的模型为robust，值为0.5780。

应在同一模态组合、同一请求缺失率下比较模型，不能用不同面板的横向位置替代相同实际信息损失。全局缺失率的分母含三个模态全部有效bin，因此文本缺失30%并不等于总信息缺失30%。下表报告全部端点，帮助区分对文本损伤敏感与对音视频损伤敏感的情况。

验证集用于建模选择，测试集用于描述最终泛化；本报告不据测试曲线重新挑选训练轮次或阈值。分类与回归曲线需共同判断，F1提升不自动意味着强度MAE下降。

| modalities | model | at_0 | at_0.4 | change | sd_at_0.4 |
| --- | --- | --- | --- | --- | --- |
| T | baseline | 0.6558 | 0.5683 | -0.0875 | 0.0032 |
| T | robust_noaug | 0.6525 | 0.5668 | -0.0857 | 0.0071 |
| T | robust | 0.6437 | 0.5613 | -0.0824 | 0.0098 |
| A | baseline | 0.6558 | 0.6565 | 0.0007 | 0.0184 |
| A | robust_noaug | 0.6525 | 0.6530 | 0.0006 | 0.0005 |
| A | robust | 0.6437 | 0.6428 | -0.0009 | 0.0194 |
| V | baseline | 0.6558 | 0.6561 | 0.0003 | 0.0169 |
| V | robust_noaug | 0.6525 | 0.6531 | 0.0006 | 0.0051 |
| V | robust | 0.6437 | 0.6431 | -0.0006 | 0.0188 |
| T+A | baseline | 0.6558 | 0.5750 | -0.0808 | 0.0116 |
| T+A | robust_noaug | 0.6525 | 0.5659 | -0.0866 | 0.0264 |
| T+A | robust | 0.6437 | 0.5770 | -0.0667 | 0.0244 |
| T+V | baseline | 0.6558 | 0.5522 | -0.1036 | 0.0229 |
| T+V | robust_noaug | 0.6525 | 0.5611 | -0.0914 | 0.0188 |
| T+V | robust | 0.6437 | 0.5780 | -0.0657 | 0.0178 |
| A+V | baseline | 0.6558 | 0.6582 | 0.0024 | 0.0136 |
| A+V | robust_noaug | 0.6525 | 0.6529 | 0.0004 | 0.0046 |
| A+V | robust | 0.6437 | 0.6422 | -0.0015 | 0.0146 |
| T+A+V | baseline | 0.6558 | 0.5786 | -0.0772 | 0.0022 |
| T+A+V | robust_noaug | 0.6525 | 0.5820 | -0.0705 | 0.0043 |
| T+A+V | robust | 0.6437 | 0.5859 | -0.0578 | 0.0222 |



## 图25 验证集局部缺失率对MAE的影响

![图25 验证集局部缺失率对MAE的影响](figures/25_rate_valid_mae.svg)

**数据来源与统计口径：**rate组，7种模态组合×4个非零缺失率×3模型×3种子；零点取完整输入。

曲线从完整输入延伸至请求缺失率0.4，图中T/A/V分别表示文本/语音/视觉。误差线为跨种子标准差，保留非单调结果；没有对曲线作平滑或强制下降。基线端点变化幅度最大的组合是T+A，MAE由0.5780变为0.6616，差值+0.0836。该组合0.4端点表现最好的模型为robust，值为0.6416。

应在同一模态组合、同一请求缺失率下比较模型，不能用不同面板的横向位置替代相同实际信息损失。全局缺失率的分母含三个模态全部有效bin，因此文本缺失30%并不等于总信息缺失30%。下表报告全部端点，帮助区分对文本损伤敏感与对音视频损伤敏感的情况。

验证集用于建模选择，测试集用于描述最终泛化；本报告不据测试曲线重新挑选训练轮次或阈值。分类与回归曲线需共同判断，F1提升不自动意味着强度MAE下降。

| modalities | model | at_0 | at_0.4 | change | sd_at_0.4 |
| --- | --- | --- | --- | --- | --- |
| T | baseline | 0.5780 | 0.6461 | 0.0681 | 0.0131 |
| T | robust_noaug | 0.5747 | 0.6451 | 0.0704 | 0.0080 |
| T | robust | 0.5675 | 0.6387 | 0.0712 | 0.0013 |
| A | baseline | 0.5780 | 0.5778 | -0.0001 | 0.0094 |
| A | robust_noaug | 0.5747 | 0.5745 | -0.0003 | 0.0118 |
| A | robust | 0.5675 | 0.5674 | -0.0000 | 0.0107 |
| V | baseline | 0.5780 | 0.5783 | 0.0003 | 0.0077 |
| V | robust_noaug | 0.5747 | 0.5746 | -0.0001 | 0.0115 |
| V | robust | 0.5675 | 0.5676 | 0.0001 | 0.0111 |
| T+A | baseline | 0.5780 | 0.6616 | 0.0836 | 0.0078 |
| T+A | robust_noaug | 0.5747 | 0.6547 | 0.0799 | 0.0203 |
| T+A | robust | 0.5675 | 0.6416 | 0.0741 | 0.0144 |
| T+V | baseline | 0.5780 | 0.6531 | 0.0751 | 0.0185 |
| T+V | robust_noaug | 0.5747 | 0.6538 | 0.0791 | 0.0338 |
| T+V | robust | 0.5675 | 0.6452 | 0.0778 | 0.0169 |
| A+V | baseline | 0.5780 | 0.5783 | 0.0004 | 0.0086 |
| A+V | robust_noaug | 0.5747 | 0.5744 | -0.0003 | 0.0115 |
| A+V | robust | 0.5675 | 0.5674 | -0.0001 | 0.0107 |
| T+A+V | baseline | 0.5780 | 0.6517 | 0.0738 | 0.0092 |
| T+A+V | robust_noaug | 0.5747 | 0.6505 | 0.0758 | 0.0114 |
| T+A+V | robust | 0.5675 | 0.6289 | 0.0614 | 0.0094 |



## 图26 测试集局部缺失率对MAE的影响

![图26 测试集局部缺失率对MAE的影响](figures/26_rate_test_mae.svg)

**数据来源与统计口径：**rate组，7种模态组合×4个非零缺失率×3模型×3种子；零点取完整输入。

曲线从完整输入延伸至请求缺失率0.4，图中T/A/V分别表示文本/语音/视觉。误差线为跨种子标准差，保留非单调结果；没有对曲线作平滑或强制下降。基线端点变化幅度最大的组合是T+A，MAE由0.6137变为0.7298，差值+0.1161。该组合0.4端点表现最好的模型为robust，值为0.7051。

应在同一模态组合、同一请求缺失率下比较模型，不能用不同面板的横向位置替代相同实际信息损失。全局缺失率的分母含三个模态全部有效bin，因此文本缺失30%并不等于总信息缺失30%。下表报告全部端点，帮助区分对文本损伤敏感与对音视频损伤敏感的情况。

验证集用于建模选择，测试集用于描述最终泛化；本报告不据测试曲线重新挑选训练轮次或阈值。分类与回归曲线需共同判断，F1提升不自动意味着强度MAE下降。

| modalities | model | at_0 | at_0.4 | change | sd_at_0.4 |
| --- | --- | --- | --- | --- | --- |
| T | baseline | 0.6137 | 0.7221 | 0.1084 | 0.0118 |
| T | robust_noaug | 0.6093 | 0.7131 | 0.1037 | 0.0106 |
| T | robust | 0.6076 | 0.7036 | 0.0960 | 0.0084 |
| A | baseline | 0.6137 | 0.6142 | 0.0005 | 0.0077 |
| A | robust_noaug | 0.6093 | 0.6095 | 0.0001 | 0.0064 |
| A | robust | 0.6076 | 0.6077 | 0.0001 | 0.0072 |
| V | baseline | 0.6137 | 0.6138 | 0.0001 | 0.0070 |
| V | robust_noaug | 0.6093 | 0.6097 | 0.0004 | 0.0067 |
| V | robust | 0.6076 | 0.6083 | 0.0007 | 0.0071 |
| T+A | baseline | 0.6137 | 0.7298 | 0.1161 | 0.0211 |
| T+A | robust_noaug | 0.6093 | 0.7238 | 0.1144 | 0.0090 |
| T+A | robust | 0.6076 | 0.7051 | 0.0974 | 0.0190 |
| T+V | baseline | 0.6137 | 0.7258 | 0.1121 | 0.0226 |
| T+V | robust_noaug | 0.6093 | 0.7278 | 0.1185 | 0.0322 |
| T+V | robust | 0.6076 | 0.7110 | 0.1034 | 0.0213 |
| A+V | baseline | 0.6137 | 0.6134 | -0.0002 | 0.0074 |
| A+V | robust_noaug | 0.6093 | 0.6093 | -0.0001 | 0.0069 |
| A+V | robust | 0.6076 | 0.6076 | -0.0001 | 0.0071 |
| T+A+V | baseline | 0.6137 | 0.7150 | 0.1013 | 0.0136 |
| T+A+V | robust_noaug | 0.6093 | 0.7083 | 0.0990 | 0.0163 |
| T+A+V | robust | 0.6076 | 0.6950 | 0.0873 | 0.0169 |



## 图27 缺失位置对分类性能的影响

![图27 缺失位置对分类性能的影响](figures/27_missing_position.svg)

**数据来源与统计口径：**请求缺失率固定0.3；仅单模态；分别按位置或连续块时长分组；3种子。

这一组比较直接回应题目对缺失位置的要求。横轴是离散实验设置，连线用于同一模型组内比较，不是连续函数。下表同时给出MAE和全局实际缺失率，便于检查名义上相同缺失率是否对应相近损伤。

前段、中段、后段与随机缺失表现的差异，只能说明当前条件下模型对时序分布敏感。未结合逐样本语义和真实词时间戳时，不能把“前段更重要”解释成所有视频的开头都含决定性情绪。

各组观察范围：

- valid/baseline/text：suffix 0.5402 至 prefix 0.5977（差0.0574）
- valid/baseline/audio：suffix 0.6074 至 middle 0.6111（差0.0037）
- valid/baseline/vision：prefix 0.6078 至 suffix 0.6125（差0.0048）
- valid/robust_noaug/text：suffix 0.5000 至 prefix 0.5976（差0.0976）
- valid/robust_noaug/audio：suffix 0.6131 至 prefix 0.6157（差0.0026）
- valid/robust_noaug/vision：suffix 0.6128 至 middle 0.6169（差0.0041）
- valid/robust/text：suffix 0.5382 至 prefix 0.5935（差0.0552）
- valid/robust/audio：random 0.6132 至 middle 0.6159（差0.0028）
- valid/robust/vision：suffix 0.6125 至 prefix 0.6171（差0.0046）
- test/baseline/text：suffix 0.5451 至 prefix 0.6124（差0.0673）
- test/baseline/audio：middle 0.6536 至 suffix 0.6562（差0.0026）
- test/baseline/vision：prefix 0.6546 至 middle 0.6569（差0.0023）
- test/robust_noaug/text：suffix 0.5181 至 prefix 0.6162（差0.0982）
- test/robust_noaug/audio：middle 0.6507 至 random 0.6543（差0.0036）
- test/robust_noaug/vision：suffix 0.6475 至 middle 0.6541（差0.0066）
- test/robust/text：suffix 0.5742 至 prefix 0.6124（差0.0382）
- test/robust/audio：middle 0.6420 至 prefix 0.6452（差0.0031）
- test/robust/vision：prefix 0.6431 至 random 0.6468（差0.0037）

| split | modality | model | position | macro_f1 | sd | mae | effective_total_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| valid | text | baseline | prefix | 0.5977 | 0.0131 | 0.6155 | 0.1071 |
| valid | text | baseline | middle | 0.5854 | 0.0124 | 0.6280 | 0.1071 |
| valid | text | baseline | suffix | 0.5402 | 0.0083 | 0.6717 | 0.1071 |
| valid | text | baseline | random | 0.5652 | 0.0238 | 0.6369 | 0.1071 |
| valid | text | robust_noaug | prefix | 0.5976 | 0.0021 | 0.6067 | 0.1071 |
| valid | text | robust_noaug | middle | 0.5866 | 0.0087 | 0.6210 | 0.1071 |
| valid | text | robust_noaug | suffix | 0.5000 | 0.0093 | 0.7058 | 0.1071 |
| valid | text | robust_noaug | random | 0.5681 | 0.0189 | 0.6361 | 0.1071 |
| valid | text | robust | prefix | 0.5935 | 0.0185 | 0.6037 | 0.1071 |
| valid | text | robust | middle | 0.5853 | 0.0080 | 0.6124 | 0.1071 |
| valid | text | robust | suffix | 0.5382 | 0.0140 | 0.6435 | 0.1071 |
| valid | text | robust | random | 0.5805 | 0.0013 | 0.6227 | 0.1071 |
| valid | audio | baseline | prefix | 0.6103 | 0.0085 | 0.5784 | 0.0988 |
| valid | audio | baseline | middle | 0.6111 | 0.0061 | 0.5784 | 0.0988 |
| valid | audio | baseline | suffix | 0.6074 | 0.0057 | 0.5772 | 0.0988 |
| valid | audio | baseline | random | 0.6097 | 0.0018 | 0.5782 | 0.0988 |
| valid | audio | robust_noaug | prefix | 0.6157 | 0.0088 | 0.5747 | 0.0988 |
| valid | audio | robust_noaug | middle | 0.6145 | 0.0054 | 0.5749 | 0.0988 |
| valid | audio | robust_noaug | suffix | 0.6131 | 0.0052 | 0.5739 | 0.0988 |
| valid | audio | robust_noaug | random | 0.6151 | 0.0042 | 0.5744 | 0.0988 |
| valid | audio | robust | prefix | 0.6135 | 0.0146 | 0.5671 | 0.0988 |
| valid | audio | robust | middle | 0.6159 | 0.0128 | 0.5677 | 0.0988 |
| valid | audio | robust | suffix | 0.6149 | 0.0149 | 0.5669 | 0.0988 |
| valid | audio | robust | random | 0.6132 | 0.0109 | 0.5671 | 0.0988 |
| valid | vision | baseline | prefix | 0.6078 | 0.0029 | 0.5785 | 0.0933 |
| valid | vision | baseline | middle | 0.6082 | 0.0037 | 0.5775 | 0.0933 |
| valid | vision | baseline | suffix | 0.6125 | 0.0025 | 0.5790 | 0.0933 |
| valid | vision | baseline | random | 0.6107 | 0.0051 | 0.5777 | 0.0933 |
| valid | vision | robust_noaug | prefix | 0.6161 | 0.0074 | 0.5748 | 0.0933 |
| valid | vision | robust_noaug | middle | 0.6169 | 0.0056 | 0.5744 | 0.0933 |
| valid | vision | robust_noaug | suffix | 0.6128 | 0.0084 | 0.5753 | 0.0933 |
| valid | vision | robust_noaug | random | 0.6162 | 0.0062 | 0.5748 | 0.0933 |
| valid | vision | robust | prefix | 0.6171 | 0.0138 | 0.5674 | 0.0933 |
| valid | vision | robust | middle | 0.6141 | 0.0131 | 0.5674 | 0.0933 |
| valid | vision | robust | suffix | 0.6125 | 0.0124 | 0.5680 | 0.0933 |
| valid | vision | robust | random | 0.6154 | 0.0122 | 0.5673 | 0.0933 |
| test | text | baseline | prefix | 0.6124 | 0.0137 | 0.6621 | 0.1075 |
| test | text | baseline | middle | 0.6108 | 0.0154 | 0.6768 | 0.1075 |
| test | text | baseline | suffix | 0.5451 | 0.0154 | 0.7560 | 0.1075 |
| test | text | baseline | random | 0.5818 | 0.0099 | 0.7075 | 0.1075 |
| test | text | robust_noaug | prefix | 0.6162 | 0.0046 | 0.6510 | 0.1075 |
| test | text | robust_noaug | middle | 0.6098 | 0.0101 | 0.6797 | 0.1075 |
| test | text | robust_noaug | suffix | 0.5181 | 0.0123 | 0.7755 | 0.1075 |
| test | text | robust_noaug | random | 0.5877 | 0.0049 | 0.7011 | 0.1075 |
| test | text | robust | prefix | 0.6124 | 0.0156 | 0.6397 | 0.1075 |
| test | text | robust | middle | 0.6029 | 0.0165 | 0.6700 | 0.1075 |
| test | text | robust | suffix | 0.5742 | 0.0154 | 0.7117 | 0.1075 |
| test | text | robust | random | 0.5946 | 0.0240 | 0.6880 | 0.1075 |
| test | audio | baseline | prefix | 0.6546 | 0.0168 | 0.6136 | 0.0988 |
| test | audio | baseline | middle | 0.6536 | 0.0108 | 0.6143 | 0.0988 |
| test | audio | baseline | suffix | 0.6562 | 0.0146 | 0.6144 | 0.0988 |
| test | audio | baseline | random | 0.6548 | 0.0168 | 0.6144 | 0.0988 |
| test | audio | robust_noaug | prefix | 0.6514 | 0.0051 | 0.6090 | 0.0988 |
| test | audio | robust_noaug | middle | 0.6507 | 0.0015 | 0.6092 | 0.0988 |
| test | audio | robust_noaug | suffix | 0.6507 | 0.0024 | 0.6101 | 0.0988 |
| test | audio | robust_noaug | random | 0.6543 | 0.0047 | 0.6093 | 0.0988 |
| test | audio | robust | prefix | 0.6452 | 0.0207 | 0.6071 | 0.0988 |
| test | audio | robust | middle | 0.6420 | 0.0185 | 0.6079 | 0.0988 |
| test | audio | robust | suffix | 0.6433 | 0.0172 | 0.6081 | 0.0988 |
| test | audio | robust | random | 0.6451 | 0.0184 | 0.6078 | 0.0988 |
| test | vision | baseline | prefix | 0.6546 | 0.0137 | 0.6137 | 0.0928 |
| test | vision | baseline | middle | 0.6569 | 0.0150 | 0.6138 | 0.0928 |
| test | vision | baseline | suffix | 0.6546 | 0.0173 | 0.6141 | 0.0928 |
| test | vision | baseline | random | 0.6553 | 0.0170 | 0.6137 | 0.0928 |
| test | vision | robust_noaug | prefix | 0.6526 | 0.0053 | 0.6092 | 0.0928 |
| test | vision | robust_noaug | middle | 0.6541 | 0.0057 | 0.6093 | 0.0928 |
| test | vision | robust_noaug | suffix | 0.6475 | 0.0054 | 0.6103 | 0.0928 |
| test | vision | robust_noaug | random | 0.6521 | 0.0037 | 0.6095 | 0.0928 |
| test | vision | robust | prefix | 0.6431 | 0.0181 | 0.6071 | 0.0928 |
| test | vision | robust | middle | 0.6432 | 0.0204 | 0.6080 | 0.0928 |
| test | vision | robust | suffix | 0.6433 | 0.0188 | 0.6080 | 0.0928 |
| test | vision | robust | random | 0.6468 | 0.0171 | 0.6077 | 0.0928 |



## 图28 缺失连续块时长对分类性能的影响

![图28 缺失连续块时长对分类性能的影响](figures/28_missing_duration.svg)

**数据来源与统计口径：**请求缺失率固定0.3；仅单模态；分别按位置或连续块时长分组；3种子。

这一组比较直接回应题目对缺失时长的要求。横轴是离散实验设置，连线用于同一模型组内比较，不是连续函数。下表同时给出MAE和全局实际缺失率，便于检查名义上相同缺失率是否对应相近损伤。

short/medium/long表示缺失连续块的设置，不是视频秒数。总请求缺失率相同而块长度不同，体现缺失集中程度；缺乏物理时间映射时不换算为秒。时长效应可能与有效长度、离散取整和片段覆盖同时相关。

各组观察范围：

- valid/baseline/text：short 0.5562 至 long 0.5692（差0.0129）
- valid/baseline/audio：short 0.6092 至 long 0.6104（差0.0012）
- valid/baseline/vision：short 0.6081 至 medium 0.6096（差0.0015）
- valid/robust_noaug/text：short 0.5434 至 long 0.5796（差0.0362）
- valid/robust_noaug/audio：short 0.6139 至 medium 0.6145（差0.0006）
- valid/robust_noaug/vision：medium 0.6142 至 long 0.6163（差0.0020）
- valid/robust/text：short 0.5488 至 medium 0.5751（差0.0263）
- valid/robust/audio：long 0.6128 至 medium 0.6159（差0.0030）
- valid/robust/vision：long 0.6145 至 short 0.6165（差0.0020）
- test/baseline/text：short 0.5797 至 long 0.6063（差0.0266）
- test/baseline/audio：short 0.6557 至 long 0.6574（差0.0017）
- test/baseline/vision：medium 0.6555 至 short 0.6562（差0.0006）
- test/robust_noaug/text：short 0.5815 至 long 0.6105（差0.0290）
- test/robust_noaug/audio：short 0.6514 至 long 0.6537（差0.0023）
- test/robust_noaug/vision：long 0.6508 至 short 0.6521（差0.0013）
- test/robust/text：medium 0.5922 至 long 0.6153（差0.0231）
- test/robust/audio：short 0.6435 至 medium 0.6464（差0.0029）
- test/robust/vision：short 0.6414 至 long 0.6437（差0.0023）

| split | modality | model | duration | macro_f1 | sd | mae | effective_total_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| valid | text | baseline | short | 0.5562 | 0.0157 | 0.6302 | 0.1071 |
| valid | text | baseline | medium | 0.5649 | 0.0068 | 0.6329 | 0.1071 |
| valid | text | baseline | long | 0.5692 | 0.0030 | 0.6227 | 0.1071 |
| valid | text | robust_noaug | short | 0.5434 | 0.0243 | 0.6445 | 0.1071 |
| valid | text | robust_noaug | medium | 0.5645 | 0.0191 | 0.6283 | 0.1071 |
| valid | text | robust_noaug | long | 0.5796 | 0.0101 | 0.6170 | 0.1071 |
| valid | text | robust | short | 0.5488 | 0.0078 | 0.6226 | 0.1071 |
| valid | text | robust | medium | 0.5751 | 0.0111 | 0.6237 | 0.1071 |
| valid | text | robust | long | 0.5738 | 0.0097 | 0.6068 | 0.1071 |
| valid | audio | baseline | short | 0.6092 | 0.0025 | 0.5776 | 0.0988 |
| valid | audio | baseline | medium | 0.6098 | 0.0025 | 0.5782 | 0.0988 |
| valid | audio | baseline | long | 0.6104 | 0.0066 | 0.5775 | 0.0988 |
| valid | audio | robust_noaug | short | 0.6139 | 0.0063 | 0.5745 | 0.0988 |
| valid | audio | robust_noaug | medium | 0.6145 | 0.0052 | 0.5750 | 0.0988 |
| valid | audio | robust_noaug | long | 0.6141 | 0.0059 | 0.5741 | 0.0988 |
| valid | audio | robust | short | 0.6147 | 0.0139 | 0.5672 | 0.0988 |
| valid | audio | robust | medium | 0.6159 | 0.0126 | 0.5678 | 0.0988 |
| valid | audio | robust | long | 0.6128 | 0.0104 | 0.5668 | 0.0988 |
| valid | vision | baseline | short | 0.6081 | 0.0028 | 0.5779 | 0.0933 |
| valid | vision | baseline | medium | 0.6096 | 0.0023 | 0.5780 | 0.0933 |
| valid | vision | baseline | long | 0.6084 | 0.0028 | 0.5780 | 0.0933 |
| valid | vision | robust_noaug | short | 0.6148 | 0.0056 | 0.5748 | 0.0933 |
| valid | vision | robust_noaug | medium | 0.6142 | 0.0075 | 0.5746 | 0.0933 |
| valid | vision | robust_noaug | long | 0.6163 | 0.0059 | 0.5745 | 0.0933 |
| valid | vision | robust | short | 0.6165 | 0.0145 | 0.5675 | 0.0933 |
| valid | vision | robust | medium | 0.6152 | 0.0139 | 0.5673 | 0.0933 |
| valid | vision | robust | long | 0.6145 | 0.0150 | 0.5673 | 0.0933 |
| test | text | baseline | short | 0.5797 | 0.0062 | 0.7149 | 0.1075 |
| test | text | baseline | medium | 0.5916 | 0.0111 | 0.6926 | 0.1075 |
| test | text | baseline | long | 0.6063 | 0.0029 | 0.6878 | 0.1075 |
| test | text | robust_noaug | short | 0.5815 | 0.0091 | 0.7079 | 0.1075 |
| test | text | robust_noaug | medium | 0.5870 | 0.0135 | 0.6909 | 0.1075 |
| test | text | robust_noaug | long | 0.6105 | 0.0120 | 0.6858 | 0.1075 |
| test | text | robust | short | 0.5955 | 0.0299 | 0.6881 | 0.1075 |
| test | text | robust | medium | 0.5922 | 0.0067 | 0.6822 | 0.1075 |
| test | text | robust | long | 0.6153 | 0.0147 | 0.6711 | 0.1075 |
| test | audio | baseline | short | 0.6557 | 0.0157 | 0.6137 | 0.0988 |
| test | audio | baseline | medium | 0.6572 | 0.0182 | 0.6133 | 0.0988 |
| test | audio | baseline | long | 0.6574 | 0.0133 | 0.6136 | 0.0988 |
| test | audio | robust_noaug | short | 0.6514 | 0.0042 | 0.6093 | 0.0988 |
| test | audio | robust_noaug | medium | 0.6515 | 0.0029 | 0.6099 | 0.0988 |
| test | audio | robust_noaug | long | 0.6537 | 0.0041 | 0.6088 | 0.0988 |
| test | audio | robust | short | 0.6435 | 0.0184 | 0.6078 | 0.0988 |
| test | audio | robust | medium | 0.6464 | 0.0194 | 0.6082 | 0.0988 |
| test | audio | robust | long | 0.6439 | 0.0185 | 0.6074 | 0.0988 |
| test | vision | baseline | short | 0.6562 | 0.0164 | 0.6134 | 0.0928 |
| test | vision | baseline | medium | 0.6555 | 0.0164 | 0.6130 | 0.0928 |
| test | vision | baseline | long | 0.6556 | 0.0158 | 0.6134 | 0.0928 |
| test | vision | robust_noaug | short | 0.6521 | 0.0060 | 0.6093 | 0.0928 |
| test | vision | robust_noaug | medium | 0.6512 | 0.0039 | 0.6094 | 0.0928 |
| test | vision | robust_noaug | long | 0.6508 | 0.0054 | 0.6095 | 0.0928 |
| test | vision | robust | short | 0.6414 | 0.0219 | 0.6078 | 0.0928 |
| test | vision | robust | medium | 0.6437 | 0.0208 | 0.6076 | 0.0928 |
| test | vision | robust | long | 0.6437 | 0.0214 | 0.6079 | 0.0928 |



## 图29 请求缺失率与实际缺失率的分母差异

![图29 请求缺失率与实际缺失率的分母差异](figures/29_missing_rate_denominators.svg)

**数据来源与统计口径：**测试集baseline，种子均值；读取effective_missing_rate及各模态对应字段。

请求缺失率针对被选中的模态有效序列；全局实际缺失率以全部模态有效bin总量作分母。只缺失一个模态时，全局比例自然小于请求值；同时缺失T+A+V时两者才较接近。离散bin数量、可用长度和取整会产生小幅偏差。

热图将0.3设置分解到各模态，未被选中模态的缺失率应为零。该核对只检查保存结果的计数与实验条件是否对应，不审核实现代码。论文需要写清分母，不能把全局比例直接称为“文本缺失率”。

本图有助于解释不同缺失组合之间的损伤强弱，但不能把被删除bin数量等同于语义信息量；文本、音频、视觉的一个bin并不具有相同信息价值。

| modalities | requested | total_effective | text | audio | vision |
| --- | --- | --- | --- | --- | --- |
| T | 0.1000 | 0.0361 | 0.1007 | 0.0000 | 0.0000 |
| T | 0.2000 | 0.0717 | 0.1998 | 0.0000 | 0.0000 |
| T | 0.3000 | 0.1075 | 0.2995 | 0.0000 | 0.0000 |
| T | 0.4000 | 0.1436 | 0.4001 | 0.0000 | 0.0000 |
| A | 0.1000 | 0.0332 | 0.0000 | 0.1005 | 0.0000 |
| A | 0.2000 | 0.0666 | 0.0000 | 0.2016 | 0.0000 |
| A | 0.3000 | 0.0988 | 0.0000 | 0.2990 | 0.0000 |
| A | 0.4000 | 0.1322 | 0.0000 | 0.3999 | 0.0000 |
| V | 0.1000 | 0.0312 | 0.0000 | 0.0000 | 0.1004 |
| V | 0.2000 | 0.0626 | 0.0000 | 0.0000 | 0.2015 |
| V | 0.3000 | 0.0928 | 0.0000 | 0.0000 | 0.2990 |
| V | 0.4000 | 0.1241 | 0.0000 | 0.0000 | 0.3997 |
| T+A | 0.1000 | 0.0694 | 0.1007 | 0.1005 | 0.0000 |
| T+A | 0.2000 | 0.1384 | 0.1998 | 0.2016 | 0.0000 |
| T+A | 0.3000 | 0.2064 | 0.2995 | 0.2990 | 0.0000 |
| T+A | 0.4000 | 0.2758 | 0.4001 | 0.3999 | 0.0000 |
| T+V | 0.1000 | 0.0673 | 0.1007 | 0.0000 | 0.1004 |
| T+V | 0.2000 | 0.1343 | 0.1998 | 0.0000 | 0.2015 |
| T+V | 0.3000 | 0.2004 | 0.2995 | 0.0000 | 0.2990 |
| T+V | 0.4000 | 0.2677 | 0.4001 | 0.0000 | 0.3997 |
| A+V | 0.1000 | 0.0644 | 0.0000 | 0.1005 | 0.1004 |
| A+V | 0.2000 | 0.1292 | 0.0000 | 0.2016 | 0.2015 |
| A+V | 0.3000 | 0.1916 | 0.0000 | 0.2990 | 0.2990 |
| A+V | 0.4000 | 0.2563 | 0.0000 | 0.3999 | 0.3997 |
| T+A+V | 0.1000 | 0.1005 | 0.1007 | 0.1005 | 0.1004 |
| T+A+V | 0.2000 | 0.2009 | 0.1998 | 0.2016 | 0.2015 |
| T+A+V | 0.3000 | 0.2992 | 0.2995 | 0.2990 | 0.2990 |
| T+A+V | 0.4000 | 0.3999 | 0.4001 | 0.3999 | 0.3997 |



## 图30 整模态消融与极端信息丢失

![图30 整模态消融与极端信息丢失](figures/30_whole_modality_ablation.svg)

**数据来源与统计口径：**ablation组，8种保留/删除组合；行标签表示被删除的模态，None表示完整输入。

整模态删除是极端压力测试，与题目主要关心的局部连续缺失不同，应作为补充消融呈现。移除T、A、V后的性能变化揭示模型对相应输入的依赖；文本删除后的下降不能独立证明文本在所有真实场景中都具有最高因果重要性。

全部删除T+A+V时，即使分类准确率非零，也可能只是类先验或常数输出的表现。该条件下Pearson数值可能来自近常数浮点波动，不能解释为模型保留了实际强度预测能力。图中MAE与F1分开显示，避免不同量纲混合。

本文消融使用固定已训练模型的输入删除结果，而非为每种保留模态重新训练专门模型。因此“只剩文本”的成绩应表述为该融合模型在只保留文本时的表现，不应写成重新训练的纯文本模型基线。

| split | model | removed | accuracy | macro_f1 | mae | pearson |
| --- | --- | --- | --- | --- | --- | --- |
| valid | baseline | None | 0.6287 | 0.6096 | 0.5780 | 0.6826 |
| valid | baseline | T | 0.3887 | 0.3738 | 0.7726 | 0.1812 |
| valid | baseline | A | 0.6296 | 0.6070 | 0.5783 | 0.6825 |
| valid | baseline | V | 0.6277 | 0.6074 | 0.5883 | 0.6741 |
| valid | baseline | T+A | 0.4016 | 0.3803 | 0.7704 | 0.1834 |
| valid | baseline | T+V | 0.3283 | 0.3112 | 0.7980 | 0.0448 |
| valid | baseline | A+V | 0.6236 | 0.6006 | 0.5901 | 0.6732 |
| valid | baseline | T+A+V | 0.3333 | 0.1643 | 0.8221 | 0.0000 |
| valid | robust_noaug | None | 0.6337 | 0.6152 | 0.5747 | 0.6804 |
| valid | robust_noaug | T | 0.3800 | 0.3672 | 0.7838 | 0.2111 |
| valid | robust_noaug | A | 0.6305 | 0.6118 | 0.5756 | 0.6796 |
| valid | robust_noaug | V | 0.6264 | 0.6070 | 0.5846 | 0.6717 |
| valid | robust_noaug | T+A | 0.3544 | 0.3018 | 0.7838 | 0.2333 |
| valid | robust_noaug | T+V | 0.3040 | 0.2453 | 0.7960 | 0.0590 |
| valid | robust_noaug | A+V | 0.6264 | 0.6070 | 0.5865 | 0.6705 |
| valid | robust_noaug | T+A+V | 0.2527 | 0.1345 | 0.8009 | 0.0116 |
| valid | robust | None | 0.6447 | 0.6138 | 0.5675 | 0.6863 |
| valid | robust | T | 0.4171 | 0.4079 | 0.7706 | 0.2262 |
| valid | robust | A | 0.6429 | 0.6130 | 0.5687 | 0.6852 |
| valid | robust | V | 0.6314 | 0.6000 | 0.5770 | 0.6768 |
| valid | robust | T+A | 0.3851 | 0.3410 | 0.7684 | 0.2459 |
| valid | robust | T+V | 0.3429 | 0.3023 | 0.7829 | 0.0674 |
| valid | robust | A+V | 0.6342 | 0.6054 | 0.5795 | 0.6748 |
| valid | robust | T+A+V | 0.2527 | 0.1345 | 0.7716 | 0.0058 |
| test | baseline | None | 0.6923 | 0.6558 | 0.6137 | 0.7022 |
| test | baseline | T | 0.4081 | 0.3740 | 0.8721 | 0.2058 |
| test | baseline | A | 0.6905 | 0.6509 | 0.6155 | 0.7005 |
| test | baseline | V | 0.6859 | 0.6520 | 0.6245 | 0.6942 |
| test | baseline | T+A | 0.4108 | 0.3603 | 0.8711 | 0.1961 |
| test | baseline | T+V | 0.3439 | 0.3223 | 0.9027 | 0.1026 |
| test | baseline | A+V | 0.6868 | 0.6488 | 0.6292 | 0.6910 |
| test | baseline | T+A+V | 0.3333 | 0.1628 | 0.9330 | -0.0270 |
| test | robust_noaug | None | 0.6896 | 0.6525 | 0.6093 | 0.7020 |
| test | robust_noaug | T | 0.3902 | 0.3704 | 0.8874 | 0.2305 |
| test | robust_noaug | A | 0.6845 | 0.6479 | 0.6119 | 0.7002 |
| test | robust_noaug | V | 0.6818 | 0.6466 | 0.6179 | 0.6955 |
| test | robust_noaug | T+A | 0.3439 | 0.2944 | 0.8903 | 0.2110 |
| test | robust_noaug | T+V | 0.3095 | 0.2646 | 0.9029 | 0.1429 |
| test | robust_noaug | A+V | 0.6790 | 0.6442 | 0.6212 | 0.6931 |
| test | robust_noaug | T+A+V | 0.2173 | 0.1190 | 0.9166 | -0.0270 |
| test | robust | None | 0.6933 | 0.6437 | 0.6076 | 0.6978 |
| test | robust | T | 0.4360 | 0.4069 | 0.8729 | 0.2375 |
| test | robust | A | 0.6887 | 0.6397 | 0.6098 | 0.6958 |
| test | robust | V | 0.6896 | 0.6419 | 0.6151 | 0.6912 |
| test | robust | T+A | 0.3998 | 0.3507 | 0.8734 | 0.2188 |
| test | robust | T+V | 0.3645 | 0.3246 | 0.8876 | 0.1581 |
| test | robust | A+V | 0.6923 | 0.6458 | 0.6178 | 0.6884 |
| test | robust | T+A+V | 0.2173 | 0.1190 | 0.8858 | -0.0135 |



## 图31 架构与训练增强的配对差异

![图31 架构与训练增强的配对差异](figures/31_paired_architecture_augmentation.svg)

**数据来源与统计口径：**同一种子、同一条件相减后，先对条件族等权平均，再对3种子计算均值与标准差。

Architecture为No aug.减Baseline；Augmentation为Robust减No aug.。F1差值为正更好，MAE差值为负更好。图中的配对保留了种子编号和缺失条件，避免把大量相关条件误认为大量独立训练。

每个条件族的权重由本次设计决定，rate组的均值并不代表真实业务缺失分布。对于ablation，去掉重复完整输入后仅汇总7个压力条件。均值接近零时应查看三个种子的符号是否一致，而不能只选择其中一次成功的结果写进论文。

该设计比历史模型横向比较更能区分架构和增强作用，但模型/增强差异的解释仍应限定在保存配置、三个种子及当前条件范围内。

| split | metric | comparison | group | mean_difference | seed_sd | seed42 | seed43 | seed44 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| valid | macro_f1 | robust_noaug - baseline | baseline | 0.0056 | 0.0050 | 0.0113 | 0.0028 | 0.0027 |
| valid | macro_f1 | robust_noaug - baseline | rate | 0.0034 | 0.0027 | 0.0046 | 0.0053 | 0.0002 |
| valid | macro_f1 | robust_noaug - baseline | position | 0.0005 | 0.0010 | 0.0007 | 0.0015 | -0.0006 |
| valid | macro_f1 | robust_noaug - baseline | duration | 0.0033 | 0.0050 | 0.0090 | 0.0001 | 0.0006 |
| valid | macro_f1 | robust_noaug - baseline | ablation | -0.0243 | 0.0165 | -0.0156 | -0.0433 | -0.0138 |
| valid | macro_f1 | robust - robust_noaug | baseline | -0.0014 | 0.0116 | -0.0050 | -0.0107 | 0.0116 |
| valid | macro_f1 | robust - robust_noaug | rate | 0.0034 | 0.0098 | -0.0033 | -0.0012 | 0.0146 |
| valid | macro_f1 | robust - robust_noaug | position | 0.0035 | 0.0105 | -0.0036 | -0.0016 | 0.0155 |
| valid | macro_f1 | robust - robust_noaug | duration | 0.0013 | 0.0110 | -0.0072 | -0.0026 | 0.0138 |
| valid | macro_f1 | robust - robust_noaug | ablation | 0.0185 | 0.0164 | 0.0155 | 0.0362 | 0.0038 |
| valid | mae | robust_noaug - baseline | baseline | -0.0032 | 0.0195 | 0.0001 | 0.0143 | -0.0242 |
| valid | mae | robust_noaug - baseline | rate | -0.0038 | 0.0177 | 0.0004 | 0.0113 | -0.0233 |
| valid | mae | robust_noaug - baseline | position | -0.0008 | 0.0192 | 0.0040 | 0.0155 | -0.0220 |
| valid | mae | robust_noaug - baseline | duration | -0.0017 | 0.0189 | 0.0044 | 0.0133 | -0.0229 |
| valid | mae | robust_noaug - baseline | ablation | -0.0012 | 0.0055 | -0.0015 | -0.0066 | 0.0044 |
| valid | mae | robust - robust_noaug | baseline | -0.0072 | 0.0037 | -0.0110 | -0.0071 | -0.0036 |
| valid | mae | robust - robust_noaug | rate | -0.0089 | 0.0043 | -0.0116 | -0.0113 | -0.0039 |
| valid | mae | robust - robust_noaug | position | -0.0121 | 0.0057 | -0.0162 | -0.0146 | -0.0056 |
| valid | mae | robust - robust_noaug | duration | -0.0089 | 0.0049 | -0.0127 | -0.0105 | -0.0034 |
| valid | mae | robust - robust_noaug | ablation | -0.0132 | 0.0029 | -0.0120 | -0.0112 | -0.0165 |
| test | macro_f1 | robust_noaug - baseline | baseline | -0.0033 | 0.0189 | -0.0184 | -0.0094 | 0.0179 |
| test | macro_f1 | robust_noaug - baseline | rate | -0.0023 | 0.0087 | -0.0114 | -0.0015 | 0.0059 |
| test | macro_f1 | robust_noaug - baseline | position | -0.0038 | 0.0143 | -0.0165 | -0.0066 | 0.0118 |
| test | macro_f1 | robust_noaug - baseline | duration | -0.0029 | 0.0133 | -0.0126 | -0.0082 | 0.0122 |
| test | macro_f1 | robust_noaug - baseline | ablation | -0.0263 | 0.0231 | -0.0349 | -0.0438 | -0.0001 |
| test | macro_f1 | robust - robust_noaug | baseline | -0.0088 | 0.0189 | -0.0282 | -0.0077 | 0.0095 |
| test | macro_f1 | robust - robust_noaug | rate | -0.0020 | 0.0134 | -0.0130 | -0.0057 | 0.0129 |
| test | macro_f1 | robust - robust_noaug | position | -0.0007 | 0.0144 | -0.0127 | -0.0048 | 0.0153 |
| test | macro_f1 | robust - robust_noaug | duration | -0.0027 | 0.0152 | -0.0190 | 0.0001 | 0.0110 |
| test | macro_f1 | robust - robust_noaug | ablation | 0.0202 | 0.0148 | 0.0223 | 0.0339 | 0.0045 |
| test | mae | robust_noaug - baseline | baseline | -0.0044 | 0.0132 | 0.0017 | 0.0047 | -0.0195 |
| test | mae | robust_noaug - baseline | rate | -0.0043 | 0.0140 | 0.0028 | 0.0047 | -0.0205 |
| test | mae | robust_noaug - baseline | position | -0.0026 | 0.0125 | 0.0040 | 0.0053 | -0.0170 |
| test | mae | robust_noaug - baseline | duration | -0.0039 | 0.0140 | 0.0041 | 0.0044 | -0.0201 |
| test | mae | robust_noaug - baseline | ablation | 0.0000 | 0.0103 | 0.0058 | -0.0119 | 0.0061 |
| test | mae | robust - robust_noaug | baseline | -0.0017 | 0.0035 | -0.0056 | 0.0012 | -0.0007 |
| test | mae | robust - robust_noaug | rate | -0.0075 | 0.0026 | -0.0091 | -0.0090 | -0.0044 |
| test | mae | robust - robust_noaug | position | -0.0093 | 0.0030 | -0.0128 | -0.0076 | -0.0076 |
| test | mae | robust - robust_noaug | duration | -0.0059 | 0.0011 | -0.0055 | -0.0070 | -0.0050 |
| test | mae | robust - robust_noaug | ablation | -0.0123 | 0.0040 | -0.0122 | -0.0082 | -0.0163 |



## 图32 局部缺失的平均性能与最差条件

![图32 局部缺失的平均性能与最差条件](figures/32_average_worst_conditions.svg)

**数据来源与统计口径：**对三种子均值再按条件等权汇总；local=49个局部缺失条件，all_stress=56个条件含整模态删除。

平均值描述实验网格的整体表现，最差值用于识别模型在何种条件下失效；两者没有相同含义。图中仅画局部缺失，避免全模态全删这种极端条件主导主要结论；下表额外列出全部压力条件供核对。

最差条件是从固定测试网格中事后选出的观察值，不是有统计置信保证的性能下界。表中准确写出对应条件ID，也列出MAE最差条件，因为分类和回归最差情形未必重合。

题目中的“稳定输出”可由缺失条件下仍有可用F1、MAE及跨种子波动共同支撑，不应仅凭程序产生数值便称为鲁棒。

| split | model | scope | conditions | mean_f1 | worst_f1 | mean_mae | worst_mae | worst_f1_condition | worst_mae_condition |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| valid | baseline | local | 49 | 0.5914 | 0.5371 | 0.6009 | 0.6717 | rate__text__r0p40__random__medium | position__text__r0p30__suffix__medium |
| valid | baseline | all_stress | 56 | 0.5718 | 0.1643 | 0.6137 | 0.8221 | ablation__text+audio+vision__r1p00__random__long | ablation__text+audio+vision__r1p00__random__long |
| valid | robust_noaug | local | 49 | 0.5940 | 0.5000 | 0.5982 | 0.7058 | position__text__r0p30__suffix__medium | position__text__r0p30__suffix__medium |
| valid | robust_noaug | all_stress | 56 | 0.5711 | 0.1345 | 0.6111 | 0.8009 | ablation__text+audio+vision__r1p00__random__long | ablation__text+audio+vision__r1p00__random__long |
| valid | robust | local | 49 | 0.5971 | 0.5382 | 0.5885 | 0.6452 | position__text__r0p30__suffix__medium | rate__text+vision__r0p40__random__medium |
| valid | robust | all_stress | 56 | 0.5761 | 0.1345 | 0.6010 | 0.7829 | ablation__text+audio+vision__r1p00__random__long | ablation__text+vision__r1p00__random__long |
| test | baseline | local | 49 | 0.6282 | 0.5451 | 0.6494 | 0.7560 | position__text__r0p30__suffix__medium | position__text__r0p30__suffix__medium |
| test | baseline | all_stress | 56 | 0.6063 | 0.1628 | 0.6655 | 0.9330 | ablation__text+audio+vision__r1p00__random__long | ablation__text+audio+vision__r1p00__random__long |
| test | robust_noaug | local | 49 | 0.6254 | 0.5181 | 0.6456 | 0.7755 | position__text__r0p30__suffix__medium | position__text__r0p30__suffix__medium |
| test | robust_noaug | all_stress | 56 | 0.6006 | 0.1190 | 0.6622 | 0.9166 | ablation__text+audio+vision__r1p00__random__long | ablation__text+audio+vision__r1p00__random__long |
| test | robust | local | 49 | 0.6236 | 0.5613 | 0.6379 | 0.7117 | rate__text__r0p40__random__medium | position__text__r0p30__suffix__medium |
| test | robust | all_stress | 56 | 0.6015 | 0.1190 | 0.6540 | 0.8876 | ablation__text+audio+vision__r1p00__random__long | ablation__text+vision__r1p00__random__long |



## 图33 文本局部缺失对各情感类别的差异影响

![图33 文本局部缺失对各情感类别的差异影响](figures/33_class_sensitivity.svg)

**数据来源与统计口径：**保存的逐类F1；文本局部缺失0至0.4，两个划分、3模型、3种子。

Macro-F1的变化可能由一个类别主导，本图用相同纵轴显示三类F1。中性类既需要与正面也需要与负面区分，其完整输入表现已偏弱；文本局部丢失后，如果中性曲线先下降，整体准确率可能因为多数正面类别相对稳定而掩盖风险。

各类F1结合precision与recall，不能单凭F1判定下降究竟来自漏识别还是误报。新增受控表只存逐类F1，未存各条件混淆矩阵；因此这里不伪造逐条件召回率或错误流向。

观察Robust相对No aug.的作用时，应说明受益类别与可能牺牲类别，而不是只报告单个平均数。完整数值随图导出。

| split | model | class | rate | f1 | sd |
| --- | --- | --- | --- | --- | --- |
| valid | baseline | negative | 0 | 0.6676 | 0.0208 |
| valid | baseline | negative | 0.1000 | 0.6501 | 0.0317 |
| valid | baseline | negative | 0.2000 | 0.6310 | 0.0387 |
| valid | baseline | negative | 0.3000 | 0.6073 | 0.0161 |
| valid | baseline | negative | 0.4000 | 0.5541 | 0.0246 |
| valid | robust_noaug | negative | 0 | 0.6675 | 0.0066 |
| valid | robust_noaug | negative | 0.1000 | 0.6491 | 0.0151 |
| valid | robust_noaug | negative | 0.2000 | 0.6160 | 0.0381 |
| valid | robust_noaug | negative | 0.3000 | 0.5840 | 0.0483 |
| valid | robust_noaug | negative | 0.4000 | 0.5486 | 0.0357 |
| valid | robust | negative | 0 | 0.6791 | 0.0052 |
| valid | robust | negative | 0.1000 | 0.6696 | 0.0070 |
| valid | robust | negative | 0.2000 | 0.6448 | 0.0436 |
| valid | robust | negative | 0.3000 | 0.6321 | 0.0035 |
| valid | robust | negative | 0.4000 | 0.5832 | 0.0207 |
| valid | baseline | neutral | 0 | 0.4580 | 0.0241 |
| valid | baseline | neutral | 0.1000 | 0.4600 | 0.0308 |
| valid | baseline | neutral | 0.2000 | 0.4690 | 0.0410 |
| valid | baseline | neutral | 0.3000 | 0.4545 | 0.0297 |
| valid | baseline | neutral | 0.4000 | 0.4358 | 0.0153 |
| valid | robust_noaug | neutral | 0 | 0.4726 | 0.0310 |
| valid | robust_noaug | neutral | 0.1000 | 0.4728 | 0.0172 |
| valid | robust_noaug | neutral | 0.2000 | 0.4488 | 0.0215 |
| valid | robust_noaug | neutral | 0.3000 | 0.4505 | 0.0170 |
| valid | robust_noaug | neutral | 0.4000 | 0.4496 | 0.0098 |
| valid | robust | neutral | 0 | 0.4375 | 0.0339 |
| valid | robust | neutral | 0.1000 | 0.4126 | 0.0294 |
| valid | robust | neutral | 0.2000 | 0.4296 | 0.0441 |
| valid | robust | neutral | 0.3000 | 0.4394 | 0.0232 |
| valid | robust | neutral | 0.4000 | 0.4096 | 0.0216 |
| valid | baseline | positive | 0 | 0.7032 | 0.0157 |
| valid | baseline | positive | 0.1000 | 0.6938 | 0.0215 |
| valid | baseline | positive | 0.2000 | 0.6623 | 0.0285 |
| valid | baseline | positive | 0.3000 | 0.6583 | 0.0336 |
| valid | baseline | positive | 0.4000 | 0.6215 | 0.0388 |
| valid | robust_noaug | positive | 0 | 0.7054 | 0.0242 |
| valid | robust_noaug | positive | 0.1000 | 0.6977 | 0.0275 |
| valid | robust_noaug | positive | 0.2000 | 0.6642 | 0.0197 |
| valid | robust_noaug | positive | 0.3000 | 0.6552 | 0.0222 |
| valid | robust_noaug | positive | 0.4000 | 0.6285 | 0.0185 |
| valid | robust | positive | 0 | 0.7248 | 0.0029 |
| valid | robust | positive | 0.1000 | 0.7107 | 0.0103 |
| valid | robust | positive | 0.2000 | 0.6977 | 0.0060 |
| valid | robust | positive | 0.3000 | 0.6785 | 0.0090 |
| valid | robust | positive | 0.4000 | 0.6571 | 0.0215 |
| test | baseline | negative | 0 | 0.7168 | 0.0127 |
| test | baseline | negative | 0.1000 | 0.6968 | 0.0097 |
| test | baseline | negative | 0.2000 | 0.6585 | 0.0214 |
| test | baseline | negative | 0.3000 | 0.6282 | 0.0273 |
| test | baseline | negative | 0.4000 | 0.6002 | 0.0145 |
| test | robust_noaug | negative | 0 | 0.7201 | 0.0045 |
| test | robust_noaug | negative | 0.1000 | 0.6685 | 0.0299 |
| test | robust_noaug | negative | 0.2000 | 0.6525 | 0.0272 |
| test | robust_noaug | negative | 0.3000 | 0.6251 | 0.0347 |
| test | robust_noaug | negative | 0.4000 | 0.5777 | 0.0454 |
| test | robust | negative | 0 | 0.7037 | 0.0166 |
| test | robust | negative | 0.1000 | 0.6793 | 0.0063 |
| test | robust | negative | 0.2000 | 0.6607 | 0.0249 |
| test | robust | negative | 0.3000 | 0.6379 | 0.0332 |
| test | robust | negative | 0.4000 | 0.6107 | 0.0057 |
| test | baseline | neutral | 0 | 0.4824 | 0.0220 |
| test | baseline | neutral | 0.1000 | 0.4510 | 0.0163 |
| test | baseline | neutral | 0.2000 | 0.4341 | 0.0108 |
| test | baseline | neutral | 0.3000 | 0.4390 | 0.0281 |
| test | baseline | neutral | 0.4000 | 0.4086 | 0.0317 |
| test | robust_noaug | neutral | 0 | 0.4709 | 0.0113 |
| test | robust_noaug | neutral | 0.1000 | 0.4617 | 0.0095 |
| test | robust_noaug | neutral | 0.2000 | 0.4278 | 0.0193 |
| test | robust_noaug | neutral | 0.3000 | 0.4561 | 0.0166 |
| test | robust_noaug | neutral | 0.4000 | 0.4332 | 0.0181 |
| test | robust | neutral | 0 | 0.4488 | 0.0545 |
| test | robust | neutral | 0.1000 | 0.4157 | 0.0418 |
| test | robust | neutral | 0.2000 | 0.4194 | 0.0340 |
| test | robust | neutral | 0.3000 | 0.4397 | 0.0396 |
| test | robust | neutral | 0.4000 | 0.3651 | 0.0339 |
| test | baseline | positive | 0 | 0.7683 | 0.0257 |
| test | baseline | positive | 0.1000 | 0.7464 | 0.0236 |
| test | baseline | positive | 0.2000 | 0.7239 | 0.0448 |
| test | baseline | positive | 0.3000 | 0.7207 | 0.0354 |
| test | baseline | positive | 0.4000 | 0.6962 | 0.0231 |
| test | robust_noaug | positive | 0 | 0.7665 | 0.0163 |
| test | robust_noaug | positive | 0.1000 | 0.7408 | 0.0279 |
| test | robust_noaug | positive | 0.2000 | 0.7221 | 0.0221 |
| test | robust_noaug | positive | 0.3000 | 0.7086 | 0.0257 |
| test | robust_noaug | positive | 0.4000 | 0.6895 | 0.0389 |
| test | robust | positive | 0 | 0.7785 | 0.0111 |
| test | robust | positive | 0.1000 | 0.7685 | 0.0171 |
| test | robust | positive | 0.2000 | 0.7560 | 0.0215 |
| test | robust | positive | 0.3000 | 0.7365 | 0.0194 |
| test | robust | positive | 0.4000 | 0.7079 | 0.0188 |



## 图34 新增九次训练的拟合与泛化过程

![图34 新增九次训练的拟合与泛化过程](figures/34_controlled_training.svg)

**数据来源与统计口径：**各模型seed_42/43/44的training_history.csv；实线验证，虚线训练。

新增训练记录与图16的30轮延长训练属于不同实验记录，不能无缝拼接成一条曲线。每个面板保留三个种子，直接观察早期提升、峰值轮次以及训练验证差距。

下表按已存验证Macro-F1定位峰值，仅用于复核历史选择，不重新训练或重新选择测试模型。若训练F1持续增大而验证F1停滞，说明继续拟合训练集并未同步提升验证性能；这比只看训练loss下降更能说明模型选择的必要性。

训练指标计算口径、正则化与增强可能影响训练验证数值差距，因此差距是诊断证据，不是对过拟合原因的唯一解释。

| model | seed | recorded_epochs | best_valid_epoch | best_valid_f1 | last_train_f1 | last_valid_f1 | last_gap |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 42 | 6 | 2.0000 | 0.6105 | 0.9757 | 0.6089 | 0.3668 |
| baseline | 43 | 7 | 3.0000 | 0.6071 | 0.9878 | 0.5893 | 0.3985 |
| baseline | 44 | 7 | 3.0000 | 0.6112 | 0.9793 | 0.5924 | 0.3869 |
| robust_noaug | 42 | 7 | 3.0000 | 0.6218 | 0.9732 | 0.6121 | 0.3612 |
| robust_noaug | 43 | 7 | 3.0000 | 0.6099 | 0.9688 | 0.5963 | 0.3725 |
| robust_noaug | 44 | 6 | 2.0000 | 0.6138 | 0.9550 | 0.5982 | 0.3568 |
| robust | 42 | 8 | 4.0000 | 0.6168 | 0.8878 | 0.5777 | 0.3101 |
| robust | 43 | 7 | 3.0000 | 0.5992 | 0.8670 | 0.5886 | 0.2784 |
| robust | 44 | 8 | 4.0000 | 0.6254 | 0.8858 | 0.6102 | 0.2756 |



## 图35 附件三aligned全量三模型输出

![图35 附件三aligned全量三模型输出](figures/35_attachment3_models.svg)

**数据来源与统计口径：**attachment3_predictions.csv，30个aligned样本×3模型；不是三种子平均。

左图0/1/2对应负面/中性/正面，右图保留每条样本的强度预测。下表列出全部30条结果，使正文图与提交文件逐条可对应。三模型分歧揭示模型选择对专项输出的影响，不能用多数票或高置信度来认定某个模型正确。

新增表没有种子列，因此不能把这180行专项预测理解为三种子集成。aligned是本轮受控训练使用的特征版本，对应结果应作为主分析。unaligned在下一图作为额外输出一致性诊断；题目要求训练与专项测试维持同一特征版本。

无真实标签时，类别分布、置信度和强度范围都是模型输出特征，不是性能评价。

| sample | baseline_class | baseline_intensity | robust_noaug_class | robust_noaug_intensity | robust_class | robust_intensity |
| --- | --- | --- | --- | --- | --- | --- |
| 01 | Negative | -0.8096 | Negative | -0.8102 | Negative | -0.9847 |
| 02 | Neutral | 0.0991 | Neutral | 0.0421 | Neutral | 0.0765 |
| 03 | Neutral | 0.2947 | Neutral | 0.1852 | Positive | 0.4207 |
| 04 | Positive | 0.5085 | Positive | 0.2682 | Positive | 0.5466 |
| 05 | Negative | -0.9567 | Negative | -1.3734 | Negative | -1.4705 |
| 06 | Positive | 0.7996 | Positive | 0.3429 | Positive | 0.5123 |
| 07 | Neutral | 0.2093 | Neutral | 0.0331 | Positive | 0.3047 |
| 08 | Negative | -0.2589 | Negative | -0.4155 | Negative | -0.2386 |
| 09 | Positive | 1.3766 | Positive | 1.3210 | Positive | 1.3531 |
| 10 | Neutral | -0.1650 | Neutral | -0.0500 | Neutral | -0.2476 |
| 11 | Positive | 0.4565 | Positive | 0.3631 | Positive | 0.7179 |
| 12 | Positive | 0.0664 | Neutral | 0.1055 | Positive | 0.5111 |
| 13 | Positive | 0.4985 | Positive | 0.4340 | Positive | 0.7090 |
| 14 | Negative | -0.7477 | Negative | -1.1105 | Negative | -0.7586 |
| 15 | Positive | 0.4696 | Positive | 0.3740 | Positive | 0.5617 |
| 16 | Positive | 1.4138 | Positive | 1.2853 | Positive | 1.3593 |
| 17 | Neutral | 0.1523 | Neutral | -0.2705 | Positive | 0.1442 |
| 18 | Positive | 0.4871 | Positive | 0.3557 | Positive | 0.5194 |
| 19 | Negative | -0.2957 | Negative | -0.4435 | Negative | -0.3030 |
| 20 | Neutral | 0.1390 | Neutral | -0.0103 | Neutral | 0.0482 |
| 21 | Positive | 0.2261 | Positive | 0.3818 | Positive | 0.6464 |
| 22 | Positive | 0.5223 | Positive | 0.4755 | Positive | 0.7022 |
| 23 | Positive | 0.5016 | Positive | 0.3756 | Positive | 0.5236 |
| 24 | Negative | -0.0345 | Negative | -0.1279 | Positive | 0.4084 |
| 25 | Neutral | 0.5216 | Positive | 0.4556 | Positive | 0.4881 |
| 26 | Positive | 0.6479 | Positive | 0.7672 | Positive | 0.8440 |
| 27 | Positive | 0.3574 | Positive | 0.5311 | Positive | 0.5572 |
| 28 | Positive | 0.3680 | Neutral | 0.2624 | Positive | 0.4434 |
| 29 | Positive | 0.5780 | Positive | 0.7346 | Positive | 0.8558 |
| 30 | Positive | 0.1990 | Neutral | 0.1221 | Positive | 0.3656 |



## 图36 附件三类别分布与格式敏感性

![图36 附件三类别分布与格式敏感性](figures/36_attachment3_formats.svg)

**数据来源与统计口径：**180行专项输出；同编号aligned/unaligned描述性配对，每模型30对。

柱状分布可以识别模型是否倾向输出某一极性，一致率与差值则量化输入版本变化的敏感性。两种格式之间的一致率高并不等于准确率高，强度差小也不代表鲁棒性已得到带标签验证。

本轮训练协议为aligned_50，unaligned输出不能替代同版本专项评价。图中A/U只表示输出来源，不能把60条版本记录当作60个独立真实样本。

完整输入缺失结构的统计还显示附件三的有效bin比例低于1，但其中可能混有填充与局部缺失，不能将1减有效比例全部解释为人为缺失率。

| model | agreement | mean_signed_intensity_difference | mean_abs_intensity_difference | max_abs_difference | aligned_counts | unaligned_counts |
| --- | --- | --- | --- | --- | --- | --- |
| baseline | 0.8667 | 0.0989 | 0.2228 | 0.6802 | Counter({'Positive': 17, 'Neutral': 7, 'Negative': 6}) | Counter({'Positive': 21, 'Negative': 5, 'Neutral': 4}) |
| robust_noaug | 0.8000 | 0.0397 | 0.2161 | 0.8441 | Counter({'Positive': 15, 'Neutral': 9, 'Negative': 6}) | Counter({'Positive': 16, 'Negative': 7, 'Neutral': 7}) |
| robust | 0.9667 | 0.0094 | 0.1871 | 0.5710 | Counter({'Positive': 22, 'Negative': 5, 'Neutral': 3}) | Counter({'Positive': 21, 'Negative': 5, 'Neutral': 4}) |



## 图37 问题三基础性能及与问题二参考结果的关系

![图37 问题三基础性能及与问题二参考结果的关系](figures/37_p3_overall.svg)

**数据来源与统计口径：**问题三验证728、测试727条；与旧问题二固定参考的描述性比较。

问题三模型的目标包含解释输出，但可解释性能力不能代替预测准确性。本图同时报告赛题四项指标，数值由预测表重新计算并与metrics.json核对一致。

问题三测试Accuracy为64.10%、Macro-F1为0.5516、MAE为0.6929、Pearson为0.6217；相较问题二固定参考，分类和回归表现均更弱。两者架构、训练与参数不同，不能将性能下降单独归因为“增加解释机制的代价”，也不能把解释输出存在视为可信性已经得到全面证明。

验证和测试应分开叙述：测试准确率略高于验证，但Macro-F1更低，提示类别分布和类别间表现不均衡的影响。下面用混淆矩阵与回归诊断解释这种表面矛盾。

| split | accuracy | macro_f1 | mae | rmse | pearson | bias | delta_accuracy | delta_macro_f1 | delta_mae | delta_pearson |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| valid | 0.6223 | 0.5767 | 0.6613 | 0.8880 | 0.5582 | -0.1327 | -0.0192 | -0.0338 | 0.0878 | -0.1283 |
| test | 0.6410 | 0.5516 | 0.6929 | 0.9141 | 0.6217 | -0.1060 | -0.0770 | -0.1192 | 0.0827 | -0.0780 |



## 图38 问题三的类别瓶颈与错误流向

![图38 问题三的类别瓶颈与错误流向](figures/38_p3_confusion.svg)

**数据来源与统计口径：**问题三全部验证/测试预测；颜色为按真实类别归一的比例，计数见下表。

验证中性类184条只识别53条，召回28.80%；测试中性类158条只识别28条，召回17.72%。测试中性误判中64条流向负面、66条流向正面，说明问题并非单向偏好，而是中性边界整体不足。

测试总误判261条，其中涉及真实或预测中性的错误为171条；直接负面与正面互换为90条。中性预测数量仅69条，明显小于真实158条。这样的混淆结构解释了Accuracy尚可而Macro-F1偏低。

这是一种数值层面的错误归因，不能在没有逐样本素材复核时进一步归因为讽刺、转写错误或表情遮挡。

| split | class | support | correct | precision | recall | f1 | predicted_count |
| --- | --- | --- | --- | --- | --- | --- | --- |
| valid | Negative | 206 | 155 | 0.5871 | 0.7524 | 0.6596 | 264 |
| valid | Neutral | 184 | 53 | 0.5196 | 0.2880 | 0.3706 | 102 |
| valid | Positive | 338 | 245 | 0.6768 | 0.7249 | 0.7000 | 362 |
| test | Negative | 207 | 158 | 0.5788 | 0.7633 | 0.6583 | 273 |
| test | Neutral | 158 | 28 | 0.4058 | 0.1772 | 0.2467 | 69 |
| test | Positive | 362 | 280 | 0.7273 | 0.7735 | 0.7497 | 385 |



## 图39 问题三的强度压缩与强情感误差

![图39 问题三的强度压缩与强情感误差](figures/39_p3_regression.svg)

**数据来源与统计口径：**回归散点保留全部样本；直线为诊断最小二乘拟合，不修改预测；下排按真实绝对强度分层。

真实值越强而预测仍集中在中间区域，反映回归收缩。斜率仅用于描述预测对标签的响应幅度，不是模型训练参数。分层MAE揭示整体均值可能掩盖强情绪样本的大误差。

零强度组与非零组的样本量不同；每个柱标注n，下表同时给出分类准确率，便于检查“极性正确但强度不准”的情况。强度高并不必然分类更困难，符号和幅值应分别评价。

问题二也存在回归压缩，因此它是现有多模态方案共有的结果现象，但不能仅靠散点判定由损失权重、标签噪声或模型容量中的哪一项导致。

| split | abs_true_band | n | mae | bias | accuracy |
| --- | --- | --- | --- | --- | --- |
| valid | 0 | 184 | 0.4816 | 0.0415 | 0.2880 |
| valid | (0,.5] | 147 | 0.4847 | -0.0616 | 0.5850 |
| valid | (.5,1] | 198 | 0.5216 | -0.2004 | 0.7222 |
| valid | (1,2] | 163 | 0.9402 | -0.3093 | 0.8528 |
| valid | (2,3] | 36 | 1.8057 | -0.1410 | 0.8889 |
| test | 0 | 158 | 0.5302 | 0.0188 | 0.1772 |
| test | (0,.5] | 130 | 0.4537 | -0.0133 | 0.5923 |
| test | (.5,1] | 202 | 0.5257 | -0.2266 | 0.7574 |
| test | (1,2] | 188 | 0.9248 | -0.2522 | 0.8777 |
| test | (2,3] | 49 | 1.6512 | 0.3033 | 0.8776 |



## 图40 问题三概率置信度与高置信错误

![图40 问题三概率置信度与高置信错误](figures/40_p3_confidence.svg)

**数据来源与统计口径：**预测表仅保存最大类置信度，使用10个等宽箱；风险曲线每点至少15样本。

验证集ECE=0.0674，测试集ECE=0.0501。置信度≥0.8的错误分别为38与42条。可靠性图能够说明置信水平与实际正确率是否一致，但ECE对分箱有依赖，也不能保证单例可靠。

风险覆盖图描述拒绝低置信样本后保留集合的表现，覆盖率下降意味着更多样本没有得到被接受的预测。这里不基于测试结果选择部署阈值。

由于问题三带标签预测表没有完整三类概率，本报告不生成该模型的逐类ROC/PR、Brier或多类NLL；仅凭最大概率无法还原另外两个类别的概率。

| split | bin_low | n | mean_confidence | accuracy |
| --- | --- | --- | --- | --- |
| valid | 0.3000 | 17 | 0.3737 | 0.3529 |
| valid | 0.4000 | 105 | 0.4547 | 0.5238 |
| valid | 0.5000 | 126 | 0.5476 | 0.5159 |
| valid | 0.6000 | 138 | 0.6498 | 0.5145 |
| valid | 0.7000 | 161 | 0.7530 | 0.7019 |
| valid | 0.8000 | 178 | 0.8441 | 0.7865 |
| valid | 0.9000 | 3 | 0.9023 | 1.0000 |
| test | 0.3000 | 16 | 0.3784 | 0.5000 |
| test | 0.4000 | 107 | 0.4541 | 0.3645 |
| test | 0.5000 | 109 | 0.5410 | 0.5229 |
| test | 0.6000 | 107 | 0.6538 | 0.5701 |
| test | 0.7000 | 160 | 0.7563 | 0.7188 |
| test | 0.8000 | 225 | 0.8475 | 0.8133 |
| test | 0.9000 | 3 | 0.9020 | 1.0000 |



## 图41 问题三50轮训练与早期验证峰值

![图41 问题三50轮训练与早期验证峰值](figures/41_p3_training.svg)

**数据来源与统计口径：**training_history.csv与problem3_metrics.json；保存选择轮为1。

保存模型在第1轮被选中，后续训练没有得到更高的已记录验证Macro-F1。第1轮训练/验证F1为0.5148/0.5767，最后一轮为0.9997/0.5525。训练loss由0.9426变为0.0211，验证loss由0.9414变为3.2540。

后期训练拟合改善而验证停滞说明继续训练并不能保证泛化。报告应使用已保存选中模型的预测，不能把第50轮训练成绩写成最终验证成绩。该问题三实验只提供seed 42，不能引用问题二的三种子波动来替代其训练稳定性证据。

历史中第1轮valid_mae与最终metrics存在约十万分之几量级差异，应以最终保存的逐样本预测为本报告评价口径，不将微小浮点或评价过程差异解释为实质提升。

| epoch | train_loss | valid_loss | train_f1 | valid_f1 | valid_mae |
| --- | --- | --- | --- | --- | --- |
| 1.0000 | 0.9426 | 0.9414 | 0.5148 | 0.5767 | 0.6613 |
| 2.0000 | 0.8019 | 0.9343 | 0.6327 | 0.5522 | 0.6119 |
| 5.0000 | 0.5980 | 1.1032 | 0.7569 | 0.5658 | 0.6232 |
| 10.0000 | 0.2744 | 1.4330 | 0.9085 | 0.5749 | 0.6444 |
| 20.0000 | 0.1020 | 2.5650 | 0.9778 | 0.5381 | 0.6694 |
| 30.0000 | 0.0363 | 3.2364 | 0.9974 | 0.5465 | 0.6229 |
| 40.0000 | 0.0238 | 3.2722 | 0.9995 | 0.5451 | 0.6490 |
| 50.0000 | 0.0211 | 3.2540 | 0.9997 | 0.5525 | 0.6367 |



## 图42 附件四全部预测与置信度

![图42 附件四全部预测与置信度](figures/42_attachment4_predictions.svg)

**数据来源与统计口径：**40行=20个编号×两种版本；颜色为极性，圆大小随置信度变化。

该图逐条呈现附件四的强度和极性，不删除边界样本。表中保留所有预测及主要参考模态，满足全量展示需要；原始详细证据字符串由专项CSV提供。

附件四没有真实标签，因此不可计算准确率，也不能把高置信预测称为正确解释。两种版本对应相同编号，只能作为配对版本记录，不能当作40条独立随机抽样。

问题三训练使用aligned版本，因此aligned输出作为主要结果，unaligned结果作描述性补充。强度靠近零且分类概率不集中的样本应被明确标为边界预测，但本次不修改其类别或引入新的中性阈值。

| variant | id | predicted_label | regression_prediction | confidence | main_modality |
| --- | --- | --- | --- | --- | --- |
| aligned | 01 | Neutral | -0.0065 | 0.4764 | text |
| aligned | 02 | Negative | -0.3556 | 0.5637 | text |
| aligned | 03 | Negative | -0.1991 | 0.4182 | text |
| aligned | 04 | Negative | -1.0964 | 0.7815 | text |
| aligned | 05 | Positive | 0.5637 | 0.6695 | vision |
| aligned | 06 | Positive | 0.6829 | 0.8041 | text |
| aligned | 07 | Positive | 0.5791 | 0.6657 | text |
| aligned | 08 | Positive | 0.3120 | 0.4924 | text |
| aligned | 09 | Negative | -1.2914 | 0.8603 | text |
| aligned | 10 | Negative | -1.0827 | 0.8262 | text |
| aligned | 11 | Negative | -0.8055 | 0.7190 | text |
| aligned | 12 | Negative | -0.7312 | 0.7035 | text |
| aligned | 13 | Neutral | 0.1989 | 0.4897 | text |
| aligned | 14 | Neutral | -0.0605 | 0.4122 | text |
| aligned | 15 | Positive | 1.0111 | 0.8664 | text |
| aligned | 16 | Negative | -1.2827 | 0.8539 | text |
| aligned | 17 | Positive | 0.8121 | 0.7801 | text |
| aligned | 18 | Negative | -0.2612 | 0.4408 | text |
| aligned | 19 | Negative | -0.6003 | 0.7186 | text |
| aligned | 20 | Positive | 0.8128 | 0.8051 | vision |
| unaligned | 01 | Neutral | 0.0707 | 0.4836 | text |
| unaligned | 02 | Negative | -0.3350 | 0.5625 | text |
| unaligned | 03 | Negative | -0.1825 | 0.4004 | text |
| unaligned | 04 | Negative | -1.2227 | 0.8059 | text |
| unaligned | 05 | Positive | 0.7016 | 0.7281 | audio |
| unaligned | 06 | Positive | 0.6717 | 0.8031 | text |
| unaligned | 07 | Positive | 0.5752 | 0.6651 | text |
| unaligned | 08 | Positive | 0.6289 | 0.6943 | text |
| unaligned | 09 | Negative | -1.2952 | 0.8610 | text |
| unaligned | 10 | Negative | -1.1009 | 0.8290 | text |
| unaligned | 11 | Negative | -0.8037 | 0.7186 | text |
| unaligned | 12 | Negative | -0.7203 | 0.7026 | text |
| unaligned | 13 | Neutral | 0.2083 | 0.4876 | text |
| unaligned | 14 | Neutral | -0.0238 | 0.4196 | text |
| unaligned | 15 | Positive | 1.0112 | 0.8586 | audio |
| unaligned | 16 | Negative | -1.2750 | 0.8543 | text |
| unaligned | 17 | Positive | 0.9497 | 0.8419 | text |
| unaligned | 18 | Negative | -0.2564 | 0.4351 | text |
| unaligned | 19 | Negative | -0.6176 | 0.7220 | text |
| unaligned | 20 | Positive | 0.7316 | 0.7644 | text |



## 图43 附件四逐样本三模态作用程度

![图43 附件四逐样本三模态作用程度](figures/43_modality_contributions.svg)

**数据来源与统计口径：**attachment4_explanations.csv，保存的归一化模态作用程度；每行三模态合计约1。

40份版本记录中36份以文本为主要参考模态，语音与视觉各2份。该比例说明此模型在当前样本的决策证据明显偏向文本，不能被描述为三模态贡献均衡。

贡献值是归一化相对量：当其他模态遮挡变化很小时，文本占比可能接近1，但并不意味着文本单独贡献了100%的预测正确性。某模态贡献为0也不能证明它在真实世界没有情绪信息，应结合有符号遮挡变化和归一化规则分析。

本图对每一条记录展示分布，避免均值抹平少数音频/视觉主导案例。主导模态的差异为选取解释卡提供依据，不应只挑选文本主导的成功例子。

| variant | modality | main_count | mean_contribution | median_contribution | zero_count |
| --- | --- | --- | --- | --- | --- |
| aligned | text | 18 | 0.8401 | 0.9686 | 0 |
| aligned | audio | 0 | 0.0759 | 0.0018 | 7 |
| aligned | vision | 2 | 0.0840 | 0.0035 | 9 |
| unaligned | text | 18 | 0.8783 | 0.9877 | 0 |
| unaligned | audio | 2 | 0.0939 | 0.0080 | 6 |
| unaligned | vision | 0 | 0.0278 | 0.0040 | 7 |



## 图44 主要参考模态内50个局部位置的重要性分布

![图44 主要参考模态内50个局部位置的重要性分布](figures/44_local_importance_heatmap.svg)

**数据来源与统计口径：**primary_modality_local_importance_samples.csv，40记录×50bin；T/A/V标注主导模态。

每行展示当前样本主要参考模态的完整50位置重要性；并非三个模态共150位置的全量矩阵。该CSV共2000行，足以覆盖主导模态的全局分布，不能扩称为6000条三模态全位置观测。

正向遮挡下降值经归一化后绘图。深色块表示在该样本内部相对集中，不适合直接比较不同样本的绝对概率下降幅度。下表提供前5位置累计质量、正值bin数量与最高位置，便于量化集中程度。

横轴严格称为bin索引，暂不标注真实秒数。零重要性既可能意味着遮挡没有降低当前预测类别概率，也可能来自负下降值被截为零；这种零值不等于该片段没有信息。

| variant | id | main_modality | sum_importance | positive_bins | top_bin | top5_mass | signed_drop_sum |
| --- | --- | --- | --- | --- | --- | --- | --- |
| aligned | 01 | text | 1.0000 | 10 | 10 | 0.8716 | -0.0706 |
| aligned | 02 | text | 1.0000 | 15 | 18 | 0.7061 | 0.2339 |
| aligned | 03 | text | 1.0000 | 34 | 24 | 0.5367 | 0.2826 |
| aligned | 04 | text | 1.0000 | 26 | 24 | 0.2961 | 0.2881 |
| aligned | 05 | vision | 1.0000 | 15 | 4 | 0.4245 | 0.0945 |
| aligned | 06 | text | 1.0000 | 22 | 24 | 0.4907 | 0.0434 |
| aligned | 07 | text | 1.0000 | 24 | 21 | 0.4730 | 0.0105 |
| aligned | 08 | text | 1.0000 | 13 | 2 | 0.5886 | 0.5283 |
| aligned | 09 | text | 1.0000 | 14 | 18 | 0.5861 | 0.0175 |
| aligned | 10 | text | 1.0000 | 42 | 30 | 0.4522 | 0.1559 |
| aligned | 11 | text | 1.0000 | 12 | 7 | 0.7421 | 0.4325 |
| aligned | 12 | text | 1.0000 | 33 | 41 | 0.3473 | 0.1241 |
| aligned | 13 | text | 1.0000 | 31 | 6 | 0.6700 | -0.0169 |
| aligned | 14 | text | 1.0000 | 19 | 2 | 0.5759 | -0.0435 |
| aligned | 15 | text | 1.0000 | 9 | 20 | 0.7072 | -0.0014 |
| aligned | 16 | text | 1.0000 | 12 | 11 | 0.6979 | 0.0353 |
| aligned | 17 | text | 1.0000 | 26 | 15 | 0.3362 | 0.3453 |
| aligned | 18 | text | 1.0000 | 30 | 29 | 0.5253 | 0.1728 |
| aligned | 19 | text | 1.0000 | 39 | 17 | 0.3213 | 0.2674 |
| aligned | 20 | vision | 1.0000 | 43 | 20 | 0.1603 | 0.0601 |
| unaligned | 01 | text | 1.0000 | 12 | 10 | 0.5944 | 0.0109 |
| unaligned | 02 | text | 1.0000 | 12 | 18 | 0.7421 | 0.1072 |
| unaligned | 03 | text | 1.0000 | 33 | 24 | 0.5467 | 0.2210 |
| unaligned | 04 | text | 1.0000 | 23 | 1 | 0.3699 | 0.0753 |
| unaligned | 05 | audio | 1.0000 | 11 | 9 | 0.6999 | 0.1163 |
| unaligned | 06 | text | 1.0000 | 19 | 24 | 0.5418 | 0.0225 |
| unaligned | 07 | text | 1.0000 | 23 | 21 | 0.4783 | 0.0044 |
| unaligned | 08 | text | 1.0000 | 9 | 2 | 0.8618 | 0.1548 |
| unaligned | 09 | text | 1.0000 | 14 | 18 | 0.5932 | 0.0104 |
| unaligned | 10 | text | 1.0000 | 22 | 11 | 0.5037 | 0.1107 |
| unaligned | 11 | text | 1.0000 | 45 | 7 | 0.7663 | 0.3963 |
| unaligned | 12 | text | 1.0000 | 28 | 41 | 0.3542 | 0.0887 |
| unaligned | 13 | text | 1.0000 | 17 | 6 | 0.6010 | -0.0035 |
| unaligned | 14 | text | 1.0000 | 20 | 2 | 0.5563 | -0.0276 |
| unaligned | 15 | audio | 1.0000 | 50 | 14 | 0.4201 | 0.0178 |
| unaligned | 16 | text | 1.0000 | 46 | 11 | 0.8356 | 0.0173 |
| unaligned | 17 | text | 1.0000 | 26 | 15 | 0.3697 | 0.1945 |
| unaligned | 18 | text | 1.0000 | 29 | 29 | 0.5427 | 0.1480 |
| unaligned | 19 | text | 1.0000 | 28 | 17 | 0.3577 | 0.1971 |
| unaligned | 20 | text | 1.0000 | 20 | 18 | 0.5415 | -0.0179 |



## 图45 按主导模态聚合的局部重要性与样本不均衡

![图45 按主导模态聚合的局部重要性与样本不均衡](figures/45_local_importance_aggregate.svg)

**数据来源与统计口径：**文本36份、音频2份、视觉2份版本记录；淡线为单记录，粗线为均值。

不同模态组的样本量极不平衡。文本曲线可以概括36份记录，但音频和视觉均只有2份，均值高度受个例影响，不能比较曲线形状后宣布某模态具有稳定的时间规律。

bin索引相同并不保证对应同一语义阶段或相同绝对秒数，跨样本平均主要用于描述索引空间中的集中位置。两种版本记录也存在同编号依赖，因此这里不画把40条当独立样本得到的置信区间。

下表的前5位置累计质量越大，说明单样本解释越集中；集中并不自动意味着解释更正确，需要与原始素材的真实定位和遮挡效应共同核验。

| modality | records | mean_top5_mass | mean_positive_bins | peak_of_mean_bin | peak_of_mean_value |
| --- | --- | --- | --- | --- | --- |
| text | 36 | 0.5578 | 23.2500 | 18 | 0.0457 |
| audio | 2 | 0.5600 | 30.5000 | 9 | 0.1212 |
| vision | 2 | 0.2924 | 29.0000 | 4 | 0.0589 |



## 图46 遮挡效应的正负方向与解释强度

![图46 遮挡效应的正负方向与解释强度](figures/46_signed_occlusion.svg)

**数据来源与统计口径：**2000个主导模态局部位置；负值表示遮挡后当前预测类别概率反而升高。

遮挡下降并不总为正。正下降值可作为当前预测的支持证据，负值表示该局部信息对该预测存在抑制或与其他信息交互；不能先截成非负再宣称所有局部片段都支持预测。

左图散点保留全部数值，箱线图不另重复画离群符号；右图给出正、零、负比例。三个模态的样本数与记录数不同，不将2000个位置当成2000个独立样本进行显著性检验。

遮挡是对模型输入的局部干预，因此可检验模型响应；它并不等于自然场景中的因果实验，且全零输入可能离开正常数据分布。报告应称为局部忠实度或模型敏感性证据。

| main_modality | positions | negative_fraction | zero_fraction | positive_fraction | min_drop | median_drop | max_drop |
| --- | --- | --- | --- | --- | --- | --- | --- |
| text | 1800 | 0.2967 | 0.2383 | 0.4650 | -0.0478 | 0.0000 | 0.1122 |
| audio | 100 | 0.3900 | 0.0000 | 0.6100 | -0.0010 | 0.0000 | 0.0210 |
| vision | 100 | 0.3500 | 0.0700 | 0.5800 | -0.0000 | 0.0010 | 0.0082 |



## 图47 注意力权重与遮挡贡献是否一致

![图47 注意力权重与遮挡贡献是否一致](figures/47_attention_occlusion_agreement.svg)

**数据来源与统计口径：**40张解释卡中的modality_attention与modality_contribution；Spearman为描述性秩相关。

两种量都呈现在[0,1]范围内，但含义不同：注意力描述内部加权，遮挡贡献描述输入扰动后的响应。散点接近对角线只能说明两种摘要接近，不代表注意力已被证明是因果解释。

图中逐模态相关反映样本间排序的一致程度；主导模态组高度不均衡、同编号版本成对，因此不报告把40点视为独立观察的显著性p值。尤其对近零模态，微小数值变化可能使秩相关看起来明显，仍应结合绝对差值。

保留两套量有助于识别“模型关注但遮挡影响不大”或“注意力小但干预响应明显”的样本，后续解释卡将展示具体证据。

| modality | n_records | spearman | mean_attention | mean_contribution | mean_abs_difference |
| --- | --- | --- | --- | --- | --- |
| text | 40 | 0.5546 | 0.8547 | 0.8592 | 0.1235 |
| audio | 40 | 0.2760 | 0.0996 | 0.0849 | 0.0931 |
| vision | 40 | 0.4732 | 0.0457 | 0.0559 | 0.0463 |



## 图48 典型解释卡：text主导的aligned_08

![图48 典型解释卡：text主导的aligned_08](figures/48_evidence_text.svg)

**数据来源与统计口径：**真实保存的解释卡；每模态5个候选位置，按原始输出顺序展示。

样本aligned_08，预测Positive，强度0.3120，置信度0.4924；主导模态text。三模态贡献分别为text=1.0000, audio=0.0000, vision=0.0000。

**原始转写：**That brings us to tonight, the Universal Design Grand Challenge

图中直接展示有符号遮挡下降，灰色表示负值。选择此例是为了覆盖主导模态类型，不表示它是准确预测或解释最可靠的样本。文本token只提供模型关注的词元，不能仅因词语看似带情感就断言其解释正确；[CLS]、标点或子词也可能进入候选列表。

语音与视觉图使用bin标注，下表原样保留输出中的秒数与帧号以供核对。现有卡片的duration_seconds与真实素材时长需单独检查，未经核实的秒数不能作为已验证的原视频证据位置。

| modality | bin | token | drop | attention | reported_start_s | reported_end_s | reported_frame_start | reported_frame_end |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| text | 2 | [CLS] | 0.1045 | 0.0520 |  |  |  |  |
| text | 3 | that | 0.0671 | 0.0486 |  |  |  |  |
| text | 12 | us | 0.0449 | 0.0535 |  |  |  |  |
| text | 0 | [CLS] | 0.0357 | 0.0487 |  |  |  |  |
| text | 4 | that | 0.0241 | 0.0474 |  |  |  |  |
| audio | 12 |  | 0.0000 | 0.0297 | 0.2400 | 0.2600 | 1 | 1 |
| audio | 0 |  | 0.0000 | 0.0271 | 0.0000 | 0.0200 | 0 | 0 |
| audio | 2 |  | -0.0214 | 0.0289 | 0.0400 | 0.0600 | 0 | 0 |
| audio | 4 |  | -0.0237 | 0.0264 | 0.0800 | 0.1000 | 0 | 0 |
| audio | 3 |  | -0.0468 | 0.0270 | 0.0600 | 0.0800 | 0 | 0 |
| vision | 12 |  | 0.0000 | 0.0095 | 0.2400 | 0.2600 | 1 | 1 |
| vision | 0 |  | 0.0000 | 0.0086 | 0.0000 | 0.0200 | 0 | 0 |
| vision | 4 |  | -0.0057 | 0.0084 | 0.0800 | 0.1000 | 0 | 0 |
| vision | 3 |  | -0.0060 | 0.0086 | 0.0600 | 0.0800 | 0 | 0 |
| vision | 2 |  | -0.0063 | 0.0092 | 0.0400 | 0.0600 | 0 | 0 |



## 图49 典型解释卡：audio主导的unaligned_05

![图49 典型解释卡：audio主导的unaligned_05](figures/49_evidence_audio.svg)

**数据来源与统计口径：**真实保存的解释卡；每模态5个候选位置，按原始输出顺序展示。

样本unaligned_05，预测Positive，强度0.7016，置信度0.7281；主导模态audio。三模态贡献分别为text=0.0519, audio=0.7644, vision=0.1837。

**原始转写：**Hi, my name is Chloe, video marketer for Red Wagon Marketing.

图中直接展示有符号遮挡下降，灰色表示负值。选择此例是为了覆盖主导模态类型，不表示它是准确预测或解释最可靠的样本。文本token只提供模型关注的词元，不能仅因词语看似带情感就断言其解释正确；[CLS]、标点或子词也可能进入候选列表。

语音与视觉图使用bin标注，下表原样保留输出中的秒数与帧号以供核对。现有卡片的duration_seconds与真实素材时长需单独检查，未经核实的秒数不能作为已验证的原视频证据位置。

| modality | bin | token | drop | attention | reported_start_s | reported_end_s | reported_frame_start | reported_frame_end |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| text | 4 | hi | 0.0017 | 0.0314 |  |  |  |  |
| text | 12 | name | -0.0049 | 0.0337 |  |  |  |  |
| text | 13 | name | -0.0076 | 0.0349 |  |  |  |  |
| text | 14 | is | -0.0124 | 0.0317 |  |  |  |  |
| text | 8 | my | -0.0126 | 0.0348 |  |  |  |  |
| audio | 4 |  | 0.0171 | 0.0261 | 0.0800 | 0.1000 | 0 | 0 |
| audio | 8 |  | 0.0087 | 0.0289 | 0.1600 | 0.1800 | 1 | 1 |
| audio | 13 |  | 0.0000 | 0.0290 | 0.2600 | 0.2800 | 1 | 1 |
| audio | 12 |  | 0.0000 | 0.0279 | 0.2400 | 0.2600 | 1 | 1 |
| audio | 14 |  | 0.0000 | 0.0263 | 0.2800 | 0.3000 | 1 | 2 |
| vision | 4 |  | 0.0033 | 0.0049 | 0.0800 | 0.1000 | 0 | 0 |
| vision | 8 |  | 0.0029 | 0.0054 | 0.1600 | 0.1800 | 1 | 1 |
| vision | 13 |  | 0.0000 | 0.0055 | 0.2600 | 0.2800 | 1 | 1 |
| vision | 12 |  | 0.0000 | 0.0053 | 0.2400 | 0.2600 | 1 | 1 |
| vision | 14 |  | 0.0000 | 0.0049 | 0.2800 | 0.3000 | 1 | 2 |



## 图50 典型解释卡：vision主导的aligned_05

![图50 典型解释卡：vision主导的aligned_05](figures/50_evidence_vision.svg)

**数据来源与统计口径：**真实保存的解释卡；每模态5个候选位置，按原始输出顺序展示。

样本aligned_05，预测Positive，强度0.5637，置信度0.6695；主导模态vision。三模态贡献分别为text=0.2650, audio=0.1024, vision=0.6326。

**原始转写：**Hi, my name is Chloe, video marketer for Red Wagon Marketing.

图中直接展示有符号遮挡下降，灰色表示负值。选择此例是为了覆盖主导模态类型，不表示它是准确预测或解释最可靠的样本。文本token只提供模型关注的词元，不能仅因词语看似带情感就断言其解释正确；[CLS]、标点或子词也可能进入候选列表。

语音与视觉图使用bin标注，下表原样保留输出中的秒数与帧号以供核对。现有卡片的duration_seconds与真实素材时长需单独检查，未经核实的秒数不能作为已验证的原视频证据位置。

| modality | bin | token | drop | attention | reported_start_s | reported_end_s | reported_frame_start | reported_frame_end |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| text | 4 | hi | 0.0106 | 0.0419 |  |  |  |  |
| text | 12 | name | 0.0032 | 0.0464 |  |  |  |  |
| text | 13 | name | -0.0024 | 0.0464 |  |  |  |  |
| text | 6 | , | -0.0055 | 0.0431 |  |  |  |  |
| text | 8 | my | -0.0134 | 0.0483 |  |  |  |  |
| audio | 6 |  | 0.0022 | 0.0100 | 0.1200 | 0.1400 | 1 | 1 |
| audio | 4 |  | 0.0016 | 0.0097 | 0.0800 | 0.1000 | 0 | 0 |
| audio | 8 |  | 0.0015 | 0.0112 | 0.1600 | 0.1800 | 1 | 1 |
| audio | 13 |  | -0.0025 | 0.0107 | 0.2600 | 0.2800 | 1 | 1 |
| audio | 12 |  | -0.0075 | 0.0107 | 0.2400 | 0.2600 | 1 | 1 |
| vision | 4 |  | 0.0082 | 0.0107 | 0.0800 | 0.1000 | 0 | 0 |
| vision | 6 |  | 0.0072 | 0.0110 | 0.1200 | 0.1400 | 1 | 1 |
| vision | 12 |  | 0.0072 | 0.0118 | 0.2400 | 0.2600 | 1 | 1 |
| vision | 8 |  | 0.0064 | 0.0123 | 0.1600 | 0.1800 | 1 | 1 |
| vision | 13 |  | 0.0039 | 0.0118 | 0.2600 | 0.2800 | 1 | 1 |



## 图51 附件四预测与解释的跨版本一致性

![图51 附件四预测与解释的跨版本一致性](figures/51_attachment4_formats.svg)

**数据来源与统计口径：**20个编号配对；矩阵行为aligned、列为unaligned。

类别一致20/20，主导模态一致17/20，强度差绝对值均值0.0520。同一个样本的预测一致不保证解释一致，反之亦然，两个层面应分别报告。

对角线外的主导模态转移揭示解释对输入版本的敏感性。但两种版本不是受控的单一因素扰动，不应据此量化某个模态缺失的因果效应。

题目要求保持训练与专项测试输入版本一致，因此本图用于界定额外输出的稳定性边界，不能用于在无标签专项集上选出所谓更准确版本。

| id | aligned_class | unaligned_class | same_class | aligned_main | unaligned_main | same_main | intensity_difference |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 01 | Neutral | Neutral | 1 | text | text | 1 | 0.0772 |
| 02 | Negative | Negative | 1 | text | text | 1 | 0.0206 |
| 03 | Negative | Negative | 1 | text | text | 1 | 0.0167 |
| 04 | Negative | Negative | 1 | text | text | 1 | -0.1263 |
| 05 | Positive | Positive | 1 | vision | audio | 0 | 0.1379 |
| 06 | Positive | Positive | 1 | text | text | 1 | -0.0111 |
| 07 | Positive | Positive | 1 | text | text | 1 | -0.0039 |
| 08 | Positive | Positive | 1 | text | text | 1 | 0.3168 |
| 09 | Negative | Negative | 1 | text | text | 1 | -0.0038 |
| 10 | Negative | Negative | 1 | text | text | 1 | -0.0182 |
| 11 | Negative | Negative | 1 | text | text | 1 | 0.0017 |
| 12 | Negative | Negative | 1 | text | text | 1 | 0.0109 |
| 13 | Neutral | Neutral | 1 | text | text | 1 | 0.0093 |
| 14 | Neutral | Neutral | 1 | text | text | 1 | 0.0367 |
| 15 | Positive | Positive | 1 | text | audio | 0 | 0.0001 |
| 16 | Negative | Negative | 1 | text | text | 1 | 0.0077 |
| 17 | Positive | Positive | 1 | text | text | 1 | 0.1376 |
| 18 | Negative | Negative | 1 | text | text | 1 | 0.0047 |
| 19 | Negative | Negative | 1 | text | text | 1 | -0.0173 |
| 20 | Positive | Positive | 1 | vision | text | 0 | -0.0813 |



## 图52 解释位置的物理时间可核验性

![图52 解释位置的物理时间可核验性](figures/52_evidence_time_consistency.svg)

**数据来源与统计口径：**只读取原始MP4容器mvhd时长，与40张解释卡的duration_seconds比较；未重跑推理。

全部40张解释卡保存的duration_seconds均为1.0秒，而本地视频容器时长范围为3.700至24.134秒。当前结果的语音秒数和视觉帧号由此不能直接视为已经验证的原素材定位。这个差异是保存成果之间的实证不一致，不涉及代码审核。

题目问题三要求关键证据可对应原始文本、语音时段或视觉关键帧。现有结果支持bin级局部重要性和token级候选证据，但当前秒数/帧号对应需要额外核验。仅将bin按视频总时长线性拉伸也不能证明正确，因为aligned特征位置未必是等时间采样；本报告不擅自修正。

因此应把“已输出解释”和“原始素材定位已验证”区分开。图48至50用bin作主图标签，同时保留原秒数供复核，避免把有问题的时间映射再包装成确定的科研结论。

| variant | id | reported_duration_s | container_duration_s | ratio_reported_to_container |
| --- | --- | --- | --- | --- |
| aligned | 01 | 1.0000 | 9.0000 | 0.1111 |
| aligned | 02 | 1.0000 | 3.7000 | 0.2703 |
| aligned | 03 | 1.0000 | 16.0000 | 0.0625 |
| aligned | 04 | 1.0000 | 9.3000 | 0.1075 |
| aligned | 05 | 1.0000 | 6.1670 | 0.1622 |
| aligned | 06 | 1.0000 | 9.8000 | 0.1020 |
| aligned | 07 | 1.0000 | 24.1340 | 0.0414 |
| aligned | 08 | 1.0000 | 8.0670 | 0.1240 |
| aligned | 09 | 1.0000 | 11.3000 | 0.0885 |
| aligned | 10 | 1.0000 | 10.9340 | 0.0915 |
| aligned | 11 | 1.0000 | 11.4240 | 0.0875 |
| aligned | 12 | 1.0000 | 20.3670 | 0.0491 |
| aligned | 13 | 1.0000 | 8.6800 | 0.1152 |
| aligned | 14 | 1.0000 | 13.9000 | 0.0719 |
| aligned | 15 | 1.0000 | 7.3340 | 0.1364 |
| aligned | 16 | 1.0000 | 5.5730 | 0.1794 |
| aligned | 17 | 1.0000 | 14.0000 | 0.0714 |
| aligned | 18 | 1.0000 | 22.9670 | 0.0435 |
| aligned | 19 | 1.0000 | 18.7340 | 0.0534 |
| aligned | 20 | 1.0000 | 18.7670 | 0.0533 |
| unaligned | 01 | 1.0000 | 9.0000 | 0.1111 |
| unaligned | 02 | 1.0000 | 3.7000 | 0.2703 |
| unaligned | 03 | 1.0000 | 16.0000 | 0.0625 |
| unaligned | 04 | 1.0000 | 9.3000 | 0.1075 |
| unaligned | 05 | 1.0000 | 6.1670 | 0.1622 |
| unaligned | 06 | 1.0000 | 9.8000 | 0.1020 |
| unaligned | 07 | 1.0000 | 24.1340 | 0.0414 |
| unaligned | 08 | 1.0000 | 8.0670 | 0.1240 |
| unaligned | 09 | 1.0000 | 11.3000 | 0.0885 |
| unaligned | 10 | 1.0000 | 10.9340 | 0.0915 |
| unaligned | 11 | 1.0000 | 11.4240 | 0.0875 |
| unaligned | 12 | 1.0000 | 20.3670 | 0.0491 |
| unaligned | 13 | 1.0000 | 8.6800 | 0.1152 |
| unaligned | 14 | 1.0000 | 13.9000 | 0.0719 |
| unaligned | 15 | 1.0000 | 7.3340 | 0.1364 |
| unaligned | 16 | 1.0000 | 5.5730 | 0.1794 |
| unaligned | 17 | 1.0000 | 14.0000 | 0.0714 |
| unaligned | 18 | 1.0000 | 22.9670 | 0.0435 |
| unaligned | 19 | 1.0000 | 18.7340 | 0.0534 |
| unaligned | 20 | 1.0000 | 18.7670 | 0.0533 |



## 图53 问题一新增汇总中的完整性与可追溯性

![图53 问题一新增汇总中的完整性与可追溯性](figures/53_p1_completeness.svg)

**数据来源与统计口径：**problem1_complete_analysis.json及problem1_validation.json；已有核验记录，非本次重跑提取。

三模态各100条样本具有预期形状且数值有限，逐模态汇总有300行，manifest有100行。这支持样本与特征的数量完整性；它不等于特征具备充分情感辨识能力，也不等于时序定位已经经过人工真值验证。

全部模态统一50位置，文本/音频/视觉有效bin为3670/4966/5000。固定存储尺寸与有效信息长度不同，模型应通过掩码区分填充。音视频所用74/35维描述并不因维度与附件二一致，就自动拥有相同物理含义或相同特征分布。

当前保存时长2.260–29.290秒与题面给出的原视频2.648–34.567秒范围不同。本文沿用已有结果的时长字段，不据此推断样本删减或擅自改数；在最终论文中应说明该字段与原视频容器时长的关系。附录提供全部100条样本的逐项统计。

| modality | samples_with_expected_shape | samples_with_finite_values | feature_dim | valid_bins | source_intervals |
| --- | --- | --- | --- | --- | --- |
| text | 100 | 100 | 768 | 3670 | 2447 |
| audio | 100 | 100 | 74 | 4966 | 77510 |
| vision | 100 | 100 | 35 | 5000 | 3936 |

