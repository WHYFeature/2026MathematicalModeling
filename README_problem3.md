# Problem 3 Explainable Multimodal Emotion Prediction

问题三使用附件二的 `aligned_50.pkl` 训练和验证可解释三模态模型，再对附件四的 aligned/unaligned 样本分别推理。附件四无标签，因此只输出预测和可复核解释，不报告准确率。

模型由三部分组成：每个模态独立的投影和时序 Transformer；三模态摘要的跨模态注意力和模态门控；融合后的时间注意力池化。模型同时输出情感极性和情感强度。解释包含两层：注意力给出模态/时间的候选证据，局部遮挡重新推理得到 predicted-class probability drop，作为证据复核分数。

文本证据映射到 BERT token；音频证据映射到视频时间区间；视觉证据映射到 5 FPS 的帧区间。每个附件四样本会保存一个 JSON evidence card，并汇总到 CSV。

先做 smoke test：

```powershell
Set-Location 'D:\E_math'
.\run_problem3.ps1 `
  -Epochs 1 -BatchSize 256 -Seed 42 `
  -SkipAttachment4 -OutputRoot 'D:\E_math\problem3_smoke'
```

正式运行：

```powershell
Set-Location 'D:\E_math'
.\run_problem3.ps1 `
  -Epochs 20 -BatchSize 64 -Seed 42 `
  -OutputRoot 'D:\E_math\problem3_outputs'
```

输出：

```text
problem3_outputs/
  problem3_checkpoint.pt
  problem3_metrics.json
  normalization.json
  training_history.json
  training_history.csv
  problem3_valid_predictions.csv
  problem3_test_predictions.csv
  attachment4_explanations.csv
  attachment4_explanations.json
  evidence_cards/aligned_01.json ... unaligned_20.json
  figures/training_curves.png/.pdf
  figures/validation_confusion.png/.pdf
  figures/modality_contributions.png/.pdf
  figures/temporal_importance.png/.pdf
  figures/primary_modality_local_importance.png/.pdf
  figures/validation_error_diagnostics.png/.pdf
  validation_error_analysis.csv
  validation_error_summary.json
  primary_modality_local_importance.csv
  primary_modality_local_importance_samples.csv
```

若训练结果已经存在，只需补生成错误归因和完整局部遮挡分布，无需重新训练：

```powershell
Set-Location 'D:\E_math'
.un_problem3_analysis.ps1 -OutputRoot 'D:\E_math\problem3_outputs'
```

该后处理会读取 `problem3_checkpoint.pt`，对附件四每个样本的 3 个模态、50 个时间 bin 逐一遮挡，记录预测类别概率下降值。`primary_modality_local_importance_samples.csv` 保存样本级完整分布，`primary_modality_local_importance.csv` 和对应 PNG/PDF 按主要参考模态聚合。验证集错误归因同时输出逐样本 CSV、汇总 JSON 和诊断图。

正文中应报告验证集 Accuracy、Macro-F1、逐类 F1、MAE 和 Pearson；附件四只能报告预测类别、强度、主要参考模态、作用程度和关键证据位置。`occlusion_drop` 越大表示遮挡该局部片段后预测类别概率下降越明显，解释卡中同时保留原始注意力值和遮挡变化，避免把注意力单独当作因果证明。
