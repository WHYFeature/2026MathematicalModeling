# 问题1代码说明

`problem1_pipeline.py` 用附件1的 100 个原始视频和 `labels_100.xlsx` 生成问题1的三模态时序特征。

Windows 默认路径：

```powershell
Set-Location D:\E_math
.\setup_problem1.ps1
.\run_problem1.ps1 -SmokeTest
```

如果 PowerShell 阻止本地脚本执行，只对当前终端临时放开：

```powershell
Set-ExecutionPolicy -Scope Process Bypass
```

`setup_problem1.ps1` 会创建 `D:\E_math\.venv_problem1`，安装 NumPy、OpenPyXL、imageio-ffmpeg、CUDA 版 PyTorch 2.11、匹配的 TorchAudio 2.11 和 Transformers。脚本针对 NVIDIA 驱动支持的 CUDA 12.6 选择 GPU wheel；`imageio-ffmpeg` 会提供 FFmpeg，如果机器已有 FFmpeg，也可以直接使用 PATH 中的版本。正式的 CTC 强制对齐还需要 TorchAudio 的 `WAV2VEC2_ASR_BASE_960H` 权重；首次正式运行时 TorchAudio 可能从 PyTorch 模型源下载该权重，之后会使用本机缓存。

正式运行需要先把 BERT 模型缓存到项目目录。下面命令会访问 Hugging Face 下载一次模型；如果网络受限，可以提前把同等目录复制到 `D:\E_math\models\bert-base-uncased`：

```powershell
$Py = "D:\E_math\.venv_problem1\Scripts\python.exe"
$env:HF_ENDPOINT = "https://hf-mirror.com"
$env:HF_HUB_DISABLE_XET = "1"
$env:HF_HUB_DOWNLOAD_TIMEOUT = "120"
& $Py -c "from transformers import AutoTokenizer, AutoModel; p=r'D:\E_math\models\bert-base-uncased'; src='google-bert/bert-base-uncased'; AutoTokenizer.from_pretrained(src).save_pretrained(p); AutoModel.from_pretrained(src).save_pretrained(p)"
```

模型会直接保存为 `D:\E_math\models\bert-base-uncased`，然后运行：

```powershell
.\run_problem1.ps1 -Limit 1 -Overwrite
```

确认单个样本无误后运行全部 100 条：

```powershell
.\run_problem1.ps1 -Overwrite
```

最终竞赛结果不能使用 `hash_fallback`，也不要把 `--text-align uniform` 当作最终时间对齐结果；这两者只适合接口冒烟或依赖未准备好的临时检查。正式运行必须使用本地 BERT 模型和默认的 CTC 强制对齐。

输出目录默认为 `D:\E_math\problem1_outputs`，包括：

```text
problem1_aligned_features.pkl  # 100条样本的50步对齐特征
manifest.csv                   # 样本、时长、有效长度和后端清单
run_metadata.json              # 运行参数、版本和对齐规则
```

输出 PKL 的每条样本包含：

```text
id, video_id, clip_id, raw_text
text       (50, 768)
text_bert (3, 50)
audio      (50, 74)
vision     (50, 35)
valid_masks
time_edges (51,)
duration_seconds, label, annotation
```

文本标签文件没有词级时间戳。正式运行默认用 TorchAudio CTC 模型将给定转写强制对齐到音频，输出 `text_word_spans`；`text_alignment=ctc` 会写入 `run_metadata.json`。如果只做接口冒烟，可显式加 `--text-align uniform`，但该模式只是均匀时间先验，不应作为论文最终的时间对齐结果。

不要把生成的特征、模型权重或视频写回 `DATA`；`problem1_outputs` 应作为独立实验结果目录。
