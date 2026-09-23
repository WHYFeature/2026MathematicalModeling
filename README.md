# E Problem Data Directory

This directory is a normalized copy of the supplied E-problem data. The original `./E题数据` directory is preserved unchanged. Only directory entries and file names were used during organization; PKL, XLSX, and MP4 contents were not inspected.

## Directory structure

```text
DATA/
├── attachment_1_raw_samples/
│   ├── labels_100.xlsx
│   └── videos/<video_id>/<clip_id>.mp4
├── attachment_2_standard_features/
│   ├── aligned_50.pkl
│   ├── unaligned_50.pkl
│   └── labels.xlsx
├── attachment_3_missing_modality/
│   ├── aligned/sample_01.pkl ... sample_30.pkl
│   └── unaligned/sample_01.pkl ... sample_30.pkl
└── attachment_4_explainability/
    ├── aligned/
    │   ├── features/sample_01.pkl ... sample_20.pkl
    │   └── videos/sample_01.mp4 ... sample_20.mp4
    └── unaligned/
        ├── features/sample_01.pkl ... sample_20.pkl
        └── videos/sample_01.mp4 ... sample_20.mp4
```

## Name mapping

| Original path or name | Organized path or name | Notes |
|---|---|---|
| `./E题数据/E题数据/` | `./DATA/` | Removed the duplicated dataset wrapper directory. |
| `附件1-数据集原始多模态样本/` | `attachment_1_raw_samples/` | Raw samples for Question 1. |
| `MOSEI数据集部分原始视频-100条/` | `attachment_1_raw_samples/` | Removed the extra descriptive wrapper level. |
| `<video_id>/<clip_id>.mp4` | `videos/<video_id>/<clip_id>.mp4` | IDs remain unchanged because they are official sample identifiers and already ASCII-safe. |
| `label-100.xlsx` | `labels_100.xlsx` | Labels for the 100 raw samples. |
| `附件2-数据集特征文件/` | `attachment_2_standard_features/` | Standard training, validation, and test features. |
| `aligned_50.pkl` | `aligned_50.pkl` | Name already ASCII-safe; unchanged. |
| `unaligned_50.pkl` | `unaligned_50.pkl` | Name already ASCII-safe; unchanged. |
| `label.xlsx` | `labels.xlsx` | Standardized plural label-file name. |
| `附件3-模态缺失特征样本/` | `attachment_3_missing_modality/` | Missing-modality task data. |
| `对齐版本/` | `aligned/` | Aligned feature variant. |
| `未对齐版本/` | `unaligned/` | Unaligned feature variant. |
| `附件3_01.pkl` ... `附件3_30.pkl` | `aligned/sample_01.pkl` ... `sample_30.pkl` | Normalized fixed-width sample names. |
| `附件3_未对齐版本_01.pkl` ... `_30.pkl` | `unaligned/sample_01.pkl` ... `sample_30.pkl` | Variant is represented by the parent directory. |
| `附件4-可解释专项视频样本与特征文件/附件4-可解释专项视频样本与特征文件/` | `attachment_4_explainability/` | Removed the duplicated attachment wrapper directory. |
| Attachment 4 `对齐版本/` | `attachment_4_explainability/aligned/` | Aligned explainability samples. |
| Attachment 4 `未对齐版本/` | `attachment_4_explainability/unaligned/` | Unaligned explainability samples. |
| Attachment 4 `01.pkl` ... `20.pkl` | `features/sample_01.pkl` ... `sample_20.pkl` | Explicitly separated features from videos. |
| Attachment 4 `videos/01.mp4` ... `20.mp4` | `videos/sample_01.mp4` ... `sample_20.mp4` | Matches the corresponding normalized feature name. |
| `.DS_Store` | Not copied | OS metadata; not part of the dataset. |

## 文件名映射检查结果

上述映射表已经覆盖本次整理中的全部改名类型：

- 两层重复的 `E题数据` 根目录；
- 附件1至附件4的中文目录名；
- 附件1中额外的“100条原始视频”包装目录；
- 附件1和附件2的标签文件名；
- 附件3的对齐、未对齐目录及两类PKL文件名；
- 附件4的重复包装目录、对齐/未对齐目录、特征文件名和视频文件名；
- 不属于数据集的 `.DS_Store` 文件。

整理后 `DATA` 内不存在中文文件名或中文目录名。附件1的 `video_id` 目录名和 `clip_id.mp4` 文件名没有修改，因为它们是赛题定义的正式样本标识，且本身只包含ASCII字符。附件3、附件4使用父目录表示数据版本，因此文件统一命名为固定宽度的 `sample_XX.pkl` 或 `sample_XX.mp4`。

## File counts after organization

| Directory | Expected files |
|---|---:|
| `attachment_1_raw_samples` | 100 MP4 and 1 XLSX |
| `attachment_2_standard_features` | 2 PKL and 1 XLSX |
| `attachment_3_missing_modality` | 60 PKL |
| `attachment_4_explainability` | 40 PKL and 40 MP4 |

Total: 244 data files, plus this README.

## Experimental use

- Keep `aligned` and `unaligned` experiments separate. Use the same variant for Attachment 2 training/validation and the corresponding Attachment 3 or Attachment 4 inference data.
- Attachment 3 and Attachment 4 are task-specific inference sets and should not be merged into training data.
- For Attachment 1, use the original `video_id` and `clip_id` as the stable identifiers. A convenient logical sample ID is `<video_id>$_$<clip_id>`.
- Do not write predictions or processed features back into this directory. Store generated artifacts in separate experiment/result directories.

## 使用整理脚本

整理脚本位于项目根目录的 `./organize_data.py`。脚本只使用Python标准库，不需要安装额外依赖；它只检查文件名和目录结构，不会反序列化PKL，也不会读取XLSX或MP4的内容。

### 准备目录

推荐将脚本、原始数据目录和输出目录放在同一个工作目录下：

```text
project/
├── organize_data.py
├── E题数据/                 # 原始数据，可以包含重复的 E题数据/E题数据 层级
└── DATA/                    # 脚本生成的规范化副本
```

原始数据目录名称或位置不同时，不需要修改代码，只需通过 `--source` 指定实际路径。目标目录通过 `--target` 指定。两个参数都支持相对路径和绝对路径，但为了方便项目迁移，建议优先使用相对路径。

### 默认运行命令

在项目根目录执行：

```bash
python organize_data.py --source "./E题数据" --target "./DATA"
```

在部分Windows环境中，也可以使用：

```powershell
py .\organize_data.py --source ".\E题数据" --target ".\DATA"
```

### 数据位于其他位置

如果其他人的原始数据目录位于项目的上一级目录：

```bash
python organize_data.py --source "../downloaded_e_problem_data" --target "./DATA"
```

也可以使用任意自定义目标目录进行试运行：

```bash
python organize_data.py --source "../downloaded_e_problem_data" --target "./DATA_TEST"
```

### 参数说明

| 参数 | 默认值 | 含义 |
|---|---|---|
| `--source` | `./E题数据` | 赛题原始数据所在目录。脚本会自动识别其内部重复的包装层级。 |
| `--target` | `./DATA` | 规范化数据输出目录。 |

查看命令帮助：

```bash
python organize_data.py --help
```

### 脚本执行规则

1. 自动寻找同时包含附件1、附件2、附件3和附件4的实际数据根目录。
2. 创建英文目录层级并按本README中的映射规则复制、重命名文件。
3. 保留原始数据，不执行移动或删除操作。
4. 忽略 `.DS_Store`。
5. 拒绝覆盖已经存在的数据文件。目标目录必须是新目录，或只包含本 `README.md`。
6. 完成后核对四个附件的文件数量；验证通过时输出 `Validation passed`。

脚本成功执行时，终端会显示识别到的源数据根目录、目标目录和验证结果。如果目标目录已经包含整理后的数据，脚本会终止而不是覆盖文件；需要重复实验时，应指定一个新的目标目录。

## Git说明

项目根目录的 `.gitignore` 使用 `/DATA/` 忽略整个本地数据目录，因此这里的原始数据副本和本README都不会进入Git提交。`organize_data.py` 位于项目根目录，不受该规则影响，可以正常提交，其他使用者可用它从各自的原始数据生成本地 `DATA`。
