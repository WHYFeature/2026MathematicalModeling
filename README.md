# E题数据目录说明

`./DATA` 是赛题E题原始数据的规范化副本，原始的 `./E题数据` 目录保持不变。整理过程只检查目录结构和文件名，没有读取PKL、XLSX或MP4的具体内容。

## 目录结构

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

## 文件名映射

| 原路径或名称 | 整理后的路径或名称 | 说明 |
|---|---|---|
| `./E题数据/E题数据/` | `./DATA/` | 消除重复的数据集包装目录。 |
| `附件1-数据集原始多模态样本/` | `attachment_1_raw_samples/` | 问题1使用的原始样本。 |
| `MOSEI数据集部分原始视频-100条/` | `attachment_1_raw_samples/` | 消除额外的描述性包装层级。 |
| `<video_id>/<clip_id>.mp4` | `videos/<video_id>/<clip_id>.mp4` | `video_id` 和 `clip_id` 是正式样本标识且已兼容ASCII，因此保持不变。 |
| `label-100.xlsx` | `labels_100.xlsx` | 100条原始样本的标签文件。 |
| `附件2-数据集特征文件/` | `attachment_2_standard_features/` | 标准训练、验证和测试特征。 |
| `aligned_50.pkl` | `aligned_50.pkl` | 原名称已兼容ASCII，保持不变。 |
| `unaligned_50.pkl` | `unaligned_50.pkl` | 原名称已兼容ASCII，保持不变。 |
| `label.xlsx` | `labels.xlsx` | 将标签文件名统一为复数形式。 |
| `附件3-模态缺失特征样本/` | `attachment_3_missing_modality/` | 模态缺失专项任务数据。 |
| `对齐版本/` | `aligned/` | 对齐特征版本。 |
| `未对齐版本/` | `unaligned/` | 未对齐特征版本。 |
| `附件3_01.pkl` ... `附件3_30.pkl` | `aligned/sample_01.pkl` ... `sample_30.pkl` | 统一为固定宽度的样本名称。 |
| `附件3_未对齐版本_01.pkl` ... `_30.pkl` | `unaligned/sample_01.pkl` ... `sample_30.pkl` | 数据版本由父目录表示。 |
| `附件4-可解释专项视频样本与特征文件/附件4-可解释专项视频样本与特征文件/` | `attachment_4_explainability/` | 消除重复的附件4包装目录。 |
| 附件4 `对齐版本/` | `attachment_4_explainability/aligned/` | 对齐版可解释性样本。 |
| 附件4 `未对齐版本/` | `attachment_4_explainability/unaligned/` | 未对齐版可解释性样本。 |
| 附件4 `01.pkl` ... `20.pkl` | `features/sample_01.pkl` ... `sample_20.pkl` | 将特征文件与视频文件明确分开存放。 |
| 附件4 `videos/01.mp4` ... `20.mp4` | `videos/sample_01.mp4` ... `sample_20.mp4` | 与对应的规范化特征文件保持同名。 |
| `.DS_Store` | 不复制 | 操作系统元数据，不属于赛题数据。 |

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

## 整理后的文件数量

| 目录 | 预期文件数 |
|---|---:|
| `attachment_1_raw_samples` | 100个MP4和1个XLSX |
| `attachment_2_standard_features` | 2个PKL和1个XLSX |
| `attachment_3_missing_modality` | 60个PKL |
| `attachment_4_explainability` | 40个PKL和40个MP4 |

共244个数据文件。本README位于项目根目录，不计入 `DATA` 的文件数量。

## 实验使用说明

- `aligned` 和 `unaligned` 实验必须分开。附件2的训练、验证数据应与附件3或附件4的推理数据使用同一种特征版本。
- 附件3和附件4是专项推理数据，不能并入训练集。
- 附件1应使用原始的 `video_id` 和 `clip_id` 作为稳定标识，推荐使用 `<video_id>$_$<clip_id>` 作为逻辑样本ID。
- 不要将预测结果或二次处理特征写回 `DATA`，应将实验生成物存放在独立的实验或结果目录中。

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

项目根目录的 `.gitignore` 使用 `/DATA/` 忽略整个本地数据目录，因此整理后的数据副本不会进入Git提交。本README和 `organize_data.py` 都位于项目根目录，不受该规则影响，可以正常提交。其他使用者克隆项目后，可根据本README运行脚本，从各自的原始数据生成本地 `DATA`。
