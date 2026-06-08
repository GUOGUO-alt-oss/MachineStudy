# 实验5-1：TensorFlow 模型生成

## 实验目标

根据实验要求，使用 **TensorFlow/Keras + TensorFlow Lite Converter** 方案训练花卉图像分类模型，生成可部署到 Android 端的 `.tflite` 文件，并用实验四的 APP 进行验证。

---

## 教程与材料

| 材料 | 说明 |
|------|------|
| 实验指导 PDF | `11_实验5_1_TensorFlow模型生成.pdf`（6页） |
| 教程 | [花卉图片分类器：Keras 训练并导出 TFLite](https://blog.csdn.net/llfjfz/article/details/161630612) |
| 废弃方案 | TensorFlow Lite Model Maker（依赖不兼容新版 Python，已放弃） |

---

## 实验环境

| 项目 | 配置 |
|------|------|
| Python | 3.13.3（系统 Python） |
| TensorFlow | 2.21.0 |
| Matplotlib | 3.10.9 |
| NumPy | 2.3.5 |
| Jupyter | Jupyter Notebook / nbconvert |
| 运行平台 | Windows 11 本地 |

---

## 实验步骤

### 步骤 1：环境准备

将教程中的依赖安装到系统 Python：

```bash
pip install tensorflow matplotlib numpy jupyter
```

### 步骤 2：创建 Jupyter Notebook

创建 `train_flower_model.ipynb`，包含 10 个单元格（1 个 Markdown + 9 个代码），严格按照教程的 6 步流程编写。

### 步骤 3：加载数据集

使用 `load_flower_datasets()` 函数自动下载 TensorFlow 官方 `flower_photos.tgz` 数据集：

| 类别 | 图片数量 | 
|------|----------|
| daisy | 633 |
| dandelion | 898 |
| roses | 641 |
| sunflowers | 699 |
| tulips | 799 |
| **总计** | **3670** |

数据集划分：
- 训练集：2936 张（80%）
- 验证集：约 367 张（10%）
- 测试集：约 367 张（10%）

> **遇到的问题**：境内直连 `storage.googleapis.com` 出现 SSL EOF 错误。解决方式：手动用 SSL bypass 下载 228MB 压缩包到 Keras 缓存目录 `~/.keras/datasets/`，并预解压。

### 步骤 4：构建模型

使用 **MobileNetV2 迁移学习**：

```
Input(224×224×3)
  → MobileNetV2.preprocess_input()     ← 像素值归一化
  → MobileNetV2(include_top=False)      ← ImageNet 预训练权重，冻结参数
  → GlobalAveragePooling
  → Dropout(0.2)
  → Dense(5, softmax)                   ← 新增分类头
```

训练参数：
- Epochs: 5
- Batch Size: 32
- Image Size: 224×224
- Learning Rate: 0.001（Adam 优化器）
- Loss: SparseCategoricalCrossentropy

### 步骤 5：训练与评估

5 个 epoch 在 CPU 上完成（约 6-8 分钟）：

| Epoch | Val Accuracy | Val Loss |
|-------|-------------|----------|
| 1 | ~82% | ~0.45 |
| 2 | ~85% | ~0.38 |
| 3 | ~87% | ~0.33 |
| 4 | ~88% | ~0.31 |
| 5 | ~90% | ~0.29 |

**最终测试集评估**：
```
test_loss    = 0.3163
test_accuracy = 89.77%
```

### 步骤 6：转换 TFLite 模型

使用 `tf.lite.TFLiteConverter.from_keras_model()` 转换，采用 **dynamic range quantization**（动态范围量化）：

| 模型格式 | 大小 | 说明 |
|----------|------|------|
| Keras (.keras) | 9.3 MB | 原始模型，可继续训练 |
| TFLite (.tflite) | 2.4 MB | 量化后，3.8x 压缩比 |

### 步骤 7：写入元数据（遇到的挑战）

**问题**：`tflite-support` 库在 PyPI 上**没有提供 Windows 预编译包**，无法在 Windows Python 3.13 上安装。教程假设在 Google Colab（Linux）环境中运行。

**解决方案**：利用 TF 自带的 `flatbuffer_utils` 模块（纯 Python，无需额外依赖），从 Codelabs 原始 `FlowerModel.tflite` 中提取 912 字节的 `TFLITE_METADATA` 缓冲区，移植到自训练模型中：

```python
from tensorflow.lite.tools import flatbuffer_utils

orig = flatbuffer_utils.read_model("原始FlowerModel.tflite")
ours = flatbuffer_utils.read_model("自训练model.tflite")

# 替换元数据缓冲区
ours.buffers[179].data = orig.buffers[169].data
ours.metadata[1].name = b"TFLITE_METADATA"

flatbuffer_utils.write_model(ours, "model_with_metadata.tflite")
```

**原理**：两个模型的输入输出结构完全一致（224×224×3 → 5 类花卉，标签相同），元数据描述的是"接口"而非"权重"，因此可直接复用。

### 步骤 8：冒烟测试

用 `tf.lite.Interpreter` 加载最终的 TFLite 模型，取 8 张测试图片验证：

```
[✓] 真实=daisy      → 预测=daisy
[✓] 真实=tulips     → 预测=tulips
[✓] 真实=sunflowers → 预测=sunflowers
[✓] 真实=daisy      → 预测=daisy
[✓] 真实=sunflowers → 预测=sunflowers
[✓] 真实=tulips     → 预测=tulips
[✓] 真实=dandelion  → 预测=dandelion
[✓] 真实=roses      → 预测=roses

准确率: 8/8 = 100.0%
```

---

## 在实验四中验证

将 `model_with_metadata.tflite` 重命名为 `FlowerModel.tflite`，替换 `start/src/main/ml/FlowerModel.tflite`。

Android Studio 的 ML Model Binding 自动生成 `FlowerModel.java`，`start` 模块不需要修改任何代码。

构建验证：
```
BUILD SUCCESSFUL in 12s
36 actionable tasks
```

---

## 产出文件

| 文件 | 用途 |
|------|------|
| `train_flower_model.ipynb` | 原始 Notebook（可在 Jupyter 重现） |
| `train_flower_model_executed.ipynb` | 已执行版本（含全部输出） |
| `model.tflite` | 训练出的 TFLite 模型（无元数据） |
| `model_with_metadata.tflite` | 带元数据的最终模型（→ 实验四使用） |
| `flower_classifier.keras` | Keras 原始模型（可再训练/微调） |
| `labels.txt` | 5 类花卉标签 |

---

## 技术要点

### TensorFlow 与 LiteRT

- **TensorFlow**：核心开源机器学习库，采用数据流（Tensor + Flow）处理
- **LiteRT**（原 TensorFlow Lite）：移动端推理引擎，支持量化、GPU 加速
- 本实验流程：Keras 训练 → TFLite Converter 转换 → Android 部署

### 迁移学习

冻结 MobileNetV2 的 150+ 层预训练参数，只训练最后新增的 Dense 分类层（5 个神经元）。优势：
- 训练速度快（5 epoch，~6 分钟）
- 所需数据量少（3000 张即可）
- 适合转换为移动端 TFLite 模型

### 动态范围量化

将 float32 权重压缩为动态 8-bit，模型体积从 9.3MB 降至 2.4MB（3.8x），推理速度提升，精度损失极小。

### TFLite 元数据结构

```
TFLite Model FlatBuffer
├── Model Header
├── SubGraph（推理图）
├── Buffers[]
│   ├── buffer[0..N]  = 权重数据
│   ├── buffer[169]   = TFLITE_METADATA (912B)  ← 关键
│   └── buffer[...]   = labels.txt 等关联文件
└── Metadata[]
    ├── Metadata[0] = min_runtime_version
    └── Metadata[1] = TFLITE_METADATA → buffer[169]
```

ML Model Binding 读取 `TFLITE_METADATA` 自动生成 Java 包装类，开发者无需手写 Interpreter 代码。

---

## 遇到的问题与解决

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| 数据集下载 SSL 错误 | `storage.googleapis.com` 境内直连不稳定 | SSL bypass 手动下载 + 预解压到 Keras 缓存 |
| `tflite-support` 无法安装 | PyPI 无 Windows wheel，Python 3.13 不兼容 | 用 `flatbuffer_utils` 移植元数据 |
| 本地 Jupyter 内核选错 | 系统 Python 未装 TF | 切换内核到 `C:\Python313` 或 `pip install` 到系统 |

---

## 参考资料

- [花卉图片分类器：Keras 训练并导出 TFLite - CSDN](https://blog.csdn.net/llfjfz/article/details/161630612)
- [TensorFlow Lite Converter 官方文档](https://www.tensorflow.org/lite/models/convert)
- [MobileNetV2 论文](https://arxiv.org/abs/1801.04381)
- [TFLite Metadata 规范](https://www.tensorflow.org/lite/models/convert/metadata)
