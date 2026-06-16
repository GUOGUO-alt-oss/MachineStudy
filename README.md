# 实验5.2: TensorFlow 石头剪刀布手势识别模型

## 实验目的

使用 TensorFlow/Keras 构建卷积神经网络（CNN），训练一个能够识别**石头（Rock）、剪刀（Scissors）、布（Paper）**三种手势的图像分类模型。

## 实验环境

| 组件 | 版本/配置 |
|------|----------|
| Python | 3.10.11 |
| TensorFlow | 2.10.0 (GPU) |
| CUDA | 11.8 |
| cuDNN | 8.9.7 |
| GPU | NVIDIA GeForce RTX 4060 Laptop (8 GB) |
| OS | Windows 11 |

> **注意**: TensorFlow 2.10 是最后一个支持 Windows 原生 GPU 加速的版本。更高版本需在 WSL2 中使用 GPU。

## 数据集

数据集来自 [Laurence Moroney](https://laurencemoroney.com/datasets.html) 提供的公开手势图片集。

| 类别 | 训练集 | 测试集 |
|------|--------|--------|
| Rock（石头） | 840 | - |
| Paper（布） | 840 | - |
| Scissors（剪刀） | 840 | - |
| **总计** | **2520** | **372** |

- 图片尺寸: 300×300 像素，24位色彩
- 预处理: 缩放至 150×150，归一化 [0,1]
- 数据增强: 旋转±40°、平移20%、剪切20%、缩放20%、水平翻转

## 模型结构

```
Sequential Model (3,473,475 parameters)
┌─────────────────────────────────────────┐
│ Conv2D(64, 3×3, ReLU) → MaxPool(2×2)    │
│ Conv2D(64, 3×3, ReLU) → MaxPool(2×2)    │
│ Conv2D(128, 3×3, ReLU) → MaxPool(2×2)   │
│ Conv2D(128, 3×3, ReLU) → MaxPool(2×2)   │
│ Flatten → Dropout(0.5)                   │
│ Dense(512, ReLU) → Dense(3, Softmax)     │
└─────────────────────────────────────────┘
```

## 训练参数

| 参数 | 值 |
|------|-----|
| 损失函数 | Categorical Crossentropy |
| 优化器 | RMSprop |
| 训练轮数 | 25 epochs |
| Batch Size | 126 |
| Steps per Epoch | 20 |

## 实验结果

| 指标 | 数值 |
|------|------|
| 训练准确率 | **97.66%** |
| 验证准确率 | **100.00%** |
| 训练时间 | ~10 分钟 (GPU) |

> 训练过程曲线见 `training_history.png`

## 文件说明

```
rps-experiment/
├── train_rps.py          # 完整训练脚本（下载→增强→训练→保存→绘图）
├── rps.h5                # 训练好的 Keras 模型 (27 MB)
├── training_history.png  # 训练准确率/损失曲线
├── README.md             # 本文件
└── data/                 # 数据集（压缩包 + 解压目录）
    ├── rps.zip           # 训练集 2520 张
    ├── rps-test-set.zip  # 测试集 372 张
    ├── rps/              # 解压后的训练集
    └── rps-test-set/     # 解压后的测试集
```

## 使用方法

### 训练模型

```bash
python train_rps.py
```

脚本会自动完成:
1. 下载数据集（如已下载则跳过）
2. 解压并验证数据
3. 创建数据增强生成器
4. 构建 CNN 模型
5. 训练 25 轮
6. 保存模型为 `rps.h5`
7. 绘制训练曲线

### 模型推理

```python
import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model('rps.h5')

# 加载并预处理图片 (150×150 RGB)
img = tf.keras.preprocessing.image.load_img('test.jpg', target_size=(150, 150))
img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# 预测
pred = model.predict(img_array)
classes = ['Rock', 'Paper', 'Scissors']
print(f'Prediction: {classes[np.argmax(pred[0])]}')
```

## 参考资料

- [TensorFlow 官方文档](https://www.tensorflow.org/)
- [CSDN 教程: TensorFlow 石头剪刀布模型生成](https://blog.csdn.net/llfjfz/article/details/129823932)
- [数据集来源](https://laurencemoroney.com/datasets.html)
