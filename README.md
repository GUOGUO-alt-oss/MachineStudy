# 实验4：基于 TensorFlow Lite 的智能图像分类 Android APP

## 项目简介

本项目是「智能图像分类APP」实验的完整实现，基于 Google 官方 TensorFlow Lite Codelab（[Recognize Flowers with TensorFlow Lite on Android](https://codelabs.developers.google.com/codelabs/recognize-flowers-with-tensorflow-on-android)）。应用利用 **CameraX** 实时获取手机摄像头画面，通过 **TensorFlow Lite** 预训练花卉识别模型进行推理，借助 **MVVM 架构**（ViewModel + LiveData + Data Binding）将 Top-3 分类结果实时展示在界面上，并支持 **GPU 硬件加速**。

---

## 实验目标

| 目标 | 说明 |
|------|------|
| 摄像头实时采集 | CameraX Preview + ImageAnalysis 连续获取摄像头帧 |
| 图像预处理 | YUV → Bitmap → TensorImage 格式转换与方向校正 |
| TFLite 模型推理 | 加载 `FlowerModel.tflite` 进行花卉分类 |
| Top-K 排序展示 | Top-3 花卉类别 + 置信度百分比，RecyclerView 动态刷新 |
| GPU 加速 | 自动检测设备 GPU 兼容性，支持时启用 GPU Delegate，否则回退 CPU |
| 代码上传 | GitHub 仓库 + 详细 README 文档 |

---

## 技术架构

### 整体架构

```
┌─────────────────────────────────────────────────────┐
│                    MainActivity                      │
│  ┌──────────┐  ┌───────────────┐  ┌──────────────┐ │
│  │  布局文件  │  │  CameraX 初始化 │  │ LiveData 观察 │ │
│  │  绑定     │  │  (权限+配置)    │  │  (RecyclerView)│ │
│  └──────────┘  └───────────────┘  └──────────────┘ │
├─────────────────────────────────────────────────────┤
│                 ImageAnalyzer (内部类)                │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────┐  │
│  │ ImageProxy │→│ toBitmap │→│ TensorImage       │  │
│  │ (YUV_420) │  │ (RGB)    │  │ (TFLite 输入格式) │  │
│  └──────────┘  └──────────┘  └──────────────────┘  │
│                  ↓                                   │
│  ┌──────────────────────────────────────────────┐   │
│  │ FlowerModel.process() → probabilityAsCategory │   │
│  │ → sortByDescending → take(3) → Recognition   │   │
│  └──────────────────────────────────────────────┘   │
│                  ↓                                   │
│          recogViewModel.updateData(items)            │
├─────────────────────────────────────────────────────┤
│     RecognitionListViewModel → LiveData → UI 刷新    │
│     RecognitionAdapter (ListAdapter + DiffUtil)      │
└─────────────────────────────────────────────────────┘
```

### 运行时数据流（单帧处理管线）

```
Camera帧 (ImageProxy)
  │  YuvToRgbConverter.yuvToRgb()
  ▼
Bitmap (方向校正: rotationMatrix.postRotate)
  │  TensorImage.fromBitmap()
  ▼
TensorImage (224×224, float32)
  │  flowerModel.process(tfImage)
  ▼
probabilityAsCategoryList → sortByDescending { score } → take(3)
  │  for (output in outputs) → Recognition(label, score)
  ▼
LiveData<List<Recognition>> → RecyclerView 刷新
```

### 技术栈

| 层次 | 技术 | 版本 |
|------|------|------|
| 语言 | Kotlin | 1.3.72 |
| 构建 | Android Gradle Plugin | 4.1.0-rc03 |
| 构建工具 | Gradle | 6.7.1 |
| 相机 | CameraX | 1.0.0-beta10 |
| 推理引擎 | TensorFlow Lite Support | 0.1.0 |
| 模型元数据 | TensorFlow Lite Metadata | 0.1.0 |
| GPU 加速 | TensorFlow Lite GPU | 2.3.0 |
| 架构 | MVVM (ViewModel + LiveData + Data Binding) | AndroidX |
| UI | RecyclerView + ListAdapter + DiffUtil | 1.1.0 |

---

## 环境要求

| 项目 | 要求 |
|------|------|
| Android Studio | 4.1 或以上版本 |
| Android SDK | compileSdk 30, minSdk 21 (Android 5.0+) |
| JDK | JDK 8（用于 Gradle 6.7.1 编译） |
| 运行设备 | **必须是真实 Android 手机**（模拟器无法获取摄像头） |
| 手机设置 | 开启「开发者选项」→「USB 调试」 |
| 网络 | 首次构建需下载 Gradle 及依赖包 |

---

## 项目结构

```
TFLClassify/
├── build.gradle                        # 根级构建脚本（仓库配置）
├── settings.gradle                     # 模块声明（:start, :finish）
├── gradle.properties                   # Gradle 全局属性（JVM/AndroidX/JDK路径）
├── gradle/wrapper/
│   ├── gradle-wrapper.jar
│   └── gradle-wrapper.properties       # Gradle 6.7.1
│
├── start/                              # ★ 实验模块（已完成所有 TODO）
│   ├── build.gradle                    # 模块构建配置
│   ├── proguard-rules.pro
│   └── src/main/
│       ├── AndroidManifest.xml         # 权限声明（CAMERA）
│       ├── ml/
│       │   └── FlowerModel.tflite      # 花卉识别模型（~13MB）
│       ├── java/org/tensorflow/lite/examples/classification/
│       │   ├── MainActivity.kt         # 主入口：CameraX + TFLite 推理
│       │   ├── ui/
│       │   │   └── RecognitionAdapter.kt  # RecyclerView 适配器
│       │   ├── util/
│       │   │   └── YuvToRgbConverter.kt   # YUV → RGB 转换
│       │   └── viewmodel/
│       │       └── RecognitionViewModel.kt # 数据层（Recognition + LiveData）
│       └── res/                        # 布局、图片、主题等资源
│
└── finish/                             # 官方参考实现（对照用）
    └── ...                             # 与 start 结构相同，代码已完整
```

---

## 快速开始

### 1. 克隆代码

```bash
git clone https://github.com/hoitab/TFLClassify.git
cd TFLClassify
```

或直接使用本仓库的 `start` 模块（已包含完成后的代码）。

### 2. 配置 JDK

本项目使用 Gradle 6.7.1，仅兼容 JDK 8~15。如果你的系统 JDK 是 17+，请在 `gradle.properties` 中指定：

```properties
org.gradle.java.home=E\:/jdk/jdk-1.8
```

### 3. 导入模型

确保 `FlowerModel.tflite` 已放置在 `start/src/main/ml/` 目录下。

### 4. 构建与运行

1. 用 Android Studio 打开项目根目录 `build.gradle`
2. **Sync Project with Gradle Files**（等待依赖下载完成）
3. 手机 USB 连接电脑，开启 USB 调试
4. 选择 `start` 模块 → **Run** → 选择物理设备
5. 首次运行授予摄像头权限
6. 将摄像头对准花卉，观察实时识别结果

### 5. 命令行构建（可选）

```bash
# Windows Git Bash / macOS / Linux
export JAVA_HOME="E:/jdk/jdk-1.8"
./gradlew :start:assembleDebug

# APK 输出路径：start/build/outputs/apk/debug/start-debug.apk
```

---

## TODO 完成情况

原项目 `start` 模块包含 6 个 TODO 占位项，均已实现：

| TODO | 内容 | 完成代码 | 知识点 |
|------|------|----------|--------|
| **TODO 1** | 初始化 TFLite 模型 | `FlowerModel.newInstance(ctx, options)` | 模型生命周期、`by lazy` 延迟初始化 |
| **TODO 2** | ImageProxy → TensorImage | `TensorImage.fromBitmap(toBitmap(imageProxy))` | UV→RGB 转换、方向校正、图像预处理 |
| **TODO 3** | 推理 + Top-K 排序 | `flowerModel.process(tfImage).probabilityAsCategoryList.apply { sortByDescending { it.score } }.take(MAX_RESULT_DISPLAY)` | 模型推理 API、Kotlin 集合操作 |
| **TODO 4** | 转 Recognition 列表 | `for (output in outputs) { items.add(Recognition(output.label, output.score)) }` | 数据建模、UI 数据绑定 |
| **TODO 5** | GPU Delegate 依赖 | build.gradle 添加 `tensorflow-lite-gpu:2.3.0` | Gradle 依赖管理、硬件加速 |
| **TODO 6** | GPU/CPU 自适应 | `CompatibilityList()` → GPU 可用则启用，否则 4 线程 CPU | 设备兼容性检测、回退策略 |

### 核心推理代码

```kotlin
// ImageAnalyzer 内部类（MainActivity.kt）
private val flowerModel: FlowerModel by lazy {
    val compatList = CompatibilityList()
    val options = if (compatList.isDelegateSupportedOnThisDevice) {
        Model.Options.Builder().setDevice(Model.Device.GPU).build()
    } else {
        Model.Options.Builder().setNumThreads(4).build()
    }
    FlowerModel.newInstance(ctx, options)
}

override fun analyze(imageProxy: ImageProxy) {
    val items = mutableListOf<Recognition>()
    val tfImage = TensorImage.fromBitmap(toBitmap(imageProxy))
    val outputs = flowerModel.process(tfImage)
        .probabilityAsCategoryList.apply { sortByDescending { it.score } }
        .take(MAX_RESULT_DISPLAY)
    for (output in outputs) {
        items.add(Recognition(output.label, output.score))
    }
    listener(items.toList())
    imageProxy.close()
}
```

---

## 实现过程中的问题与解决方案

本实验在较新的开发环境（Android Studio + Java 21）上操作时，遇到了以下兼容性问题及解决方案：

### 问题 1：Gradle 版本不兼容

**现象**：`gradle-wrapper.properties` 中为 `gradle-9.0-milestone-1`，与 AGP 4.1.0-rc03 不兼容。

**原因**：AGP 4.1.x 要求 Gradle 6.5 ~ 6.7，更高版本的 Gradle API 不兼容。

**解决**：将 `distributionUrl` 改为 `gradle-6.7.1-bin.zip`。

---

### 问题 2：JDK 版本冲突

**现象**：Gradle Sync 报错「Java 21 与 Gradle 6.7.1 不兼容，最低需要 Gradle 8.5」。

**原因**：系统 JAVA_HOME 指向 JDK 21（Android Studio 自带），但 Gradle 6.7.1 最高仅支持 JDK 15。升级 Gradle/AGP 会导致 Kotlin 版本、废弃 API 等一系列连锁问题。

**解决**：在 `gradle.properties` 中指定 JDK 8：

```properties
org.gradle.java.home=E\:/jdk/jdk-1.8
```

---

### 问题 3：TFLite 依赖 RC 版本不可用

**现象**：`tensorflow-lite-support:0.1.0-rc1` 从 Maven Central 404。

**原因**：`0.1.0-rc1` 是 release candidate 版本，已被仓库移除，仅稳定版 `0.1.0` 可解析。

**解决**：将 `tensorflow-lite-support` 和 `tensorflow-lite-metadata` 升级到 `0.1.0` 稳定版。同时保留 `jcenter()`（read-only 模式仍可提供 GPU delegate 等旧版工件）。

---

### 问题 4：Model.Options 类路径冲突

**现象**：`Cannot access class 'Options'` / `Argument type mismatch: Model.Options vs Options`。

**原因**：`tensorflow-lite-gpu:2.3.0` 传递依赖了 `tensorflow-lite:2.4.0`，其中也包含 `Model.Options` 类，与 `tensorflow-lite-support:0.1.0` 中的版本冲突。

**解决**：在 GPU 依赖中排除传递的 `tensorflow-lite` 模块：

```groovy
implementation('org.tensorflow:tensorflow-lite-gpu:2.3.0') {
    exclude group: 'org.tensorflow', module: 'tensorflow-lite'
}
```

---

### 问题 5：Debug Keystore 格式不兼容

**现象**：`packageDebug FAILED: Invalid keystore format`。

**原因**：JDK 21 生成的 `debug.keystore` 为 **PKCS12** 格式，JDK 8 的 `keytool` 仅支持 **JKS** 格式，签名时无法读取。

**解决**：

```bash
# 用 JDK 21 将 PKCS12 转为 JKS
keytool -importkeystore \
  -srckeystore debug.keystore -srcstoretype PKCS12 \
  -destkeystore debug.keystore -deststoretype JKS \
  -storepass android -keypass android -noprompt
```

---

### 问题 6：JCenter 仓库停用

**现象**：部分 TFLite 旧版工件无法从 `mavenCentral()` 下载。

**原因**：JCenter 已于 2022 年停止服务，但部分 TFLite 早期版本的工件仅存在于 JCenter read-only 存档中。

**解决**：在 `build.gradle` 中同时配置三个仓库：

```groovy
repositories {
    google()
    mavenCentral()
    jcenter()  // read-only 存档，保留以提供旧版工件
}
```

---

## 关键技术要点

### CameraX 架构

- **Preview**：仅负责将摄像头画面显示到 `PreviewView`，与 ML 推理无关
- **ImageAnalysis**：独立工作线程（`Executors.newSingleThreadExecutor()`），执行 `analyze()` 进行逐帧推理
- **背压策略**：`STRATEGY_KEEP_ONLY_LATEST`，跳帧但只处理最新帧
- **帧释放**：`imageProxy.close()` 必须调用，否则 CameraX 无法送入新帧

### MVVM 数据绑定

- `Recognition`（data class）→ `MutableLiveData`（内部可写）→ `LiveData`（对外只读）→ `Observer` 触发 `submitList()` → `DiffUtil` 增量更新
- Data Binding：`RecognitionItemBinding` 自动绑定 label + probabilityString，无需 `findViewById`

### ML Model Binding

- Android Studio 的 ML Model Binding 功能自动解析 `.tflite` 文件中的元数据
- 生成 `FlowerModel.java` 包装类，暴露 `newInstance()` / `process()` / `Outputs` 等 API
- 构建时任务：`generateDebugMlModelClass`

---

## 扩展方向

- 增加 `MAX_RESULT_DISPLAY` 显示更多分类结果
- 添加置信度阈值过滤（低于某阈值的类别不显示）
- UI 优化：调整颜色、字体、布局排列
- 增加拍照功能，保存带识别标注的图片
- 使用 [TensorFlow Lite Model Maker](https://www.tensorflow.org/lite/tutorials/model_maker_image_classification) 训练自定义花卉/物体分类模型

---

## 参考资料

| 资源 | 链接 |
|------|------|
| Google Codelabs 官方教程 | [Recognize Flowers with TensorFlow Lite on Android](https://codelabs.developers.google.com/codelabs/recognize-flowers-with-tensorflow-on-android) |
| CSDN 中文教程 | [基于 TensorFlow Lite 实现的 Android 花卉识别应用](https://blog.csdn.net/llfjfz/article/details/123899673) |
| 项目 GitHub 仓库 | [hoitab/TFLClassify](https://github.com/hoitab/TFLClassify) |
| TensorFlow Lite Model Maker | [官方文档](https://www.tensorflow.org/lite/tutorials/model_maker_image_classification) |
| CameraX 官方文档 | [Android CameraX](https://developer.android.com/training/camerax) |
| Android LiveData | [官方指南](https://developer.android.com/topic/libraries/architecture/livedata) |
| Android Data Binding | [官方指南](https://developer.android.com/topic/libraries/data-binding) |

---

## License

```
Copyright (C) 2020 The Android Open Source Project

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0
```
