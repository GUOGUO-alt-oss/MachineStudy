"""
实验5.2: TensorFlow 石头剪刀布手势模型生成
GPU 加速训练 - RTX 4060 + CUDA 11.8 + cuDNN 8.9
"""

import os
import sys
import zipfile
import ssl
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

# ========== CUDA 路径配置 ==========
CUDA_BIN = r"C:\Users\33525\.workbuddy\cuda\v11.8\bin"
CUDA_PATH = r"C:\Users\33525\.workbuddy\cuda\v11.8"
os.environ["PATH"] = CUDA_BIN + ";" + os.environ.get("PATH", "")
os.environ["CUDA_PATH"] = CUDA_PATH

# ========== 路径配置 ==========
PROJECT_DIR = Path(__file__).parent.resolve()
DATA_DIR = PROJECT_DIR / "data"
DOWNLOAD_DIR = DATA_DIR
MODEL_PATH = PROJECT_DIR / "rps.h5"
PLOT_PATH = PROJECT_DIR / "training_history.png"

# 数据集URL
RPS_URL = "https://storage.googleapis.com/learning-datasets/rps.zip"
RPS_TEST_URL = "https://storage.googleapis.com/learning-datasets/rps-test-set.zip"
RPS_ZIP = DOWNLOAD_DIR / "rps.zip"
RPS_TEST_ZIP = DOWNLOAD_DIR / "rps-test-set.zip"


def download_file(url, destination):
    """下载文件，支持断点续传"""
    if destination.exists() and destination.stat().st_size > 0:
        print(f"[SKIP] 文件已存在: {destination.name} ({destination.stat().st_size / 1024 / 1024:.1f} MB)")
        return

    temp_path = destination.with_suffix(destination.suffix + ".part")
    print(f"[下载] {destination.name} ...")

    try:
        response = urlopen(url, timeout=120)
    except URLError:
        context = ssl._create_unverified_context()
        response = urlopen(url, timeout=120, context=context)

    with response, temp_path.open("wb") as f:
        while True:
            data = response.read(1024 * 1024)
            if not data:
                break
            f.write(data)

    temp_path.replace(destination)
    print(f"[完成] {destination.name} ({destination.stat().st_size / 1024 / 1024:.1f} MB)")


def extract_zip(zip_path, extract_dir):
    """解压 zip 文件"""
    if not zip_path.exists():
        raise FileNotFoundError(f"找不到文件: {zip_path}")

    print(f"[解压] {zip_path.name} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        bad_file = zf.testzip()
        if bad_file is not None:
            raise zipfile.BadZipFile(f"文件损坏: {zip_path}")
        zf.extractall(extract_dir)
    print(f"[完成] 解压到 {extract_dir}")


def verify_dataset():
    """验证数据集完整性"""
    rock_dir = DOWNLOAD_DIR / "rps" / "rock"
    paper_dir = DOWNLOAD_DIR / "rps" / "paper"
    scissors_dir = DOWNLOAD_DIR / "rps" / "scissors"

    for d in [rock_dir, paper_dir, scissors_dir]:
        if not d.exists():
            raise FileNotFoundError(f"目录不存在: {d}。请先运行下载和解压步骤。")

    rock_count = len(list(rock_dir.iterdir()))
    paper_count = len(list(paper_dir.iterdir()))
    scissors_count = len(list(scissors_dir.iterdir()))

    print(f"[数据] 训练集: 石头={rock_count}, 布={paper_count}, 剪刀={scissors_count}, 总计={rock_count + paper_count + scissors_count}")

    test_dir = DOWNLOAD_DIR / "rps-test-set"
    if test_dir.exists():
        test_count = sum(1 for _ in test_dir.rglob("*.png"))
        print(f"[数据] 测试集: {test_count} 张")

    return rock_dir, paper_dir, scissors_dir


def build_model(input_shape=(150, 150, 3), num_classes=3):
    """构建 CNN 模型"""
    import tensorflow as tf

    model = tf.keras.models.Sequential([
        # Block 1
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu", input_shape=input_shape),
        tf.keras.layers.MaxPooling2D(2, 2),
        # Block 2
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2, 2),
        # Block 3
        tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2, 2),
        # Block 4
        tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2, 2),
        # Classifier
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.Dense(num_classes, activation="softmax"),
    ])
    return model


def create_data_generators():
    """创建数据生成器（含数据增强）"""
    import tensorflow as tf

    TRAINING_DIR = str(DOWNLOAD_DIR / "rps")
    VALIDATION_DIR = str(DOWNLOAD_DIR / "rps-test-set")

    # 训练集：强数据增强
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest",
    )

    # 验证集：仅归一化
    val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255)

    # 根据实际数据量动态调整 batch_size
    import math
    train_count = sum(1 for _ in Path(TRAINING_DIR).rglob("*.png"))
    val_count = sum(1 for _ in Path(VALIDATION_DIR).rglob("*.png"))
    
    # batch_size 设为训练数据的 1/20，确保 steps_per_epoch=20
    batch_size = max(1, train_count // 20)

    train_generator = train_datagen.flow_from_directory(
        TRAINING_DIR,
        target_size=(150, 150),
        class_mode="categorical",
        batch_size=batch_size,
    )

    validation_generator = val_datagen.flow_from_directory(
        VALIDATION_DIR,
        target_size=(150, 150),
        class_mode="categorical",
        batch_size=batch_size,
    )

    steps_per_epoch = math.ceil(train_count / batch_size)
    validation_steps = math.ceil(val_count / batch_size)

    return train_generator, validation_generator, steps_per_epoch, validation_steps


def plot_history(history, save_path):
    """绘制训练过程曲线"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = range(1, len(acc) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 准确率
    ax1.plot(epochs, acc, "r", label="训练准确率", linewidth=2)
    ax1.plot(epochs, val_acc, "b", label="验证准确率", linewidth=2)
    ax1.set_title("训练与验证准确率", fontsize=14)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)

    # 损失
    ax2.plot(epochs, loss, "r", label="训练损失", linewidth=2)
    ax2.plot(epochs, val_loss, "b", label="验证损失", linewidth=2)
    ax2.set_title("训练与验证损失", fontsize=14)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"[绘图] 训练曲线已保存: {save_path}")
    plt.close()


# ========== 主流程 ==========
def main():
    print("=" * 60)
    print("实验5.2: TensorFlow 石头剪刀布手势模型生成")
    print("=" * 60)

    # Step 1: 下载数据集
    print("\n>>> Step 1: 下载数据集")
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    download_file(RPS_URL, RPS_ZIP)
    download_file(RPS_TEST_URL, RPS_TEST_ZIP)

    # Step 2: 解压
    print("\n>>> Step 2: 解压数据集")
    extract_zip(RPS_ZIP, DOWNLOAD_DIR)
    extract_zip(RPS_TEST_ZIP, DOWNLOAD_DIR)

    # Step 3: 验证数据
    print("\n>>> Step 3: 验证数据")
    verify_dataset()

    # Step 4: 导入 TF 并检查 GPU
    print("\n>>> Step 4: 检查 GPU")
    import tensorflow as tf
    print(f"TensorFlow 版本: {tf.__version__}")
    gpus = tf.config.list_physical_devices("GPU")
    if len(gpus) > 0:
        for g in gpus:
            print(f"  GPU: {g}")
        print(f"  {len(gpus)} 个 GPU 可用，将使用 GPU 加速训练")
    else:
        print("  警告: 未检测到 GPU，将使用 CPU 训练（会很慢）")

    # Step 5: 创建数据生成器
    print("\n>>> Step 5: 创建数据生成器")
    train_gen, val_gen, steps_per_epoch, val_steps = create_data_generators()
    print(f"  steps_per_epoch={steps_per_epoch}, validation_steps={val_steps}")

    # Step 6: 构建模型
    print("\n>>> Step 6: 构建 CNN 模型")
    model = build_model()
    model.summary()

    # Step 7: 编译模型
    print("\n>>> Step 7: 编译模型")
    model.compile(
        loss="categorical_crossentropy",
        optimizer="rmsprop",
        metrics=["accuracy"],
    )

    # Step 8: 训练
    print("\n>>> Step 8: 开始训练 (25 epochs)")
    print("  使用 GPU 加速，预计耗时 10-20 分钟...")
    history = model.fit(
        train_gen,
        epochs=25,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_gen,
        validation_steps=val_steps,
        verbose=1,
    )

    # 最终准确率
    final_train_acc = history.history["accuracy"][-1]
    final_val_acc = history.history["val_accuracy"][-1]
    print(f"\n>>> 训练完成!")
    print(f"  最终训练准确率: {final_train_acc:.4f} ({final_train_acc * 100:.2f}%)")
    print(f"  最终验证准确率: {final_val_acc:.4f} ({final_val_acc * 100:.2f}%)")

    # Step 9: 保存模型
    print(f"\n>>> Step 9: 保存模型 -> {MODEL_PATH}")
    model.save(str(MODEL_PATH))
    print(f"  模型已保存: {MODEL_PATH} ({MODEL_PATH.stat().st_size / 1024 / 1024:.1f} MB)")

    # Step 10: 绘制曲线
    print("\n>>> Step 10: 绘制训练过程曲线")
    plot_history(history, PLOT_PATH)

    print("\n" + "=" * 60)
    print("实验5.2 完成!")
    print(f"  模型: {MODEL_PATH}")
    print(f"  曲线: {PLOT_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
