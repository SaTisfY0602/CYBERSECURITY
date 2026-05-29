---
title: ResNet50 对抗攻击演示
emoji: 🛡️
colorFrom: amber
colorTo: orange
sdk: streamlit
sdk_version: 1.57.0
app_file: app.py
pinned: false
---

# ResNet50-Adversarial-Lab

基于 PyTorch 的 ResNet50 图像识别与目标定向对抗攻击（Targeted Adversarial Attack）研究项目，提供交互式 Web UI 与可编程 API 两种使用方式。

## 项目简介

本项目旨在探索和复现深度卷积神经网络（CNN）在计算机视觉任务中的表现及其安全性。项目以官方预训练的 ResNet50 模型为基础，包含三个核心实验模块：

1. **基线推理（Baseline Inference）**：对本地图像数据集执行批量分类，输出 Top-1 / Top-5 置信度。
2. **对抗样本攻击（Adversarial Attack）**：利用基于梯度的算法（FGSM / PGD）生成微小扰动，实现目标定向攻击（例如：将模型高置信度识别的“犬类”图像，定向欺骗为“母鸡”）。
3. **防御加固（Defense）**：通过高斯模糊、JPEG 压缩等预处理手段破坏对抗扰动的结构化特征，恢复模型正确识别率。

## 环境兼容性

本项目**不绑定任何特定显卡型号**，已适配以下运行环境：

| 环境                    | 支持状态                    | 说明                                            |
| --------------------- | ----------------------- | --------------------------------------------- |
| **NVIDIA GPU (CUDA)** | ✅ 推荐                    | 自动检测 CUDA，任意算力版本均可（如 GTX 10 系、RTX 30/40/50 系） |
| **CPU**               | ✅ 支持                    | 无显卡时自动回退至 CPU，功能完整，推理速度较慢                     |
| **Python**            | 3.10+                   | 推荐使用 3.10 ~ 3.13                              |
| **操作系统**              | Windows / Linux / macOS | 主要开发于 Windows，跨平台兼容                           |

> **注意**：RTX 50 系列（sm_120 架构）若使用标准稳定版 PyTorch 可能出现算力兼容警告，建议安装对应 CUDA 版本的 Nightly 预览版以获得最佳性能，详见下方安装指南。

## 项目结构

```
ResNet50-Adversarial-Lab/
├── app.py                    # Streamlit Web 应用入口
├── core/
│   ├── loadModel.py          # 基线模型加载与推理（AdversarialModel）
│   ├── attack_engine.py      # 对抗攻击引擎（FGSM / PGD）
│   └── defense_engine.py     # 预处理防御引擎（高斯模糊 / JPEG 压缩）
├── components/
│   ├── attack_tab.py         # Streamlit “攻击实验室” Tab
│   ├── defense_tab.py        # Streamlit “防御加固” Tab
│   ├── visualizations.py     # Matplotlib 图表封装（热力图、柱状图、收敛曲线）
│   └── styles.py             # 自定义 CSS 主题注入
├── utils/
│   └── imagenet_labels.py    # ImageNet-1K 标签工具函数
├── tests/                    # 单元测试
├── testset/                  # 测试图像文件夹（示例图存放位置）
├── requirements.txt          # Python 依赖清单
├── NOTE.md                   # 核心原理笔记（公式、论文、关键概念）
├── EXPERIMENT_LOG.md         # 实验操作日志（时间线记录）
├── report.md                 # 实验报告（阶段成果汇总）
└── README.md                 # 本文件
```

## 快速开始

### 1. 克隆仓库

```bash
git clone <你的仓库地址>
cd ResNet50-Adversarial-Lab
```

### 2. 创建虚拟环境（推荐）

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. 安装依赖

**通用安装（默认 CPU 版本，适合所有环境）：**

```bash
pip install -r requirements.txt
```

**NVIDIA GPU 加速安装（推荐，需根据 CUDA 版本选择）：**

访问 [PyTorch 官方安装向导](https://pytorch.org/get-started/locally/) 获取精确命令。以 CUDA 12.4 为例：

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

以 RTX 50 系列 + CUDA 13.0 为例（Nightly）：

```bash
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu130
pip install streamlit pillow numpy matplotlib
```

### 4. 启动交互式 Web 界面

```bash
streamlit run app.py
```

浏览器将自动打开 `http://localhost:8501`，你可以在侧边栏上传图片、配置攻击参数（FGSM / PGD）、实时查看对抗样本与噪声热力图，并一键切换到“防御加固”Tab 测试高斯模糊或 JPEG 压缩的防御效果。

### 5. 运行基线推理（命令行）

将你的测试图片放入 `testset/` 文件夹，然后执行：

```bash
python core/loadModel.py
```

运行后，你会看到每张测试图片的 Top-1 ~ Top-5 预测结果，例如：

```
“dog.jpg” 【dog】 98.50%
    Top-2: 【cat】 0.80%
    ...
```

### 6. 编程式调用对抗攻击

以下是一个最小可运行的 Python 示例，将 `testset/dog.jpg` 定向攻击为“母鸡”（ImageNet 类别 ID 为 7）：

```python
from PIL import Image
from core.attack_engine import AttackEngine

# 1. 初始化攻击引擎（自动检测 GPU/CPU）
engine = AttackEngine()

# 2. 读取并预处理图片
image = Image.open("testset/dog.jpg").convert("RGB")
original_tensor = engine.preprocess(image)

# 3. 查看原始预测
print("原始预测:", engine.predict(original_tensor))

# 4. 生成对抗样本（epsilon 控制扰动强度，越大攻击越强）
adv_tensor, perturbation, data_grad = engine.generate_targeted_adversarial(
    original_tensor, target_id=7, epsilon=0.03
)

# 5. 查看攻击后的预测
print("对抗样本预测:", engine.predict(adv_tensor))

# 6. 将对抗样本还原为可保存的图片
adv_image = engine.tensor_to_image(adv_tensor)
adv_image.save("adversarial_dog.jpg")
print("对抗样本已保存至 adversarial_dog.jpg")
```

**预期效果**：

- 原始图片被模型高置信度识别为 `dog`。
- 对抗样本在人眼看来与原始图片几乎一样，但模型会将其错误识别为 `hen`（母鸡）。
- `epsilon` 是核心调节参数：建议从 `0.01` 开始尝试，逐步增大到 `0.05` 或 `0.1`。

### 7. 关键参数说明

| 参数          | 含义                     | 建议取值          |
| ----------- | ---------------------- | ------------- |
| `target_id` | 想误导成的目标类别（ImageNet 索引） | 7（母鸡 hen）、... |
| `epsilon`   | 扰动强度，每个像素最大变化量         | 0.01 ~ 0.1    |
| `num_iter`  | PGD 迭代次数（仅 PGD）        | 10 ~ 50       |

> **ImageNet 类别查询**：ImageNet-1K 共有 1000 个类别，索引 0~999。常见动物索引示例：7=hen（母鸡）。你可以在 Web UI 的下拉框中直接选择中文标签，无需手动查索引。

## 学习路线（按顺序执行）

本项目按阶段推进，建议按以下顺序阅读代码与运行实验：

1. **阶段一**：阅读 `core/loadModel.py`，理解模型如何加载、预处理、推理。
2. **阶段二**：阅读 `core/attack_engine.py`，理解梯度追踪、定向损失、FGSM / PGD 攻击公式。
3. **阶段三**：阅读 `core/defense_engine.py` 与 `components/defense_tab.py`，理解图像预处理防御原理。
4. **阶段四**：查看 `report.md`，汇总实验数据并生成最终报告。

## 参考资料

- FGSM 原始论文：[*Explaining and Harnessing Adversarial Examples*](https://arxiv.org/abs/1412.6572), Ian J. Goodfellow et al., ICLR 2015.
- PGD 原始论文：[*Towards Deep Learning Models Resistant to Adversarial Attacks*](https://arxiv.org/abs/1706.06083), Aleksander Madry et al., ICLR 2018.
