# Streamlit Web UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an interactive Streamlit web application that allows users to upload images, perform targeted adversarial attacks (FGSM/PGD) against ResNet50, visualize perturbations and confidence changes, and demonstrate preprocessing-based defenses.

**Architecture:** Layered architecture matching the existing codebase: `core/` for algorithm engines (model, attack, defense), `components/` for Streamlit UI tabs and shared visualizations, `utils/` for helper data, and `app.py` as the entry point. The app uses `@st.cache_resource` to cache the model and `@st.session_state` to pass adversarial samples between attack and defense tabs.

**Tech Stack:** Python 3.13, PyTorch, torchvision, Streamlit 1.57.0, Pillow, Matplotlib, NumPy, CUDA 13.0 (RTX 5060)

---

## File Structure

**New files to create:**
- `utils/imagenet_labels.py` — Load and expose ImageNet-1K human-readable labels for dropdowns
- `components/visualizations.py` — Reusable Matplotlib charts (perturbation heatmap, confidence bar chart, PGD iteration curve)
- `components/attack_tab.py` — Streamlit UI for the "Attack Lab" tab
- `components/defense_tab.py` — Streamlit UI for the "Defense Hardening" tab
- `core/defense_engine.py` — Defense algorithms (Gaussian blur, JPEG compression)
- `app.py` — Streamlit entry point with model caching and tab navigation

**Existing files to modify:**
- `core/attack_engine.py` — Extend `generate_targeted_pgd()` to optionally record per-iteration target confidence history
- `core/loadModel.py` — Uncomment/adapt the `get_adversarial_model()` cache factory for Streamlit

---

## Task 1: ImageNet Labels Utility

**Files:**
- Create: `utils/imagenet_labels.py`
- Test: `tests/test_imagenet_labels.py`

**Purpose:** Provide a clean helper that returns the 1000 ImageNet labels as a list, so Streamlit dropdowns can show human-readable names like "golden retriever" instead of raw IDs.

- [ ] **Step 1: Write the utility**

Create `utils/imagenet_labels.py`:

```python
"""
utils/imagenet_labels.py
提供 ImageNet-1K 人类可读标签列表，供 Streamlit 下拉选择框使用。
"""

import json
import urllib.request
from typing import List

LABELS_URL = (
    "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/"
    "master/imagenet-simple-labels.json"
)


def load_labels() -> List[str]:
    """从网络加载 ImageNet 标签，失败时返回数字 ID 列表作为回退。"""
    try:
        with urllib.request.urlopen(LABELS_URL, timeout=10) as response:
            labels = json.loads(response.read().decode("utf-8"))
        return labels
    except Exception:
        return [str(i) for i in range(1000)]


def get_label_options() -> List[str]:
    """返回带索引前缀的标签列表，如 ['0: tench', '1: goldfish', ...]。"""
    labels = load_labels()
    return [f"{i}: {name}" for i, name in enumerate(labels)]


def parse_label_option(option: str) -> int:
    """从 '7: cock' 格式的字符串中解析出类别索引。"""
    return int(option.split(":")[0])
```

- [ ] **Step 2: Write the test**

Create `tests/test_imagenet_labels.py`:

```python
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.imagenet_labels import load_labels, get_label_options, parse_label_option


def test_load_labels_returns_1000_items():
    labels = load_labels()
    assert len(labels) == 1000
    assert isinstance(labels[0], str)


def test_get_label_options_format():
    options = get_label_options()
    assert len(options) == 1000
    assert options[0].startswith("0: ")
    assert options[7].startswith("7: ")


def test_parse_label_option():
    assert parse_label_option("7: cock") == 7
    assert parse_label_option("123: some_label") == 123
```

- [ ] **Step 3: Run tests**

```bash
cd D:\Adversarial_example_attack
pytest tests/test_imagenet_labels.py -v
```

Expected: 3 tests PASS.

- [ ] **Step 4: Commit**

```bash
git add utils/ tests/test_imagenet_labels.py
git commit -m "feat: 添加 ImageNet 标签工具类与单元测试"
```

---

## Task 2: Visualization Components

**Files:**
- Create: `components/visualizations.py`
- Test: `tests/test_visualizations.py`

**Purpose:** Reusable Matplotlib figure generators for heatmaps, bar charts, and line charts. Streamlit can display these via `st.pyplot()`.

- [ ] **Step 1: Write the heatmap function**

Create `components/visualizations.py`:

```python
"""
components/visualizations.py
封装 Matplotlib 图表生成函数，供 Attack Tab 和 Defense Tab 共用。
"""

from typing import List, Optional

import numpy as np
import torch
from matplotlib import pyplot as plt
import matplotlib.cm as cm


def plot_perturbation_heatmap(perturbation: torch.Tensor) -> plt.Figure:
    """
    绘制扰动热力图。

    Args:
        perturbation: 扰动张量，形状 (1, C, H, W) 或 (C, H, W)，像素空间。

    Returns:
        plt.Figure: Matplotlib Figure 对象，可直接传给 st.pyplot()。
    """
    if perturbation.dim() == 4:
        perturbation = perturbation.squeeze(0)

    # 取三个通道的平均绝对值，得到单通道热力图
    heatmap = perturbation.abs().mean(dim=0).cpu().numpy()  # (H, W)

    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(heatmap, cmap="hot")
    ax.axis("off")
    ax.set_title("扰动热力图")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    return fig


def plot_confidence_bar_chart(
    original_name: str,
    original_conf: float,
    target_name: str,
    target_conf: float,
) -> plt.Figure:
    """
    绘制原始 Top-1 类别与目标类别的置信度对比柱状图。

    Args:
        original_name: 原始 Top-1 类别名称。
        original_conf: 原始 Top-1 置信度（百分比，如 87.5）。
        target_name: 目标类别名称。
        target_conf: 目标类别在对抗样本上的置信度（百分比）。

    Returns:
        plt.Figure: Matplotlib Figure 对象。
    """
    fig, ax = plt.subplots(figsize=(5, 3))
    categories = [f"原始: {original_name}", f"目标: {target_name}"]
    values = [original_conf, target_conf]
    colors = ["steelblue", "coral"]

    bars = ax.bar(categories, values, color=colors, width=0.5)
    ax.set_ylabel("置信度 (%)")
    ax.set_title("防御前后置信度对比")
    ax.set_ylim(0, 100)

    # 在柱顶标注数值
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 2,
            f"{val:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.tight_layout()
    return fig


def plot_pgd_convergence_curve(target_confs: List[float]) -> plt.Figure:
    """
    绘制 PGD 攻击过程中目标类别置信度的迭代变化曲线。

    Args:
        target_confs: 每轮迭代后目标类别的置信度列表（百分比）。

    Returns:
        plt.Figure: Matplotlib Figure 对象。
    """
    fig, ax = plt.subplots(figsize=(6, 3))
    iterations = list(range(1, len(target_confs) + 1))
    ax.plot(iterations, target_confs, marker="o", color="coral", linewidth=2)
    ax.set_xlabel("迭代轮次")
    ax.set_ylabel("目标类别置信度 (%)")
    ax.set_title("PGD 迭代收敛曲线")
    ax.set_ylim(0, 100)
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    return fig
```

- [ ] **Step 2: Write tests**

Create `tests/test_visualizations.py`:

```python
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from components.visualizations import (
    plot_perturbation_heatmap,
    plot_confidence_bar_chart,
    plot_pgd_convergence_curve,
)


def test_heatmap_returns_figure():
    perturbation = torch.randn(1, 3, 224, 224) * 0.01
    fig = plot_perturbation_heatmap(perturbation)
    assert fig is not None


def test_bar_chart_returns_figure():
    fig = plot_confidence_bar_chart("dog", 87.5, "chicken", 62.3)
    assert fig is not None


def test_pgd_curve_returns_figure():
    fig = plot_pgd_convergence_curve([5.0, 15.0, 35.0, 55.0, 62.3])
    assert fig is not None
```

- [ ] **Step 3: Run tests**

```bash
pytest tests/test_visualizations.py -v
```

Expected: 3 tests PASS.

- [ ] **Step 4: Commit**

```bash
git add components/visualizations.py tests/test_visualizations.py
git commit -m "feat: 添加可视化组件（热力图、柱状图、PGD收敛曲线）"
```

---

## Task 3: Attack Tab Component

**Files:**
- Create: `components/attack_tab.py`
- Modify: `core/attack_engine.py` (minor: add `generate_targeted_pgd_with_history` if not present)

**Purpose:** The full "Attack Lab" UI with sidebar controls, three-column image display, dynamic confidence hints, and result charts.

- [ ] **Step 1: Extend attack_engine.py with history-tracking PGD**

Modify `core/attack_engine.py` — add a new method after `generate_targeted_pgd` (around line 286):

```python
    def generate_targeted_pgd_with_history(
        self,
        original_tensor: torch.Tensor,
        target_id: int,
        epsilon: float,
        alpha: Optional[float] = None,
        num_iter: int = 10,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[float]]:
        """
        带迭代历史记录的 PGD 攻击，返回每轮目标类别的置信度。

        Returns:
            Tuple[torch.Tensor, torch.Tensor, List[float]]:
                - adv_tensor: 对抗样本（归一化空间）
                - total_perturbation: 总扰动量（像素空间）
                - history: 每轮迭代后目标类别的置信度百分比列表
        """
        if alpha is None:
            alpha = epsilon / 4.0

        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

        pixel_orig = original_tensor.clone().detach().to(self.device)
        pixel_orig = pixel_orig * std + mean

        pixel_adv = pixel_orig.clone()
        history = []

        for _ in range(num_iter):
            pixel_adv.requires_grad_(True)
            norm_input = (pixel_adv - mean) / std
            output = self.model(norm_input)

            # 记录当前轮次目标类别的置信度
            probs = torch.softmax(output, dim=1)
            target_conf = probs[0, target_id].item() * 100
            history.append(round(target_conf, 2))

            target = torch.tensor([target_id], dtype=torch.long, device=self.device)
            criterion = torch.nn.CrossEntropyLoss()
            loss = criterion(output, target)

            self.model.zero_grad()
            loss.backward()
            grad = pixel_adv.grad.data

            pixel_adv = pixel_adv - alpha * grad.sign()
            perturbation = torch.clamp(pixel_adv - pixel_orig, -epsilon, epsilon)
            pixel_adv = pixel_orig + perturbation
            pixel_adv = torch.clamp(pixel_adv, 0.0, 1.0).detach()

        total_perturbation = pixel_adv - pixel_orig
        adv_tensor = (pixel_adv - mean) / std
        return adv_tensor, total_perturbation, history
```

- [ ] **Step 2: Write attack_tab.py**

Create `components/attack_tab.py`:

```python
"""
components/attack_tab.py
Streamlit "攻击实验室" Tab 的完整 UI 与交互逻辑。
"""

from typing import Optional

import streamlit as st
from PIL import Image
import torch

from core.attack_engine import AttackEngine
from components.visualizations import (
    plot_perturbation_heatmap,
    plot_confidence_bar_chart,
    plot_pgd_convergence_curve,
)
from utils.imagenet_labels import get_label_options, parse_label_option


def render_attack_tab(model: AttackEngine) -> None:
    """渲染攻击实验室 Tab。"""
    st.header("攻击实验室")

    # ------------------ 侧边栏控制区 ------------------
    with st.sidebar:
        st.subheader("攻击参数配置")

        # 图片输入：上传 或 选择示例
        upload_option = st.radio("图片来源", ["上传图片", "选择示例图"])
        image: Optional[Image.Image] = None

        if upload_option == "上传图片":
            uploaded = st.file_uploader("上传图片", type=["jpg", "jpeg", "png", "webp"])
            if uploaded is not None:
                image = Image.open(uploaded).convert("RGB")
        else:
            import os
            testset_dir = os.path.join(os.path.dirname(__file__), "..", "testset")
            if os.path.isdir(testset_dir):
                sample_files = [
                    f for f in os.listdir(testset_dir)
                    if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
                ]
                if sample_files:
                    selected = st.selectbox("选择示例图", sample_files)
                    image = Image.open(os.path.join(testset_dir, selected)).convert("RGB")
                else:
                    st.warning("testset 目录为空")
            else:
                st.warning("testset 目录不存在")

        # 攻击算法选择
        algorithm = st.radio("攻击算法", ["FGSM", "PGD"])

        # 目标类别
        label_options = get_label_options()
        target_option = st.selectbox("目标类别", label_options, index=7)
        target_id = parse_label_option(target_option)

        # epsilon
        epsilon = st.slider("ε 强度", 0.0, 0.1, 0.03, 0.01)

        # PGD 专属参数
        num_iter = 20
        if algorithm == "PGD":
            num_iter = st.slider("迭代次数", 5, 50, 20, 5)

        generate_btn = st.button("生成对抗样本", type="primary")

    # ------------------ 主展示区 ------------------
    if image is None:
        st.info("请在侧边栏上传图片或选择示例图")
        return

    # 预处理并获取原始预测
    original_tensor = model.preprocess(image)
    original_result = model.predict(original_tensor)
    original_name = original_result["topk_names"][0]
    original_conf = original_result["topk_confs"][0]

    st.markdown(f"**【原始预测】{original_name}  {original_conf:.2f}%**")

    # 动态提示（基于原始置信度）
    if original_conf >= 80:
        st.warning("模型对原图预测非常确定，建议使用 PGD 算法或 ε ≥ 0.05")
    elif original_conf >= 50:
        st.info("模型对原图有一定置信度，FGSM 建议 ε ≥ 0.03")
    else:
        st.success("模型对原图预测不确定，FGSM 小步长即可攻击成功")

    # 显示原始图像与 Top-5
    col_orig, col_adv = st.columns(2)
    with col_orig:
        st.subheader("原始图像")
        st.image(image, use_container_width=True)
        st.markdown("**Top-5 预测：**")
        for name, conf in zip(original_result["topk_names"], original_result["topk_confs"]):
            st.write(f"- {name}: {conf:.2f}%")

    # 执行攻击
    if generate_btn:
        with st.spinner("正在计算梯度并生成对抗样本..."):
            if algorithm == "FGSM":
                adv_tensor, perturbation, _ = model.generate_targeted_adversarial(
                    original_tensor, target_id, epsilon
                )
                pgd_history = None
            else:
                adv_tensor, perturbation, pgd_history = model.generate_targeted_pgd_with_history(
                    original_tensor, target_id, epsilon, num_iter=num_iter
                )

        adv_result = model.predict(adv_tensor)
        adv_name = adv_result["topk_names"][0]
        adv_conf = adv_result["topk_confs"][0]
        target_conf = adv_result["topk_confs"][adv_result["topk_ids"].index(target_id)]

        # 保存到 session_state 供防御 Tab 使用
        st.session_state["adv_image"] = model.tensor_to_image(adv_tensor)
        st.session_state["adv_result"] = adv_result
        st.session_state["original_result"] = original_result

        # 三列展示：原始图（已显示）、热力图、对抗样本
        col_heat, col_adv_result = st.columns(2)

        with col_heat:
            st.subheader("噪声热力图")
            fig_heat = plot_perturbation_heatmap(perturbation)
            st.pyplot(fig_heat)
            max_perturbation = perturbation.abs().max().item()
            st.caption(f"最大扰动值: {max_perturbation:.4f} (ε={epsilon})")

        with col_adv_result:
            st.subheader("对抗样本")
            adv_image = model.tensor_to_image(adv_tensor)
            st.image(adv_image, use_container_width=True)
            st.markdown(f"**【对抗预测】{adv_name}  {adv_conf:.2f}%**")
            st.markdown("**Top-5 预测：**")
            for name, conf in zip(adv_result["topk_names"], adv_result["topk_confs"]):
                st.write(f"- {name}: {conf:.2f}%")

        # 置信度对比柱状图
        st.subheader("置信度对比")
        fig_bar = plot_confidence_bar_chart(original_name, original_conf, target_option.split(": ")[1], target_conf)
        st.pyplot(fig_bar)

        # PGD 迭代曲线（仅 PGD 时显示）
        if pgd_history is not None:
            st.subheader("PGD 迭代收敛曲线")
            fig_curve = plot_pgd_convergence_curve(pgd_history)
            st.pyplot(fig_curve)
```

- [ ] **Step 3: Commit**

```bash
git add core/attack_engine.py components/attack_tab.py
git commit -m "feat: 扩展 PGD 带历史记录 + 实现攻击 Tab UI"
```

---

## Task 4: Streamlit Entry Point (Round 1)

**Files:**
- Create: `app.py`
- Modify: `core/loadModel.py` (uncomment cache factory)

**Purpose:** Main Streamlit app with model caching and tab navigation.

- [ ] **Step 1: Activate cache factory in loadModel.py**

Modify `core/loadModel.py` — replace the commented block at lines 184-202 with:

```python
# ==================== Web UI 缓存适配 ====================

try:
    import streamlit as st
    _cache_decorator = st.cache_resource
except ImportError:
    def _cache_decorator(func=None, **kwargs):
        if func is not None:
            return func
        return lambda f: f


@_cache_decorator
def get_adversarial_model() -> "AdversarialModel":
    """全局缓存的模型工厂函数，供 Streamlit 调用。"""
    return AdversarialModel()
```

- [ ] **Step 2: Write app.py**

Create `app.py`:

```python
"""
app.py
Streamlit 主应用入口。
提供"攻击实验室"与"防御加固"两个 Tab，整合模型缓存与组件调用。
"""

import streamlit as st

from core.loadModel import get_adversarial_model
from components.attack_tab import render_attack_tab


def main():
    st.set_page_config(
        page_title="ResNet50 对抗攻击演示",
        page_icon="🛡️",
        layout="wide",
    )

    st.title("ResNet50 对抗攻击与防御演示")
    st.caption("基于 PyTorch 预训练 ResNet50，实现 FGSM/PGD 定向对抗样本攻击与预处理防御")

    # 缓存加载模型（单例）
    with st.spinner("正在加载 ResNet50 模型..."):
        model = get_adversarial_model()

    # Tab 导航
    tab_attack, tab_defense = st.tabs(["🎯 攻击实验室", "🛡️ 防御加固"])

    with tab_attack:
        render_attack_tab(model)

    with tab_defense:
        st.header("防御加固")
        st.info("防御功能将在第二轮实现。请先在"攻击实验室"生成对抗样本。")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run and verify Round 1**

```bash
cd D:\Adversarial_example_attack
streamlit run app.py
```

Expected behavior:
- 页面标题 "ResNet50 对抗攻击与防御演示"
- 侧边栏显示图片上传/选择、算法选择、目标类别、ε 滑块
- 上传图片后显示原始预测和动态提示
- 点击"生成对抗样本"后显示三列（原始图、热力图、对抗样本）和柱状图
- PGD 模式下额外显示迭代曲线

- [ ] **Step 4: Commit**

```bash
git add app.py core/loadModel.py
git commit -m "feat: 创建 Streamlit 主应用（第一轮：攻击 Tab）"
```

---

## Task 5: Defense Engine

**Files:**
- Create: `core/defense_engine.py`
- Test: `tests/test_defense_engine.py`

**Purpose:** Encapsulate preprocessing-based defense algorithms.

- [ ] **Step 1: Write defense_engine.py**

Create `core/defense_engine.py`:

```python
"""
core/defense_engine.py
阶段四：防御引擎，实现图像预处理防御手段。
"""

import io
from typing import Literal

from PIL import Image, ImageFilter


class DefenseEngine:
    """
    预处理防御引擎。

    提供基于图像预处理的防御方法，通过破坏对抗扰动的结构化特征
    来降低对抗样本的攻击效果。
    """

    def gaussian_defense(self, image: Image.Image, sigma: float) -> Image.Image:
        """
        高斯模糊防御。

        原理：对抗扰动通常具有高频、局部化的特征。高斯模糊通过低通滤波
        平滑图像中的高频噪声，破坏对抗扰动的空间结构，使模型恢复正确识别。

        Args:
            image: 输入图像（PIL RGB）。
            sigma: 高斯核标准差，越大模糊程度越高（0.5 ~ 5.0）。

        Returns:
            Image.Image: 模糊后的图像。
        """
        return image.filter(ImageFilter.GaussianBlur(radius=sigma))

    def jpeg_defense(self, image: Image.Image, quality: int) -> Image.Image:
        """
        JPEG 压缩防御。

        原理：JPEG 是有损压缩，在量化过程中会丢弃高频信息。
        对抗扰动中的许多精细结构属于高频成分，经 JPEG 压缩后会被抑制或消除。

        Args:
            image: 输入图像（PIL RGB）。
            quality: JPEG 质量因子（10 ~ 100），越低压缩越激进、防御效果越强。

        Returns:
            Image.Image: 经 JPEG 压缩后再解码的图像。
        """
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        return Image.open(buffer)

    def apply_defense(
        self,
        image: Image.Image,
        method: Literal["gaussian", "jpeg"],
        **kwargs,
    ) -> Image.Image:
        """
        统一防御接口，根据方法名自动分发。

        Args:
            image: 输入图像。
            method: "gaussian" 或 "jpeg"。
            **kwargs: 对应防御方法的参数（sigma 或 quality）。

        Returns:
            Image.Image: 防御处理后的图像。
        """
        if method == "gaussian":
            return self.gaussian_defense(image, kwargs.get("sigma", 1.0))
        elif method == "jpeg":
            return self.jpeg_defense(image, kwargs.get("quality", 75))
        else:
            raise ValueError(f"不支持的防御方法: {method}")
```

- [ ] **Step 2: Write tests**

Create `tests/test_defense_engine.py`:

```python
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from PIL import Image
from core.defense_engine import DefenseEngine


def test_gaussian_defense_returns_image():
    engine = DefenseEngine()
    img = Image.new("RGB", (224, 224), color=(128, 128, 128))
    result = engine.gaussian_defense(img, sigma=1.5)
    assert isinstance(result, Image.Image)
    assert result.size == (224, 224)


def test_jpeg_defense_returns_image():
    engine = DefenseEngine()
    img = Image.new("RGB", (224, 224), color=(128, 128, 128))
    result = engine.jpeg_defense(img, quality=50)
    assert isinstance(result, Image.Image)
    assert result.size == (224, 224)


def test_apply_defense_dispatcher():
    engine = DefenseEngine()
    img = Image.new("RGB", (224, 224), color=(128, 128, 128))
    result = engine.apply_defense(img, method="gaussian", sigma=2.0)
    assert isinstance(result, Image.Image)
```

- [ ] **Step 3: Run tests**

```bash
pytest tests/test_defense_engine.py -v
```

Expected: 3 tests PASS.

- [ ] **Step 4: Commit**

```bash
git add core/defense_engine.py tests/test_defense_engine.py
git commit -m "feat: 添加预处理防御引擎（高斯模糊 + JPEG 压缩）"
```

---

## Task 6: Defense Tab Component

**Files:**
- Create: `components/defense_tab.py`

**Purpose:** The full "Defense Hardening" UI with method selection, parameter controls, and before/after comparison.

- [ ] **Step 1: Write defense_tab.py**

Create `components/defense_tab.py`:

```python
"""
components/defense_tab.py
Streamlit "防御加固" Tab 的完整 UI 与交互逻辑。
"""

from typing import Optional

import streamlit as st
from PIL import Image

from core.attack_engine import AttackEngine
from core.defense_engine import DefenseEngine
from components.visualizations import plot_confidence_bar_chart


def render_defense_tab(model: AttackEngine) -> None:
    """渲染防御加固 Tab。"""
    st.header("防御加固")

    defense_engine = DefenseEngine()

    # ------------------ 输入来源 ------------------
    input_option = st.radio(
        "对抗样本来源",
        ["使用攻击实验室生成的样本", "上传对抗样本"],
        horizontal=True,
    )

    adv_image: Optional[Image.Image] = None
    adv_result = None
    original_result = None

    if input_option == "使用攻击实验室生成的样本":
        if "adv_image" in st.session_state:
            adv_image = st.session_state["adv_image"]
            adv_result = st.session_state.get("adv_result")
            original_result = st.session_state.get("original_result")
        else:
            st.warning("尚未生成对抗样本，请先在"攻击实验室"执行攻击。")
            return
    else:
        uploaded = st.file_uploader("上传对抗样本图片", type=["jpg", "jpeg", "png", "webp"])
        if uploaded is not None:
            adv_image = Image.open(uploaded).convert("RGB")
            adv_tensor = model.preprocess(adv_image)
            adv_result = model.predict(adv_tensor)
        else:
            st.info("请上传对抗样本图片")
            return

    # ------------------ 侧边栏控制区 ------------------
    with st.sidebar:
        st.subheader("防御参数配置")
        defense_method = st.radio("防御方法", ["高斯模糊", "JPEG 压缩"])

        if defense_method == "高斯模糊":
            sigma = st.slider("σ（高斯核）", 0.5, 5.0, 1.0, 0.5)
        else:
            quality = st.slider("JPEG 质量因子", 10, 100, 75, 5)

        apply_btn = st.button("应用防御处理", type="primary")

    # ------------------ 主展示区 ------------------
    # 防御前预测（如果没有则计算）
    if adv_result is None and adv_image is not None:
        adv_tensor = model.preprocess(adv_image)
        adv_result = model.predict(adv_tensor)

    if adv_result is None:
        st.error("无法获取对抗样本的预测结果")
        return

    adv_name = adv_result["topk_names"][0]
    adv_conf = adv_result["topk_confs"][0]

    st.markdown(f"**【防御前】Top-1: {adv_name}  {adv_conf:.2f}%**")

    col_before, col_after = st.columns(2)

    with col_before:
        st.subheader("对抗样本（防御前）")
        st.image(adv_image, use_container_width=True)
        st.markdown("**Top-5 预测：**")
        for name, conf in zip(adv_result["topk_names"], adv_result["topk_confs"]):
            st.write(f"- {name}: {conf:.2f}%")

    # 应用防御
    if apply_btn:
        with st.spinner("正在应用防御处理..."):
            if defense_method == "高斯模糊":
                defended_image = defense_engine.gaussian_defense(adv_image, sigma)
            else:
                defended_image = defense_engine.jpeg_defense(adv_image, quality)

        defended_tensor = model.preprocess(defended_image)
        defended_result = model.predict(defended_tensor)
        defended_name = defended_result["topk_names"][0]
        defended_conf = defended_result["topk_confs"][0]

        # 判断防御是否成功（恢复为原始预测）
        original_name = original_result["topk_names"][0] if original_result else None
        is_success = original_name is not None and defended_name == original_name

        with col_after:
            st.subheader("防御处理后")
            st.image(defended_image, use_container_width=True)
            st.markdown(f"**【防御后】Top-1: {defended_name}  {defended_conf:.2f}%**")

            if is_success:
                st.success(f"防御成功：模型已恢复为 {original_name}")
            else:
                st.error("防御失败：模型仍被误导")

            st.markdown("**Top-5 预测：**")
            for name, conf in zip(defended_result["topk_names"], defended_result["topk_confs"]):
                st.write(f"- {name}: {conf:.2f}%")

        # 防御前后置信度对比
        if original_result:
            st.subheader("防御效果对比")
            original_conf = original_result["topk_confs"][0]
            fig = plot_confidence_bar_chart(
                original_name or "原始", original_conf,
                adv_name, adv_conf,
            )
            st.pyplot(fig)
```

- [ ] **Step 2: Commit**

```bash
git add components/defense_tab.py
git commit -m "feat: 实现防御 Tab UI（高斯模糊 + JPEG 压缩）"
```

---

## Task 7: Integrate Defense Tab into App

**Files:**
- Modify: `app.py`

**Purpose:** Wire the defense tab into the main app and update placeholder text.

- [ ] **Step 1: Update app.py**

Modify `app.py` — replace the defense tab content:

```python
"""
app.py
Streamlit 主应用入口（第二轮完整版）。
提供"攻击实验室"与"防御加固"两个 Tab。
"""

import streamlit as st

from core.loadModel import get_adversarial_model
from components.attack_tab import render_attack_tab
from components.defense_tab import render_defense_tab


def main():
    st.set_page_config(
        page_title="ResNet50 对抗攻击演示",
        page_icon="🛡️",
        layout="wide",
    )

    st.title("ResNet50 对抗攻击与防御演示")
    st.caption("基于 PyTorch 预训练 ResNet50，实现 FGSM/PGD 定向对抗样本攻击与预处理防御")

    # 缓存加载模型（单例）
    with st.spinner("正在加载 ResNet50 模型..."):
        model = get_adversarial_model()

    # Tab 导航
    tab_attack, tab_defense = st.tabs(["🎯 攻击实验室", "🛡️ 防御加固"])

    with tab_attack:
        render_attack_tab(model)

    with tab_defense:
        render_defense_tab(model)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run and verify Round 2**

```bash
streamlit run app.py
```

Expected behavior:
- 两个 Tab 正常切换
- 攻击 Tab 生成对抗样本后，切换到防御 Tab 自动带入样本
- 高斯模糊和 JPEG 压缩均可应用，显示防御前后对比
- 防御成功/失败状态正确显示

- [ ] **Step 3: Commit**

```bash
git add app.py
git commit -m "feat: 集成防御 Tab，完成 Streamlit 应用（第二轮）"
```

---

## Task 8: Documentation Updates

**Files:**
- Modify: `EXPERIMENT_LOG.md`
- Modify: `NOTE.md`

**Purpose:** Update project logs per CLAUDE.md workflow requirements.

- [ ] **Step 1: Update EXPERIMENT_LOG.md**

Append to `EXPERIMENT_LOG.md`:

```markdown
## 2026-05-03 — 阶段三/四：Streamlit Web UI 开发与防御实现

### 修改内容
- 创建 Streamlit 交互式 Web 应用（`app.py`）
- 实现攻击实验室 Tab：FGSM/PGD 定向攻击、噪声热力图、置信度柱状图、PGD 迭代曲线
- 实现防御加固 Tab：高斯模糊、JPEG 压缩预处理防御、防御前后对比
- 添加 `DefenseEngine` 防御引擎、`imagenet_labels` 标签工具、可视化组件

### 实验预期
- 用户可通过浏览器直观体验对抗攻击全过程
- 预处理防御可有效恢复部分对抗样本的正确识别
```

- [ ] **Step 2: Update NOTE.md**

Append to `NOTE.md`:

```markdown
### 预处理防御原理

高斯模糊与 JPEG 压缩破坏对抗扰动的高频结构，使模型恢复正确分类。
参考：Xu et al. (2017). *Feature Squeezing: Detecting Adversarial Examples in Deep Neural Networks*. https://arxiv.org/abs/1704.01155
```

- [ ] **Step 3: Commit**

```bash
git add EXPERIMENT_LOG.md NOTE.md
git commit -m "docs: 更新实验日志与原理笔记"
```

---

## Self-Review Checklist

- [x] **Spec coverage:** All sections from the design doc are addressed:
  - File structure (Task 1-8)
  - Attack Tab layout (Task 3, 4)
  - Defense Tab layout (Task 5, 6, 7)
  - Dynamic hints (in Task 3)
  - PGD history tracking (Task 3)
  - Visualization components (Task 2)
- [x] **Placeholder scan:** No TBD, TODO, or vague steps. Every step has concrete code or commands.
- [x] **Type consistency:** Method signatures match across files:
  - `generate_targeted_pgd_with_history` returns `Tuple[torch.Tensor, torch.Tensor, List[float]]`
  - `DefenseEngine.gaussian_defense` takes `(Image.Image, float)`
  - `render_attack_tab` and `render_defense_tab` both take `AttackEngine`
