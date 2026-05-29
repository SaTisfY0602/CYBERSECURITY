# 深度学习对抗攻击与防御 — 期末汇报大纲

> **项目名称**：基于 ResNet50 的定向对抗样本攻击（FGSM/PGD）与防御研究
> **环境**：PyTorch + CUDA 13.0 + NVIDIA RTX 5060 + Streamlit

---

## 幻灯片 1：研究背景与问题定义

**核心问题**：深度神经网络虽然在图像分类任务上达到甚至超越人类水平，但对精心构造的微小扰动极其脆弱——人眼不可察觉的噪声就能让模型完全误判。

**关键词**：Adversarial Examples, Adversarial Attack, Adversarial Defense

**建议放入 PPT 的素材**：
- 一张原始图像 vs 对抗样本的视觉效果对比图（肉眼几乎相同，但模型分类完全不同）
- 一句话概括：`ResNet50 将"狗"（99.2%）→ 加 0.03 扰动 → 错判为"鸡"（87.3%）`

**对应代码**：`app.py:30-31` — 项目标题与定位
```python
st.title("ResNet50 对抗攻击与防御演示")
st.caption("基于 PyTorch 预训练 ResNet50，实现 FGSM/PGD 定向对抗样本攻击与预处理防御")
```

---

## 幻灯片 2：系统架构总览

**项目采用分层架构，共四层**：

| 层级 | 目录 | 职责 |
|------|------|------|
| 算法核心层 | `core/` | 模型加载、攻击引擎、防御引擎 |
| UI 组件层 | `components/` | Streamlit Tab 渲染、可视化图表、CSS 样式 |
| 应用入口 | `app.py` | 全局缓存、Tab 路由、页面配置 |
| 工具层 | `utils/` | ImageNet 标签加载 |

**建议放入 PPT 的素材**：
- 架构分层图（文本框+箭头即可）
- 项目文件树截图

**对应代码**：`app.py:7-12` — 模块依赖关系
```python
from core.loadModel import get_adversarial_model as _load_model
from components.attack_tab import render_attack_tab
from components.defense_tab import render_defense_tab
from components.styles import inject_custom_css
```

---

## 幻灯片 3：模型部署 — ResNet50 加载与推理

**技术要点**：

1. 使用 `ResNet50_Weights.DEFAULT` 加载 ImageNet-1K 预训练权重
2. 自动检测 CUDA 并挂载至 GPU
3. `model.eval()` 进入推理模式，保证确定性输出
4. `torch.inference_mode()` 替代 `no_grad()`，更快的推理速度

**建议放入 PPT 的素材**：
- 这段代码是模型推理的核心，建议裁剪后放入 PPT（6-8 行即可）

**对应代码**：`core/loadModel.py:32-48` — 模型初始化
```python
class AdversarialModel:
    def __init__(self, device=None):
        self.device = self.to_device(device)
        self.weights = ResNet50_Weights.DEFAULT
        self.model = resnet50(weights=self.weights)
        self.model.to(self.device)
        self.model.eval()          # 关闭 Dropout 与 BN 统计更新
        self.labels = self._load_labels()

    def predict(self, image_tensor, top_k=5):
        with torch.inference_mode():       # 比 no_grad 更快的推理模式
            outputs = self.model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
            probs, class_ids = torch.topk(probabilities, k=top_k, dim=1)
        return {
            "topk_ids": class_ids,
            "topk_names": [self.get_label(cid) for cid in class_ids],
            "topk_confs": [round(p * 100, 2) for p in probs],
        }
```

---

## 幻灯片 4：对抗攻击 — FGSM 核心原理

**FGSM（Fast Gradient Sign Method）** — Goodfellow et al., ICLR 2015

**核心公式**：

$$X_{adv} = X - \epsilon \cdot \text{sign}(\nabla_X J(X, Y_{target}))$$

- **定向攻击**：使用减号，最小化目标类别损失，迫使模型"认错"为指定类别
- **sign()**：只保留梯度方向（+1 / -1 / 0），忽略大小，确保每个像素扰动绝对值相同
- **ε（epsilon）**：控制扰动强度的超参数

**建议放入 PPT 的素材**：
- 上述公式（核心，必须出现）
- FGSM 攻击流程简图：`输入 → 前向传播 → 计算损失 → 反向传播 → 提取梯度 → sign(梯度) × ε → 构造扰动`

**对应代码**：`core/attack_engine.py:186-190` — FGSM 扰动核心
```python
# 5. 反向传播到像素空间
self.model.zero_grad()
loss.backward()
pixel_grad = pixel_input.grad.data

# 6. 像素空间 FGSM：定向攻击用减号（最小化目标损失）
perturbation = epsilon * pixel_grad.sign()
pixel_adv = pixel_input - perturbation
```

---

## 幻灯片 5：关键 Bug 修复 — 像素空间攻击

**发现的问题**：在归一化空间（约 [-2.1, 2.2]）直接执行 `torch.clamp(tensor, 0, 1)` 会导致大量合法像素被硬性截断，扰动方向严重扭曲，甚至出现置信度反弹的反常现象。

**修复方案**：将攻击下沉至像素空间 [0, 1]

| 步骤 | 操作 |
|------|------|
| 1 | 反归一化：`pixel = tensor * std + mean` |
| 2 | 在像素空间开启 `requires_grad=True` |
| 3 | 重新归一化后送入模型前向传播 |
| 4 | 反向传播得到像素空间梯度 |
| 5 | 在像素空间执行 FGSM 并 clamp |
| 6 | 再次归一化，供模型推理 |

**建议放入 PPT 的素材**：
- 修复前 vs 修复后的对比表格（实验数据来自 `report.md`）

**对应代码**：`core/attack_engine.py:169-199` — 像素空间 FGSM 完整流程
```python
# 1. 反归一化到像素空间 [0, 1]
pixel_input = original_tensor.clone().detach()
pixel_input = pixel_input * std + mean

# 2. 在像素空间开启梯度追踪
pixel_input.requires_grad_(True)

# 3. 重新归一化后送入模型
norm_input = (pixel_input - mean) / std
output = self.model(norm_input)

# 4-5. 反向传播 + FGSM 扰动（像素空间）
loss.backward()
pixel_grad = pixel_input.grad.data
pixel_adv = pixel_input - epsilon * pixel_grad.sign()

# 6. 合法像素截断 + 重新归一化
pixel_adv = torch.clamp(pixel_adv, 0.0, 1.0).detach()
adv_tensor = (pixel_adv - mean) / std
```

---

## 幻灯片 6：PGD 迭代攻击

**PGD（Projected Gradient Descent）** — Madry et al., ICLR 2018

**核心思想**："小碎步 + 反复修正"

$$X_{t+1} = \Pi_{\epsilon} \Big( \text{clamp}\big( X_t - \alpha \cdot \text{sign}(\nabla_X J(X_t, Y_{target})), 0, 1 \big) \Big)$$

- 每步步长 `α = ε / 4`（避免单步过大导致不稳定）
- 投影操作 `Π_ε`：限制当前对抗样本不超过 `ε` 的 L∞ 邻域
- 默认迭代 10~20 次

**实验结论**：对语义差距大的大类间目标（如 banana → cock），PGD 成功率远高于 FGSM

**建议放入 PPT 的素材**：
- FGSM vs PGD 对比实验数据表
- PGD 迭代收敛曲线（目标类别置信度随迭代次数上升的折线图）

**对应代码**：`core/attack_engine.py:201-243` — PGD 核心循环
```python
for _ in range(num_iter):
    pixel_adv.requires_grad_(True)
    norm_input = (pixel_adv - mean) / std
    output = self.model(norm_input)

    target = torch.tensor([target_id], dtype=torch.long, device=self.device)
    loss = self.criterion(output, target)

    self.model.zero_grad()
    loss.backward()
    grad = pixel_adv.grad.data

    pixel_adv = pixel_adv - alpha * grad.sign()                    # 小步更新
    perturbation = torch.clamp(pixel_adv - pixel_orig, -epsilon, epsilon)
    pixel_adv = pixel_orig + perturbation                          # 投影到 ε 邻域
    pixel_adv = torch.clamp(pixel_adv, 0.0, 1.0).detach()         # 合法像素截断
```

---

## 幻灯片 7：防御策略

**预处理防御原理**（对应 `core/defense_engine.py`）：

对抗扰动具有**高频、局部化**的空间结构。通过图像预处理破坏这些高频特征，可使模型恢复正确识别。

| 防御方法 | 原理 | 参数 |
|----------|------|------|
| 高斯模糊 | 低通滤波平滑高频噪声 | `sigma`：核标准差 (0.5~5.0) |
| JPEG 压缩 | DCT 量化阶段丢弃高频分量 | `quality`：质量因子 (10~100) |

**理论依据**：Feature Squeezing (Xu et al., NDSS 2018)

**建议放入 PPT 的素材**：
- 防御前 vs 防御后的预测对比截图
- 防御成功案例：攻击后 "鸡" 87% → 高斯模糊 → 恢复 "狗" 92%

**对应代码**：`core/defense_engine.py:20-53` — 两种防御实现
```python
class DefenseEngine:
    def gaussian_defense(self, image, sigma):
        """高斯模糊：低通滤波破坏高频对抗扰动"""
        return image.filter(ImageFilter.GaussianBlur(radius=sigma))

    def jpeg_defense(self, image, quality):
        """JPEG 压缩：DCT 量化丢弃高频分量"""
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        return Image.open(buffer)
```

---

## 幻灯片 8：可视化与交互设计

**三列对比布局**：原始图像 → 噪声热力图 → 对抗样本

**三种可视化图表**（对应 `components/visualizations.py`）：
1. **扰动热力图**：取三通道平均绝对值，高亮扰动集中区域
2. **置信度对比柱状图**：原始 Top-1 vs 目标类别置信度直观对比
3. **PGD 收敛曲线**：目标类别置信度随迭代次数上升的折线图

**建议放入 PPT 的素材**：
- 完整的 Web UI 截图（攻击实验室页面 + 防御加固页面）
- 热力图的可视化效果说明

**对应代码**：`components/attack_tab.py:101-168` — 三列布局与图表渲染
```python
col_orig, col_heat, col_adv = st.columns(3)

with col_orig:
    st.subheader("原始图像")
    st.image(image, use_container_width=True)

with col_heat:
    st.subheader("噪声热力图")
    fig_heat = plot_perturbation_heatmap(perturbation)
    st.pyplot(fig_heat)

with col_adv:
    st.subheader("对抗样本")
    st.image(adv_image, use_container_width=True)
```

---

## 幻灯片 9：实验结果总结

| 场景 | 算法 | ε | 结果 |
|------|------|-----|------|
| 近类攻击 (banana → lemon) | FGSM | 0.03 | 成功，epsilon 小即可 |
| 远类攻击 (banana → cock) | FGSM | 0.05 | 难以跨越决策边界 |
| 远类攻击 (banana → cock) | PGD | 0.05, 20 iter | 稳定成功 |
| 防御 (高斯模糊 σ=2.0) | — | — | 可恢复原始分类 |

**关键发现**：
1. 像素空间攻击是实现 FGSM/PGD 的正确前提
2. PGD 是大类间、高置信度目标的可靠选择
3. 预处理防御（高斯模糊/JPEG 压缩）能有效破坏对抗扰动

---

## 幻灯片 10：项目亮点与总结

**技术亮点**：
- 发现并修复了归一化空间 clamp 的关键实现缺陷
- 完整实现了 FGSM → PGD 的递进攻击链路
- 构建了可交互的 Streamlit Web 演示平台
- 采用分层架构（core → components → app），模块职责清晰

**环境亮点**：
- 针对 NVIDIA RTX 5060（sm_120）的 CUDA 13.0 + PyTorch Nightly 适配
- 已部署至 Hugging Face Spaces，可在线体验

**参考文献**：
1. Goodfellow et al., *Explaining and Harnessing Adversarial Examples*, ICLR 2015
2. Madry et al., *Towards Deep Learning Models Resistant to Adversarial Attacks*, ICLR 2018
3. Xu et al., *Feature Squeezing: Detecting Adversarial Examples*, NDSS 2018

**对应代码**：`components/styles.py` — 自定义 UI 主题（Slate + Amber 配色）
```python
st.markdown("""
<style>
.stButton > button[kind="primary"] {
    background-color: #d97706;
    border-radius: 6px;
    font-weight: 600;
}
[data-testid="stSidebar"] {
    border-right: 1px solid #e2e8f0;
}
</style>
""", unsafe_allow_html=True)
```

---

> **使用建议**：每张幻灯片 1-2 分钟讲解，总时长控制在 15-20 分钟。核心技术公式（FGSM/PGD）务必出现，配合代码片段比纯文字更有说服力。如需截取代码，建议从本文档标记的行号中精确裁剪。
