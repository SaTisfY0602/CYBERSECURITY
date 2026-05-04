# 项目上下文速览：ResNet50 定向对抗攻击（第三、四阶段已完成）

> 本文件用于快速恢复项目上下文。下次打开 Claude 时，请先阅读本文件，了解当前进度。

---

## 1. 项目总体目标

基于 PyTorch 官方预训练 ResNet50，实现 FGSM/PGD 定向对抗样本攻击，最终交付一个 Streamlit 交互式 Web 应用，让用户上传图片、选择攻击算法与参数，实时观察模型被误导至指定类别的过程。

**当前进度**：阶段一至四均已完成。2026-05-04 完成全局视觉层美化（Slate + Amber 主题）。后期任务：布局结构层美化、深度测试与优化。

---

## 2. 阶段一完成内容（基线工程）

- 加载 `torchvision.models.resnet50(weights=ResNet50_Weights.DEFAULT)`，自动检测 CUDA 并挂载至 RTX 5060。
- 实现 `AdversarialModel` 类（`core/loadModel.py`），封装预处理、Top-k 推理、设备管理。
- 预处理管道：`Resize(256)` → `CenterCrop(224)` → `ToTensor()` → `Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])`。
- 推理使用 `torch.inference_mode()`，返回结构化 Dict（`topk_ids` / `topk_names` / `topk_confs`），便于 Web 端序列化。

---

## 3. 阶段二完成内容（对抗攻击引擎）

### 3.1 核心文件

- `core/attack_engine.py`：`AttackEngine` 类，继承 `AdversarialModel`。
- `test_cli.py`：本地命令行测试脚本（未入 git），支持交互式选择 FGSM/PGD 并输入参数。

### 3.2 FGSM 单步攻击（已修复关键 Bug）

**原始公式（定向攻击）**：

```
X_adv = X - ε · sign(∇_X J(X, Y_target))
```

**关键修复：攻击必须下沉至像素空间 `[0, 1]`**

- **Bug 症状**：对 banana（置信度 70%+）执行 FGSM，ε=0.03~0.1 均失败，banana 仍为 Top-1；ε=0.1 时置信度甚至从 9.97% 反弹至 21.48%。
- **根因**：旧代码在归一化空间（约 `[-2.1, 2.2]`）直接执行 `torch.clamp(tensor, 0, 1)`，将大量合法像素截断到边界，严重扭曲扰动方向。
- **修复方案**：
  1. 反归一化到像素空间：`pixel = original * std + mean`
  2. 在像素空间开启 `requires_grad=True`
  3. 重新归一化后送入模型（乘除法可微，Autograd 能传播梯度回像素空间）
  4. 反向传播得到 `grad_pixel`
  5. 在像素空间执行 FGSM：`pixel_adv = clamp(pixel - ε · sign(grad_pixel), 0, 1)`
  6. 最后再归一化：`adv = (pixel_adv - mean) / std`

**修复后效果**：

- 近邻类别（banana → lemon）：ε=0.03 即可成功。
- 远类别（banana → cock）：单步 FGSM 仍难以跨越决策边界。

### 3.3 PGD 迭代攻击（新增）

**公式**：

```
X_{t+1} = Π_ε ( clamp( X_t - α · sign(∇_X J(X_t, Y_target)), 0, 1 ) )
```

- 步长 `α = ε / 4`（默认）
- 每步执行投影 `Π_ε`：将扰动限制在 `L_∞` 距离不超过 ε 的邻域内
- 默认迭代 10~20 次

**效果**：

- 近邻类别：ε=0.03, iter=10 稳定成功
- 远类别（banana → cock）：ε=0.05, iter=20 成功率远高于 FGSM

### 3.4 实验验证

- 使用 `test_cli.py` 在本地命令行验证。
- 输出包含：原始 Top-5、对抗 Top-5、实际最大扰动值（应与输入 ε 基本一致）。

---

## 4. 当前代码结构

```
D:\Adversarial_example_attack\
├── .streamlit/
│   └── config.toml           # Streamlit 主题配色（Slate + Amber）
├── core/
│   ├── loadModel.py          # AdversarialModel 基类 + @st.cache_resource 缓存工厂
│   ├── attack_engine.py      # AttackEngine：FGSM/PGD 攻击 + PGD 迭代历史记录
│   └── defense_engine.py     # DefenseEngine：高斯模糊 + JPEG 压缩预处理防御
├── components/
│   ├── __init__.py
│   ├── attack_tab.py         # 攻击实验室 Tab UI（侧边栏 + 三列展示 + 动态提示）
│   ├── defense_tab.py        # 防御加固 Tab UI（防御前后对比）
│   ├── styles.py             # CSS 注入模块（侧边栏/按钮/卡片/标题/页脚）
│   └── visualizations.py     # Matplotlib 图表：热力图、柱状图、PGD 收敛曲线
├── utils/
│   └── imagenet_labels.py    # ImageNet 1000 类标签（下拉选择框数据源）
├── tests/
│   ├── test_imagenet_labels.py
│   ├── test_visualizations.py
│   └── test_defense_engine.py
├── docs/
│   └── superpowers/
│       ├── specs/2026-05-04-ui-global-theme-design.md
│       └── plans/2026-05-04-ui-global-theme.md
├── app.py                    # Streamlit 主应用入口（双 Tab 导航）
├── testset/                  # 测试图片目录
├── prompt.md                 # 本文件（项目上下文速览）
├── project_plan.md           # 完整项目计划
├── report.md                 # 实验报告
├── NOTE.md                   # 核心原理笔记
├── EXPERIMENT_LOG.md         # 实验日志
└── README.md                 # 项目说明
```

---

## 5. 阶段三完成内容（Streamlit Web UI，2026-05-03）

### 5.1 应用架构

- 分层设计：`core/` 算法层 + `components/` UI 层 + `app.py` 入口
- 模型缓存：`@st.cache_resource` 缓存 `AttackEngine` 单例，避免重复加载权重
- 双 Tab 导航：「攻击实验室」与「防御加固」
- 图片输入纯内存处理（BytesIO），不写入硬盘

### 5.2 攻击实验室 Tab

- 侧边栏：图片来源（上传/示例）、算法选择（FGSM/PGD）、目标类别下拉框（1000 类）、ε 滑块（0~0.1）、PGD 迭代次数（条件显示）
- 动态提示：基于原始 Top-1 置信度给出算法和参数建议
- 主展示区三列布局：原始图像 + 噪声热力图（Matplotlib heatmap）+ 对抗样本
- 置信度对比柱状图 + PGD 迭代收敛曲线
- 通过 `st.session_state` 传递对抗样本到防御 Tab

### 5.3 防御加固 Tab

- 输入来源：自动继承攻击 Tab 样本 或 手动上传
- 防御方法：高斯模糊（可调 σ） / JPEG 压缩（可调质量因子）
- 防御前后双列对比：图像 + Top-5 预测
- 防御成功/失败判定（恢复为原始 Top-1 类别即为成功）

### 5.4 关键 Bug 修复记录

| 问题                                                              | 根因                                                        | 修复                                                                     |
| --------------------------------------------------------------- | --------------------------------------------------------- | ---------------------------------------------------------------------- |
| `AttributeError: no attribute 'generate_targeted_adversarial'`  | `get_adversarial_model()` 返回基类 `AdversarialModel`         | 改为返回 `AttackEngine()`，函数内懒导入避免循环依赖                                     |
| `RuntimeError: Can't call numpy() on Tensor that requires grad` | FGSM 返回的 `adv_tensor` 未 `.detach()`                       | `pixel_adv.clamp(0,1).detach()` + `tensor_to_image` 内部加 `.detach()` 兜底 |
| `List` 未导入                                                      | `generate_targeted_pgd_with_history` 返回类型用了 `List[float]` | `from typing import List`                                              |
| 中文字体缺失（Matplotlib 警告）                                           | DejaVu Sans 不含 CJK 字形                                     | 图表标题改为英文                                                               |

---

## 6. 后期任务

### 6.1 深度测试与优化

- 边界测试：ε=0 和 ε=0.1 极值情况、PGD 高迭代次数（50+）时的显存与耗时
- 多图片批量测试：testset 全量对抗样本生成，统计攻击成功率
- 性能优化：PGD 迭代中是否可复用梯度计算、显存释放策略
- 对比实验：FGSM vs PGD 在不同 ε 和类别跨度下的成功率统计

### 6.2 UI 美化

#### A. 全局视觉层（2026-05-04 完成）

- 配色方案：Slate + Amber 现代极简风
  - `primaryColor="#d97706"` / `backgroundColor="#f8fafc"` / `textColor="#1e293b"`
- 新增文件：
  - `.streamlit/config.toml` — 主题配色
  - `components/styles.py` — CSS 注入模块（侧边栏/按钮/卡片/标题/页脚）
- Matplotlib 图表颜色同步（柱状图 `#1e293b`/`#d97706`，收敛曲线 `#d97706`）
- 设计文档：`docs/superpowers/specs/2026-05-04-ui-global-theme-design.md`
- 实现计划：`docs/superpowers/plans/2026-05-04-ui-global-theme.md`

#### B. 布局结构层（待完成）

- 页面布局优化：卡片式容器（`st.container(border=True)`）、响应式间距
- 添加 Logo / 页脚 / 说明文档折叠区

---

## 7. 关键技术约束

1. **像素空间攻击**：任何 `clamp` 操作必须在像素空间 `[0, 1]` 执行
2. **梯度追踪**：在像素空间开启 `requires_grad=True`
3. **显存管理**：RTX 5060 笔记本 GPU 显存有限，注意释放
4. **缓存机制**：`@st.cache_resource` 缓存模型实例
5. **代码风格**：面向对象编程，关键步骤有中文注释

---

## 8. 关键公式速查

**FGSM 定向攻击**：

```
X_adv = X - ε · sign(∇_X J(X, Y_target))
```

**PGD 迭代攻击**：

```
X_{t+1} = Π_ε ( clamp( X_t - α · sign(∇_X J(X_t, Y_target)), 0, 1 ) )
其中 α = ε / 4，Π_ε 为 L_∞ 投影
```

**反归一化**：

```
pixel = normalized × std + mean
mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
```

---

## 9. 参考文献

1. Goodfellow et al. (2015). *Explaining and Harnessing Adversarial Examples*. ICLR 2015. https://arxiv.org/abs/1412.6572
2. Madry et al. (2018). *Towards Deep Learning Models Resistant to Adversarial Attacks*. ICLR 2018. https://arxiv.org/abs/1706.06083
3. Xu et al. (2017). *Feature Squeezing: Detecting Adversarial Examples in Deep Neural Networks*. NDSS 2018. https://arxiv.org/abs/1704.01155
