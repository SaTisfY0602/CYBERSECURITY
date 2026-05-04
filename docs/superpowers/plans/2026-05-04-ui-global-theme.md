# UI Global Theme Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Apply Modern Minimal (Slate + Amber) theme to the Streamlit app via config.toml, CSS injection, and Matplotlib color sync.

**Architecture:** Three layers work together — `config.toml` sets Streamlit's native theme variables; `styles.py` injects a `<style>` block for elements Streamlit doesn't expose; `visualizations.py` syncs Matplotlib rcParams to the same palette.

**Tech Stack:** Streamlit theme config, CSS, Matplotlib rcParams

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `.streamlit/config.toml` | Create | 5 theme variables (primaryColor, backgroundColor, secondaryBackgroundColor, textColor, font) |
| `components/styles.py` | Create | `inject_custom_css()` returns `<style>` string for sidebar, buttons, cards, titles, footer |
| `app.py` | Modify | Import + call `inject_custom_css()` after `set_page_config` |
| `components/visualizations.py` | Modify | Set `plt.rcParams` + update bar/line colors to slate-amber palette |

---

### Task 1: Create Streamlit theme config

**Files:**
- Create: `.streamlit/config.toml`

- [ ] **Step 1: Write the config file**

```toml
[theme]
primaryColor = "#d97706"
backgroundColor = "#f8fafc"
secondaryBackgroundColor = "#ffffff"
textColor = "#1e293b"
font = "sans serif"
```

- [ ] **Step 2: Verify Streamlit loads the config**

Run: `streamlit run app.py` (start, then Ctrl+C after it boots)
Expected: No errors. Streamlit prints "You can now view your Streamlit app..." with no config-related warnings.

- [ ] **Step 3: Commit**

```bash
git add .streamlit/config.toml
git commit -m "feat: add Streamlit theme config (Slate + Amber palette)"
```

---

### Task 2: Create CSS injection module

**Files:**
- Create: `components/styles.py`

- [ ] **Step 1: Write the CSS module**

```python
"""
components/styles.py
注入自定义 CSS 以精修 Streamlit 原生组件样式。
配色基于 Slate + Amber 现代极简风。
"""

import streamlit as st


def inject_custom_css() -> None:
    """将精修样式注入 Streamlit 页面。"""
    st.markdown(
        """
        <style>
        /* ===== 侧边栏 ===== */
        [data-testid="stSidebar"] {
            border-right: 1px solid #e2e8f0;
        }
        [data-testid="stSidebar"] .stRadio > div {
            gap: 0.5rem;
        }
        [data-testid="stSidebar"] .stSlider {
            margin-bottom: 1.25rem;
        }

        /* ===== 主按钮 ===== */
        .stButton > button[kind="primary"] {
            background-color: #d97706;
            border: none;
            border-radius: 6px;
            color: white;
            font-weight: 600;
            transition: background-color 0.15s ease;
        }
        .stButton > button[kind="primary"]:hover {
            background-color: #b45309;
        }

        /* ===== 卡片容器 (border=True 的 container) ===== */
        [data-testid="stVerticalBlockBorderWrapper"] {
            background-color: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 6px;
            box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04);
        }

        /* ===== 标题层级 ===== */
        h1 {
            font-weight: 700;
            color: #0f172a;
        }
        h2 {
            font-weight: 600;
            color: #1e293b;
        }
        h3 {
            font-weight: 500;
            color: #334155;
        }

        /* ===== 页脚 ===== */
        footer {
            text-align: center;
            font-size: 0.8rem;
            color: #94a3b8;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
```

- [ ] **Step 2: Verify syntax**

Run: `python -m py_compile components/styles.py`
Expected: No output (compile success).

- [ ] **Step 3: Commit**

```bash
git add components/styles.py
git commit -m "feat: add CSS injection module for UI polish"
```

---

### Task 3: Wire CSS injection into app.py

**Files:**
- Modify: `app.py`


当前 `app.py` 内容：
```python
"""
app.py
Streamlit 主应用入口。
提供"攻击实验室"与"防御加固"两个 Tab，整合模型缓存与组件调用。
"""

import streamlit as st

from core.loadModel import get_adversarial_model
from components.attack_tab import render_attack_tab
from components.defense_tab import render_defense_tab


def main():
    st.set_page_config(
        page_title="ResNet50 对抗攻击演示",
        page_icon=None,
        layout="wide",
    )

    st.title("ResNet50 对抗攻击与防御演示")
```

- [ ] **Step 1: Add import and CSS call**

In `app.py`, add the import line:

```python
from components.styles import inject_custom_css
```

After `st.set_page_config(...)` add:

```python
inject_custom_css()
```

Complete `main()` function after changes:

```python
def main():
    st.set_page_config(
        page_title="ResNet50 对抗攻击演示",
        page_icon=None,
        layout="wide",
    )

    inject_custom_css()

    st.title("ResNet50 对抗攻击与防御演示")
    # ... rest remains unchanged
```

- [ ] **Step 2: Verify app starts**

Run: `python -m streamlit run app.py` (start, then Ctrl+C after it boots)
Expected: No import errors, app launches normally.

- [ ] **Step 3: Commit**

```bash
git add app.py
git commit -m "feat: wire CSS injection into app entry point"
```

---

### Task 4: Update Matplotlib chart colors

**Files:**
- Modify: `components/visualizations.py`

- [ ] **Step 1: Add rcParams block at module level**

After the existing imports, add:

```python
import matplotlib as mpl

# 全局 Matplotlib 样式，与 Slate + Amber 主题统一
mpl.rcParams.update({
    "axes.edgecolor": "#e2e8f0",
    "axes.labelcolor": "#475569",
    "xtick.color": "#64748b",
    "ytick.color": "#64748b",
    "figure.dpi": 120,
    "font.size": 10,
})
```

- [ ] **Step 2: Update bar chart colors**

In `plot_confidence_bar_chart`, change:
```python
colors = ["steelblue", "coral"]
```
to:
```python
colors = ["#1e293b", "#d97706"]
```

- [ ] **Step 3: Update convergence curve color**

In `plot_pgd_convergence_curve`, change:
```python
ax.plot(iterations, target_confs, marker="o", color="coral", linewidth=2)
```
to:
```python
ax.plot(iterations, target_confs, marker="o", color="#d97706", linewidth=2)
```

And change grid color — add to the existing `ax.grid(...)` line:
```python
ax.grid(True, linestyle="--", alpha=0.5, color="#e2e8f0")
```

- [ ] **Step 4: Verify visually**

Run: `python -m streamlit run app.py`
Expected: Charts render with:
- Bar chart: slate (`#1e293b`) left bar, amber (`#d97706`) right bar
- Heatmap: hot colormap (unchanged)
- Convergence curve: amber (`#d97706`) line, light gray grid

- [ ] **Step 5: Commit**

```bash
git add components/visualizations.py
git commit -m "feat: sync Matplotlib chart colors with Slate + Amber theme"
```

---

### Task 5: Integration verification

- [ ] **Step 1: Full app startup**

Run: `python -m streamlit run app.py`
Expected: No import errors, no config warnings, no CSS injection errors, no Matplotlib warnings.

- [ ] **Step 2: Check browser rendering**

Open `http://localhost:8501` and verify:
- Primary button ("生成对抗样本") is amber (#d97706)
- Sidebar has right border
- Cards have white bg + subtle shadow
- Title h1 is bold, h2/h3 lighter
- Page background is light gray (#f8fafc)

- [ ] **Step 3: Test attack flow**

Upload an image or select sample, run FGSM attack. Verify:
- Bar chart shows slate + amber bars
- Heatmap renders correctly
- Page layout is not broken

- [ ] **Step 4: Final commit (if any tweaks needed)**

```bash
git add -A
git commit -m "feat: verify UI global theme integration"
```
