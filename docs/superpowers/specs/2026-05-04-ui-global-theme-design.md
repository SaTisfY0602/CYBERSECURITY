# UI Global Theme Design — Modern Minimal (Slate + Amber)

**Date:** 2026-05-04
**Scope:** `.streamlit/config.toml` + CSS 注入 + Matplotlib 配色统一
**Status:** approved

## 1. Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Visual style | Modern Minimal (Light) | Clean, academic, professional |
| Primary color | `#d97706` (Amber) | Warm accent, clear attack/defense state signaling |
| Text color | `#1e293b` (Slate-800) | Softer than pure black, reduces eye strain |
| Background | `#f8fafc` (Slate-50) | Light gray with subtle warmth |
| Card background | `#ffffff` (White) | Maximizes contrast for image display and charts |
| Danger | `#ef4444` (Red-500) | Standard error/danger signaling |
| Success | `#16a34a` (Green-600) | Standard success signaling |

## 2. Files

### 2.1 New: `.streamlit/config.toml`

```toml
[theme]
primaryColor = "#d97706"
backgroundColor = "#f8fafc"
secondaryBackgroundColor = "#ffffff"
textColor = "#1e293b"
font = "sans serif"
```

### 2.2 New: `components/styles.py`

Function `inject_custom_css()` returning a `<style>` block string covering:
- Sidebar: top divider line, parameter group spacing
- Primary button: amber bg, 6px radius, hover darken
- Card containers: white bg, light border, 4px radius, subtle shadow
- Title hierarchy: h1 weight 700, h2 weight 600, h3 weight 500
- Footer: centered, small, light gray

### 2.3 Modified: `app.py`

- Import `inject_custom_css` from `components/styles`
- Call `inject_custom_css()` right after `st.set_page_config()`, wrapping in `st.markdown(..., unsafe_allow_html=True)`

### 2.4 Modified: `components/visualizations.py`

- Set `matplotlib.rcParams` at module level:
  - `axes.edgecolor`: `#e2e8f0`
  - `axes.labelcolor`: `#475569`
  - `xtick.color` / `ytick.color`: `#64748b`
  - `figure.dpi`: `120`
- Bar chart: `#1e293b` for original category, `#d97706` for target/adversarial
- Convergence line: `#d97706`
- Grid: `#f1f5f9`
- All English titles preserved (no CJK font issue)

## 3. What we do NOT touch

- No HTML component replacement
- No Streamlit native widget styling beyond CSS classes
- No layout restructuring (columns, tabs stay as-is)
- No chart type changes

## 4. Verification

- [ ] `streamlit run app.py` starts without errors
- [ ] Sidebar shows amber-accented primary button
- [ ] Main area cards have white bg + subtle shadow
- [ ] Matplotlib charts use slate + amber palette (bar chart)
- [ ] No console warnings from Streamlit or Matplotlib
- [ ] Theme renders correctly in browser (Chrome)
