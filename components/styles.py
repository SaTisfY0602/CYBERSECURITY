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
