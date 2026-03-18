"""氣密測試 DOE 實驗系統 — 主入口程式。

此 Streamlit 應用程式包含四個核心模組：
1. DOE 實驗表格產生器
2. 響應變數計算
3. 可調式參數優選決策
4. 界線判定與風險驗證
"""

import streamlit as st

# ── 頁面設定 ──
st.set_page_config(
    page_title="氣密測試 DOE 實驗系統",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── 全域樣式 ──
st.markdown(
    """
    <style>
    /* 主標題區塊 */
    .main-header {
        background: linear-gradient(135deg, #1a237e 0%, #283593 50%, #3949ab 100%);
        color: white;
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.15);
    }
    .main-header h1 {
        color: white !important;
        margin: 0 !important;
        font-size: 1.8rem !important;
    }
    .main-header p {
        color: #c5cae9 !important;
        margin: 0.3rem 0 0 0 !important;
        font-size: 0.95rem !important;
    }

    /* 分頁標籤美化 */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 12px 24px;
        font-size: 1rem;
        font-weight: 600;
    }

    /* Metric 卡片風格 */
    div[data-testid="stMetric"] {
        background: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 8px;
        padding: 12px 16px;
    }

    /* 成功訊息強調 */
    .stSuccess {
        border-left: 4px solid #4CAF50;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── 標題 ──
st.markdown(
    """
    <div class="main-header">
        <h1>🔬 氣密測試 DOE 實驗系統</h1>
        <p>Design of Experiments — 尋找最佳氣密測試參數，訂定安全判定界線</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── 初始化 session_state ──
if "doe_df" not in st.session_state:
    st.session_state["doe_df"] = None
if "raw_df" not in st.session_state:
    st.session_state["raw_df"] = None
if "result_df" not in st.session_state:
    st.session_state["result_df"] = None
if "best_param_row" not in st.session_state:
    st.session_state["best_param_row"] = None
if "axial_df" not in st.session_state:
    st.session_state["axial_df"] = None
if "merged_doe_df" not in st.session_state:
    st.session_state["merged_doe_df"] = None
if "ccd_selected_factors" not in st.session_state:
    st.session_state["ccd_selected_factors"] = []
if "ccd_design_type" not in st.session_state:
    st.session_state["ccd_design_type"] = "CCC"

# ── 模組分頁 ──
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "📋 模組一：DOE 表格產生器",
        "📐 模組二：響應變數計算",
        "🎯 模組三：參數優選決策",
        "⚖️ 模組四：界線判定",
        "📊 模組五：基礎統計分析",
    ]
)

# 動態匯入各模組（避免循環匯入）
from modules.doe_generator import render_doe_generator
from modules.response_calc import render_response_calc
from modules.param_selector import render_param_selector
from modules.threshold import render_threshold_determination
from modules.stats_analysis import render_stats_analysis

with tab1:
    render_doe_generator()

with tab2:
    render_response_calc()

with tab3:
    render_param_selector()

with tab4:
    render_threshold_determination()

with tab5:
    render_stats_analysis()

# ── 頁尾 ──
st.divider()
st.caption("© 2026 氣密測試 DOE 實驗系統 | 版本 1.0.0")
