"""模組五：基礎統計分析 (Basic Statistical Analysis)。

提供使用者針對現有計算出的響應變數，進行二階迴歸模型分析，
展示 R-sq、ANOVA 表格與柏拉圖 (Pareto Chart)，快速釐清各因子顯著性。
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import statsmodels.formula.api as smf
import statsmodels.api as sm


def render_stats_analysis() -> None:
    """渲染模組五的 Streamlit UI。"""
    st.header("📊 模組五：基礎統計分析")
    st.caption("分析四個控制因子 (Pvac, Tvac, Tstab, Ttest) 對指定響應變數的主效應與交互作用顯著性。")

    # 檢查是否有資料
    if st.session_state.get("result_df") is None:
        st.warning("⚠️ 尚無計算資料！請先至「模組二：響應變數計算」上傳檔案並執行計算。")
        return

    result_df = st.session_state["result_df"]

    # 定義響應變數的選項
    response_options = {
        "Y1 (總測試時間)": "Y1_TotalTime",
        "AUC (鑑別正確率)": "Y2_AUC",
        "GapQ (邊界安全裕度)": "Y3_GapQ",
        "Y4 (穩定性全距)": "Y4_Stability",
    }

    col_sel, _ = st.columns([1, 2])
    with col_sel:
        selected_label = st.selectbox(
            "📍 選擇欲分析的響應變數：",
            options=list(response_options.keys()),
        )

    target_col = response_options[selected_label]

    st.divider()

    # 檢查該變數變異數是否為零
    if result_df[target_col].var() == 0:
        st.error(f"❌ 選擇的響應變數 ({selected_label}) 變異數為零（所有數值皆相同），無法進行統計迴歸分析。")
        return

    # 構建 OLS 迴歸模型：主效應 + 二階交互作用
    # 變數：Pvac, Tvac, Tstab, Ttest
    # 確保資料中沒有 NaN 且型別正確
    model_data = result_df[["Pvac", "Tvac", "Tstab", "Ttest", target_col]].copy()
    model_data.columns = ["Pvac", "Tvac", "Tstab", "Ttest", "Target"]
    model_data = model_data.dropna()

    if len(model_data) < 5:
        st.error("❌ 有效數據筆數過少，無法配適模型。")
        return

    formula = "Target ~ (Pvac + Tvac + Tstab + Ttest)**2"
    
    with st.spinner("正在計算統計模型..."):
        try:
            model = smf.ols(formula=formula, data=model_data).fit()
            
            # 顯示模型 R-square
            col_r2, _ = st.columns([1, 4])
            with col_r2:
                st.metric(
                    label="R-squred ($R^2$)",
                    value=f"{model.rsquared * 100:.2f}%",
                    help="模型解釋變異的比例，越接近 100% 代表配適度越好。",
                )
            
            # --- 產生 ANOVA 表 ---
            anova_table = sm.stats.anova_lm(model, typ=2)
            # rename for readability
            anova_table = anova_table.rename(columns={"PR(>F)": "p-value", "sum_sq": "Sum Sq", "df": "DF", "F": "F-Value"})

            st.subheader("📋 變異數分析 (ANOVA)")
            
            # 建立 p-value 格式化條件
            def highlight_pvalue(val):
                color = '#ff4b4b' if pd.notna(val) and val < 0.05 else ''
                font_weight = 'bold' if pd.notna(val) and val < 0.05 else 'normal'
                return f'color: {color}; font-weight: {font_weight}'

            formatted_anova = anova_table.style.format({
                "Sum Sq": "{:.4f}",
                "F-Value": "{:.2f}",
                "p-value": "{:.4f}",
                "DF": "{:.0f}"
            }).applymap(highlight_pvalue, subset=["p-value"])
            
            st.dataframe(formatted_anova, use_container_width=True)
            st.caption("🔴 p-value < 0.05 代表該因子/交互作用具備統計顯著性。")

            st.divider()

            # --- 繪製 Pareto Chart (柏拉圖) ---
            st.subheader("📉 標準化效應柏拉圖 (Pareto Chart)")

            # 取絕對 t 值
            t_values = abs(model.tvalues).drop("Intercept")
            t_values = t_values.sort_values(ascending=False)
            
            # t 分配自由度
            df_resid = model.df_resid
            # α = 0.05 的臨界 t 值 (雙尾)
            t_critical = stats.t.ppf(1 - 0.025, df_resid)
            
            # 建立圖表 DataFrame
            pareto_df = pd.DataFrame({
                "Term": t_values.index,
                "Effect": t_values.values,
            })

            # 使用 Plotly
            fig = px.bar(
                pareto_df, 
                x="Effect", 
                y="Term", 
                orientation="h",
                text="Effect",
                labels={"Effect": "標準化效應 (絕對 t 值)", "Term": "因子/交互作用"},
            )
            
            fig.update_traces(
                texttemplate='%{text:.2f}', 
                textposition='outside',
                marker_color='#5c6bc0'
            )
            
            # 加入紅色基準線
            fig.add_vline(
                x=t_critical, 
                line_width=2, 
                line_dash="dash", 
                line_color="red",
                annotation_text=f"p=0.05 (Crit: {t_critical:.2f})",
                annotation_position="top right"
            )

            fig.update_layout(
                yaxis={'categoryorder': 'total ascending'},
                height=500,
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(showgrid=True, gridcolor='#e0e0e0'),
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"❌ 模型計算發生錯誤：{e}")
            st.info("請確認數據中變數是否皆有正常值，或樣本數量是否充足。")
