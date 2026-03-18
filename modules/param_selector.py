"""模組三：可調式參數優選決策 (Parameter Selection Filter)。

根據四階層過濾器（AUC、Y4 穩定性、GapQ 安全裕度、決策偏好），
自動篩選並推薦最佳氣密測試參數組合。
"""

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


def render_param_selector() -> None:
    """渲染模組三的 Streamlit UI。"""
    st.header("🎯 模組三：參數優選決策")
    st.caption("四階層過濾器篩選最佳測試參數，支援產能與安全度之間的彈性取捨")

    if st.session_state.get("result_df") is None:
        st.info("💡 請先完成模組二的響應變數計算，才能進行參數優選。")
        return

    result_df: pd.DataFrame = st.session_state["result_df"].copy()

    # 確認必要欄位
    needed_cols = ["Y1_TotalTime", "Y2_AUC", "Y3_GapQ", "Y4_Stability"]
    missing = [c for c in needed_cols if c not in result_df.columns]
    if missing:
        st.error(f"❌ 資料中缺少必要欄位：{', '.join(missing)}，請回模組二重新計算。")
        return

    # ── Y4 分布圖（供使用者參考） ──
    st.subheader("📊 Y4 穩定性分布概覽")
    fig_y4 = px.histogram(
        result_df,
        x="Y4_Stability",
        nbins=15,
        title="Y4 穩定性全距分布",
        labels={"Y4_Stability": "Y4 (穩定性全距)"},
        color_discrete_sequence=["#4472C4"],
    )
    q3_y4 = float(result_df["Y4_Stability"].quantile(0.75))
    fig_y4.add_vline(
        x=q3_y4,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Q3 = {q3_y4:.4f}",
    )
    st.plotly_chart(fig_y4, width="stretch")
    st.caption("Y4 上限預設採 Q3 作為經驗門檻，屬啟發式設定，建議仍依工程風險自行調整。")

    st.divider()

    # ── 四階層過濾器 ──
    st.subheader("🔧 過濾條件設定")

    col_filter1, col_filter2 = st.columns(2)

    with col_filter1:
        # 第一階：AUC 下限
        auc_min = st.number_input(
            "🥇 第一階：AUC 下限（鑑別力護欄）",
            min_value=0.0,
            max_value=1.0,
            value=1.0,
            step=0.05,
            format="%.2f",
            help="低於此 AUC 值的參數組合將被淘汰。預設 1.0 表示只保留 100% 正確鑑別的參數。",
            key="auc_min_filter",
        )

        # 第三階：GapQ 下限
        gap_q_min = st.number_input(
            "🛡️ 第三階：GapQ 下限（安全裕度護欄）",
            value=0.0,
            step=0.001,
            format="%.4f",
            help="GapQ > 0 代表 OK/NG 分佈不重疊，越大越安全。",
            key="gap_q_min_filter",
        )

    with col_filter2:
        # 第二階：Y4 上限
        y4_max = st.number_input(
            "📏 第二階：Y4 上限（穩定性護欄）",
            value=round(q3_y4, 4),
            step=0.001,
            format="%.4f",
            help=f"預設為 Q3 值 ({q3_y4:.4f})。Y4 越小代表設備越穩定。",
            key="y4_max_filter",
        )

        # 第四階：決策偏好
        sort_pref = st.radio(
            "⚖️ 第四階：決策偏好排序",
            options=[
                "優先追求產線產能 (依 Y1 遞增排序)",
                "優先追求測試安全度 (依 GapQ 遞減排序)",
            ],
            index=0,
            key="sort_preference",
        )

    st.divider()

    # ── 動態過濾 ──
    filtered_df = result_df.copy()

    # 第一階
    filtered_df = filtered_df[filtered_df["Y2_AUC"] >= auc_min]
    after_stage1 = len(filtered_df)

    # 第二階
    filtered_df = filtered_df[filtered_df["Y4_Stability"] <= y4_max]
    after_stage2 = len(filtered_df)

    # 第三階
    filtered_df = filtered_df[filtered_df["Y3_GapQ"] >= gap_q_min]
    after_stage3 = len(filtered_df)

    # 過濾進程
    st.subheader("📉 過濾進程")
    p1, p2, p3, p4 = st.columns(4)
    with p1:
        st.metric("原始總數", f"{len(result_df)} 組")
    with p2:
        st.metric(
            "AUC 篩後",
            f"{after_stage1} 組",
            delta=f"-{len(result_df) - after_stage1}",
            delta_color="inverse",
        )
    with p3:
        st.metric(
            "Y4 篩後",
            f"{after_stage2} 組",
            delta=f"-{after_stage1 - after_stage2}",
            delta_color="inverse",
        )
    with p4:
        st.metric(
            "GapQ 篩後",
            f"{after_stage3} 組",
            delta=f"-{after_stage2 - after_stage3}",
            delta_color="inverse",
        )

    if len(filtered_df) == 0:
        st.warning("⚠️ 沒有參數組合通過所有篩選條件！請放寬過濾門檻。")
        return

    # 第四階：排序
    if "Y1" in sort_pref:
        filtered_df = filtered_df.sort_values("Y1_TotalTime", ascending=True)
    else:
        filtered_df = filtered_df.sort_values("Y3_GapQ", ascending=False)

    filtered_df = filtered_df.reset_index(drop=True)

    # ── 候選參數展示 ──
    st.subheader("📋 候選參數表")
    display_cols = [
        "Run_Order", "Std_Order", "Point_Type",
        "Pvac", "Tvac", "Tstab", "Ttest",
        "Y1_TotalTime", "Y2_AUC", "Y3_GapQ", "Y4_Stability",
    ]
    st.dataframe(
        filtered_df[display_cols].style.format(
            {
                "Y1_TotalTime": "{:.1f}",
                "Y2_AUC": "{:.3f}",
                "Y3_GapQ": "{:.4f}",
                "Y4_Stability": "{:.4f}",
                "Pvac": "{:.1f}",
                "Tvac": "{:.1f}",
                "Tstab": "{:.1f}",
                "Ttest": "{:.1f}",
            }
        ),
        width="stretch",
    )

    # ── 最佳參數推薦 ──
    best = filtered_df.iloc[0]
    st.divider()

    st.success("### 👑 系統推薦最佳參數組合")
    b1, b2, b3, b4 = st.columns(4)
    with b1:
        st.metric("Pvac", f"{best['Pvac']:.1f} kPa")
    with b2:
        st.metric("Tvac", f"{best['Tvac']:.1f} sec")
    with b3:
        st.metric("Tstab", f"{best['Tstab']:.1f} sec")
    with b4:
        st.metric("Ttest", f"{best['Ttest']:.1f} sec")

    r1, r2, r3, r4 = st.columns(4)
    with r1:
        st.metric("Y1 總時間", f"{best['Y1_TotalTime']:.1f} sec")
    with r2:
        st.metric("AUC 鑑別率", f"{best['Y2_AUC']:.3f}")
    with r3:
        st.metric("GapQ 安全裕度", f"{best['Y3_GapQ']:.4f}")
    with r4:
        st.metric("Y4 穩定性全距", f"{best['Y4_Stability']:.4f}")

    # 儲存最佳參數到 session_state 供模組四使用
    st.session_state["best_param_row"] = best
    st.session_state["filtered_df"] = filtered_df
