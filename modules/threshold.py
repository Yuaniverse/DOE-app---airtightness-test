"""模組四：界線判定與風險驗證 (Threshold Determination)。

選定最佳參數後，透過互動式滑桿模擬判定界線 (Threshold) 的位置，
即時計算 False Negative / False Positive 風險，並以混淆矩陣可視化。
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


def calc_confusion_matrix(
    ok_dps: np.ndarray,
    ng_dps: np.ndarray,
    threshold: float,
) -> dict[str, int]:
    """根據判定界線計算混淆矩陣。

    判定規則：dP >= threshold → 系統判定 NG (Reject)；dP < threshold → 系統判定 OK (Pass)。

    Args:
        ok_dps: OK 良品的 dP 陣列。
        ng_dps: NG 不良品的 dP 陣列。
        threshold: 判定界線 t。

    Returns:
        包含 TP, TN, FP, FN 數值的字典。
    """
    # 真實 NG 且系統判定 NG (True Positive)
    tp = int(np.sum(ng_dps >= threshold))
    # 真實 NG 但系統判定 OK (False Negative - 漏網之魚)
    fn = int(np.sum(ng_dps < threshold))
    # 真實 OK 但系統判定 NG (False Positive - 誤殺良品)
    fp = int(np.sum(ok_dps >= threshold))
    # 真實 OK 且系統判定 OK (True Negative)
    tn = int(np.sum(ok_dps < threshold))

    return {"TP": tp, "TN": tn, "FP": fp, "FN": fn}


def render_threshold_determination() -> None:
    """渲染模組四的 Streamlit UI。"""
    st.header("⚖️ 模組四：界線判定與風險驗證")
    st.caption("拖動滑桿設定 dP 判定界線，即時觀察誤判風險與混淆矩陣")

    if st.session_state.get("best_param_row") is None:
        st.info("💡 請先完成模組三的參數優選，選出最佳參數組合後再進行界線判定。")
        return

    if st.session_state.get("result_df") is None or "ok_ids" not in st.session_state:
        st.error("❌ 缺少必要的計算數據，請重新執行模組二。")
        return

    best_row = st.session_state["best_param_row"]
    result_df = st.session_state["result_df"]
    ok_ids = st.session_state["ok_ids"]
    ng_ids = st.session_state["ng_ids"]

    # 顯示最佳參數
    st.info(
        f"🎯 當前最佳參數：**Pvac={best_row['Pvac']:.1f}**, "
        f"**Tvac={best_row['Tvac']:.1f}**, "
        f"**Tstab={best_row['Tstab']:.1f}**, "
        f"**Ttest={best_row['Ttest']:.1f}**"
    )

    # ── 提取最佳參數行的 10 台樣品 dP 值 ──
    step_id_cols = [f"Step{i}_ID" for i in range(1, 13)]
    step_dp_cols = [f"Step{i}_dP" for i in range(1, 13)]

    # 找到最佳參數對應的行
    best_run = best_row["Run_Order"]
    best_row_data = result_df[result_df["Run_Order"] == best_run]

    if len(best_row_data) == 0:
        st.error("❌ 無法找到最佳參數對應的實驗數據行。")
        return

    best_row_data = best_row_data.iloc[0]

    # 建立 ID → dP 映射
    id_dp_map: dict[str, float] = {}
    for id_col, dp_col in zip(step_id_cols, step_dp_cols):
        sample_id = str(best_row_data[id_col]).strip()
        dp_val = best_row_data[dp_col]
        if sample_id and pd.notna(dp_val):
            try:
                id_dp_map[sample_id] = float(dp_val)
            except (ValueError, TypeError):
                continue

    ok_dps = np.array([id_dp_map.get(sid, np.nan) for sid in ok_ids])
    ng_dps = np.array([id_dp_map.get(sid, np.nan) for sid in ng_ids])

    ok_valid = ok_dps[~np.isnan(ok_dps)]
    ng_valid = ng_dps[~np.isnan(ng_dps)]

    if len(ok_valid) == 0 or len(ng_valid) == 0:
        st.error("❌ 無法從最佳參數行提取足夠的 OK/NG dP 數據。")
        return

    # 顯示原始數據
    st.subheader("📊 最佳參數行 dP 數據")
    data_col1, data_col2 = st.columns(2)
    with data_col1:
        ok_data = pd.DataFrame({"樣品 ID": ok_ids, "dP 值": ok_dps})
        st.markdown("**✅ OK 良品 dP**")
        st.dataframe(ok_data, hide_index=True, use_container_width=True)
    with data_col2:
        ng_data = pd.DataFrame({"樣品 ID": ng_ids, "dP 值": ng_dps})
        st.markdown("**❌ NG 不良品 dP**")
        st.dataframe(ng_data, hide_index=True, use_container_width=True)

    st.divider()

    # ── 動態滑桿 ──
    all_dps = np.concatenate([ok_valid, ng_valid])
    dp_min = float(np.min(all_dps))
    dp_max = float(np.max(all_dps))

    # 確保有合理範圍
    if dp_min == dp_max:
        st.warning("⚠️ 所有 dP 值相同，無法進行界線判定。")
        return

    # 預設界線設在中間值
    default_threshold = (dp_min + dp_max) / 2

    st.subheader("🎚️ 判定界線設定")
    threshold = st.slider(
        "拖動滑桿設定判定界線 (t)",
        min_value=dp_min,
        max_value=dp_max,
        value=default_threshold,
        step=0.001,
        format="%.3f",
        help="dP ≥ t → 系統判定 NG (Reject)；dP < t → 系統判定 OK (Pass)",
        key="threshold_slider",
    )

    # ── 計算混淆矩陣 ──
    cm = calc_confusion_matrix(ok_valid, ng_valid, threshold)

    st.divider()

    # ── 風險指標儀表板 ──
    st.subheader("🚨 風險指標")
    risk_col1, risk_col2 = st.columns(2)

    with risk_col1:
        fn_color = "🔴" if cm["FN"] > 0 else "🟢"
        st.metric(
            f"{fn_color} 放過不良數 (False Negative)",
            f"{cm['FN']} 台",
            delta=f"漏網之魚" if cm["FN"] > 0 else "零漏放 ✓",
            delta_color="inverse" if cm["FN"] > 0 else "off",
        )
        if cm["FN"] > 0:
            st.error(
                f"⚠️ **嚴重警告**：有 {cm['FN']} 台不良品被系統誤判為良品放行！"
                f"這是氣密測試中最致命的錯誤，可能導致客訴！"
            )

    with risk_col2:
        fp_color = "🟡" if cm["FP"] > 0 else "🟢"
        st.metric(
            f"{fp_color} 誤殺良品數 (False Positive)",
            f"{cm['FP']} 台",
            delta=f"重工成本↑" if cm["FP"] > 0 else "零誤殺 ✓",
            delta_color="inverse" if cm["FP"] > 0 else "off",
        )
        if cm["FP"] > 0:
            st.warning(f"⚠️ 有 {cm['FP']} 台良品被系統誤殺，會增加產線重工成本。")

    st.divider()

    # ── 混淆矩陣 ──
    st.subheader("📊 混淆矩陣 (Confusion Matrix)")

    # 使用 Plotly 畫 2x2 混淆矩陣
    z_values = [[cm["TN"], cm["FP"]], [cm["FN"], cm["TP"]]]
    z_text = [
        [f"TN\n{cm['TN']} 台\n(正確放行)", f"FP\n{cm['FP']} 台\n(誤殺良品)"],
        [f"FN\n{cm['FN']} 台\n(漏放不良)", f"TP\n{cm['TP']} 台\n(正確攔截)"],
    ]

    # 配色：TN/TP 綠色，FN 紅色，FP 橙色
    color_scale = [
        [0, "#E8F5E9"],  # 0 → 淺綠
        [0.5, "#FFF9C4"],  # 中間 → 淺黃
        [1, "#FFCDD2"],  # 高 → 淺紅
    ]

    fig_cm = go.Figure(
        data=go.Heatmap(
            z=z_values,
            x=["系統判定: OK (Pass)", "系統判定: NG (Reject)"],
            y=["真實: OK 良品", "真實: NG 不良品"],
            text=z_text,
            texttemplate="%{text}",
            textfont={"size": 16},
            colorscale=color_scale,
            showscale=False,
            hoverinfo="skip",
        )
    )
    fig_cm.update_layout(
        title=f"判定界線 t = {threshold:.3f}",
        xaxis_title="系統判定結果",
        yaxis_title="真實狀態",
        height=350,
        yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(fig_cm, use_container_width=True)

    # ── dP 分布圖 (含界線) ──
    st.subheader("📈 dP 分布與判定界線")
    fig_dist = go.Figure()

    # OK 樣品散點
    fig_dist.add_trace(
        go.Scatter(
            x=list(range(1, len(ok_valid) + 1)),
            y=ok_valid,
            mode="markers+text",
            marker=dict(color="#4CAF50", size=14, symbol="circle"),
            name="OK 良品",
            text=[f"{v:.3f}" for v in ok_valid],
            textposition="top center",
        )
    )

    # NG 樣品散點
    fig_dist.add_trace(
        go.Scatter(
            x=list(range(len(ok_valid) + 1, len(ok_valid) + len(ng_valid) + 1)),
            y=ng_valid,
            mode="markers+text",
            marker=dict(color="#F44336", size=14, symbol="x"),
            name="NG 不良品",
            text=[f"{v:.3f}" for v in ng_valid],
            textposition="top center",
        )
    )

    # 判定界線
    fig_dist.add_hline(
        y=threshold,
        line_dash="dash",
        line_color="orange",
        line_width=3,
        annotation_text=f"t = {threshold:.3f}",
        annotation_position="top left",
    )

    fig_dist.update_layout(
        title="dP 散布圖與判定界線",
        xaxis_title="樣品序號",
        yaxis_title="dP 值",
        height=400,
        showlegend=True,
    )
    st.plotly_chart(fig_dist, use_container_width=True)
