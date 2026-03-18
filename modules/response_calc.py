"""模組二：響應變數計算 (Response Variables Calculation)。

讀取測試完畢的數據，計算四個核心響應變數 (Y1, Y2/AUC, Y3/GapQ, Y4)，
供工程師決策最佳氣密測試參數。
"""

from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st

from utils.excel_export import export_formatted_xlsx


# ─────────────────────────── 核心計算函式 ───────────────────────────


def calc_y1(df: pd.DataFrame) -> pd.Series:
    """計算 Y1 (總測試時間)：Y1 = Tvac + Tstab + Ttest。

    Args:
        df: 含 Tvac, Tstab, Ttest 欄位的 DataFrame。

    Returns:
        Y1 的 Series。
    """
    return df["Tvac"] + df["Tstab"] + df["Ttest"]


def calc_y4(df: pd.DataFrame) -> pd.Series:
    """計算 Y4 (穩定性全距)：基準良品三點定錨值的 Max - Min。

    使用 Step1_dP, Step6_dP, Step12_dP（基準良品位置）。

    Args:
        df: 含 Step1_dP, Step6_dP, Step12_dP 數值欄位的 DataFrame。

    Returns:
        Y4 的 Series。
    """
    anchor_cols = ["Step1_dP", "Step6_dP", "Step12_dP"]
    anchor_values = df[anchor_cols].astype(float)
    return anchor_values.max(axis=1) - anchor_values.min(axis=1)


def calc_auc_score(
    df: pd.DataFrame,
    ok_ids: list[str],
    ng_ids: list[str],
) -> pd.Series:
    """計算 AUC (鑑別正確率/Y2)：向量化的 25 次配對比較。

    使用 NumPy broadcasting 避免雙重迴圈。
    記分規則：NG_dP > OK_dP → 1, NG_dP == OK_dP → 0.5, NG_dP < OK_dP → 0。

    Args:
        df: 含量測數據（Step*_ID, Step*_dP）的 DataFrame。
        ok_ids: 5 台 OK 良品的 ID 清單。
        ng_ids: 5 台 NG 不良品的 ID 清單。

    Returns:
        AUC 的 Series（每組實驗一個值）。
    """
    n_rows = len(df)
    auc_scores = np.zeros(n_rows)

    # 建立 ID → dP 的映射（每列）
    step_id_cols = [f"Step{i}_ID" for i in range(1, 13)]
    step_dp_cols = [f"Step{i}_dP" for i in range(1, 13)]

    for row_idx in range(n_rows):
        # 取出該列所有 (ID, dP) 配對
        id_dp_map: dict[str, float] = {}
        for id_col, dp_col in zip(step_id_cols, step_dp_cols):
            sample_id = str(df.iloc[row_idx][id_col]).strip()
            dp_val = df.iloc[row_idx][dp_col]
            if sample_id and pd.notna(dp_val):
                try:
                    id_dp_map[sample_id] = float(dp_val)
                except (ValueError, TypeError):
                    continue

        # 提取 OK 與 NG 的 dP 陣列
        ok_dps = np.array([id_dp_map.get(sid, np.nan) for sid in ok_ids])
        ng_dps = np.array([id_dp_map.get(sid, np.nan) for sid in ng_ids])

        # 過濾 NaN
        ok_valid = ok_dps[~np.isnan(ok_dps)]
        ng_valid = ng_dps[~np.isnan(ng_dps)]

        if len(ok_valid) == 0 or len(ng_valid) == 0:
            auc_scores[row_idx] = np.nan
            continue

        # 向量化配對比較 (broadcasting)
        # ng_valid[:, None] → (ng_count, 1), ok_valid[None, :] → (1, ok_count)
        diff = ng_valid[:, None] - ok_valid[None, :]
        score_matrix = np.where(diff > 0, 1.0, np.where(diff == 0, 0.5, 0.0))
        total_pairs = len(ng_valid) * len(ok_valid)
        auc_scores[row_idx] = score_matrix.sum() / total_pairs

    return pd.Series(auc_scores, index=df.index, name="AUC")


def calc_gap_q(
    df: pd.DataFrame,
    ok_ids: list[str],
    ng_ids: list[str],
) -> pd.Series:
    """計算 GapQ (邊界安全裕度/Y3)：min(NG_dP) - max(OK_dP)。

    GapQ > 0 代表分佈不重疊，絕對安全；<= 0 代表高誤判風險。

    Args:
        df: 含量測數據的 DataFrame。
        ok_ids: 5 台 OK 良品的 ID 清單。
        ng_ids: 5 台 NG 不良品的 ID 清單。

    Returns:
        GapQ 的 Series。
    """
    n_rows = len(df)
    gap_q_values = np.zeros(n_rows)

    step_id_cols = [f"Step{i}_ID" for i in range(1, 13)]
    step_dp_cols = [f"Step{i}_dP" for i in range(1, 13)]

    for row_idx in range(n_rows):
        id_dp_map: dict[str, float] = {}
        for id_col, dp_col in zip(step_id_cols, step_dp_cols):
            sample_id = str(df.iloc[row_idx][id_col]).strip()
            dp_val = df.iloc[row_idx][dp_col]
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
            gap_q_values[row_idx] = np.nan
        else:
            gap_q_values[row_idx] = float(np.min(ng_valid) - np.max(ok_valid))

    return pd.Series(gap_q_values, index=df.index, name="GapQ")


def extract_all_sample_ids(df: pd.DataFrame) -> list[str]:
    """從 DataFrame 中提取所有不重複的樣品 ID。

    Args:
        df: 含 Step*_ID 欄位的 DataFrame。

    Returns:
        去重後的樣品 ID 排序清單。
    """
    step_id_cols = [f"Step{i}_ID" for i in range(1, 13)]
    all_ids: set[str] = set()
    for col in step_id_cols:
        if col in df.columns:
            ids = df[col].dropna().astype(str).str.strip()
            all_ids.update(ids[ids != ""])
    return sorted(all_ids)


# ─────────────────────────── UI 渲染 ───────────────────────────


def render_response_calc() -> None:
    """渲染模組二的 Streamlit UI。"""
    st.header("📐 模組二：響應變數計算")
    st.caption("上傳測試結果，計算 Y1 (時間)、AUC (鑑別力)、GapQ (安全裕度)、Y4 (穩定性)")

    # ── 檔案上傳 ──
    uploaded_file = st.file_uploader(
        "上傳已填入 dP 數據的實驗表格 (.xlsx)",
        type=["xlsx"],
        key="upload_xlsx",
    )

    if uploaded_file is None:
        st.info("💡 請先至模組一產生實驗表格，填入測試數據後再上傳至此。")
        return

    # ── 讀取檔案 ──
    try:
        raw_df = pd.read_excel(uploaded_file, engine="openpyxl")
        st.session_state["raw_df"] = raw_df
    except Exception as e:
        st.error(f"❌ 檔案讀取失敗：{e}")
        st.info("請確認檔案格式為 .xlsx，且包含模組一產出的標準欄位。")
        return

    # 驗證必要欄位
    required_cols = ["Run_Order", "Std_Order", "Point_Type", "Pvac", "Tvac", "Tstab", "Ttest"]
    step_id_cols = [f"Step{i}_ID" for i in range(1, 13)]
    step_dp_cols = [f"Step{i}_dP" for i in range(1, 13)]
    all_required = required_cols + step_id_cols + step_dp_cols

    missing = [c for c in all_required if c not in raw_df.columns]
    if missing:
        st.error(f"❌ 檔案缺少以下必要欄位：{', '.join(missing)}")
        st.info("請確認上傳的是模組一產出的 DOE 實驗表格，並已填入所有 dP 數據。")
        return

    # 檢查 dP 數據是否為空
    dp_filled = raw_df[step_dp_cols].notna().sum().sum()
    total_dp_cells = len(raw_df) * len(step_dp_cols)
    st.info(f"📊 已讀取 {len(raw_df)} 組實驗，dP 數據填入率：{dp_filled}/{total_dp_cells} ({dp_filled/total_dp_cells*100:.1f}%)")

    if dp_filled == 0:
        st.warning("⚠️ 尚未偵測到任何 dP 數據，請確認已將測試結果填入 Excel 表格中。")
        return

    st.divider()

    # ── 樣品標籤定義 ──
    st.subheader("🏷️ 樣品 OK/NG 標籤定義")
    all_sample_ids = extract_all_sample_ids(raw_df)

    if len(all_sample_ids) < 10:
        st.warning(f"⚠️ 偵測到的樣品 ID 數量不足（僅 {len(all_sample_ids)} 台），請確認數據完整性。")

    col1, col2 = st.columns(2)
    with col1:
        ok_ids = st.multiselect(
            "✅ 選擇 5 台 OK 良品",
            options=all_sample_ids,
            default=[],
            max_selections=5,
            key="ok_ids_select",
        )
    with col2:
        ng_ids = st.multiselect(
            "❌ 選擇 5 台 NG 不良品",
            options=all_sample_ids,
            default=[],
            max_selections=5,
            key="ng_ids_select",
        )

    # 防呆驗證
    if len(ok_ids) != 5 or len(ng_ids) != 5:
        st.warning(f"⚠️ OK 需選取 5 台（目前 {len(ok_ids)} 台），NG 需選取 5 台（目前 {len(ng_ids)} 台）。")
        return

    overlap = set(ok_ids) & set(ng_ids)
    if overlap:
        st.error(f"❌ 以下樣品同時被標記為 OK 與 NG，請修正：{', '.join(overlap)}")
        return

    st.success("✅ OK/NG 標籤設定正確！")

    st.divider()

    # ── 執行計算 ──
    if st.button("🧮 開始計算響應變數", type="primary", key="calc_response"):
        with st.spinner("正在計算中..."):
            result_df = raw_df.copy()

            # Y1: 總時間
            result_df["Y1_TotalTime"] = calc_y1(result_df)

            # Y4: 穩定性全距
            result_df["Y4_Stability"] = calc_y4(result_df)

            # AUC (Y2): 鑑別正確率
            result_df["Y2_AUC"] = calc_auc_score(result_df, ok_ids, ng_ids)

            # GapQ (Y3): 邊界安全裕度
            result_df["Y3_GapQ"] = calc_gap_q(result_df, ok_ids, ng_ids)

            st.session_state["result_df"] = result_df
            st.session_state["ok_ids"] = ok_ids
            st.session_state["ng_ids"] = ng_ids

        st.success("✅ 計算完成！")

    # ── 結果展示 ──
    if st.session_state.get("result_df") is not None:
        result_df = st.session_state["result_df"]

        st.subheader("📊 計算結果總表")

        # 顯示重點欄位
        display_cols = [
            "Run_Order", "Std_Order", "Point_Type",
            "Pvac", "Tvac", "Tstab", "Ttest",
            "Y1_TotalTime", "Y2_AUC", "Y3_GapQ", "Y4_Stability",
        ]
        display_df = result_df[display_cols].copy()

        # 條件格式化
        def style_results(df: pd.DataFrame):
            """套用條件格式化至結果 DataFrame。"""
            styler = df.style

            # Y4 越大越不穩定 → 背景梯度紅色
            styler = styler.background_gradient(
                subset=["Y4_Stability"],
                cmap="Reds",
                vmin=0,
            )

            # GapQ 越小越危險 → 背景梯度紅到綠
            styler = styler.background_gradient(
                subset=["Y3_GapQ"],
                cmap="RdYlGn",
            )

            # AUC 越高越好 → 背景梯度綠色
            styler = styler.background_gradient(
                subset=["Y2_AUC"],
                cmap="Greens",
                vmin=0,
                vmax=1,
            )

            styler = styler.format(
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
            )
            return styler

        st.dataframe(style_results(display_df), use_container_width=True)

        # 關鍵指標摘要
        st.subheader("📈 關鍵指標摘要")
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("最佳 AUC", f"{result_df['Y2_AUC'].max():.3f}")
        with m2:
            st.metric("AUC=1.0 組數", f"{(result_df['Y2_AUC'] == 1.0).sum()} / {len(result_df)}")
        with m3:
            st.metric("最大 GapQ", f"{result_df['Y3_GapQ'].max():.4f}")
        with m4:
            st.metric("最小 Y1 (時間)", f"{result_df['Y1_TotalTime'].min():.1f} sec")

        # ── 資料匯出 ──
        st.divider()
        st.subheader("📥 測試結果匯出")

        col_dl1, col_dl2, _ = st.columns([1, 1, 2])

        with col_dl1:
            # 一般總表匯出 (全部欄位)
            excel_data_full = export_formatted_xlsx(
                df=result_df,
                highlight_cols=["Y1_TotalTime", "Y2_AUC", "Y3_GapQ", "Y4_Stability"],
            )

            st.download_button(
                label="📥 下載完整計算總表 (.xlsx)",
                data=excel_data_full,
                file_name="DOE_Results_Full.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                help="包含所有量測細節與計算結果",
            )

        with col_dl2:
            # Minitab 專用格式匯出
            # 保留欄位並重新命名
            minitab_cols = [
                "Std_Order", "Run_Order", "Point_Type", 
                "Pvac", "Tvac", "Tstab", "Ttest",
                "Y1_TotalTime", "Y2_AUC", "Y3_GapQ", "Y4_Stability"
            ]
            minitab_df = result_df[minitab_cols].rename(columns={
                "Std_Order": "StdOrder",
                "Run_Order": "RunOrder",
                "Point_Type": "PtType",
                "Y1_TotalTime": "Y1",
                "Y2_AUC": "AUC",
                "Y3_GapQ": "GapQ",
                "Y4_Stability": "Y4",
            })
            
            excel_data_minitab = export_formatted_xlsx(
                df=minitab_df,
                highlight_cols=["Y1", "AUC", "GapQ", "Y4"],
                header_color="#388E3C",  # 用綠色標題列以示區別
            )

            st.download_button(
                label="📊 下載 Minitab 專用格式",
                data=excel_data_minitab,
                file_name="DOE_Results_Minitab.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                help="已過濾量測細節，可直接貼入 Minitab 跑分析",
                type="primary",
            )
