"""模組一：DOE 實驗表格產生器 (Excel Generator)。

負責接收使用者的實驗條件，產生完全隨機化且含三點定錨法的空白實驗紀錄表。
亦提供 CCD 軸點（Axial Points）產生與合併功能，供模組五延伸使用。
"""

import itertools
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st

from utils.excel_export import export_formatted_xlsx


FACTOR_COLUMNS = ["Pvac", "Tvac", "Tstab", "Ttest"]


# ─────────────────────────── 核心演算法 ───────────────────────────


def generate_doe_matrix(
    factors: dict[str, tuple[float, float]],
) -> pd.DataFrame:
    """產生 2^4 全因子設計矩陣 + 6 個中心點。

    Args:
        factors: 因子名稱 → (低水準, 高水準) 的字典。

    Returns:
        包含 22 列實驗條件的 DataFrame，含 Std_Order 與 Point_Type。
    """
    factor_names = list(factors.keys())
    levels = [factors[name] for name in factor_names]

    # 2^4 = 16 角落點
    corner_points = list(itertools.product(*levels))
    df_corner = pd.DataFrame(corner_points, columns=factor_names)
    df_corner["Point_Type"] = "Corner"
    df_corner["Std_Order"] = range(1, len(df_corner) + 1)

    # 6 個中心點
    center_values = {name: (lo + hi) / 2 for name, (lo, hi) in factors.items()}
    df_center = pd.DataFrame([center_values] * 6)
    df_center["Point_Type"] = "Center"
    df_center["Std_Order"] = range(len(df_corner) + 1, len(df_corner) + 7)

    df = pd.concat([df_corner, df_center], ignore_index=True)
    return df


def randomize_run_order(df: pd.DataFrame, seed: Optional[int] = None) -> pd.DataFrame:
    """將實驗順序完全隨機化。

    Args:
        df: 含 Std_Order 的 DOE 矩陣。
        seed: 隨機種子（可選，用於重現）。

    Returns:
        新增 Run_Order 欄的 DataFrame。
    """
    rng = np.random.default_rng(seed)
    n = len(df)
    run_order = rng.permutation(n) + 1
    df = df.copy()
    df["Run_Order"] = run_order
    df = df.sort_values("Run_Order").reset_index(drop=True)
    return df


def build_measurement_sequence(
    df: pd.DataFrame,
    golden_sample_id: str,
    other_sample_ids: list[str],
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """為每列實驗建立 12 步量測序列（三點定錨法）。

    定錨點：Step 1, 6, 12 固定為基準良品。
    隨機點：Step 2-5, 7-11 隨機塞入其餘 9 台樣品。

    Args:
        df: DOE 矩陣。
        golden_sample_id: 基準良品 ID。
        other_sample_ids: 其餘 9 台樣品的 ID 清單。
        seed: 隨機種子。

    Returns:
        新增 Step1_ID ~ Step12_ID 與 Step1_dP ~ Step12_dP 欄位的 DataFrame。
    """
    rng = np.random.default_rng(seed)
    df = df.copy()

    # 定錨位置 (0-indexed): 0, 5, 11 即 Step 1, 6, 12
    anchor_positions = [0, 5, 11]
    random_positions = [i for i in range(12) if i not in anchor_positions]

    if len(other_sample_ids) < len(random_positions):
        raise ValueError(
            f"other_sample_ids 至少需要 {len(random_positions)} 台樣品，"
            f"目前只有 {len(other_sample_ids)} 台。"
        )

    step_id_cols = [f"Step{i+1}_ID" for i in range(12)]
    step_dp_cols = [f"Step{i+1}_dP" for i in range(12)]

    # 預分配所有 ID 與 dP 欄位
    for col in step_id_cols + step_dp_cols:
        df[col] = ""

    for row_idx in range(len(df)):
        sequence = [""] * 12

        # 定錨點
        for pos in anchor_positions:
            sequence[pos] = golden_sample_id

        # 隨機點
        shuffled = other_sample_ids.copy()
        rng.shuffle(shuffled)
        for i, pos in enumerate(random_positions):
            sequence[pos] = shuffled[i]

        for step_idx in range(12):
            df.at[row_idx, step_id_cols[step_idx]] = sequence[step_idx]
            df.at[row_idx, step_dp_cols[step_idx]] = ""  # 空白待填

    return df


def generate_axial_points(
    factors: dict[str, tuple[float, float]],
    selected_factors: list[str],
    design_type: str = "CCC",
    start_std_order: int = 1,
) -> pd.DataFrame:
    """產生 CCD 軸點（Axial / Star Points）。

    Args:
        factors: 因子名稱 → (實際最小值, 實際最大值) 的字典。
        selected_factors: 要納入 CCD 的因子名稱。
        design_type: CCD 類型，支援 CCC / CCF。
        start_std_order: 軸點起始標準順序。

    Returns:
        軸點 DataFrame，含 Axial_Factor / Axial_Sign / CCD_Type / Alpha。
    """
    valid_factors = [factor for factor in selected_factors if factor in factors]
    if not valid_factors:
        return pd.DataFrame(columns=FACTOR_COLUMNS + [
            "Point_Type", "Std_Order", "Axial_Factor", "Axial_Sign", "CCD_Type", "Alpha"
        ])

    design_type = design_type.upper()
    if design_type not in {"CCC", "CCF"}:
        raise ValueError("design_type 必須為 'CCC' 或 'CCF'")

    original_design_k = len([factor for factor in FACTOR_COLUMNS if factor in factors]) or len(factors)
    alpha = float(2 ** (original_design_k / 4)) if design_type == "CCC" else 1.0

    centers = {
        name: (float(bounds[0]) + float(bounds[1])) / 2
        for name, bounds in factors.items()
    }

    rows: list[dict[str, float | int | str]] = []
    std_order = start_std_order

    for factor in valid_factors:
        min_val, max_val = factors[factor]
        center = centers[factor]
        half_range = (float(max_val) - float(min_val)) / 2

        for sign in (-1, 1):
            row: dict[str, float | int | str] = {name: centers[name] for name in factors}
            row[factor] = center + sign * alpha * abs(half_range)
            row["Point_Type"] = "Axial"
            row["Std_Order"] = std_order
            row["Axial_Factor"] = factor
            row["Axial_Sign"] = "-" if sign < 0 else "+"
            row["CCD_Type"] = design_type
            row["Alpha"] = alpha
            rows.append(row)
            std_order += 1

    return pd.DataFrame(rows)


def merge_doe_with_axial(
    base_df: pd.DataFrame,
    axial_df: pd.DataFrame,
    seed: Optional[int] = None,
    randomize_new_runs: bool = True,
) -> pd.DataFrame:
    """將軸點表格合併回原始 DOE 表格。

    設計原則：
    - 保留原始已完成實驗的 Run_Order 不變
    - 只對新增軸點分配新的 Run_Order
    - 新增欄位（如 Axial_Factor / CCD_Type）會自動補齊到原始表格

    Args:
        base_df: 原始 DOE 表格。
        axial_df: 已建立量測序列的軸點表格。
        seed: 新增軸點 Run_Order 的隨機種子。
        randomize_new_runs: 是否只對新增軸點順序做隨機化。

    Returns:
        合併後的完整 DOE DataFrame。
    """
    if axial_df.empty:
        return base_df.copy()

    merged_base = base_df.copy()
    merged_axial = axial_df.copy()

    max_run_order = int(pd.to_numeric(merged_base["Run_Order"], errors="coerce").max())
    new_run_orders = np.arange(max_run_order + 1, max_run_order + 1 + len(merged_axial))

    if randomize_new_runs and len(new_run_orders) > 1:
        rng = np.random.default_rng(seed)
        new_run_orders = rng.permutation(new_run_orders)

    merged_axial["Run_Order"] = new_run_orders

    base_cols = list(merged_base.columns)
    extra_cols = [col for col in merged_axial.columns if col not in base_cols]
    all_cols = base_cols + extra_cols

    for col in all_cols:
        if col not in merged_base.columns:
            merged_base[col] = ""
        if col not in merged_axial.columns:
            merged_axial[col] = ""

    merged_df = pd.concat(
        [merged_base[all_cols], merged_axial[all_cols]],
        ignore_index=True,
    )
    merged_df = merged_df.sort_values("Run_Order").reset_index(drop=True)
    return merged_df


# ─────────────────────────── UI 渲染 ───────────────────────────


def render_doe_generator() -> None:
    """渲染模組一的 Streamlit UI。"""
    st.header("📋 模組一：DOE 實驗表格產生器")
    st.caption("設定控制因子與樣品資訊，自動產生隨機化的實驗紀錄表")

    # ── 控制因子設定區 ──
    st.subheader("🔧 控制因子設定")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**低水準 (Low Level)**")
        pvac_lo = st.number_input("Pvac 低 (kPa)", value=-30.0, key="pvac_lo", format="%.1f")
        tvac_lo = st.number_input("Tvac 低 (sec)", value=3.0, key="tvac_lo", format="%.1f")
        tstab_lo = st.number_input("Tstab 低 (sec)", value=3.0, key="tstab_lo", format="%.1f")
        ttest_lo = st.number_input("Ttest 低 (sec)", value=3.0, key="ttest_lo", format="%.1f")

    with col2:
        st.markdown("**高水準 (High Level)**")
        pvac_hi = st.number_input("Pvac 高 (kPa)", value=-60.0, key="pvac_hi", format="%.1f")
        tvac_hi = st.number_input("Tvac 高 (sec)", value=8.0, key="tvac_hi", format="%.1f")
        tstab_hi = st.number_input("Tstab 高 (sec)", value=8.0, key="tstab_hi", format="%.1f")
        ttest_hi = st.number_input("Ttest 高 (sec)", value=8.0, key="ttest_hi", format="%.1f")

    st.divider()

    # ── 樣品 ID 輸入區 ──
    st.subheader("🏷️ 樣品資訊設定")
    sample_input = st.text_input(
        "輸入 10 台樣品 ID（以逗號分隔）",
        value="A01, A02, A03, A04, A05, B01, B02, B03, B04, B05",
        help="例如：A01, B02, C03, 114, 225",
        key="sample_ids_input",
    )

    # 解析並清理
    sample_ids = [s.strip() for s in sample_input.split(",") if s.strip()]

    if len(sample_ids) != 10:
        st.warning(f"⚠️ 請輸入剛好 10 台樣品 ID（目前偵測到 {len(sample_ids)} 台）")
        return

    st.success(f"✅ 已偵測到 10 台樣品：{', '.join(sample_ids)}")

    # ── 基準良品指定 ──
    golden_sample = st.selectbox(
        "🌟 指定基準良品 (Golden Sample)",
        options=sample_ids,
        help="此樣品將用於監測設備熱飄移與系統穩定性，固定於 Step 1, 6, 12 量測",
        key="golden_sample_select",
    )

    st.divider()

    # ── 隨機種子（可選） ──
    use_seed = st.checkbox("使用固定隨機種子（便於重現結果）", value=False, key="use_seed")
    seed_value = None
    if use_seed:
        seed_value = st.number_input("隨機種子值", value=42, step=1, key="seed_value")

    # ── 產生按鈕 ──
    if st.button("🚀 產生 DOE 實驗表格", type="primary", key="generate_doe"):
        factors = {
            "Pvac": (pvac_lo, pvac_hi),
            "Tvac": (tvac_lo, tvac_hi),
            "Tstab": (tstab_lo, tstab_hi),
            "Ttest": (ttest_lo, ttest_hi),
        }

        other_ids = [sid for sid in sample_ids if sid != golden_sample]

        if len(other_ids) != 9:
            st.error("❌ 基準良品外的樣品數量應為 9 台，請檢查是否有重複的 ID。")
            return

        with st.spinner("正在產生 DOE 矩陣..."):
            # Step 1: 產生矩陣
            doe_df = generate_doe_matrix(factors)

            # Step 2: 隨機化
            doe_df = randomize_run_order(doe_df, seed=seed_value)

            # Step 3: 建立量測序列
            doe_df = build_measurement_sequence(
                doe_df, golden_sample, other_ids, seed=seed_value
            )

            # 整理欄位順序
            factor_cols = ["Run_Order", "Std_Order", "Point_Type", "Pvac", "Tvac", "Tstab", "Ttest"]
            step_cols = []
            for i in range(1, 13):
                step_cols.extend([f"Step{i}_ID", f"Step{i}_dP"])
            doe_df = doe_df[factor_cols + step_cols]

            # 儲存到 session_state
            st.session_state["doe_df"] = doe_df

        st.success("✅ DOE 實驗表格產生完成！")

    # ── 預覽與下載 ──
    if st.session_state.get("doe_df") is not None:
        doe_df = st.session_state["doe_df"]

        st.subheader("📊 實驗表格預覽（前 5 筆）")
        st.dataframe(doe_df.head(5), width="stretch")

        st.metric("總實驗組數", f"{len(doe_df)} 組")

        # 匯出 Excel
        highlight_cols = [f"Step{i}_dP" for i in range(1, 13)]
        xlsx_bytes = export_formatted_xlsx(
            doe_df,
            highlight_cols=highlight_cols,
            highlight_color="#E2EFDA",
        )

        st.download_button(
            label="📥 下載 DOE 實驗表格 (.xlsx)",
            data=xlsx_bytes,
            file_name="DOE_Experiment_Table.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type="primary",
        )
