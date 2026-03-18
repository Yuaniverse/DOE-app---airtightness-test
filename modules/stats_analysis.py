"""模組五：基礎統計分析 (Basic Statistical Analysis)。

提供使用者針對現有計算出的響應變數，進行統計建模、彎曲檢定，
並在需要時產生 CCD（CCC / CCF）軸點表格，協助升級為二階響應曲面分析。
"""

from collections import Counter

import pandas as pd
import streamlit as st
import plotly.express as px
from scipy import stats
import statsmodels.formula.api as smf
import statsmodels.api as sm

from modules.doe_generator import (
    FACTOR_COLUMNS,
    build_measurement_sequence,
    generate_axial_points,
    merge_doe_with_axial,
)
from utils.excel_export import export_formatted_xlsx


RESPONSE_OPTIONS = {
    "Y1 (總測試時間)": "Y1_TotalTime",
    "AUC (鑑別正確率)": "Y2_AUC",
    "GapQ (邊界安全裕度)": "Y3_GapQ",
    "Y4 (穩定性全距)": "Y4_Stability",
}


def _style_pvalue(val: float) -> str:
    """格式化顯著 p-value 樣式。"""
    color = "#ff4b4b" if pd.notna(val) and val < 0.05 else ""
    font_weight = "bold" if pd.notna(val) and val < 0.05 else "normal"
    return f"color: {color}; font-weight: {font_weight}"


def _build_first_order_formula() -> str:
    """建立一階線性模型公式。"""
    return "Target ~ " + " + ".join(FACTOR_COLUMNS)


def _run_curvature_test(df: pd.DataFrame, target_col: str) -> dict[str, float | int | bool] | None:
    """以標準 DOE curvature F-test 執行彎曲檢定。"""
    if "Point_Type" not in df.columns or target_col not in df.columns:
        return None

    test_df = df[df["Point_Type"].isin(["Corner", "Center"])].copy()
    if test_df.empty:
        return None

    required_cols = FACTOR_COLUMNS + [target_col]
    for col in required_cols:
        test_df[col] = pd.to_numeric(test_df[col], errors="coerce")
    test_df = test_df.dropna(subset=required_cols)

    corner = test_df.loc[test_df["Point_Type"] == "Corner", target_col]
    center = test_df.loc[test_df["Point_Type"] == "Center", target_col]

    if len(corner) < 2 or len(center) < 2:
        return None

    model_data = test_df[FACTOR_COLUMNS + [target_col]].copy()
    model_data.columns = FACTOR_COLUMNS + ["Target"]

    first_order_model = smf.ols(formula=_build_first_order_formula(), data=model_data).fit()
    mse = float(first_order_model.mse_resid)
    df_den = float(first_order_model.df_resid)

    if not pd.notna(mse) or mse <= 0 or df_den <= 0:
        return None

    mean_corner = float(corner.mean())
    mean_center = float(center.mean())
    diff = mean_center - mean_corner
    n_corner = int(len(corner))
    n_center = int(len(center))
    curvature_ss = (n_corner * n_center * (mean_corner - mean_center) ** 2) / (n_corner + n_center)
    f_stat = curvature_ss / mse
    p_value = float(stats.f.sf(f_stat, 1, df_den))

    return {
        "n_corner": n_corner,
        "n_center": n_center,
        "mean_corner": mean_corner,
        "mean_center": mean_center,
        "diff": diff,
        "curvature_ss": float(curvature_ss),
        "mse": mse,
        "f_stat": float(f_stat),
        "df_num": 1,
        "df_den": float(df_den),
        "p_value": p_value,
        "significant": bool(pd.notna(p_value) and p_value < 0.05),
    }


def _detect_quadratic_factors(df: pd.DataFrame) -> list[str]:
    """從資料中偵測哪些因子已補過軸點，供 RSM 模型加入二次項。"""
    if "Point_Type" not in df.columns:
        return []

    axial_mask = df["Point_Type"].astype(str).str.strip().eq("Axial")
    if not axial_mask.any():
        return []

    if "Axial_Factor" in df.columns:
        factors = (
            df.loc[axial_mask, "Axial_Factor"]
            .dropna()
            .astype(str)
            .str.strip()
            .tolist()
        )
        unique_factors = [factor for factor in FACTOR_COLUMNS if factor in set(factors)]
        if unique_factors:
            return unique_factors

    return FACTOR_COLUMNS.copy()


def _build_formula(quadratic_factors: list[str]) -> str:
    """建立回歸公式。"""
    base_formula = "Target ~ (" + " + ".join(FACTOR_COLUMNS) + ")**2"
    if not quadratic_factors:
        return base_formula

    quadratic_terms = " + ".join(f"I({factor} ** 2)" for factor in quadratic_factors)
    return f"{base_formula} + {quadratic_terms}"


def _extract_main_effect_pvalues(anova_table: pd.DataFrame) -> dict[str, float]:
    """抽取主效應 p-value。"""
    result: dict[str, float] = {}
    for factor in FACTOR_COLUMNS:
        if factor in anova_table.index:
            value = anova_table.loc[factor, "p-value"]
            result[factor] = float(value) if pd.notna(value) else float("nan")
    return result


def _extract_factor_bounds(df: pd.DataFrame) -> dict[str, tuple[float, float]]:
    """從非軸點資料估計原始因子範圍。"""
    source_df = df.copy()
    if "Point_Type" in source_df.columns:
        non_axial = source_df[~source_df["Point_Type"].astype(str).str.strip().eq("Axial")]
        if not non_axial.empty:
            source_df = non_axial

    bounds: dict[str, tuple[float, float]] = {}
    for factor in FACTOR_COLUMNS:
        values = pd.to_numeric(source_df[factor], errors="coerce").dropna()
        bounds[factor] = (float(values.min()), float(values.max()))
    return bounds


def _infer_measurement_setup(df: pd.DataFrame) -> tuple[str | None, list[str]]:
    """從現有 DOE 表格推回 Golden Sample 與其餘樣品 ID。"""
    step_id_cols = [f"Step{i}_ID" for i in range(1, 13)]
    anchor_cols = ["Step1_ID", "Step6_ID", "Step12_ID"]

    anchor_ids: list[str] = []
    for col in anchor_cols:
        if col in df.columns:
            vals = df[col].dropna().astype(str).str.strip()
            anchor_ids.extend([val for val in vals if val])

    golden_sample = Counter(anchor_ids).most_common(1)[0][0] if anchor_ids else None

    all_ids: list[str] = []
    seen: set[str] = set()
    for col in step_id_cols:
        if col not in df.columns:
            continue
        for raw_val in df[col].dropna().astype(str).str.strip():
            if raw_val and raw_val not in seen:
                seen.add(raw_val)
                all_ids.append(raw_val)

    other_ids = [sample_id for sample_id in all_ids if sample_id != golden_sample]
    return golden_sample, other_ids


def _select_base_experiment_df() -> pd.DataFrame | None:
    """選擇用來合併軸點的原始實驗表格。"""
    if st.session_state.get("raw_df") is not None:
        return st.session_state["raw_df"].copy()

    if st.session_state.get("doe_df") is not None:
        return st.session_state["doe_df"].copy()

    result_df = st.session_state.get("result_df")
    if result_df is None:
        return None

    removable_cols = ["Y1_TotalTime", "Y2_AUC", "Y3_GapQ", "Y4_Stability"]
    remaining_cols = [col for col in result_df.columns if col not in removable_cols]
    return result_df[remaining_cols].copy()


def render_stats_analysis() -> None:
    """渲染模組五的 Streamlit UI。"""
    st.header("📊 模組五：基礎統計分析")
    st.caption("分析控制因子顯著性、檢查模型是否有彎曲，必要時補 CCD 軸點升級為二階模型。")

    if st.session_state.get("result_df") is None:
        st.warning("⚠️ 尚無計算資料！請先至「模組二：響應變數計算」上傳檔案並執行計算。")
        return

    result_df = st.session_state["result_df"].copy()

    col_sel, _ = st.columns([1, 2])
    with col_sel:
        selected_label = st.selectbox(
            "📍 選擇欲分析的響應變數：",
            options=list(RESPONSE_OPTIONS.keys()),
        )

    target_col = RESPONSE_OPTIONS[selected_label]

    if result_df[target_col].var() == 0:
        st.error(f"❌ 選擇的響應變數 ({selected_label}) 變異數為零（所有數值皆相同），無法進行統計迴歸分析。")
        return

    curvature_result = _run_curvature_test(result_df, target_col)
    quadratic_factors = _detect_quadratic_factors(result_df)
    formula = _build_formula(quadratic_factors)

    model_data = result_df[FACTOR_COLUMNS + [target_col]].copy()
    model_data.columns = FACTOR_COLUMNS + ["Target"]
    for col in FACTOR_COLUMNS + ["Target"]:
        model_data[col] = pd.to_numeric(model_data[col], errors="coerce")
    model_data = model_data.dropna()

    if len(model_data) < 5:
        st.error("❌ 有效數據筆數過少，無法配適模型。")
        return

    st.divider()
    st.subheader("📈 統計模型")

    if quadratic_factors:
        st.info(
            "已偵測到軸點資料，模型將加入以下二次項："
            f" {', '.join(quadratic_factors)}。"
        )
    else:
        st.caption("目前使用第一輪模型：主效應 + 二階交互作用。若彎曲顯著，可再補 CCD 軸點。")

    with st.expander("查看目前模型公式"):
        st.code(formula)

    with st.spinner("正在計算統計模型..."):
        try:
            model = smf.ols(formula=formula, data=model_data).fit()

            col_r2, col_adj_r2, _ = st.columns([1, 1, 3])
            with col_r2:
                st.metric(
                    label="R-squared ($R^2$)",
                    value=f"{model.rsquared * 100:.2f}%",
                    help="模型解釋變異的比例，越接近 100% 代表配適度越好。",
                )
            with col_adj_r2:
                st.metric(
                    label="Adj. $R^2$",
                    value=f"{model.rsquared_adj * 100:.2f}%",
                    help="考慮模型項數後的調整後解釋力。",
                )

            anova_table = sm.stats.anova_lm(model, typ=3)
            anova_table = anova_table.rename(
                columns={"PR(>F)": "p-value", "sum_sq": "Sum Sq", "df": "DF", "F": "F-Value"}
            )

            st.subheader("📋 變異數分析 (ANOVA)")
            formatted_anova = anova_table.copy().round(
                {"Sum Sq": 4, "F-Value": 2, "p-value": 4, "DF": 0}
            )
            st.dataframe(formatted_anova, width="stretch")
            st.caption("🔴 p-value < 0.05 代表該因子/交互作用具備統計顯著性。此處採 Type III SS，較接近 DOE / Minitab 慣用做法。")

            st.divider()
            st.subheader("📉 標準化效應柏拉圖 (Pareto Chart)")

            t_values = model.tvalues.drop(labels=["Intercept"], errors="ignore").abs().dropna()
            t_values = t_values.sort_values(ascending=False)
            df_resid = model.df_resid
            t_critical = stats.t.ppf(1 - 0.025, df_resid)

            pareto_df = pd.DataFrame({"Term": t_values.index, "Effect": t_values.values})
            fig = px.bar(
                pareto_df,
                x="Effect",
                y="Term",
                orientation="h",
                text="Effect",
                labels={"Effect": "標準化效應 (絕對 t 值)", "Term": "因子/交互作用"},
            )
            fig.update_traces(texttemplate="%{text:.2f}", textposition="outside", marker_color="#5c6bc0")
            fig.add_vline(
                x=t_critical,
                line_width=2,
                line_dash="dash",
                line_color="red",
                annotation_text=f"p=0.05 (Crit: {t_critical:.2f})",
                annotation_position="top right",
            )
            fig.update_layout(
                yaxis={"categoryorder": "total ascending"},
                height=500,
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(showgrid=True, gridcolor="#e0e0e0"),
            )
            st.plotly_chart(fig, width="stretch")

        except Exception as e:
            st.error(f"❌ 模型計算發生錯誤：{e}")
            st.info("請確認數據中變數是否皆有正常值，或樣本數量是否充足。")
            return

    st.divider()
    st.subheader("🔄 彎曲檢定 (Curvature Test)")

    if curvature_result is None:
        st.warning("⚠️ 缺少足夠的 Corner / Center 資料，暫時無法執行彎曲檢定。")
        return

    cv1, cv2, cv3, cv4 = st.columns(4)
    with cv1:
        st.metric("Corner 平均", f"{curvature_result['mean_corner']:.4f}")
    with cv2:
        st.metric("Center 平均", f"{curvature_result['mean_center']:.4f}")
    with cv3:
        st.metric("平均差 (Center-Corner)", f"{curvature_result['diff']:.4f}")
    with cv4:
        st.metric("p-value", f"{curvature_result['p_value']:.4f}")
    st.caption(
        f"Curvature F-test：F = {curvature_result['f_stat']:.4f} "
        f"(df = {int(curvature_result['df_num'])}, {int(curvature_result['df_den'])})，"
        f"MSE = {curvature_result['mse']:.4f}。"
    )

    curvature_plot_df = result_df[result_df["Point_Type"].isin(["Corner", "Center"])].copy()
    curvature_plot_df[target_col] = pd.to_numeric(curvature_plot_df[target_col], errors="coerce")
    curvature_plot_df = curvature_plot_df.dropna(subset=[target_col])
    fig_curvature = px.box(
        curvature_plot_df,
        x="Point_Type",
        y=target_col,
        color="Point_Type",
        points="all",
        color_discrete_map={"Corner": "#5c6bc0", "Center": "#ef5350"},
        labels={"Point_Type": "點型態", target_col: selected_label},
        title="Corner vs Center 響應分佈比較",
    )
    st.plotly_chart(fig_curvature, width="stretch")

    if curvature_result["significant"]:
        st.warning(
            "⚠️ 已偵測到顯著彎曲（p-value < 0.05）。 代表一階模型可能不足，建議補 CCD 軸點以估計二次項。"
        )
    else:
        st.success("✅ 目前未偵測到顯著彎曲。 若工程上仍懷疑存在局部彎曲，可視需求保守補點。")

    st.divider()
    st.subheader("🎯 CCD 因子建議")

    main_effect_pvalues = _extract_main_effect_pvalues(anova_table)
    significant_factors = [
        factor for factor, p_value in main_effect_pvalues.items() if pd.notna(p_value) and p_value < 0.05
    ]

    suggestion_rows = []
    for factor in FACTOR_COLUMNS:
        p_value = main_effect_pvalues.get(factor, float("nan"))
        suggestion_rows.append(
            {
                "因子": factor,
                "主效應 p-value": p_value,
                "建議": "建議保留" if pd.notna(p_value) and p_value < 0.05 else "可考慮固定在中心值",
            }
        )
    suggestion_df = pd.DataFrame(suggestion_rows)
    suggestion_display_df = suggestion_df.copy().round({"主效應 p-value": 4})
    st.dataframe(suggestion_display_df, width="stretch")

    if curvature_result["significant"] and not significant_factors:
        default_selected_factors = FACTOR_COLUMNS.copy()
        st.info(
            "偵測到彎曲，但主效應尚未出現明顯顯著因子。 這種情況可能來自純二次效應，因此預設先保留全部因子較安全。"
        )
    else:
        default_selected_factors = significant_factors.copy()

    st.caption(
        "實務建議：第一輪不顯著的因子可以考慮拿掉，但不要只因線性主效應不顯著就武斷刪除。 如果整體彎曲顯著，該因子仍可能透過二次項產生影響。"
    )

    selected_factors = st.multiselect(
        "選擇要納入 CCD 補點的因子",
        options=FACTOR_COLUMNS,
        default=default_selected_factors,
        help="預設依主效應顯著性建議；你也可以依工程知識自行調整。",
        key="ccd_selected_factors",
    )

    st.divider()
    st.subheader("⭐ CCD 軸點產生器")

    base_experiment_df = _select_base_experiment_df()
    if base_experiment_df is None:
        st.warning("⚠️ 找不到可合併的原始 DOE 表格。")
        return

    factor_bounds = _extract_factor_bounds(base_experiment_df)
    default_design_index = 0 if st.session_state.get("ccd_design_type") == "CCC" else 1
    design_type = st.radio(
        "選擇 CCD 類型",
        options=["CCC", "CCF"],
        index=default_design_index,
        horizontal=True,
        help="CCC = Circumscribed（軸點可能超出原始範圍）；CCF = Face-Centered（軸點落在原始邊界上）。",
        key="ccd_design_type_radio",
    )

    use_seed_for_ccd = st.checkbox("新增軸點 Run_Order 使用固定隨機種子", value=False, key="ccd_use_seed")
    ccd_seed = None
    if use_seed_for_ccd:
        ccd_seed = st.number_input("CCD 隨機種子", value=42, step=1, key="ccd_seed_value")

    if selected_factors:
        next_std_order = int(pd.to_numeric(base_experiment_df["Std_Order"], errors="coerce").max()) + 1
        axial_preview_df = generate_axial_points(
            factors=factor_bounds,
            selected_factors=selected_factors,
            design_type=design_type,
            start_std_order=next_std_order,
        )

        alpha_value = float(axial_preview_df["Alpha"].iloc[0]) if not axial_preview_df.empty else 0.0
        st.info(f"本次將新增 {len(axial_preview_df)} 個軸點，CCD 類型 = {design_type}，α = {alpha_value:.4f}。")

        if design_type == "CCC":
            st.warning("CCC 軸點可能超出原始因子上下限，請先確認設備與製程允許此範圍。")

        preview_cols = ["Std_Order", "Point_Type", "Axial_Factor", "Axial_Sign", "CCD_Type", "Alpha"] + FACTOR_COLUMNS
        st.dataframe(axial_preview_df[preview_cols], width="stretch")

        if st.button("🚀 產生並合併 CCD 軸點表格", type="primary", key="generate_ccd_table"):
            golden_sample, other_sample_ids = _infer_measurement_setup(base_experiment_df)

            if not golden_sample or len(other_sample_ids) < 9:
                st.error("❌ 無法從現有表格推回 Golden Sample / 其餘樣品 ID，請先確認原始 DOE 欄位完整。")
                return

            other_sample_ids = other_sample_ids[:9]
            axial_with_sequence = build_measurement_sequence(
                axial_preview_df,
                golden_sample_id=golden_sample,
                other_sample_ids=other_sample_ids,
                seed=ccd_seed,
            )

            merged_df = merge_doe_with_axial(
                base_df=base_experiment_df,
                axial_df=axial_with_sequence,
                seed=ccd_seed,
                randomize_new_runs=True,
            )

            st.session_state["axial_df"] = axial_with_sequence
            st.session_state["merged_doe_df"] = merged_df
            st.session_state["ccd_selected_factors"] = selected_factors
            st.session_state["ccd_design_type"] = design_type

            st.success("✅ 已完成 CCD 軸點表格產生，並合併回原始實驗表。")
    else:
        st.info("請至少選擇 1 個因子，才能產生 CCD 軸點。")

    if st.session_state.get("merged_doe_df") is not None:
        merged_doe_df = st.session_state["merged_doe_df"]
        st.subheader("📄 合併後 DOE 表格預覽")
        st.dataframe(merged_doe_df.head(10), width="stretch")
        st.metric("合併後總實驗組數", f"{len(merged_doe_df)} 組")

        highlight_cols = [f"Step{i}_dP" for i in range(1, 13)]
        merged_xlsx = export_formatted_xlsx(
            merged_doe_df,
            highlight_cols=highlight_cols,
            highlight_color="#E2EFDA",
        )
        st.download_button(
            label="📥 下載合併後 CCD 實驗表格 (.xlsx)",
            data=merged_xlsx,
            file_name="DOE_Experiment_Table_With_CCD.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type="primary",
        )
