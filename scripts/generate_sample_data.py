from pathlib import Path
import itertools

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "sample_data"
OUT_PATH = OUT_DIR / "DOE_TestData_Filled.xlsx"


def generate_doe_matrix(factors: dict[str, tuple[float, float]]) -> pd.DataFrame:
    factor_names = list(factors.keys())
    levels = [factors[name] for name in factor_names]

    corner_points = list(itertools.product(*levels))
    df_corner = pd.DataFrame(corner_points, columns=factor_names)
    df_corner["Point_Type"] = "Corner"
    df_corner["Std_Order"] = range(1, len(df_corner) + 1)

    center_values = {name: (lo + hi) / 2 for name, (lo, hi) in factors.items()}
    df_center = pd.DataFrame([center_values] * 6)
    df_center["Point_Type"] = "Center"
    df_center["Std_Order"] = range(len(df_corner) + 1, len(df_corner) + 7)

    return pd.concat([df_corner, df_center], ignore_index=True)


def randomize_run_order(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    result = df.copy()
    result["Run_Order"] = rng.permutation(len(result)) + 1
    return result.sort_values("Run_Order").reset_index(drop=True)


def build_measurement_sequence(
    df: pd.DataFrame,
    golden_sample_id: str,
    other_sample_ids: list[str],
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    result = df.copy()

    anchor_positions = [0, 5, 11]
    random_positions = [i for i in range(12) if i not in anchor_positions]

    step_id_cols = [f"Step{i + 1}_ID" for i in range(12)]
    step_dp_cols = [f"Step{i + 1}_dP" for i in range(12)]
    for col in step_id_cols + step_dp_cols:
        result[col] = ""

    for row_idx in range(len(result)):
        sequence = [""] * 12
        for pos in anchor_positions:
            sequence[pos] = golden_sample_id

        shuffled = other_sample_ids.copy()
        rng.shuffle(shuffled)
        for i, pos in enumerate(random_positions):
            sequence[pos] = shuffled[i]

        for step_idx in range(12):
            result.at[row_idx, step_id_cols[step_idx]] = sequence[step_idx]
            result.at[row_idx, step_dp_cols[step_idx]] = ""

    return result


def norm01(value: float, low: float, high: float) -> float:
    if high == low:
        return 0.0
    return (value - low) / (high - low)


def calc_run_profile(row: pd.Series) -> tuple[float, float, float]:
    x1 = norm01(abs(float(row["Pvac"])), 30.0, 60.0)
    x2 = norm01(float(row["Tvac"]), 3.0, 8.0)
    x3 = norm01(float(row["Tstab"]), 3.0, 8.0)
    x4 = norm01(float(row["Ttest"]), 3.0, 8.0)

    common = 0.020 + 0.004 * x1 + 0.003 * x2 + 0.002 * x3 + 0.003 * x4
    sep_bonus = 0.001 + 0.002 * x1 + 0.003 * x2 + 0.004 * x3 + 0.004 * x4 + 0.002 * x2 * x3 - 0.001 * x1 * x4
    stability = max(0.00035, 0.0016 - 0.00025 * x2 - 0.00045 * x3 - 0.00035 * x4 + 0.00015 * x1)
    return common, sep_bonus, stability


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)

    factors = {
        "Pvac": (-30.0, -60.0),
        "Tvac": (3.0, 8.0),
        "Tstab": (3.0, 8.0),
        "Ttest": (3.0, 8.0),
    }
    sample_ids = ["A01", "A02", "A03", "A04", "A05", "B01", "B02", "B03", "B04", "B05"]
    golden_sample = "A01"
    other_ids = [sample_id for sample_id in sample_ids if sample_id != golden_sample]

    df = generate_doe_matrix(factors)
    df = randomize_run_order(df, seed=42)
    df = build_measurement_sequence(df, golden_sample, other_ids, seed=42)

    factor_cols = ["Run_Order", "Std_Order", "Point_Type", "Pvac", "Tvac", "Tstab", "Ttest"]
    step_cols: list[str] = []
    for i in range(1, 13):
        step_cols.extend([f"Step{i}_ID", f"Step{i}_dP"])
    df = df[factor_cols + step_cols]

    ok_offsets = {"A01": 0.000, "A02": 0.003, "A03": 0.006, "A04": 0.008, "A05": 0.011}
    ng_offsets = {"B01": 0.010, "B02": 0.013, "B03": 0.016, "B04": 0.020, "B05": 0.026}

    for row_idx in range(len(df)):
        row = df.iloc[row_idx]
        common, sep_bonus, stability = calc_run_profile(row)
        seed = int(row["Run_Order"]) * 100 + int(row["Std_Order"])
        rng = np.random.default_rng(seed)

        for step in range(1, 13):
            id_col = f"Step{step}_ID"
            dp_col = f"Step{step}_dP"
            sample_id = str(df.at[row_idx, id_col]).strip()

            if sample_id in ok_offsets:
                nominal = common + ok_offsets[sample_id]
                noise_sigma = stability if sample_id == golden_sample and step in (1, 6, 12) else 0.00045
                value = nominal + rng.normal(0, noise_sigma)
            elif sample_id in ng_offsets:
                nominal = common + ng_offsets[sample_id] + sep_bonus
                value = nominal + rng.normal(0, 0.00055)
            else:
                value = np.nan

            df.at[row_idx, dp_col] = round(float(value), 4)

    with pd.ExcelWriter(OUT_PATH, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="DOE_Experiment")

    print(OUT_PATH)
    print(df[["Run_Order", "Std_Order", "Point_Type", "Step1_ID", "Step1_dP", "Step2_ID", "Step2_dP"]].head(5).to_string(index=False))


if __name__ == "__main__":
    main()