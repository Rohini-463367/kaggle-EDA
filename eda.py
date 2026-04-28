from __future__ import annotations

from pathlib import Path
from typing import Iterable
import warnings

import numpy as np
import pandas as pd


warnings.filterwarnings("ignore")

DATA_DIR = Path("data")
OUTPUT_DIR = Path("output")


def find_csv(data_dir: Path | str = DATA_DIR, csv_file: str | None = None) -> Path:
    data_dir = Path(data_dir)
    if csv_file:
        path = data_dir / csv_file
        if not path.exists():
            raise FileNotFoundError(f"Could not find dataset: {path}")
        return path

    csv_files = sorted(data_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            "No CSV file found in the data folder. Add a Kaggle CSV to ./data and rerun."
        )
    return csv_files[0]


def load_data(csv_file: str | None = None, data_dir: Path | str = DATA_DIR) -> tuple[Path, pd.DataFrame]:
    csv_path = find_csv(data_dir=data_dir, csv_file=csv_file)
    df = pd.read_csv(csv_path)
    return csv_path, df


def set_plot_style() -> None:
    plt, sns = _load_plot_libs()
    sns.set_theme(style="whitegrid", palette="deep")
    plt.rcParams["figure.figsize"] = (10, 6)
    plt.rcParams["axes.titlesize"] = 14
    plt.rcParams["axes.labelsize"] = 11


def _load_plot_libs():
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError as exc:
        raise ImportError(
            "Plotting packages are missing. Install the requirements file before running chart cells."
        ) from exc
    return plt, sns


def profile_overview(df: pd.DataFrame) -> dict[str, object]:
    memory_mb = float(round(df.memory_usage(deep=True).sum() / (1024 ** 2), 2))
    overview = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_cells": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "memory_mb": memory_mb,
    }
    return overview


def column_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = pd.DataFrame(
        {
            "dtype": df.dtypes.astype(str),
            "non_null": df.notna().sum(),
            "missing": df.isna().sum(),
            "missing_pct": (df.isna().mean() * 100).round(2),
            "unique": df.nunique(dropna=False),
        }
    )
    return summary.sort_values(["missing_pct", "unique"], ascending=[False, False])


def get_column_groups(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    return numeric_cols, categorical_cols


def prepare_dates(df: pd.DataFrame, threshold: float = 0.8) -> tuple[pd.DataFrame, list[str]]:
    parsed_cols: list[str] = []
    object_cols = df.select_dtypes(include=["object"]).columns

    for col in object_cols:
        sample = df[col].dropna().astype(str).head(200)
        if sample.empty or sample.nunique() <= 1:
            continue

        parsed = pd.to_datetime(sample, errors="coerce")
        success_rate = parsed.notna().mean()

        if success_rate >= threshold:
            df[col] = pd.to_datetime(df[col], errors="coerce")
            parsed_cols.append(col)

    return df, parsed_cols


def numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols, _ = get_column_groups(df)
    if not numeric_cols:
        return pd.DataFrame()

    summary = df[numeric_cols].describe().T
    summary["missing"] = df[numeric_cols].isna().sum()
    summary["missing_pct"] = (df[numeric_cols].isna().mean() * 100).round(2)
    summary["skew"] = df[numeric_cols].skew(numeric_only=True).round(2)
    return summary.round(2)


def categorical_summary(df: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    _, categorical_cols = get_column_groups(df)
    if not categorical_cols:
        return pd.DataFrame()

    rows = []
    for col in categorical_cols:
        mode = df[col].mode(dropna=True)
        rows.append(
            {
                "column": col,
                "unique": int(df[col].nunique(dropna=False)),
                "missing": int(df[col].isna().sum()),
                "missing_pct": round(df[col].isna().mean() * 100, 2),
                "top_value": None if mode.empty else mode.iloc[0],
                "top_count": 0 if mode.empty else int((df[col] == mode.iloc[0]).sum()),
            }
        )

    summary = pd.DataFrame(rows).sort_values(["missing_pct", "unique"], ascending=[False, False])
    return summary.head(top_n) if top_n else summary


def date_summary(df: pd.DataFrame) -> pd.DataFrame:
    date_cols = df.select_dtypes(include=["datetime64[ns]", "datetimetz"]).columns.tolist()
    if not date_cols:
        return pd.DataFrame()

    rows = []
    for col in date_cols:
        rows.append(
            {
                "column": col,
                "min_date": df[col].min(),
                "max_date": df[col].max(),
                "missing": int(df[col].isna().sum()),
            }
        )
    return pd.DataFrame(rows)


def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        pd.DataFrame(
            {
                "missing": df.isna().sum(),
                "missing_pct": (df.isna().mean() * 100).round(2),
            }
        )
        .query("missing > 0")
        .sort_values("missing_pct", ascending=False)
    )
    return summary


def outlier_summary(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols, _ = get_column_groups(df)
    if not numeric_cols:
        return pd.DataFrame()

    rows = []
    for col in numeric_cols:
        series = df[col].dropna()
        if series.empty:
            continue
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            outliers = 0
        else:
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outliers = int(((series < lower) | (series > upper)).sum())
        rows.append(
            {
                "column": col,
                "outliers": outliers,
                "outlier_pct": round((outliers / len(series)) * 100, 2),
            }
        )

    return pd.DataFrame(rows).sort_values("outlier_pct", ascending=False)


def grouped_numeric_summary(
    df: pd.DataFrame,
    max_categorical: int = 3,
    max_numeric: int = 3,
    max_categories: int = 10,
) -> dict[str, pd.DataFrame]:
    numeric_cols, categorical_cols = get_column_groups(df)
    tables: dict[str, pd.DataFrame] = {}

    usable_cats = [col for col in categorical_cols if 1 < df[col].nunique(dropna=True) <= max_categories]
    for cat_col in usable_cats[:max_categorical]:
        agg = (
            df.groupby(cat_col)[numeric_cols[:max_numeric]]
            .mean(numeric_only=True)
            .round(2)
            .sort_index()
        )
        if not agg.empty:
            tables[cat_col] = agg
    return tables


def ensure_output_dir(output_dir: Path | str = OUTPUT_DIR) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def save_plot(fig, name: str, output_dir: Path | str = OUTPUT_DIR, show: bool = True) -> None:
    plt, _ = _load_plot_libs()
    output_path = ensure_output_dir(output_dir)
    fig.tight_layout()
    fig.savefig(output_path / name, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_missingness(df: pd.DataFrame, output_dir: Path | str = OUTPUT_DIR, show: bool = True) -> None:
    plt, _ = _load_plot_libs()
    summary = missing_summary(df)
    if summary.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    summary["missing_pct"].plot(kind="bar", ax=ax, color="#d95f02")
    ax.set_title("Missing Values by Column")
    ax.set_ylabel("Missing Percentage")
    ax.set_xlabel("Columns")
    ax.tick_params(axis="x", rotation=45)
    save_plot(fig, "missing_values.png", output_dir=output_dir, show=show)


def _iter_chunks(items: Iterable[str], size: int) -> list[list[str]]:
    items = list(items)
    return [items[i : i + size] for i in range(0, len(items), size)]


def plot_numeric_distributions(
    df: pd.DataFrame,
    output_dir: Path | str = OUTPUT_DIR,
    show: bool = True,
    max_cols: int = 8,
) -> None:
    plt, sns = _load_plot_libs()
    numeric_cols, _ = get_column_groups(df)
    numeric_cols = numeric_cols[:max_cols]
    if not numeric_cols:
        return

    for idx, chunk in enumerate(_iter_chunks(numeric_cols, 4), start=1):
        rows = len(chunk)
        fig, axes = plt.subplots(rows, 1, figsize=(10, 4 * rows))
        if rows == 1:
            axes = [axes]
        for ax, col in zip(axes, chunk):
            sns.histplot(df[col].dropna(), kde=True, ax=ax, color="#1f77b4")
            ax.set_title(f"Distribution of {col}")
        save_plot(fig, f"numeric_dist_{idx}.png", output_dir=output_dir, show=show)


def plot_boxplots(
    df: pd.DataFrame,
    output_dir: Path | str = OUTPUT_DIR,
    show: bool = True,
    max_cols: int = 8,
) -> None:
    plt, sns = _load_plot_libs()
    numeric_cols, _ = get_column_groups(df)
    numeric_cols = numeric_cols[:max_cols]
    if not numeric_cols:
        return

    for idx, chunk in enumerate(_iter_chunks(numeric_cols, 4), start=1):
        rows = len(chunk)
        fig, axes = plt.subplots(rows, 1, figsize=(10, 4 * rows))
        if rows == 1:
            axes = [axes]
        for ax, col in zip(axes, chunk):
            sns.boxplot(x=df[col], ax=ax, color="#66c2a5")
            ax.set_title(f"Boxplot of {col}")
        save_plot(fig, f"boxplot_{idx}.png", output_dir=output_dir, show=show)


def plot_correlation_heatmap(
    df: pd.DataFrame,
    output_dir: Path | str = OUTPUT_DIR,
    show: bool = True,
    max_cols: int = 12,
) -> None:
    plt, sns = _load_plot_libs()
    numeric_cols, _ = get_column_groups(df)
    numeric_cols = numeric_cols[:max_cols]
    if len(numeric_cols) < 2:
        return

    corr = df[numeric_cols].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, cmap="coolwarm", annot=True, fmt=".2f", ax=ax)
    ax.set_title("Correlation Heatmap")
    save_plot(fig, "correlation_heatmap.png", output_dir=output_dir, show=show)


def plot_categorical_counts(
    df: pd.DataFrame,
    output_dir: Path | str = OUTPUT_DIR,
    show: bool = True,
    max_cols: int = 6,
    max_categories: int = 15,
) -> None:
    plt, sns = _load_plot_libs()
    _, categorical_cols = get_column_groups(df)
    selected = [col for col in categorical_cols if df[col].nunique(dropna=True) <= max_categories][:max_cols]
    if not selected:
        return

    for idx, col in enumerate(selected, start=1):
        counts = df[col].fillna("Missing").astype(str).value_counts().head(max_categories)
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(x=counts.values, y=counts.index, ax=ax, color="#7570b3")
        ax.set_title(f"Top Categories in {col}")
        ax.set_xlabel("Count")
        ax.set_ylabel(col)
        save_plot(fig, f"category_{idx}_{col}.png", output_dir=output_dir, show=show)


def run_eda(csv_file: str | None = None, data_dir: Path | str = DATA_DIR, output_dir: Path | str = OUTPUT_DIR) -> dict[str, object]:
    set_plot_style()
    csv_path, df = load_data(csv_file=csv_file, data_dir=data_dir)
    df, parsed_dates = prepare_dates(df)
    numeric_cols, categorical_cols = get_column_groups(df)

    print(f"Dataset: {csv_path}")
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"Numeric columns: {len(numeric_cols)}")
    print(f"Categorical columns: {len(categorical_cols)}")
    if parsed_dates:
        print(f"Parsed date columns: {', '.join(parsed_dates)}")

    print("\nOverview")
    print(profile_overview(df))

    missing = missing_summary(df)
    if not missing.empty:
        print("\nMissing values")
        print(missing.head(10).to_string())

    outliers = outlier_summary(df)
    if not outliers.empty:
        print("\nPotential outliers")
        print(outliers.head(10).to_string(index=False))

    plot_missingness(df, output_dir=output_dir, show=False)
    plot_numeric_distributions(df, output_dir=output_dir, show=False)
    plot_boxplots(df, output_dir=output_dir, show=False)
    plot_correlation_heatmap(df, output_dir=output_dir, show=False)
    plot_categorical_counts(df, output_dir=output_dir, show=False)

    return {
        "dataset_path": str(csv_path),
        "overview": profile_overview(df),
        "column_summary": column_summary(df),
        "numeric_summary": numeric_summary(df),
        "categorical_summary": categorical_summary(df),
        "date_summary": date_summary(df),
        "missing_summary": missing,
        "outlier_summary": outliers,
        "grouped_summary": grouped_numeric_summary(df),
    }


if __name__ == "__main__":
    run_eda()
