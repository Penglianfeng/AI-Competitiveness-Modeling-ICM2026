#!/usr/bin/env python3
"""\
合并（仅本文件夹）AI人才数据为宽表
================================

用途：
- 只处理本文件夹 `ai_talent_data_v2/processed/combined_data_*.csv`
- 输出国家-年份面板宽表：一行=国家+年份；列=各指标
- 额外导出一份建模常用字段子集 + 摘要报告

运行：
  python .\"Supply, Mobility and Quality of AI Talents"\merge_ai_talent_wide.py
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime
import pandas as pd


INDICATOR_RENAME = {
    "每百万人研究人员数": "researchers_per_million",
    "每百万人研发技术人员数": "technicians_per_million",
    "高等教育毛入学率(%)": "tertiary_gross_enrollment_pct",
    "高等教育在校生总数": "tertiary_enrollment_total",
    "高等教育女性占比(%)": "tertiary_female_share_pct",
    "教育支出占GDP比例(%)": "education_expenditure_pct_gdp",
    "高等教育生均支出占人均GDP比例(%)": "tertiary_spend_per_student_pct_gdp_pc",
    "R&D支出占GDP比例(%)": "rd_expenditure_pct_gdp",
    "总人口": "population_total",
    "15-64岁人口占比(%)": "pop_15_64_pct",
    "每百万人研究人员(FTE)": "researchers_per_million_fte",
    "STEM毕业生占比(%)": "stem_graduates_pct",
    "25-34岁高等教育完成率(%)": "tertiary_completion_25_34_pct",
}


def find_latest_file(directory: Path, pattern: str) -> Path | None:
    if not directory.exists():
        return None
    files = list(directory.glob(pattern))
    if not files:
        return None
    return max(files, key=lambda p: p.stat().st_mtime)


def load_latest_long_table(base_dir: Path) -> pd.DataFrame:
    processed_dir = base_dir / "ai_talent_data_v2" / "processed"
    latest = find_latest_file(processed_dir, "combined_data_*.csv")
    if latest is None:
        raise FileNotFoundError(f"未找到 {processed_dir} 下的 combined_data_*.csv")

    df = pd.read_csv(latest)

    required = {"country_code", "year", "indicator_name_cn", "value"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"数据缺少必要列: {missing}. 实际列: {df.columns.tolist()}")

    df = df.copy()
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    # 仅保留目标列（其他列保留作补充信息）
    return df


def long_to_wide(df_long: pd.DataFrame) -> pd.DataFrame:
    df = df_long.copy()

    # 指标短名
    df["indicator"] = df["indicator_name_cn"].map(INDICATOR_RENAME).fillna(df["indicator_name_cn"])

    # 去重：同一国家-年份-指标，取均值（防止不同source重复）
    key_cols = ["country_code", "year", "indicator"]
    df = df.dropna(subset=["country_code", "year", "indicator"])

    wide = (
        df.pivot_table(
            index=["country_code", "year"],
            columns="indicator",
            values="value",
            aggfunc="mean",
        )
        .reset_index()
    )

    # 国家中文/英文名
    if "country_cn" in df.columns:
        cn_map = (
            df.dropna(subset=["country_code", "country_cn"])
            .drop_duplicates(subset=["country_code"])
            .set_index("country_code")["country_cn"]
            .to_dict()
        )
        wide["country_cn"] = wide["country_code"].map(cn_map)

    if "country_name" in df.columns:
        en_map = (
            df.dropna(subset=["country_code", "country_name"])
            .drop_duplicates(subset=["country_code"])
            .set_index("country_code")["country_name"]
            .to_dict()
        )
        wide["country_en"] = wide["country_code"].map(en_map)

    # 列顺序：国家/年份/名称在前
    front = [c for c in ["country_code", "country_cn", "country_en", "year"] if c in wide.columns]
    rest = [c for c in wide.columns if c not in front]
    wide = wide[front + rest]

    # 排序
    wide = wide.sort_values(["country_code", "year"]).reset_index(drop=True)
    return wide


def build_model_features(wide: pd.DataFrame) -> pd.DataFrame:
    """挑一份建模常用字段子集（可按需调整）"""
    preferred = [
        "country_code",
        "country_cn",
        "country_en",
        "year",
        "rd_expenditure_pct_gdp",
        "researchers_per_million",
        "technicians_per_million",
        "tertiary_gross_enrollment_pct",
        "education_expenditure_pct_gdp",
        "population_total",
        "pop_15_64_pct",
        "stem_graduates_pct",
        "tertiary_completion_25_34_pct",
    ]
    cols = [c for c in preferred if c in wide.columns]
    return wide[cols].copy()


def write_summary(wide: pd.DataFrame, out_dir: Path) -> Path:
    lines: list[str] = []
    lines.append("=" * 70)
    lines.append("AI人才宽表数据摘要")
    lines.append("=" * 70)
    lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"总行数(国家-年份): {len(wide)}")

    if "country_code" in wide.columns:
        lines.append(f"覆盖国家数: {wide['country_code'].nunique()}")
    if "year" in wide.columns:
        lines.append(f"年份范围: {int(wide['year'].min())} - {int(wide['year'].max())}")

    feature_cols = [c for c in wide.columns if c not in {"country_code", "country_cn", "country_en", "year"}]
    lines.append(f"指标列数: {len(feature_cols)}")
    lines.append("")
    lines.append("各指标非空率(Top 15):")
    for col in sorted(feature_cols, key=lambda c: wide[c].notna().mean(), reverse=True)[:15]:
        pct = wide[col].notna().mean() * 100
        lines.append(f"- {col}: {pct:.1f}%")

    path = out_dir / "summary_report.txt"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> None:
    base_dir = Path(__file__).resolve().parent

    df_long = load_latest_long_table(base_dir)
    wide = long_to_wide(df_long)

    out_dir = base_dir / "merged_wide"
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    wide_csv = out_dir / f"ai_talent_wide_{ts}.csv"
    wide.to_csv(wide_csv, index=False, encoding="utf-8-sig")

    # Excel（可选）
    wide_xlsx = out_dir / f"ai_talent_wide_{ts}.xlsx"
    try:
        with pd.ExcelWriter(wide_xlsx, engine="openpyxl") as writer:
            wide.to_excel(writer, sheet_name="wide", index=False)
            build_model_features(wide).to_excel(writer, sheet_name="model_features", index=False)
    except Exception:
        # 没装 openpyxl 时不阻塞主流程
        wide_xlsx = None

    model_df = build_model_features(wide)
    model_csv = out_dir / f"model_features_{ts}.csv"
    model_df.to_csv(model_csv, index=False, encoding="utf-8-sig")

    summary_path = write_summary(wide, out_dir)

    print("\n" + "=" * 70)
    print("AI人才宽表生成完成")
    print("=" * 70)
    print(f"宽表CSV: {wide_csv}")
    if wide_xlsx is not None:
        print(f"宽表Excel: {wide_xlsx}")
    else:
        print("宽表Excel: (跳过，可能未安装 openpyxl)")
    print(f"建模子集CSV: {model_csv}")
    print(f"摘要报告: {summary_path}")

    show_cols = [c for c in ["country_code", "country_cn", "year", "rd_expenditure_pct_gdp", "researchers_per_million"] if c in wide.columns]
    if show_cols:
        print("\n预览(前10行):")
        print(wide[show_cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
