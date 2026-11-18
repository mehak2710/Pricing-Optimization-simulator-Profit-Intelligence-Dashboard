
"""
STAGE 3.2 — DATA PREPROCESSING
Project: ShopSphere Pricing Intelligence System

Run:
python src/preprocess.py

Input:
data/raw/

Output:
data/processed/
"""

import pandas as pd
import numpy as np
from pathlib import Path

from data_validation import validate_datasets


# ============================================================
# 1. PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. LOAD DATASETS
# ============================================================

def load_datasets():

    print("\nLoading raw datasets...")

    products = pd.read_csv(
        RAW_DIR / "products.csv"
    )

    sales = pd.read_csv(
        RAW_DIR / "sales.csv"
    )

    competitors = pd.read_csv(
        RAW_DIR / "competitor_pricing.csv"
    )

    market = pd.read_csv(
        RAW_DIR / "market_conditions.csv"
    )

    dates = pd.read_csv(
        RAW_DIR / "date_dimension.csv"
    )

    print("All datasets loaded successfully.")

    return products, sales, competitors, market, dates


# ============================================================
# 3. STANDARDIZE COLUMN VALUES AND DATA TYPES
# ============================================================

def standardize_data(
    products,
    sales,
    competitors,
    market,
    dates
):

    print("\nStandardizing datasets...")

    # Remove accidental whitespace from text fields.
    datasets = [
        products, sales, competitors, market, dates
    ]

    for df in datasets:

        text_columns = df.select_dtypes(
            include=["object"]
        ).columns

        for column in text_columns:

            df[column] = df[column].str.strip()

    # Standardize dates.
    sales["Date"] = pd.to_datetime(
        sales["Date"], errors="raise"
    )

    competitors["Observation_Date"] = pd.to_datetime(
        competitors["Observation_Date"],
        errors="raise"
    )

    market["Date"] = pd.to_datetime(
        market["Date"], errors="raise"
    )

    dates["Date"] = pd.to_datetime(
        dates["Date"], errors="raise"
    )

    # Convert numeric fields.
    numeric_columns = {

        "products": [
            "Unit_Cost",
            "Base_Price",
            "Minimum_Margin_Percentage",
            "Maximum_Price_Change_Percentage",
            "Base_Daily_Demand",
            "Price_Elasticity"
        ],

        "sales": [
            "Selling_Price",
            "Units_Sold",
            "Discount_Percentage",
            "Revenue",
            "Profit",
            "Unit_Cost_At_Sale",
            "List_Price"
        ],

        "competitors": [
            "Competitor_Price"
        ],

        "market": [
            "Demand_Index",
            "Seasonality",
            "Inventory_Level",
            "Promotion_Intensity"
        ],

        "dates": [
            "Day", "Week", "Month", "Year"
        ]
    }

    mapping = {
        "products": products,
        "sales": sales,
        "competitors": competitors,
        "market": market,
        "dates": dates
    }

    for dataset_name, columns in numeric_columns.items():

        df = mapping[dataset_name]

        for column in columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="raise"
            )

    # Integer fields.
    sales["Units_Sold"] = sales["Units_Sold"].astype("int64")

    market["Inventory_Level"] = (
        market["Inventory_Level"].astype("int64")
    )

    dates["Day"] = dates["Day"].astype("int64")
    dates["Month"] = dates["Month"].astype("int64")
    dates["Year"] = dates["Year"].astype("int64")

    print("Data types standardized.")

    return products, sales, competitors, market, dates


# ============================================================
# 4. REMOVE EXACT DUPLICATE ROWS
# ============================================================

def remove_exact_duplicates(
    products,
    sales,
    competitors,
    market,
    dates
):

    print("\nChecking exact duplicates...")

    datasets = {
        "Products": products,
        "Sales": sales,
        "Competitors": competitors,
        "Market": market,
        "Dates": dates
    }

    cleaned = {}

    for name, df in datasets.items():

        before = len(df)

        df = df.drop_duplicates().copy()

        after = len(df)

        print(
            f"{name}: Removed {before - after} exact duplicates."
        )

        cleaned[name] = df

    return (
        cleaned["Products"],
        cleaned["Sales"],
        cleaned["Competitors"],
        cleaned["Market"],
        cleaned["Dates"]
    )


# ============================================================
# 5. FINANCIAL DATA VALIDATION
# ============================================================

def validate_financial_calculations(sales):

    print("\nValidating financial calculations...")

    expected_revenue = (
        sales["Selling_Price"]
        * sales["Units_Sold"]
    )

    expected_profit = (
        sales["Selling_Price"]
        - sales["Unit_Cost_At_Sale"]
    ) * sales["Units_Sold"]

    revenue_difference = (
        sales["Revenue"] - expected_revenue
    ).abs()

    profit_difference = (
        sales["Profit"] - expected_profit
    ).abs()

    if (revenue_difference > 0.02).any():

        raise ValueError(
            "Revenue calculation mismatch detected."
        )

    if (profit_difference > 0.02).any():

        raise ValueError(
            "Profit calculation mismatch detected."
        )

    # Validate selling price against list price and discount.
    calculated_discount = (
        1 - sales["Selling_Price"] / sales["List_Price"]
    )

    discount_difference = (
        calculated_discount - sales["Discount_Percentage"]
    ).abs()

    if (discount_difference > 0.015).any():

        raise ValueError(
            "Selling price and discount percentage mismatch."
        )

    print("Revenue, profit and discount calculations verified.")


# ============================================================
# 6. PRODUCT FEATURE ENGINEERING
# ============================================================

def engineer_product_features(products):

    print("\nCreating product-level features...")

    products = products.copy()

    # Gross margin at base price.
    products["Base_Gross_Margin_Percentage"] = (
        (
            products["Base_Price"]
            - products["Unit_Cost"]
        )
        / products["Base_Price"]
    )

    # Ground-truth synthetic elasticity segmentation.
    # This is NOT an estimated elasticity result.

    absolute_elasticity = products[
        "Price_Elasticity"
    ].abs()

    products["Synthetic_Elasticity_Segment"] = np.select(
        [
            absolute_elasticity >= 2.0,
            absolute_elasticity >= 1.2
        ],
        [
            "High",
            "Medium"
        ],
        default="Low"
    )

    products["Base_Gross_Margin_Percentage"] = (
        products["Base_Gross_Margin_Percentage"].round(4)
    )

    return products


# ============================================================
# 7. SALES FEATURE ENGINEERING
# ============================================================

def engineer_sales_features(sales):

    print("\nCreating sales analytical features...")

    sales = sales.copy()

    # Revenue per transaction line.
    sales["Revenue"] = (
        sales["Selling_Price"] * sales["Units_Sold"]
    ).round(2)

    # Profit per transaction line.
    sales["Profit"] = (
        (
            sales["Selling_Price"]
            - sales["Unit_Cost_At_Sale"]
        ) * sales["Units_Sold"]
    ).round(2)

    # Total discount amount.
    sales["Discount_Amount"] = (
        (
            sales["List_Price"]
            - sales["Selling_Price"]
        ) * sales["Units_Sold"]
    ).round(2)

    # Transaction-level profit margin.
    sales["Profit_Margin_Percentage"] = np.where(
        sales["Revenue"] != 0,
        sales["Profit"] / sales["Revenue"],
        0
    )

    # Cost percentage.
    sales["Cost_Percentage"] = np.where(
        sales["Revenue"] != 0,
        (
            sales["Unit_Cost_At_Sale"]
            * sales["Units_Sold"]
        ) / sales["Revenue"],
        0
    )

    # Actual selling price relative to list price.
    sales["Price_Realization_Percentage"] = (
        sales["Selling_Price"] / sales["List_Price"]
    )

    # Date features.
    sales["Year"] = sales["Date"].dt.year
    sales["Month"] = sales["Date"].dt.month
    sales["Month_Name"] = sales["Date"].dt.strftime("%B")
    sales["Quarter"] = (
        "Q" + sales["Date"].dt.quarter.astype(str)
    )

    # Round calculated ratios.
    ratio_columns = [
        "Profit_Margin_Percentage",
        "Cost_Percentage",
        "Price_Realization_Percentage"
    ]

    sales[ratio_columns] = (
        sales[ratio_columns].round(4)
    )

    return sales


# ============================================================
# 8. COMPETITOR FEATURE ENGINEERING
# ============================================================

def engineer_competitor_features(
    competitors,
    products
):

    print("\nCreating competitor pricing features...")

    competitors = competitors.copy()

    # Add company reference price.
    competitors = competitors.merge(
        products[
            ["Product_ID", "Base_Price"]
        ],
        on="Product_ID",
        how="left",
        validate="many_to_one"
    )

    competitors.rename(
        columns={
            "Base_Price": "Company_Base_Price"
        },
        inplace=True
    )

    # Absolute competitor price difference.
    competitors["Price_Difference_vs_Base"] = (
        competitors["Competitor_Price"]
        - competitors["Company_Base_Price"]
    )

    # Percentage difference.
    competitors["Price_Difference_Percentage"] = (
        competitors["Price_Difference_vs_Base"]
        / competitors["Company_Base_Price"]
    )

    # Competitive positioning.
    competitors["Competitive_Position"] = np.select(
        [
            competitors["Price_Difference_Percentage"] < -0.02,
            competitors["Price_Difference_Percentage"] > 0.02
        ],
        [
            "Competitor Cheaper",
            "Competitor More Expensive"
        ],
        default="Price Similar"
    )

    competitors[
        [
            "Price_Difference_vs_Base",
            "Price_Difference_Percentage"
        ]
    ] = competitors[
        [
            "Price_Difference_vs_Base",
            "Price_Difference_Percentage"
        ]
    ].round(4)

    return competitors


# ============================================================
# 9. MARKET CONDITIONS CLEANING
# ============================================================

def clean_market_data(market):

    print("\nCleaning market conditions...")

    market = market.copy()

    # Sort by product and date.
    market.sort_values(
        ["Product_ID", "Date"],
        inplace=True
    )

    # Standardize campaign text.
    market["Marketing_Campaign"] = (
        market["Marketing_Campaign"].str.strip()
    )

    return market


# ============================================================
# 10. DATE DIMENSION CLEANING
# ============================================================

def clean_date_dimension(dates):

    print("\nCleaning date dimension...")

    dates = dates.copy()

    dates.sort_values(
        "Date",
        inplace=True
    )

    dates["Is_Weekend"] = (
        dates["Is_Weekend"].astype("int64")
    )

    dates["Is_Festival_Season"] = (
        dates["Is_Festival_Season"].astype("int64")
    )

    return dates


# ============================================================
# 11. OUTLIER DETECTION
# ============================================================

def detect_sales_outliers(sales):

    print("\nDetecting unusual sales observations...")

    sales = sales.copy()

    def calculate_iqr_flags(series):

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)

        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        return (
            (series < lower) |
            (series > upper)
        )

    # Detect unusual prices within each product.
    sales["Price_Outlier_Flag"] = (
        sales.groupby("Product_ID")["Selling_Price"]
        .transform(calculate_iqr_flags)
    )

    # Detect unusually large transaction quantities.
    sales["Quantity_Outlier_Flag"] = (
        sales.groupby("Product_ID")["Units_Sold"]
        .transform(calculate_iqr_flags)
    )

    print(
        "Price outliers:",
        int(sales["Price_Outlier_Flag"].sum())
    )

    print(
        "Quantity outliers:",
        int(sales["Quantity_Outlier_Flag"].sum())
    )

    # We flag unusual records but do NOT delete them.
    return sales


# ============================================================
# 12. DATA QUALITY REPORT AND FINAL VALIDATION
# ============================================================

def final_validation(
    products,
    sales,
    competitors,
    market,
    dates
):

    print("\nRunning final validation...")

    report = validate_datasets(
        products,
        sales,
        competitors,
        market,
        dates
    )

    critical_errors = report[
        (report["Status"] == "FAIL") &
        (report["Severity"] == "ERROR")
    ]

    if not critical_errors.empty:

        raise ValueError(
            "Processed data failed validation. "
            "Check data_quality_report.csv."
        )

    print("Final validation passed.")


# ============================================================
# 13. EXPORT PROCESSED DATASETS
# ============================================================

def export_datasets(
    products,
    sales,
    competitors,
    market,
    dates
):

    print("\nExporting processed datasets...")

    datasets = {
        "products_clean.csv": products,
        "sales_clean.csv": sales,
        "competitor_pricing_clean.csv": competitors,
        "market_conditions_clean.csv": market,
        "date_dimension_clean.csv": dates
    }

    for filename, df in datasets.items():

        output_path = PROCESSED_DIR / filename

        df.to_csv(
            output_path,
            index=False,
            date_format="%Y-%m-%d"
        )

        print(
            f"{filename}: {len(df):,} rows exported."
        )


# ============================================================
# 14. FINAL SUMMARY
# ============================================================

def display_summary(
    products,
    sales,
    competitors,
    market,
    dates
):

    print("\n" + "=" * 60)
    print("STAGE 3 — PREPROCESSING SUMMARY")
    print("=" * 60)

    print("\nDATASET SIZES")

    print("Products:", f"{len(products):,}")
    print("Sales:", f"{len(sales):,}")
    print("Competitors:", f"{len(competitors):,}")
    print("Market:", f"{len(market):,}")
    print("Date Dimension:", f"{len(dates):,}")

    print("\nFINANCIAL SUMMARY")

    revenue = sales["Revenue"].sum()
    profit = sales["Profit"].sum()

    margin = profit / revenue if revenue else 0

    print(f"Revenue: ₹{revenue:,.2f}")
    print(f"Profit: ₹{profit:,.2f}")
    print(f"Profit Margin: {margin:.2%}")

    print("\nOUTLIER SUMMARY")

    print(
        "Price outliers:",
        int(sales["Price_Outlier_Flag"].sum())
    )

    print(
        "Quantity outliers:",
        int(sales["Quantity_Outlier_Flag"].sum())
    )

    print("\nPROCESSED FILES LOCATION:")
    print(PROCESSED_DIR)

    print("\nSTAGE 3 COMPLETED SUCCESSFULLY!")


# ============================================================
# 15. MAIN PIPELINE
# ============================================================

def main():

    print("=" * 60)
    print("SHOPSPHERE DATA PREPROCESSING PIPELINE")
    print("=" * 60)

    (
        products,
        sales,
        competitors,
        market,
        dates
    ) = load_datasets()

    (
        products,
        sales,
        competitors,
        market,
        dates
    ) = standardize_data(
        products,
        sales,
        competitors,
        market,
        dates
    )

    (
        products,
        sales,
        competitors,
        market,
        dates
    ) = remove_exact_duplicates(
        products,
        sales,
        competitors,
        market,
        dates
    )

    validate_financial_calculations(sales)

    products = engineer_product_features(products)

    sales = engineer_sales_features(sales)

    competitors = engineer_competitor_features(
        competitors,
        products
    )

    market = clean_market_data(market)

    dates = clean_date_dimension(dates)

    sales = detect_sales_outliers(sales)

    final_validation(
        products,
        sales,
        competitors,
        market,
        dates
    )

    export_datasets(
        products,
        sales,
        competitors,
        market,
        dates
    )

    display_summary(
        products,
        sales,
        competitors,
        market,
        dates
    )


if __name__ == "__main__":
    main()
