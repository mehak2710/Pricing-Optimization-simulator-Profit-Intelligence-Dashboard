import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

PROCESSED.mkdir(parents=True, exist_ok=True)

FILES = {
    "products": "products.csv",
    "sales": "sales.csv",
    "competitors": "competitor_pricing.csv",
    "market": "market_conditions.csv",
    "dates": "date_dimension.csv"
}

REQUIRED_COLUMNS = {
    "products": [
        "Product_ID", "Product_Name", "Category",
        "Unit_Cost", "Base_Price", "Price_Elasticity"
    ],
    "sales": [
        "Transaction_ID", "Product_ID", "Date",
        "Selling_Price", "Units_Sold",
        "Discount_Percentage", "Revenue", "Profit",
        "Unit_Cost_At_Sale", "List_Price"
    ],
    "competitors": [
        "Product_ID", "Competitor_Name",
        "Competitor_Price", "Observation_Date"
    ],
    "market": [
        "Date", "Product_ID", "Demand_Index",
        "Seasonality", "Inventory_Level",
        "Marketing_Campaign"
    ],
    "dates": [
        "Date", "Day", "Month", "Year",
        "Quarter", "Financial_Year"
    ]
}


def add_result(report, dataset, check, status, severity, details):
    report.append({
        "Dataset": dataset,
        "Check": check,
        "Status": status,
        "Severity": severity,
        "Details": details
    })


def validate_datasets(products, sales, competitors, market, dates):

    report = []

    datasets = {
        "Products": products,
        "Sales": sales,
        "Competitor Pricing": competitors,
        "Market Conditions": market,
        "Date Dimension": dates
    }

    # --------------------------------------------------
    # 1. REQUIRED COLUMN CHECK
    # --------------------------------------------------

    for name, df in datasets.items():

        key = {
            "Products": "products",
            "Sales": "sales",
            "Competitor Pricing": "competitors",
            "Market Conditions": "market",
            "Date Dimension": "dates"
        }[name]

        missing_columns = set(REQUIRED_COLUMNS[key]) - set(df.columns)

        add_result(
            report,
            name,
            "Required columns",
            "FAIL" if missing_columns else "PASS",
            "ERROR" if missing_columns else "INFO",
            f"Missing columns: {list(missing_columns)}"
            if missing_columns else "All required columns present"
        )

    # Stop if a required column is missing.
    if any(r["Severity"] == "ERROR" and r["Status"] == "FAIL"
           for r in report):
        return pd.DataFrame(report)

    # --------------------------------------------------
    # 2. MISSING VALUE CHECK
    # --------------------------------------------------

    for name, df in datasets.items():

        missing = int(df.isna().sum().sum())

        add_result(
            report,
            name,
            "Missing values",
            "FAIL" if missing else "PASS",
            "ERROR" if missing else "INFO",
            f"Total missing cells: {missing}"
        )

    # --------------------------------------------------
    # 3. DUPLICATE CHECKS
    # --------------------------------------------------

    duplicate_checks = {
        "Products": (
            products,
            ["Product_ID"]
        ),
        "Sales": (
            sales,
            ["Transaction_ID"]
        ),
        "Competitor Pricing": (
            competitors,
            ["Product_ID", "Competitor_Name", "Observation_Date"]
        ),
        "Market Conditions": (
            market,
            ["Product_ID", "Date"]
        ),
        "Date Dimension": (
            dates,
            ["Date"]
        )
    }

    for name, (df, columns) in duplicate_checks.items():

        count = int(df.duplicated(columns).sum())

        add_result(
            report,
            name,
            "Duplicate business keys",
            "FAIL" if count else "PASS",
            "ERROR" if count else "INFO",
            f"Duplicate records: {count}"
        )

    # --------------------------------------------------
    # 4. PRODUCT VALIDATION
    # --------------------------------------------------

    invalid_cost = int((products["Unit_Cost"] <= 0).sum())

    invalid_price = int(
        (products["Base_Price"] <= products["Unit_Cost"]).sum()
    )

    add_result(
        report, "Products", "Positive unit cost",
        "FAIL" if invalid_cost else "PASS",
        "ERROR" if invalid_cost else "INFO",
        f"Invalid costs: {invalid_cost}"
    )

    add_result(
        report, "Products", "Base price above unit cost",
        "FAIL" if invalid_price else "PASS",
        "ERROR" if invalid_price else "INFO",
        f"Invalid products: {invalid_price}"
    )

    # --------------------------------------------------
    # 5. SALES VALIDATION
    # --------------------------------------------------

    invalid_units = int((sales["Units_Sold"] <= 0).sum())

    invalid_selling_price = int(
        (sales["Selling_Price"] <= 0).sum()
    )

    invalid_discount = int(
        (~sales["Discount_Percentage"].between(0, 0.50)).sum()
    )

    unknown_products = int(
        (~sales["Product_ID"].isin(products["Product_ID"])).sum()
    )

    revenue_difference = (
        sales["Revenue"]
        - sales["Selling_Price"] * sales["Units_Sold"]
    ).abs()

    profit_difference = (
        sales["Profit"]
        - (
            sales["Selling_Price"]
            - sales["Unit_Cost_At_Sale"]
        ) * sales["Units_Sold"]
    ).abs()

    checks = [
        ("Positive units sold", invalid_units),
        ("Positive selling price", invalid_selling_price),
        ("Valid discount percentage", invalid_discount),
        ("Valid Product_ID", unknown_products),
        ("Revenue calculation", int((revenue_difference > 0.02).sum())),
        ("Profit calculation", int((profit_difference > 0.02).sum()))
    ]

    for check, count in checks:

        add_result(
            report,
            "Sales",
            check,
            "FAIL" if count else "PASS",
            "ERROR" if count else "INFO",
            f"Invalid records: {count}"
        )

    # --------------------------------------------------
    # 6. COMPETITOR VALIDATION
    # --------------------------------------------------

    invalid_competitor_prices = int(
        (competitors["Competitor_Price"] <= 0).sum()
    )

    unknown_competitors = int(
        (~competitors["Product_ID"].isin(products["Product_ID"])).sum()
    )

    add_result(
        report,
        "Competitor Pricing",
        "Positive competitor prices",
        "FAIL" if invalid_competitor_prices else "PASS",
        "ERROR" if invalid_competitor_prices else "INFO",
        f"Invalid records: {invalid_competitor_prices}"
    )

    add_result(
        report,
        "Competitor Pricing",
        "Valid Product_ID",
        "FAIL" if unknown_competitors else "PASS",
        "ERROR" if unknown_competitors else "INFO",
        f"Unknown products: {unknown_competitors}"
    )

    competitor_counts = competitors.groupby(
        ["Product_ID", "Observation_Date"]
    )["Competitor_Name"].nunique()

    incomplete_observations = int(
        (competitor_counts != 3).sum()
    )

    add_result(
        report,
        "Competitor Pricing",
        "Three competitors per product-date",
        "FAIL" if incomplete_observations else "PASS",
        "ERROR" if incomplete_observations else "INFO",
        f"Incomplete product-date groups: {incomplete_observations}"
    )

    # --------------------------------------------------
    # 7. MARKET VALIDATION
    # --------------------------------------------------

    invalid_inventory = int(
        (market["Inventory_Level"] < 0).sum()
    )

    invalid_demand = int(
        (market["Demand_Index"] <= 0).sum()
    )

    invalid_seasonality = int(
        (market["Seasonality"] <= 0).sum()
    )

    unknown_market_products = int(
        (~market["Product_ID"].isin(products["Product_ID"])).sum()
    )

    market_checks = [
        ("Non-negative inventory", invalid_inventory),
        ("Positive demand index", invalid_demand),
        ("Positive seasonality", invalid_seasonality),
        ("Valid Product_ID", unknown_market_products)
    ]

    for check, count in market_checks:

        add_result(
            report,
            "Market Conditions",
            check,
            "FAIL" if count else "PASS",
            "ERROR" if count else "INFO",
            f"Invalid records: {count}"
        )

    expected_market_rows = len(products) * len(dates)

    market_coverage = len(market) == expected_market_rows

    add_result(
        report,
        "Market Conditions",
        "Complete daily coverage",
        "PASS" if market_coverage else "FAIL",
        "ERROR" if market_coverage else "INFO",
        f"Expected: {expected_market_rows}, Actual: {len(market)}"
    )

    # --------------------------------------------------
    # 8. SAVE QUALITY REPORT
    # --------------------------------------------------

    report_df = pd.DataFrame(report)

    report_path = PROCESSED / "data_quality_report.csv"

    report_df.to_csv(report_path, index=False)

    print("\nDATA QUALITY REPORT")
    print("=" * 60)

    print(
        report_df.groupby(["Status", "Severity"])
        .size()
        .to_string()
    )

    errors = report_df[
        (report_df["Status"] == "FAIL") &
        (report_df["Severity"] == "ERROR")
    ]

    print("\nReport saved to:", report_path)

    if not errors.empty:
        print("\nCRITICAL DATA QUALITY ISSUES FOUND!")
        print(errors[["Dataset", "Check", "Details"]].to_string(index=False))
    else:
        print("\nALL CRITICAL VALIDATION CHECKS PASSED!")

    return report_df


def main():

    data = {}

    for key, filename in FILES.items():

        path = RAW / filename

        if not path.exists():
            raise FileNotFoundError(
                f"Missing file: {path}. Run Stage 2 first."
            )

        data[key] = pd.read_csv(path)

    report = validate_datasets(
        data["products"],
        data["sales"],
        data["competitors"],
        data["market"],
        data["dates"]
    )

    if (
        (report["Status"] == "FAIL") &
        (report["Severity"] == "ERROR")
    ).any():
        raise ValueError(
            "Data validation failed. Resolve the reported issues."
        )


if __name__ == "__main__":
    main()
