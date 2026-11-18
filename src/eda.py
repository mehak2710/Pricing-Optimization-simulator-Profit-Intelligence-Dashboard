
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# 1. PATH CONFIGURATION
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT_DIR / "data" / "processed"
REPORT_DIR = ROOT_DIR / "reports"
FIGURE_DIR = REPORT_DIR / "figures"
SUMMARY_DIR = REPORT_DIR / "summaries"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")


# ============================================================
# 2. LOAD PROCESSED DATA
# ============================================================

def load_data():

    products = pd.read_csv(
        DATA_DIR / "products_clean.csv"
    )

    sales = pd.read_csv(
        DATA_DIR / "sales_clean.csv"
    )

    competitors = pd.read_csv(
        DATA_DIR / "competitor_pricing_clean.csv"
    )

    market = pd.read_csv(
        DATA_DIR / "market_conditions_clean.csv"
    )

    dates = pd.read_csv(
        DATA_DIR / "date_dimension_clean.csv"
    )

    sales["Date"] = pd.to_datetime(sales["Date"])
    market["Date"] = pd.to_datetime(market["Date"])

    competitors["Observation_Date"] = pd.to_datetime(
        competitors["Observation_Date"]
    )

    dates["Date"] = pd.to_datetime(dates["Date"])

    return products, sales, competitors, market, dates


# ============================================================
# 3. PREPARE ANALYTICS DATASET
# ============================================================

def prepare_analysis_data(
    products, sales, competitors, market
):

    # Add product information to sales.

    df = sales.merge(
        products,
        on="Product_ID",
        how="left",
        validate="many_to_one"
    )

    # Add daily market information.
    # Market has one record per product and date.

    market_columns = [
        "Product_ID",
        "Date",
        "Demand_Index",
        "Seasonality",
        "Inventory_Level",
        "Marketing_Campaign",
        "Festival_Flag",
        "Promotion_Intensity",
    ]

    df = df.merge(
        market[market_columns],
        on=["Product_ID", "Date"],
        how="left",
        validate="many_to_one"
    )

    # Average competitor prices by product and date.
    # This avoids multiplying sales rows by three competitors.

    competitor_daily = (
        competitors.groupby(
            ["Product_ID", "Observation_Date"],
            as_index=False
        )
        .agg(
            Average_Competitor_Price=(
                "Competitor_Price", "mean"
            ),
            Lowest_Competitor_Price=(
                "Competitor_Price", "min"
            ),
            Highest_Competitor_Price=(
                "Competitor_Price", "max"
            ),
        )
        .rename(
            columns={"Observation_Date": "Date"}
        )
    )

    df = df.merge(
        competitor_daily,
        on=["Product_ID", "Date"],
        how="left",
        validate="many_to_one"
    )

    # Calculate competitor price difference
    # against the actual company selling price.

    df["Competitor_Price_Difference"] = (
        df["Selling_Price"]
        - df["Average_Competitor_Price"]
    )

    df["Competitor_Price_Difference_Percentage"] = (
        df["Competitor_Price_Difference"]
        / df["Average_Competitor_Price"]
    ) * 100

    # Recalculate analytical margin.

    df["Profit_Margin_Percentage"] = (
        df["Profit"] / df["Revenue"]
    ) * 100

    df["Year_Month"] = df["Date"].dt.to_period(
        "M"
    ).astype(str)

    return df


# ============================================================
# 4. GENERATE BUSINESS SUMMARY
# ============================================================

def generate_summary(df, products):

    total_revenue = df["Revenue"].sum()
    total_profit = df["Profit"].sum()
    total_units = df["Units_Sold"].sum()

    overall_margin = (
        total_profit / total_revenue * 100
        if total_revenue else 0
    )

    summary = pd.DataFrame({
        "Metric": [
            "Total Revenue",
            "Total Profit",
            "Total Units Sold",
            "Average Selling Price",
            "Overall Profit Margin",
            "Number of Products",
            "Number of Transactions",
        ],
        "Value": [
            total_revenue,
            total_profit,
            total_units,
            df["Revenue"].sum() / total_units
            if total_units else 0,
            overall_margin,
            products["Product_ID"].nunique(),
            df["Transaction_ID"].nunique(),
        ]
    })

    summary.to_csv(
        SUMMARY_DIR / "overall_business_summary.csv",
        index=False
    )

    print("\nOVERALL BUSINESS SUMMARY")
    print("=" * 55)
    print(summary.to_string(index=False))


# ============================================================
# 5. CATEGORY PERFORMANCE
# ============================================================

def category_analysis(df):

    category_summary = (
        df.groupby("Category", as_index=False)
        .agg(
            Revenue=("Revenue", "sum"),
            Profit=("Profit", "sum"),
            Units_Sold=("Units_Sold", "sum"),
            Average_Selling_Price=(
                "Selling_Price", "mean"
            ),
        )
    )

    category_summary["Profit_Margin_Percentage"] = (
        category_summary["Profit"]
        / category_summary["Revenue"]
    ) * 100

    category_summary.to_csv(
        SUMMARY_DIR / "category_performance.csv",
        index=False
    )

    print("\nCATEGORY PERFORMANCE")
    print(category_summary.to_string(index=False))

    # Revenue by category

    plt.figure(figsize=(10, 6))

    sns.barplot(
        data=category_summary,
        x="Category",
        y="Revenue"
    )

    plt.title("Revenue by Product Category")
    plt.xlabel("Category")
    plt.ylabel("Revenue (₹)")
    plt.xticks(rotation=15)

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / "revenue_by_category.png",
        dpi=300
    )

    plt.close()

    # Profit margin by category

    plt.figure(figsize=(10, 6))

    sns.barplot(
        data=category_summary,
        x="Category",
        y="Profit_Margin_Percentage"
    )

    plt.title("Profit Margin by Category")
    plt.xlabel("Category")
    plt.ylabel("Profit Margin (%)")
    plt.xticks(rotation=15)

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / "profit_margin_by_category.png",
        dpi=300
    )

    plt.close()


# ============================================================
# 6. MONTHLY REVENUE AND PROFIT TRENDS
# ============================================================

def monthly_trend_analysis(df):

    monthly = (
        df.groupby("Year_Month", as_index=False)
        .agg(
            Revenue=("Revenue", "sum"),
            Profit=("Profit", "sum"),
            Units_Sold=("Units_Sold", "sum"),
        )
    )

    monthly.to_csv(
        SUMMARY_DIR / "monthly_performance.csv",
        index=False
    )

    plt.figure(figsize=(14, 6))

    sns.lineplot(
        data=monthly,
        x="Year_Month",
        y="Revenue",
        marker="o",
        label="Revenue"
    )

    sns.lineplot(
        data=monthly,
        x="Year_Month",
        y="Profit",
        marker="o",
        label="Profit"
    )

    plt.title("Monthly Revenue and Profit Trend")
    plt.xlabel("Month")
    plt.ylabel("Amount (₹)")
    plt.xticks(rotation=60)

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / "monthly_revenue_profit.png",
        dpi=300
    )

    plt.close()


# ============================================================
# 7. PRICING AND DISCOUNT ANALYSIS
# ============================================================

def pricing_analysis(df):

    pricing_summary = (
        df.groupby("Category", as_index=False)
        .agg(
            Average_List_Price=("List_Price", "mean"),
            Average_Selling_Price=(
                "Selling_Price", "mean"
            ),
            Average_Discount=(
                "Discount_Percentage", "mean"
            ),
            Average_Profit_Margin=(
                "Profit_Margin_Percentage", "mean"
            ),
        )
    )

    pricing_summary["Average_Discount"] *= 100

    pricing_summary.to_csv(
        SUMMARY_DIR / "pricing_analysis.csv",
        index=False
    )

    plt.figure(figsize=(10, 6))

    sns.boxplot(
        data=df,
        x="Category",
        y="Selling_Price"
    )

    plt.title("Selling Price Distribution by Category")
    plt.xlabel("Category")
    plt.ylabel("Selling Price (₹)")
    plt.xticks(rotation=15)

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / "selling_price_distribution.png",
        dpi=300
    )

    plt.close()

    plt.figure(figsize=(10, 6))

    sns.scatterplot(
        data=df.sample(
            min(10000, len(df)),
            random_state=42
        ),
        x="Discount_Percentage",
        y="Profit_Margin_Percentage",
        hue="Category",
        alpha=0.6
    )

    plt.title("Discount vs Profit Margin")
    plt.xlabel("Discount Percentage (decimal)")
    plt.ylabel("Profit Margin (%)")

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / "discount_vs_profit_margin.png",
        dpi=300
    )

    plt.close()


# ============================================================
# 8. COMPETITOR PRICING ANALYSIS
# ============================================================

def competitor_analysis(df):

    competitor_summary = (
        df.groupby("Category", as_index=False)
        .agg(
            Average_Company_Price=(
                "Selling_Price", "mean"
            ),
            Average_Competitor_Price=(
                "Average_Competitor_Price", "mean"
            ),
            Lowest_Competitor_Price=(
                "Lowest_Competitor_Price", "mean"
            ),
            Highest_Competitor_Price=(
                "Highest_Competitor_Price", "mean"
            ),
        )
    )

    competitor_summary.to_csv(
        SUMMARY_DIR / "competitor_analysis.csv",
        index=False
    )

    plt.figure(figsize=(10, 6))

    sns.barplot(
        data=competitor_summary,
        x="Category",
        y="Average_Company_Price",
        color="steelblue",
        label="Company"
    )

    plt.title("Average Company Selling Price by Category")
    plt.xlabel("Category")
    plt.ylabel("Average Selling Price (₹)")

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / "company_average_price.png",
        dpi=300
    )

    plt.close()

    plt.figure(figsize=(10, 6))

    sns.barplot(
        data=competitor_summary,
        x="Category",
        y="Average_Competitor_Price"
    )

    plt.title("Average Competitor Price by Category")
    plt.xlabel("Category")
    plt.ylabel("Average Competitor Price (₹)")

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / "competitor_average_price.png",
        dpi=300
    )

    plt.close()


# ============================================================
# 9. MARKETING CAMPAIGN ANALYSIS
# ============================================================

def campaign_analysis(df):

    campaign_summary = (
        df.groupby("Marketing_Campaign", as_index=False)
        .agg(
            Revenue=("Revenue", "sum"),
            Profit=("Profit", "sum"),
            Units_Sold=("Units_Sold", "sum"),
            Average_Discount=(
                "Discount_Percentage", "mean"
            ),
        )
    )

    campaign_summary["Profit_Margin_Percentage"] = (
        campaign_summary["Profit"]
        / campaign_summary["Revenue"]
    ) * 100

    campaign_summary["Average_Discount"] *= 100

    campaign_summary.to_csv(
        SUMMARY_DIR / "campaign_performance.csv",
        index=False
    )

    plt.figure(figsize=(12, 6))

    sns.barplot(
        data=campaign_summary,
        x="Marketing_Campaign",
        y="Revenue"
    )

    plt.title("Revenue by Marketing Campaign")
    plt.xlabel("Marketing Campaign")
    plt.ylabel("Revenue (₹)")
    plt.xticks(rotation=20)

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / "campaign_revenue.png",
        dpi=300
    )

    plt.close()


# ============================================================
# 10. PRICE AND DEMAND RELATIONSHIP
# ============================================================

def price_demand_analysis(df):

    # Aggregate to product-day level so multiple
    # transactions do not count as separate demand observations.

    daily_product = (
        df.groupby(
            ["Product_ID", "Date"],
            as_index=False
        )
        .agg(
            Selling_Price=("Selling_Price", "mean"),
            Units_Sold=("Units_Sold", "sum"),
            Demand_Index=("Demand_Index", "first"),
            Category=("Category", "first"),
        )
    )

    daily_product.to_csv(
        SUMMARY_DIR / "daily_product_demand.csv",
        index=False
    )

    sample = daily_product.sample(
        min(10000, len(daily_product)),
        random_state=42
    )

    plt.figure(figsize=(10, 6))

    sns.scatterplot(
        data=sample,
        x="Selling_Price",
        y="Units_Sold",
        hue="Category",
        alpha=0.5
    )

    plt.title("Selling Price vs Units Sold")
    plt.xlabel("Selling Price (₹)")
    plt.ylabel("Units Sold per Product-Day")

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / "price_vs_demand.png",
        dpi=300
    )

    plt.close()

    correlation = daily_product[
        ["Selling_Price", "Units_Sold"]
    ].corr().iloc[0, 1]

    print(
        "\nOverall price-demand correlation:",
        round(correlation, 4)
    )


# ============================================================
# 11. CORRELATION ANALYSIS
# ============================================================

def correlation_analysis(df):

    numeric_columns = [
        "Selling_Price",
        "Units_Sold",
        "Discount_Percentage",
        "Revenue",
        "Profit",
        "Demand_Index",
        "Seasonality",
        "Inventory_Level",
        "Promotion_Intensity",
        "Average_Competitor_Price",
    ]

    correlation = df[numeric_columns].corr()

    correlation.to_csv(
        SUMMARY_DIR / "correlation_matrix.csv"
    )

    plt.figure(figsize=(12, 9))

    sns.heatmap(
        correlation,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        linewidths=0.5
    )

    plt.title("Pricing and Business Metrics Correlation")

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / "correlation_heatmap.png",
        dpi=300
    )

    plt.close()


# ============================================================
# 12. MAIN EXECUTION
# ============================================================

def main():

    print("\n" + "=" * 60)
    print("STAGE 4: EXPLORATORY DATA ANALYSIS")
    print("=" * 60)

    print("\nLoading processed datasets...")

    products, sales, competitors, market, dates = load_data()

    print("Preparing combined analytics dataset...")

    df = prepare_analysis_data(
        products,
        sales,
        competitors,
        market
    )

    print(f"Analytics records: {len(df):,}")

    print("\nRunning business analyses...")

    generate_summary(df, products)

    category_analysis(df)

    monthly_trend_analysis(df)

    pricing_analysis(df)

    competitor_analysis(df)

    campaign_analysis(df)

    price_demand_analysis(df)

    correlation_analysis(df)

    print("\n" + "=" * 60)
    print("EDA COMPLETED SUCCESSFULLY")
    print("=" * 60)

    print(f"\nCharts saved in: {FIGURE_DIR}")
    print(f"Summary files saved in: {SUMMARY_DIR}")


if __name__ == "__main__":
    main()
