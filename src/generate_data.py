
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT_DIR / "data" / "raw"

RAW_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)

START_DATE = "2025-01-01"
END_DATE = "2026-10-31"

DATES = pd.date_range(START_DATE, END_DATE, freq="D")


# ============================================================
# 2. CATEGORY CONFIGURATION
# ============================================================

CATEGORY_CONFIG = {
    "Electronics": {
        "price_range": (8000, 85000),
        "cost_ratio": (0.68, 0.82),
        "elasticity": (1.2, 2.2),
        "demand": (8, 25),
        "margin": (0.15, 0.28),
    },
    "Fashion": {
        "price_range": (500, 8000),
        "cost_ratio": (0.40, 0.65),
        "elasticity": (1.5, 2.8),
        "demand": (15, 45),
        "margin": (0.25, 0.45),
    },
    "Home Appliances": {
        "price_range": (3000, 45000),
        "cost_ratio": (0.65, 0.82),
        "elasticity": (1.0, 2.0),
        "demand": (6, 20),
        "margin": (0.16, 0.30),
    },
    "Beauty": {
        "price_range": (200, 5000),
        "cost_ratio": (0.35, 0.60),
        "elasticity": (1.4, 2.5),
        "demand": (12, 40),
        "margin": (0.25, 0.45),
    },
}


# ============================================================
# 3. CAMPAIGN CONFIGURATION
# ============================================================

CAMPAIGNS = {
    "No Campaign": {
        "demand_factor": 1.00,
        "discount_range": (0.00, 0.00),
    },
    "Performance Ads": {
        "demand_factor": 1.12,
        "discount_range": (0.02, 0.07),
    },
    "Festival Sale": {
        "demand_factor": 1.35,
        "discount_range": (0.08, 0.20),
    },
    "Flash Sale": {
        "demand_factor": 1.45,
        "discount_range": (0.10, 0.25),
    },
    "Influencer Campaign": {
        "demand_factor": 1.18,
        "discount_range": (0.03, 0.10),
    },
}

CAMPAIGN_NAMES = list(CAMPAIGNS.keys())

CAMPAIGN_PROBABILITIES = [
    0.72,  # No Campaign
    0.10,  # Performance Ads
    0.08,  # Festival Sale
    0.05,  # Flash Sale
    0.05,  # Influencer Campaign
]


# ============================================================
# 4. COMPETITOR CONFIGURATION
# ============================================================

COMPETITORS = {
    "MarketHub": 0.98,
    "QuickCart": 1.04,
    "ValueMart": 0.95,
}


# ============================================================
# 5. FESTIVAL CALENDAR
# ============================================================

FESTIVAL_MONTHS = [1, 3, 8, 10, 11]

# Festival periods are approximated for synthetic data.
# They are not intended to represent an exact holiday calendar.


# ============================================================
# 6. PRODUCT DATA GENERATION
# ============================================================

def generate_products():

    products = []

    product_counter = 1

    for category, config in CATEGORY_CONFIG.items():

        for i in range(25):

            base_price = round(
                rng.uniform(*config["price_range"]), 2
            )

            cost_ratio = rng.uniform(*config["cost_ratio"])

            unit_cost = round(base_price * cost_ratio, 2)

            product = {
                "Product_ID": f"P{product_counter:03d}",
                "Product_Name": f"{category.split()[0]} Product {i + 1:02d}",
                "Category": category,
                "Brand": f"Brand_{(i % 5) + 1}",
                "Unit_Cost": unit_cost,
                "Base_Price": base_price,
                "Minimum_Margin_Percentage": round(
                    rng.uniform(*config["margin"]), 4
                ),
                "Maximum_Price_Change_Percentage": round(
                    rng.uniform(0.10, 0.25), 4
                ),
                "Base_Daily_Demand": int(
                    rng.integers(
                        config["demand"][0],
                        config["demand"][1] + 1
                    )
                ),
                "Price_Elasticity": round(
                    rng.uniform(*config["elasticity"]), 4
                ),
            }

            products.append(product)

            product_counter += 1

    return pd.DataFrame(products)


# ============================================================
# 7. DATE DIMENSION GENERATION
# ============================================================

def generate_date_dimension():

    records = []

    for date in DATES:

        is_festival = int(date.month in FESTIVAL_MONTHS)

        financial_year_start = (
            date.year if date.month >= 4 else date.year - 1
        )

        financial_year = (
            f"{financial_year_start}-"
            f"{financial_year_start + 1}"
        )

        records.append({
            "Date": date.strftime("%Y-%m-%d"),
            "Day": date.day,
            "Day_Name": date.day_name(),
            "Week": int(date.isocalendar().week),
            "Month": date.month,
            "Month_Name": date.strftime("%B"),
            "Quarter": f"Q{date.quarter}",
            "Year": date.year,
            "Financial_Year": financial_year,
            "Is_Weekend": int(date.dayofweek >= 5),
            "Is_Festival_Season": is_festival,
        })

    return pd.DataFrame(records)


# ============================================================
# 8. SEASONALITY GENERATION
# ============================================================

def get_seasonality(date, category):

    month = date.month

    seasonal_factors = {
        "Electronics": {
            1: 0.95, 2: 0.95, 3: 1.00,
            4: 1.00, 5: 1.05, 6: 1.00,
            7: 0.98, 8: 1.05, 9: 1.10,
            10: 1.25, 11: 1.20, 12: 1.10,
        },
        "Fashion": {
            1: 0.95, 2: 0.95, 3: 1.00,
            4: 1.00, 5: 1.05, 6: 0.95,
            7: 1.00, 8: 1.05, 9: 1.10,
            10: 1.25, 11: 1.20, 12: 1.15,
        },
        "Home Appliances": {
            1: 0.95, 2: 0.95, 3: 1.00,
            4: 1.05, 5: 1.10, 6: 1.05,
            7: 1.00, 8: 1.00, 9: 1.05,
            10: 1.20, 11: 1.15, 12: 1.05,
        },
        "Beauty": {
            1: 1.00, 2: 1.00, 3: 1.05,
            4: 1.00, 5: 1.00, 6: 0.95,
            7: 1.00, 8: 1.05, 9: 1.10,
            10: 1.20, 11: 1.20, 12: 1.15,
        },
    }

    return seasonal_factors[category][month]


# ============================================================
# 9. MARKET CONDITIONS + COMPETITOR DATA
# ============================================================

def generate_market_and_competitor_data(products):

    market_records = []
    competitor_records = []

    for product in products.itertuples(index=False):

        product_id = product.Product_ID
        base_price = product.Base_Price
        category = product.Category
        base_demand = product.Base_Daily_Demand
        elasticity = product.Price_Elasticity

        unit_cost = product.Unit_Cost

        for date in DATES:

            date_string = date.strftime("%Y-%m-%d")

            # --------------------------------------------
            # Demand and seasonality
            # --------------------------------------------

            demand_index = float(
                np.clip(rng.normal(1.0, 0.16), 0.55, 1.55)
            )

            seasonality = get_seasonality(date, category)

            if date.dayofweek >= 5:
                weekday_factor = 1.08
            else:
                weekday_factor = 1.00

            # --------------------------------------------
            # Marketing campaign
            # --------------------------------------------

            campaign = rng.choice(
                CAMPAIGN_NAMES,
                p=CAMPAIGN_PROBABILITIES
            )

            # IMPORTANT:
            # No Campaign is an explicit category.
            # Marketing_Campaign will never be NaN.

            campaign_config = CAMPAIGNS[campaign]

            campaign_factor = campaign_config["demand_factor"]

            min_discount, max_discount = (
                campaign_config["discount_range"]
            )

            discount = float(
                rng.uniform(min_discount, max_discount)
            )

            # --------------------------------------------
            # Inventory
            # --------------------------------------------

            inventory = int(
                rng.integers(
                    max(10, base_demand),
                    max(20, base_demand * 3) + 1
                )
            )

            # --------------------------------------------
            # Competitor pricing
            # --------------------------------------------

            competitor_prices = []

            for competitor_name, multiplier in COMPETITORS.items():

                competitor_price = round(
                    base_price
                    * multiplier
                    * rng.uniform(0.97, 1.03),
                    2
                )

                competitor_price = max(0.01, competitor_price)

                competitor_prices.append(competitor_price)

                competitor_records.append({
                    "Product_ID": product_id,
                    "Competitor_Name": competitor_name,
                    "Competitor_Price": competitor_price,
                    "Observation_Date": date_string,
                })

            average_competitor_price = np.mean(
                competitor_prices
            )

            # If competitors are more expensive,
            # our product receives a small demand advantage.

            competitor_factor = np.clip(
                1 + (
                    (average_competitor_price - base_price)
                    / base_price
                ) * 0.5,
                0.90,
                1.10
            )

            # --------------------------------------------
            # Expected demand
            # --------------------------------------------

            selling_price = round(
                base_price * (1 - discount), 2
            )

            price_ratio = selling_price / base_price

            price_demand_factor = price_ratio ** (-elasticity)

            expected_demand = (
                base_demand
                * price_demand_factor
                * demand_index
                * seasonality
                * weekday_factor
                * campaign_factor
                * competitor_factor
            )

            expected_demand = max(0, expected_demand)

            units_demanded = int(
                rng.poisson(expected_demand)
            )

            units_sold = min(
                units_demanded,
                inventory
            )

            promotion_intensity = round(
                discount + rng.uniform(0, 0.04),
                4
            )

            festival_flag = int(
                date.month in FESTIVAL_MONTHS
            )

            # --------------------------------------------
            # Market conditions record
            # --------------------------------------------

            market_records.append({
                "Date": date_string,
                "Product_ID": product_id,
                "Demand_Index": round(demand_index, 4),
                "Seasonality": round(seasonality, 4),
                "Inventory_Level": inventory,
                "Marketing_Campaign": campaign,
                "Festival_Flag": festival_flag,
                "Day_Of_Week": date.day_name(),
                "Promotion_Intensity": promotion_intensity,
            })

            # --------------------------------------------
            # Sales transactions
            # --------------------------------------------

            if units_sold <= 0:
                continue

            # Split daily sales into one or two transactions.

            if units_sold >= 2 and rng.random() < 0.45:

                first_quantity = int(
                    rng.integers(1, units_sold)
                )

                quantities = [
                    first_quantity,
                    units_sold - first_quantity
                ]

            else:
                quantities = [units_sold]

            for quantity in quantities:

                revenue = round(
                    selling_price * quantity, 2
                )

                profit = round(
                    (selling_price - unit_cost) * quantity,
                    2
                )

                yield {
                    "Product_ID": product_id,
                    "Date": date_string,
                    "Selling_Price": selling_price,
                    "Units_Sold": quantity,
                    "Discount_Percentage": round(discount, 4),
                    "Revenue": revenue,
                    "Profit": profit,
                    "Unit_Cost_At_Sale": unit_cost,
                    "List_Price": base_price,
                    "Order_Channel": rng.choice(
                        ["Website", "Mobile App", "Marketplace"],
                        p=[0.35, 0.45, 0.20]
                    ),
                }


# ============================================================
# 10. SALES DATA GENERATION
# ============================================================

def generate_sales_data(products):

    sales_records = []

    for record in generate_market_and_competitor_data(products):
        sales_records.append(record)

    sales_df = pd.DataFrame(sales_records)

    if sales_df.empty:
        raise ValueError("Sales dataset is empty.")

    sales_df.insert(
        0,
        "Transaction_ID",
        [
            f"T{i:08d}"
            for i in range(1, len(sales_df) + 1)
        ]
    )

    return sales_df


# ============================================================
# 11. COMPLETE DATA GENERATION
# ============================================================

def main():

    print("\n" + "=" * 60)
    print("REAL-TIME PRICING OPTIMIZATION DATA GENERATOR")
    print("=" * 60)

    print("\nGenerating products...")

    products_df = generate_products()

    print("Generating date dimension...")

    dates_df = generate_date_dimension()

    print("Generating market, competitor and sales data...")

    # Generate the daily market and competitor records once.
    market_records = []
    competitor_records = []
    sales_records = []

    for product in products_df.itertuples(index=False):

        product_id = product.Product_ID
        base_price = product.Base_Price
        category = product.Category
        base_demand = product.Base_Daily_Demand
        elasticity = product.Price_Elasticity
        unit_cost = product.Unit_Cost

        for date in DATES:

            date_string = date.strftime("%Y-%m-%d")

            demand_index = float(
                np.clip(rng.normal(1.0, 0.16), 0.55, 1.55)
            )

            seasonality = get_seasonality(date, category)

            weekday_factor = (
                1.08 if date.dayofweek >= 5 else 1.00
            )

            campaign = rng.choice(
                CAMPAIGN_NAMES,
                p=CAMPAIGN_PROBABILITIES
            )

            campaign_config = CAMPAIGNS[campaign]

            min_discount, max_discount = (
                campaign_config["discount_range"]
            )

            discount = float(
                rng.uniform(min_discount, max_discount)
            )

            campaign_factor = campaign_config["demand_factor"]

            inventory = int(
                rng.integers(
                    max(10, base_demand),
                    max(20, base_demand * 3) + 1
                )
            )

            competitor_prices = []

            for competitor_name, multiplier in COMPETITORS.items():

                competitor_price = round(
                    base_price
                    * multiplier
                    * rng.uniform(0.97, 1.03),
                    2
                )

                competitor_price = max(0.01, competitor_price)

                competitor_prices.append(competitor_price)

                competitor_records.append({
                    "Product_ID": product_id,
                    "Competitor_Name": competitor_name,
                    "Competitor_Price": competitor_price,
                    "Observation_Date": date_string,
                })

            average_competitor_price = np.mean(
                competitor_prices
            )

            competitor_factor = np.clip(
                1 + (
                    (average_competitor_price - base_price)
                    / base_price
                ) * 0.5,
                0.90,
                1.10
            )

            selling_price = round(
                base_price * (1 - discount), 2
            )

            price_demand_factor = (
                selling_price / base_price
            ) ** (-elasticity)

            expected_demand = (
                base_demand
                * price_demand_factor
                * demand_index
                * seasonality
                * weekday_factor
                * campaign_factor
                * competitor_factor
            )

            units_demanded = int(
                rng.poisson(max(0, expected_demand))
            )

            units_sold = min(units_demanded, inventory)

            promotion_intensity = round(
                discount + rng.uniform(0, 0.04),
                4
            )

            festival_flag = int(
                date.month in FESTIVAL_MONTHS
            )

            # Explicitly store No Campaign instead of NaN.

            market_records.append({
                "Date": date_string,
                "Product_ID": product_id,
                "Demand_Index": round(demand_index, 4),
                "Seasonality": round(seasonality, 4),
                "Inventory_Level": inventory,
                "Marketing_Campaign": campaign,
                "Festival_Flag": festival_flag,
                "Day_Of_Week": date.day_name(),
                "Promotion_Intensity": promotion_intensity,
            })

            if units_sold == 0:
                continue

            if units_sold >= 2 and rng.random() < 0.45:

                first_quantity = int(
                    rng.integers(1, units_sold)
                )

                quantities = [
                    first_quantity,
                    units_sold - first_quantity
                ]

            else:
                quantities = [units_sold]

            for quantity in quantities:

                revenue = round(
                    selling_price * quantity, 2
                )

                profit = round(
                    (selling_price - unit_cost) * quantity,
                    2
                )

                sales_records.append({
                    "Product_ID": product_id,
                    "Date": date_string,
                    "Selling_Price": selling_price,
                    "Units_Sold": quantity,
                    "Discount_Percentage": round(discount, 4),
                    "Revenue": revenue,
                    "Profit": profit,
                    "Unit_Cost_At_Sale": unit_cost,
                    "List_Price": base_price,
                    "Order_Channel": rng.choice(
                        ["Website", "Mobile App", "Marketplace"],
                        p=[0.35, 0.45, 0.20]
                    ),
                })

    # Convert records to DataFrames.

    market_df = pd.DataFrame(market_records)

    competitor_df = pd.DataFrame(competitor_records)

    sales_df = pd.DataFrame(sales_records)

    if sales_df.empty:
        raise ValueError("No sales transactions were generated.")

    sales_df.insert(
        0,
        "Transaction_ID",
        [
            f"T{i:08d}"
            for i in range(1, len(sales_df) + 1)
        ]
    )

    # ========================================================
    # 12. FINAL DATA QUALITY SAFEGUARDS
    # ========================================================

    # Ensure no missing campaign values.

    market_df["Marketing_Campaign"] = (
        market_df["Marketing_Campaign"]
        .fillna("No Campaign")
    )

    # Ensure date fields are consistently formatted.

    sales_df["Date"] = pd.to_datetime(
        sales_df["Date"]
    ).dt.strftime("%Y-%m-%d")

    market_df["Date"] = pd.to_datetime(
        market_df["Date"]
    ).dt.strftime("%Y-%m-%d")

    competitor_df["Observation_Date"] = pd.to_datetime(
        competitor_df["Observation_Date"]
    ).dt.strftime("%Y-%m-%d")

    # Verify the expected data structure.

    assert len(products_df) == 100

    assert len(dates_df) == len(DATES)

    assert len(market_df) == len(products_df) * len(DATES)

    assert market_df["Marketing_Campaign"].notna().all()

    assert market_df["Product_ID"].notna().all()

    assert competitor_df["Competitor_Name"].nunique() == 3

    assert sales_df["Transaction_ID"].is_unique

    assert (sales_df["Selling_Price"] > 0).all()

    assert (sales_df["Units_Sold"] > 0).all()

    assert (sales_df["Revenue"] > 0).all()

    # ========================================================
    # 13. EXPORT CSV FILES
    # ========================================================

    products_df.to_csv(
        RAW_DIR / "products.csv",
        index=False
    )

    sales_df.to_csv(
        RAW_DIR / "sales.csv",
        index=False
    )

    competitor_df.to_csv(
        RAW_DIR / "competitor_pricing.csv",
        index=False
    )

    market_df.to_csv(
        RAW_DIR / "market_conditions.csv",
        index=False
    )

    dates_df.to_csv(
        RAW_DIR / "date_dimension.csv",
        index=False
    )

    # ========================================================
    # 14. GENERATION SUMMARY
    # ========================================================

    total_revenue = sales_df["Revenue"].sum()

    total_profit = sales_df["Profit"].sum()

    total_units = sales_df["Units_Sold"].sum()

    profit_margin = (
        total_profit / total_revenue * 100
        if total_revenue > 0 else 0
    )

    print("\n" + "=" * 60)
    print("DATA GENERATION COMPLETED")
    print("=" * 60)

    print(f"\nProducts: {len(products_df):,}")
    print(f"Sales transactions: {len(sales_df):,}")
    print(f"Competitor records: {len(competitor_df):,}")
    print(f"Market condition records: {len(market_df):,}")
    print(f"Date records: {len(dates_df):,}")

    print("\nMARKETING CAMPAIGN DISTRIBUTION")

    print(
        market_df["Marketing_Campaign"]
        .value_counts()
        .to_string()
    )

    print("\nFINANCIAL SUMMARY")

    print(f"Total Revenue: ₹{total_revenue:,.2f}")
    print(f"Total Profit: ₹{total_profit:,.2f}")
    print(f"Total Units Sold: {total_units:,}")
    print(f"Profit Margin: {profit_margin:.2f}%")

    print("\nMISSING VALUE CHECK")

    for name, df in {
        "Products": products_df,
        "Sales": sales_df,
        "Competitor Pricing": competitor_df,
        "Market Conditions": market_df,
        "Date Dimension": dates_df,
    }.items():

        print(
            f"{name}: {df.isnull().sum().sum()} missing cells"
        )

    print(f"\nFiles saved in: {RAW_DIR}")

    print("\nNext step: Run python src/data_validation.py")


# ============================================================
# 15. EXECUTION
# ============================================================

if __name__ == "__main__":
    main()
