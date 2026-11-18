
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_poisson_deviance


# ============================================================
# 1. PATHS AND CONFIGURATION
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT_DIR / "data" / "processed"
OUTPUT_DIR = ROOT_DIR / "reports" / "elasticity"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
MIN_OBSERVATIONS = 100


# ============================================================
# 2. LOAD DATA
# ============================================================

def load_data():

    products = pd.read_csv(
        DATA_DIR / "products_clean.csv"
    )

    sales = pd.read_csv(
        DATA_DIR / "sales_clean.csv"
    )

    market = pd.read_csv(
        DATA_DIR / "market_conditions_clean.csv"
    )

    competitors = pd.read_csv(
        DATA_DIR / "competitor_pricing_clean.csv"
    )

    for df in [sales, market]:
        df["Date"] = pd.to_datetime(df["Date"])

    competitors["Observation_Date"] = pd.to_datetime(
        competitors["Observation_Date"]
    )

    return products, sales, market, competitors


# ============================================================
# 3. CREATE PRODUCT-DAY ANALYTICS DATASET
# ============================================================

def create_product_day_dataset(
    products, sales, market, competitors
):

    # Aggregate transactions to product-day level.

    daily_sales = (
        sales.groupby(
            ["Product_ID", "Date"],
            as_index=False
        )
        .agg(
            Units_Sold=("Units_Sold", "sum"),
            Observed_Selling_Price=(
                "Selling_Price", "mean"
            )
        )
    )

    # Average competitor price for each product-day.

    daily_competitors = (
        competitors.groupby(
            ["Product_ID", "Observation_Date"],
            as_index=False
        )
        .agg(
            Average_Competitor_Price=(
                "Competitor_Price", "mean"
            )
        )
        .rename(
            columns={"Observation_Date": "Date"}
        )
    )

    # Start with market conditions so zero-sales days
    # are retained in the modelling dataset.

    panel = market.merge(
        products,
        on="Product_ID",
        how="left",
        validate="many_to_one"
    )

    panel = panel.merge(
        daily_sales,
        on=["Product_ID", "Date"],
        how="left",
        validate="one_to_one"
    )

    panel = panel.merge(
        daily_competitors,
        on=["Product_ID", "Date"],
        how="left",
        validate="one_to_one"
    )

    panel["Units_Sold"] = (
        panel["Units_Sold"].fillna(0)
    )

    # Observed price is available on days with sales.
    # On zero-sales days, use an approximate price based
    # on base price and promotion intensity.

    proxy_price = (
        panel["Base_Price"]
        * (1 - panel["Promotion_Intensity"])
    )

    panel["Selling_Price"] = (
        panel["Observed_Selling_Price"]
        .fillna(proxy_price)
    )

    panel["Selling_Price"] = (
        panel["Selling_Price"].clip(lower=0.01)
    )

    # Check that the model has the required fields.

    if panel["Average_Competitor_Price"].isna().any():
        raise ValueError(
            "Some product-day records have no competitor price."
        )

    if panel["Base_Price"].isna().any():
        raise ValueError(
            "Some products have no matching product information."
        )

    if panel["Units_Sold"].isna().any():
        raise ValueError(
            "Units_Sold contains missing values."
        )

    # Log features for the Poisson log-link model.

    panel["Log_Selling_Price"] = np.log(
        panel["Selling_Price"]
    )

    panel["Log_Demand_Index"] = np.log(
        panel["Demand_Index"].clip(lower=0.01)
    )

    panel["Log_Seasonality"] = np.log(
        panel["Seasonality"].clip(lower=0.01)
    )

    panel["Competitor_Price_Ratio"] = (
        panel["Selling_Price"]
        / panel["Average_Competitor_Price"]
    )

    panel["Is_Weekend"] = (
        pd.to_datetime(panel["Date"]).dt.dayofweek >= 5
    ).astype(int)

    panel["Festival_Flag"] = (
        panel["Festival_Flag"].astype(int)
    )

    # Save the complete product-day panel.

    panel.to_csv(
        OUTPUT_DIR / "product_day_modelling_data.csv",
        index=False
    )

    return panel


# ============================================================
# 4. MODEL FEATURES
# ============================================================

def prepare_features(df):

    campaign_features = pd.get_dummies(
        df["Marketing_Campaign"],
        prefix="Campaign",
        dtype=int
    )

    features = pd.concat(
        [
            df[
                [
                    "Log_Selling_Price",
                    "Log_Demand_Index",
                    "Log_Seasonality",
                    "Competitor_Price_Ratio",
                    "Promotion_Intensity",
                    "Is_Weekend",
                    "Festival_Flag",
                ]
            ].reset_index(drop=True),

            campaign_features.reset_index(drop=True),
        ],
        axis=1
    )

    return features.astype(float)


# ============================================================
# 5. ELASTICITY CLASSIFICATION
# ============================================================

def classify_elasticity(value):

    if value < -1.05:
        return "Elastic"

    elif value > -0.95:
        return "Inelastic"

    else:
        return "Approximately Unit Elastic"


# ============================================================
# 6. PRODUCT-LEVEL ELASTICITY MODELLING
# ============================================================

def estimate_product_elasticity(panel):

    results = []
    performance = []

    for product_id, product_df in panel.groupby(
        "Product_ID"
    ):

        product_df = (
            product_df.sort_values("Date")
            .reset_index(drop=True)
        )

        if len(product_df) < MIN_OBSERVATIONS:
            continue

        # Use chronological split to avoid future data leakage.

        split_index = int(len(product_df) * 0.80)

        train = product_df.iloc[:split_index]
        test = product_df.iloc[split_index:]

        X = prepare_features(product_df)

        X_train = X.iloc[:split_index]
        X_test = X.iloc[split_index:]

        y_train = train["Units_Sold"].astype(float)
        y_test = test["Units_Sold"].astype(float)

        # A Poisson model supports zero-sales observations.
        # With a log link, the coefficient on log price
        # estimates the price elasticity parameter.

        model = PoissonRegressor(
            alpha=0.1,
            max_iter=1000,
            tol=1e-7
        )

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        predictions = np.clip(
            predictions, 1e-9, None
        )

        price_index = X.columns.get_loc(
            "Log_Selling_Price"
        )

        elasticity = float(
            model.coef_[price_index]
        )

        # Assess price variation.

        price_variation = (
            product_df["Selling_Price"].std()
            / product_df["Selling_Price"].mean()
        )

        # A very small price variation makes elasticity
        # difficult to identify reliably.

        if price_variation < 0.005:
            reliability = "Low - Limited Price Variation"
        elif len(product_df) >= 300:
            reliability = "Moderate - Observational Estimate"
        else:
            reliability = "Low - Limited Observations"

        results.append({
            "Product_ID": product_id,
            "Category": product_df["Category"].iloc[0],
            "Estimated_Price_Elasticity": round(
                elasticity, 4
            ),
            "Elasticity_Segment": classify_elasticity(
                elasticity
            ),
            "Observation_Count": len(product_df),
            "Price_Variation_CV": round(
                price_variation, 4
            ),
            "Reliability": reliability,
        })

        deviance = mean_poisson_deviance(
            y_test, predictions
        )

        performance.append({
            "Product_ID": product_id,
            "Test_Observations": len(test),
            "Mean_Poisson_Deviance": round(
                deviance, 4
            ),
            "Mean_Actual_Demand": round(
                y_test.mean(), 4
            ),
            "Mean_Predicted_Demand": round(
                predictions.mean(), 4
            ),
        })

    elasticity_df = pd.DataFrame(results)

    performance_df = pd.DataFrame(performance)

    if elasticity_df.empty:
        raise ValueError(
            "No products had enough observations for modelling."
        )

    elasticity_df.to_csv(
        OUTPUT_DIR / "product_elasticity_estimates.csv",
        index=False
    )

    performance_df.to_csv(
        OUTPUT_DIR / "product_model_performance.csv",
        index=False
    )

    return elasticity_df, performance_df


# ============================================================
# 7. CATEGORY-LEVEL ELASTICITY SUMMARY
# ============================================================

def category_elasticity_summary(elasticity_df):

    category_summary = (
        elasticity_df.groupby(
            "Category", as_index=False
        )
        .agg(
            Average_Elasticity=(
                "Estimated_Price_Elasticity", "mean"
            ),
            Median_Elasticity=(
                "Estimated_Price_Elasticity", "median"
            ),
            Product_Count=(
                "Product_ID", "nunique"
            ),
        )
    )

    category_summary["Demand_Sensitivity"] = (
        category_summary["Average_Elasticity"]
        .apply(classify_elasticity)
    )

    category_summary.to_csv(
        OUTPUT_DIR / "category_elasticity_summary.csv",
        index=False
    )

    return category_summary


# ============================================================
# 8. MAIN EXECUTION
# ============================================================

def main():

    print("\n" + "=" * 60)
    print("STAGE 5: PRICE ELASTICITY MODELLING")
    print("=" * 60)

    print("\nLoading processed datasets...")

    products, sales, market, competitors = load_data()

    print("Creating product-day modelling dataset...")

    panel = create_product_day_dataset(
        products,
        sales,
        market,
        competitors
    )

    print(
        f"Product-day observations: {len(panel):,}"
    )

    print(
        f"Products: {panel['Product_ID'].nunique():,}"
    )

    print(
        f"Zero-sales product-days: "
        f"{(panel['Units_Sold'] == 0).sum():,}"
    )

    print("\nEstimating product-level elasticity...")

    elasticity_df, performance_df = (
        estimate_product_elasticity(panel)
    )

    print("\nGenerating category-level summary...")

    category_summary = category_elasticity_summary(
        elasticity_df
    )

    print("\nPRODUCT ELASTICITY RESULTS")

    print(
        elasticity_df.head(10).to_string(index=False)
    )

    print("\nCATEGORY ELASTICITY SUMMARY")

    print(
        category_summary.to_string(index=False)
    )

    print("\nMODEL PERFORMANCE SUMMARY")

    print(
        performance_df[
            ["Mean_Poisson_Deviance"]
        ].describe().round(4)
    )

    print("\n" + "=" * 60)
    print("PRICE ELASTICITY MODELLING COMPLETED")
    print("=" * 60)

    print(f"\nResults saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
