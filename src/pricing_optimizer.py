
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT_DIR / "data" / "processed"
ELASTICITY_DIR = ROOT_DIR / "reports" / "elasticity"
OUTPUT_DIR = ROOT_DIR / "reports" / "optimization"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LOOKBACK_DAYS = 30
PRICE_GRID_POINTS = 51
COMPETITIVE_UNDERCUT = 0.01


# ============================================================
# 2. LOAD INPUT DATA
# ============================================================

def load_data():

    products = pd.read_csv(
        DATA_DIR / "products_clean.csv"
    )

    panel = pd.read_csv(
        ELASTICITY_DIR / "product_day_modelling_data.csv",
        parse_dates=["Date"]
    )

    elasticity = pd.read_csv(
        ELASTICITY_DIR / "product_elasticity_estimates.csv"
    )

    required_product_columns = [
        "Product_ID",
        "Unit_Cost",
        "Base_Price",
        "Minimum_Margin_Percentage",
        "Maximum_Price_Change_Percentage",
    ]

    required_panel_columns = [
        "Product_ID",
        "Date",
        "Units_Sold",
        "Selling_Price",
        "Inventory_Level",
        "Demand_Index",
        "Average_Competitor_Price",
    ]

    for column in required_product_columns:
        if column not in products.columns:
            raise ValueError(
                f"Missing product column: {column}"
            )

    for column in required_panel_columns:
        if column not in panel.columns:
            raise ValueError(
                f"Missing product-day column: {column}"
            )

    if "Estimated_Price_Elasticity" not in elasticity.columns:
        raise ValueError(
            "Elasticity estimates file is missing its elasticity column."
        )

    return products, panel, elasticity


# ============================================================
# 3. PREPARE ELASTICITY ESTIMATES
# ============================================================

def prepare_elasticity(products, elasticity):

    valid = elasticity[
        elasticity["Estimated_Price_Elasticity"].between(
            -4.0, -0.05
        )
    ].copy()

    # Use category-level medians only as a fallback
    # when a product estimate is invalid or unavailable.

    category_fallback = (
        valid.groupby("Category")[
            "Estimated_Price_Elasticity"
        ].median().to_dict()
    )

    elasticity_lookup = elasticity.set_index(
        "Product_ID"
    )["Estimated_Price_Elasticity"].to_dict()

    records = []

    for product in products.itertuples(index=False):

        estimated = elasticity_lookup.get(
            product.Product_ID, np.nan
        )

        if pd.notna(estimated) and -4.0 <= estimated <= -0.05:
            chosen = float(estimated)
            source = "Product Model Estimate"

        else:
            chosen = category_fallback.get(
                product.Category, -1.2
            )
            source = "Category/Default Fallback"

        records.append({
            "Product_ID": product.Product_ID,
            "Model_Elasticity": chosen,
            "Elasticity_Source": source,
        })

    return pd.DataFrame(records)


# ============================================================
# 4. BUILD PRODUCT BASELINES
# ============================================================

def build_product_baselines(panel, products, elasticity_data):

    panel = panel.copy()

    panel["Date"] = pd.to_datetime(panel["Date"])

    latest_date = panel["Date"].max()

    cutoff_date = (
        latest_date - pd.Timedelta(days=LOOKBACK_DAYS - 1)
    )

    recent = panel[
        panel["Date"] >= cutoff_date
    ].copy()

    baselines = (
        recent.groupby("Product_ID", as_index=False)
        .agg(
            Baseline_Daily_Demand=(
                "Units_Sold", "mean"
            ),
            Baseline_Price=(
                "Selling_Price", "mean"
            ),
            Average_Inventory=(
                "Inventory_Level", "mean"
            ),
            Average_Demand_Index=(
                "Demand_Index", "mean"
            ),
            Average_Competitor_Price=(
                "Average_Competitor_Price", "mean"
            ),
            Recent_Units_Sold=(
                "Units_Sold", "sum"
            ),
        )
    )

    # Merge product costs and constraints.

    baselines = baselines.merge(
        products[
            [
                "Product_ID",
                "Category",
                "Unit_Cost",
                "Base_Price",
                "Minimum_Margin_Percentage",
                "Maximum_Price_Change_Percentage",
            ]
        ],
        on="Product_ID",
        how="left",
        validate="one_to_one"
    )

    baselines = baselines.merge(
        elasticity_data,
        on="Product_ID",
        how="left",
        validate="one_to_one"
    )

    # Calculate minimum allowable price from margin rule.

    baselines["Minimum_Allowed_Price"] = (
        baselines["Unit_Cost"]
        / (
            1 - baselines["Minimum_Margin_Percentage"]
        )
    )

    # Maximum price movement is measured from base price.

    baselines["Price_Floor"] = np.maximum(
        baselines["Minimum_Allowed_Price"],
        baselines["Base_Price"] * (
            1 - baselines["Maximum_Price_Change_Percentage"]
        )
    )

    baselines["Price_Ceiling"] = (
        baselines["Base_Price"]
        * (
            1 + baselines["Maximum_Price_Change_Percentage"]
        )
    )

    # Ensure baseline price is positive.

    baselines["Baseline_Price"] = (
        baselines["Baseline_Price"].clip(lower=0.01)
    )

    # If constraints conflict, no feasible price exists.

    baselines["Feasible"] = (
        baselines["Price_Floor"]
        <= baselines["Price_Ceiling"]
    )

    baselines.to_csv(
        OUTPUT_DIR / "product_pricing_baselines.csv",
        index=False
    )

    return baselines


# ============================================================
# 5. DEMAND AND PROFIT SIMULATION
# ============================================================

def simulate_price(
    price,
    baseline_demand,
    baseline_price,
    elasticity,
    unit_cost,
    inventory
):

    price_ratio = price / baseline_price

    expected_demand = (
        baseline_demand
        * (price_ratio ** elasticity)
    )

    expected_units = min(
        max(expected_demand, 0),
        max(inventory, 0)
    )

    expected_revenue = price * expected_units

    expected_profit = (
        price - unit_cost
    ) * expected_units

    margin = (
        expected_profit / expected_revenue * 100
        if expected_revenue > 0 else 0
    )

    return {
        "Recommended_Price": price,
        "Expected_Daily_Demand": expected_demand,
        "Expected_Daily_Units_Sold": expected_units,
        "Expected_Daily_Revenue": expected_revenue,
        "Expected_Daily_Profit": expected_profit,
        "Expected_Profit_Margin_Percentage": margin,
    }


# ============================================================
# 6. GENERATE THREE PRICING STRATEGIES
# ============================================================

def optimize_product(row):

    product_id = row["Product_ID"]

    if not row["Feasible"]:
        return []

    baseline_demand = max(
        float(row["Baseline_Daily_Demand"]), 0
    )

    baseline_price = max(
        float(row["Baseline_Price"]), 0.01
    )

    elasticity = float(row["Model_Elasticity"])

    unit_cost = float(row["Unit_Cost"])

    inventory = max(
        float(row["Average_Inventory"]), 0
    )

    competitor_price = float(
        row["Average_Competitor_Price"]
    )

    demand_index = float(
        row["Average_Demand_Index"]
    )

    price_floor = float(row["Price_Floor"])

    price_ceiling = float(row["Price_Ceiling"])

    # --------------------------------------------
    # Strategy 1: Competitive Pricing
    # --------------------------------------------

    competitive_target = (
        competitor_price * (1 - COMPETITIVE_UNDERCUT)
    )

    competitive_price = np.clip(
        competitive_target,
        price_floor,
        price_ceiling
    )

    # --------------------------------------------
    # Strategy 2: Demand-Based Pricing
    # --------------------------------------------

    if demand_index >= 1.10:
        demand_adjustment = 0.05

    elif demand_index <= 0.90:
        demand_adjustment = -0.05

    else:
        demand_adjustment = 0.00

    demand_target = (
        baseline_price * (1 + demand_adjustment)
    )

    demand_price = np.clip(
        demand_target,
        price_floor,
        price_ceiling
    )

    # --------------------------------------------
    # Strategy 3: Profit Maximization
    # --------------------------------------------

    candidate_prices = np.linspace(
        price_floor,
        price_ceiling,
        PRICE_GRID_POINTS
    )

    candidate_results = []

    for candidate_price in candidate_prices:

        result = simulate_price(
            candidate_price,
            baseline_demand,
            baseline_price,
            elasticity,
            unit_cost,
            inventory
        )

        candidate_results.append(result)

    candidate_df = pd.DataFrame(candidate_results)

    best_index = candidate_df[
        "Expected_Daily_Profit"
    ].idxmax()

    profit_max_price = candidate_df.loc[
        best_index, "Recommended_Price"
    ]

    strategy_prices = {
        "Competitive Pricing": competitive_price,
        "Demand-Based Pricing": demand_price,
        "Profit-Maximization Pricing": profit_max_price,
    }

    # --------------------------------------------
    # Evaluate each strategy
    # --------------------------------------------

    results = []

    baseline_result = simulate_price(
        baseline_price,
        baseline_demand,
        baseline_price,
        elasticity,
        unit_cost,
        inventory
    )

    for strategy, price in strategy_prices.items():

        metrics = simulate_price(
            float(price),
            baseline_demand,
            baseline_price,
            elasticity,
            unit_cost,
            inventory
        )

        revenue_change = (
            (
                metrics["Expected_Daily_Revenue"]
                / baseline_result["Expected_Daily_Revenue"]
                - 1
            ) * 100
            if baseline_result["Expected_Daily_Revenue"] > 0
            else 0
        )

        profit_change = (
            (
                metrics["Expected_Daily_Profit"]
                / baseline_result["Expected_Daily_Profit"]
                - 1
            ) * 100
            if baseline_result["Expected_Daily_Profit"] > 0
            else 0
        )

        results.append({
            "Product_ID": product_id,
            "Category": row["Category"],
            "Strategy": strategy,
            "Baseline_Price": baseline_price,
            "Average_Competitor_Price": competitor_price,
            "Unit_Cost": unit_cost,
            "Estimated_Elasticity": elasticity,
            "Elasticity_Source": row["Elasticity_Source"],
            "Price_Floor": price_floor,
            "Price_Ceiling": price_ceiling,
            **metrics,
            "Revenue_Change_Percentage": revenue_change,
            "Profit_Change_Percentage": profit_change,
        })

    return results


# ============================================================
# 7. BUILD FULL PRICE SCENARIO GRID
# ============================================================

def build_scenario_grid(baselines):

    records = []

    for row in baselines.itertuples(index=False):

        if not row.Feasible:
            continue

        candidate_prices = np.linspace(
            row.Price_Floor,
            row.Price_Ceiling,
            PRICE_GRID_POINTS
        )

        for price in candidate_prices:

            metrics = simulate_price(
                price,
                row.Baseline_Daily_Demand,
                row.Baseline_Price,
                row.Model_Elasticity,
                row.Unit_Cost,
                row.Average_Inventory
            )

            records.append({
                "Product_ID": row.Product_ID,
                "Category": row.Category,
                "Scenario_Price": price,
                "Estimated_Elasticity": row.Model_Elasticity,
                **{
                    key: value
                    for key, value in metrics.items()
                    if key != "Recommended_Price"
                }
            })

    scenario_df = pd.DataFrame(records)

    scenario_df.to_csv(
        OUTPUT_DIR / "price_scenario_grid.csv",
        index=False
    )

    return scenario_df


# ============================================================
# 8. RUN OPTIMIZATION FOR ALL PRODUCTS
# ============================================================

def run_optimization(baselines):

    all_results = []

    for _, row in baselines.iterrows():

        all_results.extend(
            optimize_product(row)
        )

    recommendations = pd.DataFrame(all_results)

    if recommendations.empty:
        raise ValueError(
            "No feasible pricing recommendations were generated."
        )

    recommendations.to_csv(
        OUTPUT_DIR / "pricing_recommendations.csv",
        index=False
    )

    return recommendations


# ============================================================
# 9. STRATEGY PERFORMANCE SUMMARY
# ============================================================

def generate_strategy_summary(recommendations):

    summary = (
        recommendations.groupby(
            "Strategy", as_index=False
        )
        .agg(
            Product_Count=("Product_ID", "nunique"),
            Average_Recommended_Price=(
                "Recommended_Price", "mean"
            ),
            Total_Expected_Daily_Revenue=(
                "Expected_Daily_Revenue", "sum"
            ),
            Total_Expected_Daily_Profit=(
                "Expected_Daily_Profit", "sum"
            ),
            Average_Profit_Margin=(
                "Expected_Profit_Margin_Percentage", "mean"
            ),
            Average_Profit_Change=(
                "Profit_Change_Percentage", "mean"
            ),
        )
    )

    summary.to_csv(
        OUTPUT_DIR / "strategy_performance_summary.csv",
        index=False
    )

    return summary


# ============================================================
# 10. MAIN EXECUTION
# ============================================================

def main():

    print("\n" + "=" * 65)
    print("STAGE 6: PRICING OPTIMIZATION ENGINE")
    print("=" * 65)

    print("\nLoading data...")

    products, panel, elasticity = load_data()

    print("Preparing elasticity estimates...")

    elasticity_data = prepare_elasticity(
        products,
        elasticity
    )

    print("Building product pricing baselines...")

    baselines = build_product_baselines(
        panel,
        products,
        elasticity_data
    )

    feasible_count = int(
        baselines["Feasible"].sum()
    )

    print(
        f"\nProducts with feasible price ranges: "
        f"{feasible_count}"
    )

    print("Generating pricing recommendations...")

    recommendations = run_optimization(
        baselines
    )

    print("Generating price scenario grid...")

    scenario_df = build_scenario_grid(
        baselines
    )

    print("Summarizing strategy performance...")

    summary = generate_strategy_summary(
        recommendations
    )

    print("\nPRICING STRATEGY SUMMARY")

    print(summary.to_string(index=False))

    print("\nSAMPLE RECOMMENDATIONS")

    print(
        recommendations.head(10).to_string(index=False)
    )

    print("\nSCENARIO GRID RECORDS:", len(scenario_df))

    print("\n" + "=" * 65)
    print("PRICING OPTIMIZATION COMPLETED")
    print("=" * 65)

    print(f"\nResults saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
