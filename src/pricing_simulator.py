
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data" / "processed"
OPT_DIR = ROOT / "reports" / "optimization"
ELASTICITY_DIR = ROOT / "reports" / "elasticity"

st.set_page_config(
    page_title="ShopSphere | Pricing Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# 2. CUSTOM DESIGN
# ============================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background-color: #F5F7FB;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1450px;
}

[data-testid="stSidebar"] {
    background-color: #14243A;
}

[data-testid="stSidebar"] * {
    color: #F3F6FA !important;
}

[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background-color: #263B55;
    border: 1px solid #53677E;
}

[data-testid="stSidebar"] label {
    color: #D5DFEA !important;
}

h1, h2, h3 {
    color: #14243A;
    font-weight: 700;
}

.main-header {
    background: linear-gradient(115deg, #14243A, #234B68);
    padding: 30px 34px;
    border-radius: 16px;
    margin-bottom: 25px;
    color: white;
}

.main-header h1 {
    color: white;
    font-size: 31px;
    margin: 0;
    letter-spacing: -0.8px;
}

.main-header p {
    color: #C7D6E5;
    font-size: 14px;
    margin: 9px 0 0 0;
}

.header-tag {
    display: inline-block;
    background: rgba(255,255,255,0.13);
    color: #DCEAF5;
    padding: 5px 12px;
    border-radius: 20px;
    font-size: 11px;
    letter-spacing: 1px;
    margin-bottom: 13px;
}

.section-heading {
    color: #14243A;
    font-size: 21px;
    font-weight: 700;
    margin-top: 22px;
    margin-bottom: 5px;
}

.section-subtitle {
    color: #78879A;
    font-size: 13px;
    margin-bottom: 17px;
}

.kpi-card {
    background: white;
    border: 1px solid #E6EBF2;
    border-radius: 13px;
    padding: 18px 19px;
    min-height: 115px;
    box-shadow: 0 3px 12px rgba(20,36,58,0.035);
}

.kpi-label {
    color: #7A899B;
    font-size: 12px;
    font-weight: 600;
    margin-bottom: 10px;
}

.kpi-value {
    color: #14243A;
    font-size: 25px;
    font-weight: 800;
    letter-spacing: -0.5px;
}

.kpi-note {
    color: #8492A3;
    font-size: 11px;
    margin-top: 6px;
}

.panel {
    background: white;
    border: 1px solid #E6EBF2;
    border-radius: 14px;
    padding: 20px 22px;
    box-shadow: 0 3px 12px rgba(20,36,58,0.035);
}

.simulator-panel {
    background: white;
    border: 1px solid #DCE5EF;
    border-radius: 16px;
    padding: 24px;
    box-shadow: 0 5px 20px rgba(20,36,58,0.045);
}

.price-display {
    background: #14243A;
    color: white;
    padding: 22px;
    border-radius: 13px;
    text-align: center;
}

.price-display .small {
    color: #B9C9D9;
    font-size: 12px;
}

.price-display .amount {
    color: white;
    font-size: 34px;
    font-weight: 800;
    margin-top: 5px;
}

.insight-box {
    background: #EAF4F3;
    border-left: 4px solid #25A18E;
    border-radius: 8px;
    padding: 16px 18px;
    color: #244B4A;
    font-size: 13px;
    line-height: 1.6;
}

.warning-box {
    background: #FFF5E7;
    border-left: 4px solid #E5A33B;
    border-radius: 8px;
    padding: 14px 16px;
    color: #75501D;
    font-size: 13px;
}

.footer {
    color: #8794A5;
    text-align: center;
    font-size: 11px;
    padding-top: 18px;
}

div[data-testid="stDownloadButton"] button {
    background: #14243A;
    color: white;
    border: none;
    border-radius: 8px;
    padding: 10px 18px;
}

div[data-testid="stDownloadButton"] button:hover {
    background: #2878B5;
    color: white;
}

.stButton button {
    border-radius: 8px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# 3. HELPER FUNCTIONS
# ============================================================

def currency(value):
    return f"₹{value:,.2f}"


def make_kpi(label, value, note=""):
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def section(title, subtitle=None):
    st.markdown(
        f'<div class="section-heading">{title}</div>',
        unsafe_allow_html=True
    )

    if subtitle:
        st.markdown(
            f'<div class="section-subtitle">{subtitle}</div>',
            unsafe_allow_html=True
        )


# ============================================================
# 4. LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    products = pd.read_csv(
        DATA_DIR / "products_clean.csv"
    )

    baselines = pd.read_csv(
        OPT_DIR / "product_pricing_baselines.csv"
    )

    recommendations = pd.read_csv(
        OPT_DIR / "pricing_recommendations.csv"
    )

    elasticity = pd.read_csv(
        ELASTICITY_DIR / "product_elasticity_estimates.csv"
    )

    # Select only required baseline columns

    baseline_columns = [
        "Product_ID",
        "Baseline_Daily_Demand",
        "Baseline_Price",
        "Average_Inventory",
        "Average_Demand_Index",
        "Average_Competitor_Price",
        "Price_Floor",
        "Price_Ceiling",
        "Model_Elasticity"
    ]

    baselines = baselines[baseline_columns]

    # Convert long-format strategy records to wide format

    def normalize_strategy(strategy):

        strategy = str(strategy).strip().lower()

        if "competitive" in strategy:
            return "Competitive_Price"

        if "demand" in strategy:
            return "Demand_Based_Price"

        if "profit" in strategy:
            return "Profit_Maximizing_Price"

        return None

    recommendations["Strategy_Column"] = (
        recommendations["Strategy"].apply(
            normalize_strategy
        )
    )

    recommendations = recommendations.dropna(
        subset=["Strategy_Column"]
    )

    recommendation_pivot = recommendations.pivot_table(
        index="Product_ID",
        columns="Strategy_Column",
        values="Recommended_Price",
        aggfunc="first"
    ).reset_index()

    recommendation_pivot.columns.name = None

    # Merge datasets

    data = products.merge(
        baselines,
        on="Product_ID",
        how="inner"
    )

    data = data.merge(
        recommendation_pivot,
        on="Product_ID",
        how="left"
    )

    data = data.merge(
        elasticity[
            ["Product_ID", "Estimated_Price_Elasticity"]
        ],
        on="Product_ID",
        how="left"
    )

    # Elasticity fallback

    data["Estimated_Price_Elasticity"] = (
        data["Estimated_Price_Elasticity"]
        .fillna(data["Model_Elasticity"])
    )

    numeric_columns = [
        "Unit_Cost",
        "Base_Price",
        "Minimum_Margin_Percentage",
        "Maximum_Price_Change_Percentage",
        "Baseline_Daily_Demand",
        "Baseline_Price",
        "Average_Inventory",
        "Average_Demand_Index",
        "Average_Competitor_Price",
        "Price_Floor",
        "Price_Ceiling",
        "Estimated_Price_Elasticity",
        "Competitive_Price",
        "Demand_Based_Price",
        "Profit_Maximizing_Price"
    ]

    for column in numeric_columns:

        data[column] = pd.to_numeric(
            data[column],
            errors="coerce"
        )

    return data


# ============================================================
# 5. LOAD AND VALIDATE
# ============================================================

try:

    data = load_data()

    data = data.replace(
        [np.inf, -np.inf],
        np.nan
    )

    required_columns = [
        "Product_ID",
        "Product_Name",
        "Category",
        "Unit_Cost",
        "Base_Price",
        "Minimum_Margin_Percentage",
        "Maximum_Price_Change_Percentage",
        "Baseline_Daily_Demand",
        "Baseline_Price",
        "Average_Inventory",
        "Average_Demand_Index",
        "Average_Competitor_Price",
        "Price_Floor",
        "Price_Ceiling",
        "Estimated_Price_Elasticity",
        "Competitive_Price",
        "Demand_Based_Price",
        "Profit_Maximizing_Price"
    ]

    missing = [
        col for col in required_columns
        if col not in data.columns
    ]

    if missing:
        st.error(f"Missing columns: {missing}")
        st.stop()

    data = data.dropna(
        subset=required_columns
    ).drop_duplicates(
        subset=["Product_ID"]
    )

    if data.empty:
        st.error("No valid products found.")
        st.stop()

except FileNotFoundError as error:

    st.error(f"Dataset not found: {error}")
    st.info("Run Stage 5 and Stage 6 before launching the simulator.")
    st.stop()

except Exception as error:

    st.error(f"Unable to load project data: {error}")
    st.stop()


# ============================================================
# 6. SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div style="padding:10px 0 25px 0;">
            <div style="font-size:25px;font-weight:800;color:white;">
                ShopSphere
            </div>
            <div style="font-size:11px;color:#AFC1D3;letter-spacing:1px;">
                PRICING INTELLIGENCE
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### Simulation Controls")

    st.markdown("**Product Selection**")

    categories = sorted(
        data["Category"].dropna().unique()
    )

    selected_category = st.selectbox(
        "Category",
        categories,
        label_visibility="collapsed"
    )

    category_data = data[
        data["Category"] == selected_category
    ].copy()

    category_data["Product_Display"] = (
        category_data["Product_Name"].astype(str)
        + " ("
        + category_data["Product_ID"].astype(str)
        + ")"
    )

    selected_product_display = st.selectbox(
        "Product",
        category_data["Product_Display"].tolist()
    )

    product = category_data[
        category_data["Product_Display"]
        == selected_product_display
    ].iloc[0]

    st.divider()

    st.markdown("**Product Details**")

    st.caption(f"Product ID: {product['Product_ID']}")

    st.caption(f"Brand: {product['Brand']}")

    st.caption(f"Category: {product['Category']}")

    st.divider()

    st.markdown(
        """
        <div style="font-size:11px;color:#AFC1D3;line-height:1.7;">
        MODEL-BASED SIMULATION<br>
        Uses historical demand, price elasticity,
        inventory and pricing strategy estimates.
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# 7. HEADER
# ============================================================

st.markdown(
    """
    <div class="main-header">
        <div class="header-tag">PRICING & PROFIT INTELLIGENCE</div>
        <h1>Pricing Optimization Simulator</h1>
        <p>
            Explore pricing scenarios, understand demand sensitivity
            and evaluate their estimated financial impact.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# 8. PRODUCT PARAMETERS
# ============================================================

price_floor = float(product["Price_Floor"])
price_ceiling = float(product["Price_Ceiling"])
baseline_price = float(product["Baseline_Price"])
unit_cost = float(product["Unit_Cost"])

baseline_demand = max(
    float(product["Baseline_Daily_Demand"]),
    0
)

inventory = max(
    float(product["Average_Inventory"]),
    0
)

elasticity_value = float(
    product["Estimated_Price_Elasticity"]
)

minimum_margin = float(
    product["Minimum_Margin_Percentage"]
)

if (
    not np.isfinite(price_floor)
    or not np.isfinite(price_ceiling)
    or price_floor <= 0
    or price_floor > price_ceiling
):
    st.error("This product has an invalid feasible price range.")
    st.stop()

if baseline_price <= 0:
    st.error("Baseline price must be positive.")
    st.stop()


# ============================================================
# 9. PRODUCT OVERVIEW
# ============================================================

section(
    "Product Overview",
    "Current product economics and market reference points."
)

k1, k2, k3, k4 = st.columns(4)

with k1:
    make_kpi(
        "BASE PRICE",
        currency(product["Base_Price"]),
        "Original product list price"
    )

with k2:
    make_kpi(
        "UNIT COST",
        currency(unit_cost),
        "Estimated product cost"
    )

with k3:
    make_kpi(
        "PRICE ELASTICITY",
        f"{elasticity_value:.2f}",
        "Estimated demand sensitivity"
    )

with k4:
    make_kpi(
        "COMPETITOR AVERAGE",
        currency(product["Average_Competitor_Price"]),
        "Average observed competitor price"
    )


# ============================================================
# 10. DEMAND AND FINANCIAL CALCULATIONS
# ============================================================

def calculate_scenario(price):

    price = float(price)

    if price <= 0:
        demand = 0

    else:
        demand = (
            baseline_demand
            * (price / baseline_price) ** elasticity_value
        )

    demand = min(
        max(demand, 0),
        inventory
    )

    revenue = price * demand

    profit = (
        (price - unit_cost) * demand
    )

    margin = (
        profit / revenue * 100
        if revenue > 0
        else 0
    )

    return {
        "Price": price,
        "Expected Demand": demand,
        "Revenue": revenue,
        "Profit": profit,
        "Profit Margin (%)": margin
    }


# ============================================================
# 11. INTERACTIVE PRICE SIMULATOR
# ============================================================

st.markdown("<br>", unsafe_allow_html=True)

section(
    "Interactive Price Simulator",
    "Change the selling price and immediately evaluate the estimated impact."
)

with st.container(border=True):

    left, right = st.columns([1.7, 1])

    with left:

        default_price = float(
            np.clip(
                product["Profit_Maximizing_Price"],
                price_floor,
                price_ceiling
            )
        )

        selected_price = st.slider(
            "Adjust Selling Price (₹)",
            min_value=float(price_floor),
            max_value=float(price_ceiling),
            value=default_price,
            step=1.0,
            help="The slider is restricted to the feasible price range calculated in Stage 6."
        )

        st.caption(
            f"Allowed range: {currency(price_floor)} to "
            f"{currency(price_ceiling)}"
        )

        st.caption(
            f"Historical baseline price: {currency(baseline_price)}"
        )

    with right:

        st.markdown(
            f"""
            <div class="price-display">
                <div class="small">YOUR SELECTED SELLING PRICE</div>
                <div class="amount">{currency(selected_price)}</div>
                <div class="small">Per unit</div>
            </div>
            """,
            unsafe_allow_html=True
        )

selected_result = calculate_scenario(
    selected_price
)

expected_demand = selected_result["Expected Demand"]
expected_revenue = selected_result["Revenue"]
expected_profit = selected_result["Profit"]
profit_margin = selected_result["Profit Margin (%)"]


# ============================================================
# 12. FINANCIAL IMPACT
# ============================================================

section(
    "Simulated Business Impact",
    "Estimated daily performance at your selected price."
)

m1, m2, m3, m4 = st.columns(4)

with m1:
    make_kpi(
        "EXPECTED DAILY DEMAND",
        f"{expected_demand:,.1f}",
        "Units per day"
    )

with m2:
    make_kpi(
        "EXPECTED DAILY REVENUE",
        currency(expected_revenue),
        "Estimated sales value"
    )

with m3:
    make_kpi(
        "EXPECTED DAILY PROFIT",
        currency(expected_profit),
        "Revenue less product cost"
    )

with m4:
    make_kpi(
        "PROFIT MARGIN",
        f"{profit_margin:.2f}%",
        "Estimated gross profit margin"
    )

minimum_allowed_price = (
    unit_cost / (1 - minimum_margin)
    if 0 <= minimum_margin < 1
    else float("inf")
)

margin_valid = (
    selected_price >= minimum_allowed_price
    and profit_margin >= minimum_margin * 100
)

if margin_valid:

    st.markdown(
        """
        <div class="insight-box">
            <b>Pricing constraint satisfied.</b><br>
            Your selected price meets the configured minimum
            gross margin requirement.
        </div>
        """,
        unsafe_allow_html=True
    )

else:

    st.markdown(
        """
        <div class="warning-box">
            <b>Margin constraint warning.</b><br>
            This scenario does not meet the configured
            minimum gross margin requirement.
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# 13. STRATEGY COMPARISON
# ============================================================

st.markdown("<br>", unsafe_allow_html=True)

section(
    "Pricing Strategy Comparison",
    "Compare the estimated financial outcomes of the three strategies against your selected price."
)

strategy_prices = {

    "Competitive Pricing": product["Competitive_Price"],

    "Demand-Based Pricing": product["Demand_Based_Price"],

    "Profit-Maximization Pricing": product["Profit_Maximizing_Price"],

    "Your Selected Price": selected_price

}

scenario_results = []

for strategy, price in strategy_prices.items():

    price = float(
        np.clip(
            price,
            price_floor,
            price_ceiling
        )
    )

    result = calculate_scenario(price)

    result["Strategy"] = strategy

    scenario_results.append(result)

scenario_df = pd.DataFrame(
    scenario_results
)

scenario_df = scenario_df[
    [
        "Strategy",
        "Price",
        "Expected Demand",
        "Revenue",
        "Profit",
        "Profit Margin (%)"
    ]
]

chart1, chart2 = st.columns(2)

with chart1:

    fig_profit = px.bar(
        scenario_df,
        x="Strategy",
        y="Profit",
        color="Strategy",
        text_auto=".2s",
        title="Expected Daily Profit"
    )

    fig_profit.update_layout(
        template="plotly_white",
        showlegend=False,
        height=370,
        margin=dict(l=10, r=10, t=55, b=10),
        xaxis_title="",
        yaxis_title="Profit (₹)"
    )

    fig_profit.update_traces(
        marker_line_width=0
    )

    st.plotly_chart(
        fig_profit,
        use_container_width=True
    )

with chart2:

    fig_revenue = px.bar(
        scenario_df,
        x="Strategy",
        y="Revenue",
        color="Strategy",
        text_auto=".2s",
        title="Expected Daily Revenue"
    )

    fig_revenue.update_layout(
        template="plotly_white",
        showlegend=False,
        height=370,
        margin=dict(l=10, r=10, t=55, b=10),
        xaxis_title="",
        yaxis_title="Revenue (₹)"
    )

    fig_revenue.update_traces(
        marker_line_width=0
    )

    st.plotly_chart(
        fig_revenue,
        use_container_width=True
    )


# ============================================================
# 14. SCENARIO COMPARISON TABLE
# ============================================================

section(
    "Scenario Performance",
    "Detailed comparison of the pricing alternatives."
)

display_df = scenario_df.copy()

display_df["Price"] = display_df["Price"].map(
    currency
)

display_df["Revenue"] = display_df["Revenue"].map(
    currency
)

display_df["Profit"] = display_df["Profit"].map(
    currency
)

display_df["Expected Demand"] = (
    display_df["Expected Demand"].map(
        lambda x: f"{x:,.1f}"
    )
)

display_df["Profit Margin (%)"] = (
    display_df["Profit Margin (%)"].map(
        lambda x: f"{x:.2f}%"
    )
)

st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# 15. PRICE VS DEMAND CURVE
# ============================================================

st.markdown("<br>", unsafe_allow_html=True)

section(
    "Price vs Demand Analysis",
    "Explore the estimated relationship between selling price and daily demand."
)

price_range = np.linspace(
    price_floor,
    price_ceiling,
    60
)

demand_curve = (
    baseline_demand
    * (price_range / baseline_price) ** elasticity_value
)

demand_curve = np.minimum(
    np.maximum(demand_curve, 0),
    inventory
)

curve_df = pd.DataFrame({
    "Selling Price": price_range,
    "Expected Demand": demand_curve
})

fig_demand = px.line(
    curve_df,
    x="Selling Price",
    y="Expected Demand",
    title="Estimated Price-Demand Curve"
)

fig_demand.add_trace(
    go.Scatter(
        x=[selected_price],
        y=[expected_demand],
        mode="markers",
        name="Selected Price",
        marker=dict(
            size=13,
            color="#E5A33B",
            line=dict(
                width=2,
                color="white"
            )
        )
    )
)

fig_demand.update_layout(
    template="plotly_white",
    height=420,
    hovermode="x unified",
    xaxis_title="Selling Price (₹)",
    yaxis_title="Expected Daily Demand (Units)",
    margin=dict(l=10, r=10, t=60, b=10)
)

st.plotly_chart(
    fig_demand,
    use_container_width=True
)


# ============================================================
# 16. PRICE AND PROFIT CHANGE
# ============================================================

section(
    "Impact of Your Pricing Decision",
    "Compare the selected scenario with the historical baseline."
)

price_change = (
    (selected_price - baseline_price)
    / baseline_price * 100
)

baseline_profit = (
    (baseline_price - unit_cost)
    * min(baseline_demand, inventory)
)

profit_change = (
    expected_profit - baseline_profit
)

demand_change = (
    expected_demand - baseline_demand
)

i1, i2, i3 = st.columns(3)

with i1:
    make_kpi(
        "PRICE CHANGE",
        f"{price_change:+.2f}%",
        "Compared with baseline price"
    )

with i2:
    make_kpi(
        "PROFIT CHANGE",
        currency(profit_change),
        "Estimated daily profit difference"
    )

with i3:
    make_kpi(
        "DEMAND CHANGE",
        f"{demand_change:+,.1f}",
        "Units compared with baseline"
    )


# ============================================================
# 17. PRICING DECISION SUMMARY
# ============================================================

section(
    "Pricing Decision Summary",
    "A quick interpretation of the scenario results."
)

best_strategy = scenario_df.loc[
    scenario_df["Profit"].idxmax()
]

best_strategy_name = best_strategy["Strategy"]

best_strategy_profit = best_strategy["Profit"]

profit_difference = (
    best_strategy_profit - expected_profit
)

if profit_difference > 0:

    decision_text = (
        f"The highest estimated daily profit among the displayed "
        f"scenarios is generated by <b>{best_strategy_name}</b>, "
        f"with an estimated profit of <b>"
        f"{currency(best_strategy_profit)}</b>. "
        f"This is {currency(profit_difference)} higher than "
        f"your selected price scenario."
    )

else:

    decision_text = (
        f"Your selected price scenario currently delivers the "
        f"highest estimated daily profit among the displayed "
        f"alternatives, at <b>{currency(expected_profit)}</b>."
    )

st.markdown(
    f"""
    <div class="insight-box">
        <b>Business insight</b><br>
        {decision_text}
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# 18. DOWNLOAD REPORT
# ============================================================

st.markdown("<br>", unsafe_allow_html=True)

section(
    "Export Scenario Report",
    "Download the pricing analysis for further business review."
)

export_df = scenario_df.copy()

export_df["Product_ID"] = product["Product_ID"]

export_df["Product_Name"] = product["Product_Name"]

export_df["Category"] = product["Category"]

export_df = export_df[
    [
        "Product_ID",
        "Product_Name",
        "Category",
        "Strategy",
        "Price",
        "Expected Demand",
        "Revenue",
        "Profit",
        "Profit Margin (%)"
    ]
]

csv_data = export_df.to_csv(
    index=False
).encode("utf-8")

st.download_button(
    label="Download Pricing Scenario Report",
    data=csv_data,
    file_name=f"{product['Product_ID']}_pricing_scenarios.csv",
    mime="text/csv"
)


# ============================================================
# 19. FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        <b>ShopSphere</b> · Pricing Optimization & Profit Intelligence<br>
        Model-based estimates · Historical data · Scenario analysis
    </div>
    """,
    unsafe_allow_html=True
)
