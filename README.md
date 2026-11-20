# Real-Time Pricing Optimization & Profit Intelligence System
A data-driven pricing analytics project that combines demand analysis,
price-elasticity estimation, pricing recommendations, scenario
simulation, and business intelligence for a fictional Indian e-commerce
business.

## Description
Pricing decisions influence sales volume, revenue, profit, competitive
positioning, and demand. This project brings product, sales,
competitor-pricing, and market-condition data into one workflow to
evaluate pricing decisions.

The pipeline prepares and validates data, estimates product-level price
sensitivity, evaluates candidate prices and pricing strategies, and
generates recommendations based on expected profitability and configured
pricing constraints. A Streamlit application supports interactive
scenario exploration, while Power BI presents the resulting business
insights.

## Overview
### Workflow
1. Generate synthetic product, sales, competitor-pricing, and
market-condition data.
2.Clean and validate datasets.
3.Explore revenue, profit, demand, pricing, and competition.
4.Estimate product-level price elasticity and sensitivity segments.
5.Evaluate candidate prices and pricing strategies.
6.Generate recommended prices and expected outcomes.
7.Explore pricing scenarios in Streamlit.
8.Review business performance and optimization insights in Power BI.

### Business questions
1. Which products and categories contribute most
to revenue and profit?
2. How do our prices compare with competitor prices?
3. How sensitive are products to price changes?
4. How do pricing strategies compare on expected revenue and profit?
5. Which products
receive the largest recommended price changes?

## Features 
- Synthetic e-commerce data generation.
- Data cleaning and validation checks.
- Exploratory analysis of sales, revenue, profit, demand, and prices.
- Product-level price-elasticity estimation and segmentation.
- Candidate-price evaluation and pricing recommendations.
- Comparison of profit-maximization, competitive, and demand-based
strategies.
- Interactive Streamlit pricing simulator.
- Four-page Power BI report covering overview, category performance,
pricing optimization, and elasticity.
- Business metrics including revenue, profit, margin, units sold,
expected daily revenue, expected daily profit, and recommended price
change.

## Tech Stack

| Tool | How it is used in this project |
|---|---|
|**Python**| Programming |
|**pandas**| Data cleaning, transformation, manipulation, aggregation, and generating analytical CSV reports |
| **NumPy**| Numerical calculations, demand simulations, pricing scenarios, and mathematical operations |
| **Streamlit**| Interactive pricing simulator and profit intelligence dashboard where users can explore pricing scenarios and evaluate their expected business impact|
| **PowerBI**| Interactive business intelligence dashboard with KPI cards, category performance, pricing strategy comparisons, and price elasticity insights |
| **DAX**| Used to create calculated measures for revenue, profit, profit margins, expected daily performance, and pricing metrics in Power BI |

## Project Structure

```
pricing-optimization/
├── data/
│   ├── raw/
│   └── processed/
├── reports/
│   ├── optimization/
│   └── elasticity/
├── notebooks/
├── src/
│   └── pricing_simulator.py
├── generate_data.py
├── requirements.txt
├── README.md
└── README_POWERBI.md
```

## Setup & usage
### 1. Clone the repository
```bash
git clone <your-repository-url>
cd pricing-optimization
```

### 2. Create a virtual environment
#### Windows PowerShell
```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```
#### macOS/Linux
```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Generate datasets
```bash
python generate_data.py
```

### Launch the Streamlit application
```bash
streamlit run src/pricing_simulator.py
```

## Streamlit Dashboard and simulator
The Streamlit application is the interactive decision-support layer of
the project. It allows a user to explore pricing scenarios and inspect
their estimated business impact.

A typical workflow is:
1. Select a product or product group using the controls available in
   the app.
2. Review its baseline price and relevant product assumptions.
3. Adjust a candidate price or scenario input.
4. Run or update the scenario.
5. Review the estimated demand, units sold, revenue, and profit.
6. Compare the scenario with the baseline or recommendation.
Key outputs may include: baseline price, scenario price, recommended
price, expected units sold, expected revenue, expected profit, and
percentage price change. The exact controls and outputs should be kept
consistent with the current implementation of
src/pricing_simulator.py.

## PowerBI Report
The Power BI report contains four analytical pages:
1. Pricing Overview -- revenue, profit, profit margin, units sold,
   and monthly trends.
2. Category Performance -- category-level revenue and profitability
   comparisons.
3. Pricing Strategy & Optimization -- expected outcomes by strategy
   and recommended product price changes.
4. Price Elasticity Analysis -- product sensitivity, category
   elasticity, and demand patterns.
See [README_POWERBI.md](README_POWERBI.md) for the page-by-page report
explanation.

## Future Enhancements
- Integrate refreshed competitor-price data.
- Add time-aware demand forecasting and stronger causal analysis.
- Include inventory-aware and promotion-aware optimization.
- Add scenario history, report exports, and configurable business
constraints.
- Deploy the Streamlit app and automate data refresh.
- Add model monitoring and recommendation explainability.
