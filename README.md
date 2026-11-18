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
