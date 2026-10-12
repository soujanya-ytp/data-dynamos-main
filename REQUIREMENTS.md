# Project 2: Investment Portfolio Analysis & Wealth Intelligence Agent

## Purpose

Go beyond basic portfolio tracking — analyze performance, stress-test against market scenarios, monitor live news sentiment for held stocks, predict short-term price movement signals, detect tax-loss harvesting opportunities, and deliver a personalised wealth report with actionable rebalancing steps.

## Example Flow

> Portfolio: 51.9% AAPL, 48.1% VTSAX, $120K total → Sharpe ratio 1.42, max drawdown -18% → Overweight tech (62% vs 30% benchmark) → News sentiment for AAPL drops to -0.6 (earnings miss risk) → Stress test: 2008 scenario would lose 34% → Tax-loss harvest opportunity: sell TSLA at -$4,200 loss to offset capital gains → Rebalance: reduce AAPL to 35%, add MSFT and bonds → Projected post-rebalance Sharpe: 1.68

## Input / Output

| | |
|---|---|
| **Input** | Holdings (ticker, quantity, purchase price, date), risk appetite, investment goals, time horizon |
| **Output** | Performance metrics, risk scores, news sentiment per holding, price signal alerts, tax-loss harvest opportunities, stress-test results, rebalancing plan |
| **Data Source** | Portfolio CSV, Yahoo Finance / Alpha Vantage API (live prices), financial news feed, historical market data |
| **Success Criteria** | Metrics calculated, sentiment & signals surfaced, stress test run, tax opportunity found, rebalancing plan generated |

## Tools (9)

| Tool | Description |
|---|---|
| **Portfolio Validator** | Validate holdings completeness and resolve tickers to current market data |
| **Metrics Calculator** | Calculate returns, Sharpe ratio, beta, alpha, and max drawdown vs benchmark |
| **Diversification Analyzer** | Analyze sector, geography, and asset-class allocation; flag over-concentration |
| **News Sentiment Analyzer** | Fetch recent news for each holding and score sentiment (positive/neutral/negative) |
| **Price Signal Detector** | Identify technical signals (RSI overbought/oversold, MACD crossover, 52-week high/low) per ticker |
| **Risk & Stress Tester** | Simulate portfolio under historical crash scenarios (2008, COVID-2020, dot-com) and custom shocks |
| **Tax-Loss Harvest Finder** | Identify positions with unrealised losses that can offset capital gains within tax rules |
| **Rebalancing Planner** | Generate step-by-step rebalancing trades with target weights, estimated costs, and projected Sharpe improvement |
| **Wealth Report Generator** | Consolidated report: portfolio scorecard, sentiment alerts, signals, stress results, tax tips, and rebalancing plan |
