# Data Dynamos — Investment Portfolio Analysis & Wealth Intelligence Agent

**Hackathon:** Agentic Orchestration Hackathon
**Domain:** Finance
**Team:** Data Dynamos

## Overview

An autonomous Finance agent that ingests a portfolio CSV and produces a full wealth intelligence report covering performance metrics, sentiment, price signals, stress tests, tax-loss opportunities, and a rebalancing plan.

Follows the agentic pattern: **Observe → Reason → Plan → Act → Verify → Iterate**

## Structure

```
agent-finance/          ← main agent implementation
├── src/
│   ├── agent.py        ← orchestration loop
│   ├── state.py        ← AgentState, Holding dataclasses
│   ├── tools/          ← 9 domain tools
│   └── verification/   ← 3-layer verification system
├── tests/              ← unit + integration tests
├── data/               ← sample portfolio CSV
└── docs/               ← architecture + data schema

REQUIREMENTS.md         ← project requirements (from brief)
docs/superpowers/plans/ ← implementation plan
```

## Tools (9)

| Tool | Category |
|---|---|
| Portfolio Validator | Retrieval + Verification |
| Metrics Calculator | Processing |
| Diversification Analyzer | Processing |
| News Sentiment Analyzer | Analysis |
| Price Signal Detector | Analysis |
| Risk & Stress Tester | Analysis |
| Tax-Loss Harvest Finder | Analysis |
| Rebalancing Planner | Execution |
| Wealth Report Generator | Reporting |

## Quick Start

```bash
cd agent-finance
pip install -r requirements.txt
pytest tests/ -v
```

## Data Sources
- Market prices: yfinance (free, no API key)
- News: yfinance Ticker.news (free)
- Sentiment: TextBlob (local)
- Portfolio: CSV file (`agent-finance/data/sample_portfolio.csv`)
