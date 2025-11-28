# UK Debt Sustainability Analysis

## Expert-Corrected Implementation (November 2025)

This repository contains a comprehensive debt sustainability analysis (DSA) for the United Kingdom, incorporating methodological corrections based on expert review.

---

## 📋 Overview

The analysis evaluates UK public debt sustainability through multiple complementary frameworks:

1. **Econometric Testing** - Unit root, cointegration, and structural break tests
2. **Bohn Fiscal Reaction Test** - With HAC-corrected standard errors
3. **Debt Dynamics Decomposition** - r-g differential analysis
4. **Monte Carlo Simulation** - MLE-calibrated fat-tailed distributions
5. **Gross Financing Needs** - Rollover risk assessment
6. **Scenario Analysis** - Stress testing under alternative assumptions

---

## 🔧 Installation

```bash
# Clone the repository
git clone https://github.com/your-repo/uk-dsa.git
cd uk-dsa

# Install dependencies
pip install -r requirements.txt
```

### Requirements
- Python 3.9+
- pandas >= 2.0.0
- numpy >= 1.24.0
- scipy >= 1.10.0
- matplotlib >= 3.7.0
- openpyxl >= 3.1.0
- xlsxwriter >= 3.1.0

---

## 🚀 Quick Start

```python
from src.main import run_complete_analysis

# Run full analysis
results = run_complete_analysis()

# Generate all outputs
results.generate_excel_workbook('outputs/UK_DSA_Analysis.xlsx')
results.generate_all_figures('outputs/figures/')
```

Or run from command line:

```bash
python -m src.run_analysis
```

---

## 📁 Project Structure

```
uk_dsa_final/
├── src/
│   ├── __init__.py
│   ├── config.py                 # Configuration & November 2025 Budget data
│   ├── data_loader.py            # Historical data loading utilities
│   ├── econometric_tests.py      # Unit root, cointegration, HAC SEs
│   ├── bohn_test.py              # Fiscal reaction function tests
│   ├── monte_carlo.py            # MLE-calibrated stochastic simulation
│   ├── debt_dynamics.py          # Decomposition analysis
│   ├── sustainability_assessment.py  # Overall assessment framework
│   ├── visualization.py          # Publication-quality figures
│   └── main.py                   # Main analysis orchestration
├── outputs/
│   ├── figures/                  # Generated visualizations
│   ├── tables/                   # Summary tables
│   └── data/                     # Processed data exports
├── requirements.txt
└── README.md
```

---

## 📊 Key Findings

### Bohn Fiscal Reaction Test

| Specification | β (debt response) | HAC t-stat | Result |
|---------------|-------------------|------------|--------|
| Basic         | -0.099            | -1.57      | ❌ FAIL |
| Augmented     | -0.085            | -1.42      | ❌ FAIL |
| Non-linear    | Negative          | -          | ❌ FAIL |

**Interpretation**: The UK government has historically **NOT** exhibited debt-stabilizing fiscal behavior. A negative β means higher debt is associated with *worse* primary balances, not better.

### Monte Carlo Simulation (10,000 simulations, 10-year horizon)

| Metric | OBR Baseline | Fiscal Reaction |
|--------|--------------|-----------------|
| Mean Terminal Debt | 102.5% | 162.8% |
| P(>100% terminal) | 55.8% | 100.0% |
| VaR 95% | 122.3% | 189.7% |
| VaR 99% | 137.4% | 219.7% |

**Key insight**: If fiscal behavior reverts to historical patterns (negative Bohn β), debt spirals are virtually certain.

### Fat-Tail Impact

- MLE-estimated degrees of freedom: 2.1-2.3 (much fatter than assumed 5-7)
- GDP-interest correlation: +0.40 (procyclical, not -0.30 countercyclical)
- Fat tails add ~9pp to P(>100%) vs. normal distribution

---

## ⚠️ Expert Corrections Applied

This codebase incorporates corrections from methodological review:

| Issue | Original | Corrected |
|-------|----------|-----------|
| Standard errors | OLS only | Newey-West HAC |
| Sample size | 49 obs (stated) | 32 obs (actual) |
| Distribution params | Assumed df=5-7 | MLE-estimated df≈2-3 |
| Correlations | Literature values | UK-estimated from data |
| Fiscal space | Ghosh single-country | **REMOVED** (invalid) |
| Non-linear constant | Missing | Added |

---

## 📈 Generated Outputs

### Figures (12 publication-quality visualizations)

1. `fan_chart.png` - Monte Carlo debt path projections
2. `debt_decomposition.png` - Historical dynamics breakdown
3. `bohn_scatter.png` - Fiscal reaction function
4. `r_g_differential.png` - Interest-growth gap
5. `gfn_analysis.png` - Gross financing needs
6. `scenario_comparison.png` - Stress test results
7. `fat_tail_comparison.png` - Normal vs. Student-t distributions
8. `terminal_distribution.png` - Histogram of terminal debt
9. `debt_composition.png` - Gilt portfolio structure
10. `ilg_sensitivity.png` - Inflation-linked exposure
11. `fiscal_rules_history.png` - Timeline of abandoned rules
12. `sustainability_dashboard.png` - Summary assessment

### Excel Workbook

Comprehensive workbook with sheets for:
- Executive Summary
- Bohn Test Results
- Debt Dynamics
- Monte Carlo Statistics
- GFN Analysis
- Scenario Comparison
- Raw Data

---

## 🔬 Methodology

### Econometric Framework

```
1. Unit Root Testing
   - ADF test: H₀ = unit root
   - KPSS test: H₀ = stationarity
   - Combined inference for I(d) determination

2. Cointegration (Engle-Granger)
   - Two-step residual-based test
   - EG-augmented critical values

3. Structural Break Detection
   - Chow test at known dates (1997, 2008, 2020)
   - Sup-Wald for unknown breaks

4. HAC Standard Errors
   - Newey-West with Bartlett kernel
   - Automatic bandwidth selection
```

### Bohn Test Specification

```
Basic:      pb_t = α + β·d_{t-1} + ε_t
Augmented:  pb_t = α + β·d_{t-1} + γ₁·YGAP + γ₂·GVAR + ε_t
Non-linear: pb_t = α + β₁·d + β₂·d² + γ·YGAP + ε_t
Dynamic:    pb_t = α + ρ·pb_{t-1} + β·d_{t-1} + γ·YGAP + ε_t
```

Sustainability requires β > 0 and statistically significant.

### Monte Carlo Simulation

```python
# Shock generation with fat tails
gdp_shock = t.rvs(df=3.0, loc=0, scale=σ_g)
rate_shock = t.rvs(df=4.0, loc=0, scale=σ_r)

# Cholesky decomposition for correlation
L = cholesky(Σ)
correlated_shocks = L @ independent_shocks

# Debt dynamics
d_{t+1} = (1 + r_t - g_t)/(1 + g_t) · d_t - pb_t + sfa_t
```

---

## 📚 Data Sources

| Data | Source | Frequency |
|------|--------|-----------|
| Public Sector Net Debt | ONS | Monthly |
| Primary Balance | ONS/OBR | Annual |
| Gilt Yields | DMO/BoE | Daily |
| GDP | ONS | Quarterly |
| Inflation (RPI/CPI) | ONS | Monthly |
| Budget Forecasts | OBR EFO March 2025 | - |

---

## 🎯 Sustainability Verdict

### Operational Definition

Debt is **SUSTAINABLE** if:
1. Debt/GDP stabilizes or declines with P > 50%
2. Government retains market access under stress
3. Primary balance achieves debt-stabilizing level
4. Fiscal reaction to debt is positive (Bohn β > 0)

### UK Assessment

| Criterion | Status | Notes |
|-----------|--------|-------|
| Fiscal reaction | ❌ FAIL | β = -0.099 (negative) |
| Debt path | ❌ FAIL | P(>100%) = 55.8% |
| Primary balance | ⚠️ IF achieved | pb = 1.4% ≥ pb* = 0.9% |
| Financing needs | ✅ PASS | GFN = 10.3% < 15% |

**Overall: CONDITIONALLY SUSTAINABLE**

Sustainable *only if* policy commitments are achieved. Historical behavior provides NO evidence of automatic stabilization.

---

## 📖 References

- Bohn, H. (1998). "The Behavior of U.S. Public Debt and Deficits." *Quarterly Journal of Economics*.
- Blanchard, O. (2019). "Public Debt and Low Interest Rates." *American Economic Review*.
- IMF (2022). *Staff Guidance Note on the Sovereign Risk and Debt Sustainability Framework*.
- OBR (2025). *Economic and Fiscal Outlook - March 2025*.
- HM Treasury (2025). *Autumn Budget 2024 / Spring Statement 2025*.

---

## 📝 License

MIT License - See LICENSE file for details.

---

## 👥 Contributors

UK DSA Research Project - Imperial College London UROP

---

*Last updated: November 2025*
