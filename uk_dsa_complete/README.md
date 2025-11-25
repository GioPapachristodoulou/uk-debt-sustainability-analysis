# UK Debt Sustainability Analysis - Complete Codebase

## Overview

This is a comprehensive, institutional-grade debt sustainability analysis (DSA) framework for the United Kingdom. It implements:

1. **Fiscal Reaction Function (Bohn Test)** - The canonical econometric test of fiscal sustainability
2. **Fiscal Space Calculation** - IMF methodology (Ghosh et al. 2013)
3. **Gross Financing Needs (GFN)** - Rollover risk analysis with IMF thresholds
4. **Fat-Tailed Monte Carlo** - 10,000 stochastic simulations with Student-t distributions
5. **Scenario Stress Tests** - Multiple adverse and benign scenarios
6. **Debt Dynamics Decomposition** - Primary balance, interest-growth, stock-flow components

## Key Findings (November 2025)

| Metric | Value | Assessment |
|--------|-------|------------|
| Current Debt/GDP | 96.0% | ⚠️ High |
| Bohn Test | β = -0.017 | ❌ FAIL |
| Fiscal Space | 18pp | ⚠️ Limited |
| GFN/GDP | 10.3% | ✓ Below threshold |
| P(Debt > 100%) | 40%/61% | ⚠️ Elevated |
| VaR 99% | 135% | ⚠️ Severe tail risk |

**OVERALL VERDICT: MARGINALLY SUSTAINABLE - HIGH RISK**

## Project Structure

```
uk_dsa/
├── src/
│   ├── __init__.py
│   ├── advanced_analysis.py      # Bohn test, Fiscal space, GFN
│   ├── calibrated_monte_carlo.py # Fat-tailed Monte Carlo
│   ├── config.py                 # Parameters and historical data
│   ├── data_loader.py            # Data loading utilities
│   ├── debt_dynamics.py          # Debt projection model
│   ├── monte_carlo.py            # Base Monte Carlo simulation
│   ├── sustainability_assessment.py # Assessment framework
│   ├── visualizations_complete.py   # All figures
│   └── master_execution.py       # Main execution script
├── data/                         # Data directory (load from project files)
├── outputs/                      # Generated outputs
└── README.md
```

## Installation

```bash
# Requirements
pip install numpy pandas scipy matplotlib openpyxl
```

## Usage

### Quick Start

```python
# Run complete analysis
python src/master_execution.py
```

### Individual Components

```python
# 1. Bohn Test
from src.advanced_analysis import BohnTest
bohn = BohnTest()
results = bohn.run_all_tests()

# 2. Fiscal Space
from src.advanced_analysis import FiscalSpaceCalculator
fs = FiscalSpaceCalculator()
fs.calculate_fiscal_space(current_debt=96.0)
fs.print_results()

# 3. Gross Financing Needs
from src.advanced_analysis import GrossFinancingNeeds
gfn = GrossFinancingNeeds()
gfn.calculate_gfn()
gfn.print_results()

# 4. Fat-Tailed Monte Carlo
from src.calibrated_monte_carlo import CalibratedFatTailMC
mc = CalibratedFatTailMC(n_simulations=10000)
mc.simulate()
mc.print_results()
```

## Module Documentation

### 1. advanced_analysis.py

#### BohnTest
Implements the Bohn (1998) fiscal sustainability test:
- Basic regression: `pb_t = α + β·d_{t-1} + ε_t`
- Augmented with business cycle controls
- Non-linear (quadratic) specification
- Newey-West HAC standard errors

**Key output:** β coefficient - positive and significant indicates sustainability

#### FiscalSpaceCalculator
Implements IMF fiscal space methodology:
- Estimates fiscal reaction curve
- Calculates debt limit where fiscal fatigue sets in
- Fiscal space = debt limit - current debt

#### GrossFinancingNeeds
Calculates annual financing requirements:
- GFN = Primary deficit + Interest + Maturing debt
- Compares against IMF thresholds (15%/20% GDP)
- Refinancing risk index

### 2. calibrated_monte_carlo.py

#### CalibratedFatTailMC
Enhanced Monte Carlo with:
- Student-t distributions (df=5-7) for fat tails
- Gaussian copula for correlation structure
- Calibrated to match OBR baseline as median
- AR(1) dynamics with automatic stabilizers

**Parameters:**
- `n_simulations`: Number of paths (default 10,000)
- `horizon_years`: Projection horizon (default 10)
- `random_seed`: For reproducibility

### 3. config.py

Contains all configuration parameters:
- Historical debt/GDP series (1997-2024)
- OBR forecast parameters
- Scenario definitions
- Economic assumptions

### 4. visualizations_complete.py

Generates 12 publication-quality figures:
1. Historical debt trajectory
2. Monte Carlo fan chart
3. Scenario stress tests
4. Debt decomposition
5. Interest-growth differential
6. Interest burden analysis
7. Debt composition
8. ILG sensitivity
9. Bohn test results
10. Fiscal space analysis
11. Gross financing needs
12. Fat-tail impact comparison

## Outputs

### Figures (PNG, 300 DPI)
- `fig1_historical_debt.png` - Historical debt/GDP with events
- `fig2_fan_chart.png` - Monte Carlo confidence intervals
- `fig3_scenarios.png` - Stress test comparison
- `fig4_decomposition.png` - Debt dynamics decomposition
- `fig5_r_g_differential.png` - Interest-growth gap
- `fig6_interest_burden.png` - Interest/GDP and Interest/Revenue
- `fig7_debt_composition.png` - Debt by instrument and maturity
- `fig8_ilg_sensitivity.png` - ILG inflation sensitivity
- `fig9_bohn_test.png` - Fiscal reaction function
- `fig10_fiscal_space.png` - Fiscal space analysis
- `fig11_gfn.png` - Gross financing needs
- `fig12_fat_tail_impact.png` - Normal vs fat-tailed distributions

### Excel Workbook
`UK_DSA_Complete_Analysis.xlsx` with sheets:
- Executive Summary
- Bohn Test Results
- Fiscal Space Analysis
- GFN Analysis
- Monte Carlo Results
- Fan Chart Data
- Scenarios

### Report
`UK_DSA_Complete_Report.md` - Comprehensive markdown report

## Data Sources

- **ONS:** Public Sector Finances (monthly/annual)
- **OBR:** Economic and Fiscal Outlook (March 2025)
- **DMO:** Gilt market data, redemption profile
- **Bank of England:** Interest rate statistics, yield curves

## Methodology Notes

### Bohn Test Interpretation
- β > 0 and significant: Sustainable (government responds to debt)
- β ≤ 0 or insignificant: Not sustainable (no systematic response)
- UK result: β = -0.017, NOT significant → FAIL

### Fiscal Space Caveats
- Model-dependent estimates
- Market sentiment can shift limits abruptly
- UK-specific factors (reserve currency, BoE) may extend limits

### Fat-Tailed Monte Carlo
- Student-t df=5 for GDP implies ~10% probability of events beyond ±2σ
- Excess kurtosis captures crisis-type outcomes
- 40% probability of exceeding 100% GDP reflects genuine risk

## References

1. Bohn, H. (1998). "The Behavior of U.S. Public Debt and Deficits." Quarterly Journal of Economics.
2. Ghosh, A., Kim, J., Mendoza, E., Ostry, J., & Qureshi, M. (2013). "Fiscal Fatigue, Fiscal Space and Debt Sustainability in Advanced Economies." Economic Journal.
3. IMF (2013). "Staff Guidance Note for Public Debt Sustainability Analysis in Market-Access Countries."
4. OBR (2025). "Economic and Fiscal Outlook - March 2025."

## License

For academic and policy research purposes.

## Author

UK DSA Project - November 2025
