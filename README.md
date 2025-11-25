# 🇬🇧 UK Debt Sustainability Analysis

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.XXXXXXX-blue)](https://doi.org/10.5281/zenodo.XXXXXXX)

A comprehensive, institutional-grade debt sustainability analysis framework for the United Kingdom, implementing advanced econometric methods used by the IMF, central banks, and fiscal policy institutions.

<p align="center">
  <img src="outputs/fig2_fan_chart.png" alt="UK Debt Sustainability Fan Chart" width="700">
</p>

## 📋 Table of Contents

- [Executive Summary](#-executive-summary)
- [Key Findings](#-key-findings)
- [Features](#-features)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Project Structure](#-project-structure)
- [Methodology](#-methodology)
- [Data Sources](#-data-sources)
- [Outputs](#-outputs)
- [Documentation](#-documentation)
- [Citation](#-citation)
- [Contributing](#-contributing)
- [License](#-license)
- [Acknowledgements](#-acknowledgements)

## 📊 Executive Summary

This repository provides a complete toolkit for assessing the sustainability of UK public debt using four complementary methodological approaches:

1. **Bohn Fiscal Reaction Function Test** - Tests whether the government systematically responds to debt accumulation
2. **Fiscal Space Calculation** - Estimates headroom before reaching debt limits (Ghosh et al., 2013)
3. **Gross Financing Needs Analysis** - Assesses rollover risk against IMF thresholds
4. **Fat-Tailed Monte Carlo Simulation** - 10,000-path stochastic projections with Student's t-distributions

The analysis produces publication-ready figures, comprehensive Excel workbooks, and detailed reports suitable for academic research, policy analysis, or financial market assessment.

## 🔑 Key Findings

| Metric | Value | Assessment |
|--------|-------|------------|
| **Bohn Test β Coefficient** | -0.017 | ❌ **FAIL** - No debt-stabilising response |
| **Fiscal Space** | 18 pp | ⚠️ Limited headroom to 114% debt limit |
| **P(Debt > 100%)** | 40.1% | ⚠️ Substantial tail risk |
| **VaR 99%** | 134.6% | ⚠️ Severe downside scenarios |
| **GFN/GDP** | 10.3% avg | ✅ Below IMF 15% threshold |

**Overall Verdict: MARGINALLY SUSTAINABLE** - Conditional on achieving OBR-projected surpluses, avoiding major shocks, and maintaining market confidence.

### Critical Finding: Bohn Test Failure

Unlike the United States (where Bohn, 1998 found positive fiscal reaction), the UK shows **no systematic debt-stabilising fiscal response**. The negative β coefficient implies sustainability depends entirely on explicit policy commitment, not historical behavioural patterns.

## ✨ Features

- **Canonical Econometric Tests**: Bohn (1998) fiscal reaction function with Newey-West HAC standard errors
- **IMF-Standard Fiscal Space**: Ghosh et al. (2013) cubic reaction function methodology
- **Fat-Tailed Distributions**: Student's t-distributions (df=5-7) capturing crisis-frequency events
- **Gaussian Copula Dependence**: Proper correlation structure among macroeconomic shocks
- **12 Publication-Quality Figures**: 300 DPI PNG outputs ready for journals
- **Comprehensive Excel Workbook**: Multi-sheet analysis with conditional formatting
- **Scenario Stress Testing**: 6 deterministic scenarios including stagflation and combined adverse
- **Full Reproducibility**: All code, data, and parameters documented

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Option 1: Clone and Install

```bash
# Clone the repository
git clone https://github.com/yourusername/uk-debt-sustainability.git
cd uk-debt-sustainability

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Option 2: Install as Package

```bash
pip install uk-debt-sustainability
```

### Dependencies

```
numpy>=1.21.0
pandas>=1.3.0
scipy>=1.7.0
matplotlib>=3.4.0
seaborn>=0.11.0
openpyxl>=3.0.0
statsmodels>=0.13.0
```

## 🏃 Quick Start

### Run Complete Analysis

```python
from uk_dsa import run_full_analysis

# Run all analyses and generate outputs
results = run_full_analysis(
    output_dir='./outputs',
    n_simulations=10000,
    forecast_years=10
)

# Access key results
print(f"Bohn β coefficient: {results['bohn_test']['beta']:.4f}")
print(f"Fiscal space: {results['fiscal_space']['space_pp']:.1f} pp")
print(f"P(Debt > 100%): {results['monte_carlo']['prob_exceed_100']:.1%}")
```

### Run Individual Components

```python
from uk_dsa import BohnTest, FiscalSpace, MonteCarloSimulation

# Bohn Test
bohn = BohnTest(data_path='data/')
bohn_results = bohn.run_all_specifications()
bohn.plot_results('outputs/bohn_test.png')

# Fiscal Space
fs = FiscalSpace(r=0.045, g=0.035)
space = fs.calculate(current_debt=96.0)
fs.plot_fiscal_space('outputs/fiscal_space.png')

# Monte Carlo
mc = MonteCarloSimulation(n_paths=10000, horizon=10, fat_tails=True)
mc_results = mc.run()
mc.plot_fan_chart('outputs/fan_chart.png')
```

### Command Line Interface

```bash
# Run full analysis
python -m uk_dsa.run --output-dir ./outputs --simulations 10000

# Run specific component
python -m uk_dsa.run --component bohn_test --output-dir ./outputs

# Generate report only (using cached results)
python -m uk_dsa.run --report-only --format docx
```

## 📁 Project Structure

```
uk-debt-sustainability/
│
├── 📂 src/uk_dsa/                 # Main package
│   ├── __init__.py                # Package initialization
│   ├── config.py                  # Configuration and parameters
│   ├── data_loader.py             # Data loading utilities
│   ├── debt_dynamics.py           # Core debt projection model
│   ├── bohn_test.py               # Bohn fiscal reaction function
│   ├── fiscal_space.py            # Ghosh et al. fiscal space
│   ├── gfn_analysis.py            # Gross financing needs
│   ├── monte_carlo.py             # Standard Monte Carlo
│   ├── fat_tailed_mc.py           # Fat-tailed Monte Carlo
│   ├── scenario_analysis.py       # Deterministic scenarios
│   ├── visualizations.py          # All plotting functions
│   ├── report_generator.py        # Report generation
│   └── run.py                     # CLI entry point
│
├── 📂 data/                       # Data files
│   ├── 📂 ons/                    # ONS Public Sector Finances
│   ├── 📂 obr/                    # OBR forecasts
│   ├── 📂 boe/                    # Bank of England data
│   ├── 📂 dmo/                    # Debt Management Office
│   └── README.md                  # Data documentation
│
├── 📂 outputs/                    # Generated outputs
│   ├── 📂 figures/                # PNG figures (300 DPI)
│   ├── 📂 tables/                 # CSV/Excel tables
│   └── 📂 reports/                # Generated reports
│
├── 📂 docs/                       # Documentation
│   ├── methodology.md             # Detailed methodology
│   ├── api_reference.md           # API documentation
│   ├── data_dictionary.md         # Variable definitions
│   └── replication_guide.md       # Replication instructions
│
├── 📂 tests/                      # Unit tests
│   ├── test_bohn.py
│   ├── test_fiscal_space.py
│   ├── test_monte_carlo.py
│   └── test_integration.py
│
├── 📂 notebooks/                  # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_bohn_test_analysis.ipynb
│   ├── 03_monte_carlo_deep_dive.ipynb
│   └── 04_sensitivity_analysis.ipynb
│
├── 📂 paper/                      # Academic paper
│   ├── UK_DSA_Academic_Paper.docx
│   ├── UK_DSA_Academic_Paper.md
│   └── figures/                   # Paper figures
│
├── .github/                       # GitHub configuration
│   ├── workflows/ci.yml           # CI/CD pipeline
│   ├── ISSUE_TEMPLATE/
│   └── PULL_REQUEST_TEMPLATE.md
│
├── requirements.txt               # Python dependencies
├── setup.py                       # Package setup
├── pyproject.toml                 # Modern Python packaging
├── LICENSE                        # MIT License
├── CONTRIBUTING.md                # Contribution guidelines
├── CODE_OF_CONDUCT.md             # Code of conduct
├── CHANGELOG.md                   # Version history
└── README.md                      # This file
```

## 📐 Methodology

### 1. Bohn Fiscal Reaction Function Test

Tests whether fiscal policy satisfies the sustainability condition β > 0:

```
pb_t = α + β·d_{t-1} + γ₁·YGAP_t + γ₂·GVAR_t + ε_t
```

Where:
- `pb_t` = Primary balance (% GDP)
- `d_{t-1}` = Lagged debt ratio
- `YGAP_t` = Output gap
- `GVAR_t` = Temporary spending deviation

**Specifications implemented:**
- Basic OLS
- Augmented with cyclical controls
- Non-linear (quadratic debt response)
- Newey-West HAC standard errors

### 2. Fiscal Space (Ghosh et al., 2013)

Estimates debt limit from intersection of fiscal reaction curve and debt-stabilising requirement:

```
Fiscal Reaction:     pb = f(d) = α + β₁·d + β₂·d² + β₃·d³
Debt-Stabilising:    pb* = [(r-g)/(1+g)]·d
Debt Limit:          f(d̄) = pb*(d̄) and f'(d̄) < pb*'(d̄)
Fiscal Space:        d̄ - d_current
```

### 3. Gross Financing Needs

```
GFN_t = Primary Deficit_t + Interest_t + Maturing Debt_t
```

Assessed against IMF thresholds:
- **15% GDP**: Elevated risk
- **20% GDP**: High risk

### 4. Fat-Tailed Monte Carlo

10,000 stochastic paths with:

- **Marginal distributions**: Student's t (df: GDP=5, Inflation=5, Rates=7)
- **Dependence structure**: Gaussian copula
- **Correlation matrix**: Calibrated to UK historical data
- **Dynamics**: AR(1) with automatic stabilisers

**Risk measures computed:**
- VaR (95%, 99%)
- Expected Shortfall
- Threshold breach probabilities
- Distribution moments

## 📊 Data Sources

| Source | Data | Frequency | Series |
|--------|------|-----------|--------|
| **ONS** | Public Sector Finances | Monthly | PSND, PSNB, receipts, expenditure |
| **OBR** | Economic & Fiscal Outlook | Biannual | Forecasts, fan charts |
| **Bank of England** | Interest rates, yields | Daily/Monthly | Bank Rate, gilt yields |
| **DMO** | Gilt market data | Daily | Issuance, redemptions, holdings |
| **ONS** | National Accounts | Quarterly | GDP, deflators |

All data are publicly available. See `data/README.md` for download instructions and data dictionary.

## 📈 Outputs

### Figures Generated

| Figure | Description | Section |
|--------|-------------|---------|
| `fig1_historical_debt.png` | Historical debt/GDP 1997-2035 | 3.2 |
| `fig2_fan_chart.png` | Monte Carlo fan chart | 5.4 |
| `fig3_scenarios.png` | Scenario stress tests | 6.2 |
| `fig4_decomposition.png` | Debt dynamics decomposition | Appendix |
| `fig5_r_g_differential.png` | Interest-growth differential | Appendix |
| `fig6_interest_burden.png` | Interest payment analysis | Appendix |
| `fig7_debt_composition.png` | Debt by instrument/maturity | Appendix |
| `fig8_ilg_sensitivity.png` | Index-linked gilt sensitivity | 8.2 |
| `fig9_bohn_test.png` | Bohn test scatter/regression | 5.1 |
| `fig10_fiscal_space.png` | Fiscal space diagram | 5.2 |
| `fig11_gfn.png` | Gross financing needs | 5.3 |
| `fig12_fat_tail_impact.png` | Fat-tail distribution comparison | 5.4 |

### Excel Workbook Sheets

- **Executive Summary**: Key metrics and verdicts
- **Bohn Test**: Regression results and diagnostics
- **Fiscal Space**: Scenarios and sensitivity
- **GFN Analysis**: Annual projections and risk index
- **Monte Carlo**: Distribution statistics and probabilities
- **Fan Chart Data**: Percentile paths for charting
- **Scenarios**: Stress test trajectories

## 📚 Documentation

Detailed documentation available in the `docs/` folder:

- **[Methodology Guide](docs/methodology.md)**: Complete mathematical framework
- **[API Reference](docs/api_reference.md)**: Function and class documentation
- **[Data Dictionary](docs/data_dictionary.md)**: Variable definitions and sources
- **[Replication Guide](docs/replication_guide.md)**: Step-by-step replication instructions

## 📝 Citation

If you use this code in academic work, please cite:

```bibtex
@software{uk_debt_sustainability_2025,
  author = {[Your Name]},
  title = {UK Debt Sustainability Analysis: A Comprehensive Framework},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/yourusername/uk-debt-sustainability},
  version = {1.0.0}
}
```

For the accompanying paper:

```bibtex
@article{author_uk_debt_2025,
  title = {Debt Sustainability in the United Kingdom: A Comprehensive Assessment 
           Using Advanced Econometric Methods},
  author = {[Your Name]},
  journal = {Imperial College Business School Working Paper},
  year = {2025},
  month = {November}
}
```

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Ways to Contribute

- 🐛 Report bugs and issues
- 💡 Suggest new features or methodologies
- 📖 Improve documentation
- 🔧 Submit pull requests
- 📊 Add data sources or countries

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgements

### Methodological References

- **Bohn, H. (1998)** - "The Behavior of U.S. Public Debt and Deficits" - *Quarterly Journal of Economics*
- **Ghosh, A.R. et al. (2013)** - "Fiscal Fatigue, Fiscal Space and Debt Sustainability" - *Economic Journal*
- **IMF (2013)** - "Staff Guidance Note for Public Debt Sustainability Analysis"
- **Blanchard, O.J. (1990)** - "Suggestions for a New Set of Fiscal Indicators" - *OECD Working Papers*

### Data Providers

- Office for National Statistics (ONS)
- Office for Budget Responsibility (OBR)
- Bank of England
- UK Debt Management Office (DMO)

### Institutional Context

This analysis was developed following methodologies employed by:
- International Monetary Fund (IMF)
- European Commission
- HM Treasury
- Bank of England Financial Stability Division

---

<p align="center">
  <b>⭐ Star this repository if you find it useful! ⭐</b>
</p>

<p align="center">
  Made with 📊 for fiscal policy research
</p>
