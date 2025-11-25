# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Planned: Support for additional countries (Germany, France, Italy)
- Planned: Real-time data fetching from ONS API
- Planned: Interactive dashboard using Streamlit

### Changed
- None

### Deprecated
- None

### Removed
- None

### Fixed
- None

### Security
- None

## [1.0.0] - 2025-11-25

### Added

#### Core Analysis Framework
- **Bohn Fiscal Reaction Function Test**
  - Basic OLS specification
  - Augmented specification with output gap and spending gap controls
  - Non-linear (quadratic) specification
  - Newey-West HAC standard errors for robust inference
  - Comprehensive diagnostics (DW statistic, R², F-test)

- **Fiscal Space Calculation (Ghosh et al., 2013)**
  - Cubic fiscal reaction function estimation
  - Debt limit calculation via intersection method
  - Multi-scenario sensitivity analysis
  - Configurable interest rate and growth assumptions

- **Gross Financing Needs Analysis**
  - Component breakdown (primary deficit, interest, maturing debt)
  - IMF threshold assessment (15%, 20% GDP)
  - Composite refinancing risk index
  - 10-year projection capability

- **Fat-Tailed Monte Carlo Simulation**
  - 10,000-path stochastic simulation
  - Student's t-distributions (configurable degrees of freedom)
  - Gaussian copula for correlation structure
  - VaR and Expected Shortfall calculation
  - Threshold breach probability computation

#### Scenario Analysis
- Six deterministic stress scenarios:
  - Baseline (OBR March 2025)
  - Adverse interest rates (+200bp)
  - Recession (-3% GDP shock)
  - Stagflation (+4% RPI, -2% GDP)
  - Consolidation (+1% GDP surplus)
  - Combined adverse

#### Visualization
- 12 publication-quality figures (300 DPI PNG)
- Fan chart with confidence intervals
- Bohn test scatter and regression plots
- Fiscal space diagrams
- GFN component charts
- Fat-tail distribution comparisons
- Debt composition pie and bar charts

#### Output Generation
- Comprehensive Excel workbook (7 sheets)
- Markdown report with executive summary
- Academic paper template (DOCX)
- CSV data exports

#### Data Infrastructure
- ONS Public Sector Finances integration
- OBR Economic and Fiscal Outlook parsing
- Bank of England yield curve data
- DMO gilt market statistics
- 162+ historical time series

### Technical Features
- Type hints throughout codebase
- Comprehensive docstrings (Google style)
- Modular architecture
- Configuration via YAML
- Logging with loguru
- Progress bars for long operations

### Documentation
- Complete README with examples
- API reference documentation
- Methodology guide
- Data dictionary
- Replication instructions
- Contributing guidelines

### Testing
- Unit tests for all core functions
- Integration tests for full pipeline
- Test coverage reporting
- Hypothesis-based property testing

## [0.2.0] - 2025-11-20 (Internal)

### Added
- Initial Monte Carlo simulation framework
- Basic debt dynamics model
- Scenario analysis module
- First set of visualizations

### Changed
- Refactored data loading for efficiency
- Improved configuration management

## [0.1.0] - 2025-11-15 (Internal)

### Added
- Project structure and scaffolding
- Basic Bohn test implementation
- Data loading utilities
- Initial documentation

---

## Version History Summary

| Version | Date | Highlights |
|---------|------|------------|
| 1.0.0 | 2025-11-25 | Full release with all four methodologies |
| 0.2.0 | 2025-11-20 | Monte Carlo and scenarios |
| 0.1.0 | 2025-11-15 | Initial structure |

## Upgrade Guide

### From 0.x to 1.0.0

Version 1.0.0 is the first public release. If you were using internal versions:

1. Update all imports to use the new package structure:
   ```python
   # Old
   from dsa import BohnTest
   
   # New
   from uk_dsa import BohnTest
   ```

2. Configuration files should be updated to YAML format.

3. The Monte Carlo API has changed:
   ```python
   # Old
   mc = MonteCarlo(n=10000)
   
   # New
   mc = MonteCarloSimulation(n_paths=10000, horizon=10, fat_tails=True)
   ```

---

[Unreleased]: https://github.com/yourusername/uk-debt-sustainability/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/yourusername/uk-debt-sustainability/releases/tag/v1.0.0
[0.2.0]: https://github.com/yourusername/uk-debt-sustainability/releases/tag/v0.2.0
[0.1.0]: https://github.com/yourusername/uk-debt-sustainability/releases/tag/v0.1.0
