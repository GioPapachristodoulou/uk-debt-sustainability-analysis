# UK Debt Sustainability Analysis
## Comprehensive Assessment Report
### November 2025

---

## Executive Summary

| Metric | Value | Assessment |
|--------|-------|------------|
| Current Debt/GDP | 96.0% | ⚠️ High |
| Bohn Test | β = -0.017 | ❌ FAIL |
| Fiscal Space | 18pp | ⚠️ Limited |
| GFN/GDP | 10.3% | ✓ Below threshold |
| P(Debt > 100%) | 40%/61% | ⚠️ Elevated |
| VaR 99% | 135% | ⚠️ Severe tail risk |

**OVERALL VERDICT: MARGINALLY SUSTAINABLE - HIGH RISK**

UK public debt sustainability is conditional on:
1. Achieving primary surpluses as projected by OBR
2. Avoiding major adverse shocks
3. Maintaining market confidence
4. No further expansion of fiscal policy

---

## 1. Bohn Fiscal Reaction Test

The Bohn test examines whether the government systematically responds to higher debt 
by improving the primary balance. A positive, statistically significant coefficient (β) 
indicates sustainable fiscal behavior.

### Results

| Test Variant | β Coefficient | t-statistic | p-value | Result |
|--------------|---------------|-------------|---------|--------|
| Basic | -0.0321 | -1.72 | 0.0922 | ❌ FAIL |
| Augmented | -0.0166 | -2.54 | 0.0146 | ❌ FAIL |
| Newey-West | -0.0166 | -0.21 | 0.8360 | ❌ FAIL |

### Interpretation

The **negative** Bohn coefficient is a critical finding. It indicates that the UK 
government has **not** historically responded to higher debt by running larger 
primary surpluses. Instead, the relationship is weakly negative - higher debt 
has been associated with slightly *worse* primary balances.

This contrasts with countries like the United States, where Bohn (1998) found 
positive fiscal response, supporting sustainability.

**Implication:** Past fiscal behavior does not provide evidence that debt will 
be stabilized through automatic fiscal adjustment. Policy commitment is required.

---

## 2. Fiscal Space Analysis

Following the Ghosh et al. (2013) methodology used by the IMF, we estimate 
fiscal space as the gap between current debt and the debt limit.

### Key Results

| Metric | Value |
|--------|-------|
| Current Debt/GDP | 96.0% |
| Estimated Debt Limit | 114% |
| **FISCAL SPACE** | **18pp** |

### Scenario Sensitivity

| Scenario | r (%) | g (%) | Debt Limit | Fiscal Space |
|----------|-------|-------|------------|--------------|
| baseline | 4.5 | 3.5 | 114% | 18pp |
| low_growth | 4.5 | 2.0 | 124% | 28pp |
| high_rates | 6.0 | 3.5 | 123% | 27pp |
| adverse | 6.0 | 2.0 | 133% | 37pp |
| benign | 3.5 | 4.5 | 101% | 5pp |

### Interpretation

Fiscal space of 18pp means the UK could, in principle, absorb additional debt 
of up to 18% of GDP before reaching the estimated debt limit. However:

- This limit is model-dependent and uncertain
- Market sentiment can shift limits abruptly (cf. 2022 mini-budget)
- The benign scenario shows fiscal space could shrink to just 5pp
- UK-specific factors (reserve currency, BoE) may extend limits

---

## 3. Gross Financing Needs (GFN)

GFN measures rollover risk: the total amount the government must raise each year 
to cover deficits, interest, and maturing debt.

### Annual GFN Projections

| Year | GFN (£bn) | GFN/GDP |
|------|-----------|---------|
| 2025 | 385 | 13.4% |
| 2026 | 388 | 13.0% |
| 2027 | 363 | 11.7% |
| 2028 | 362 | 11.2% |
| 2029 | 368 | 11.0% |
| 2030 | 343 | 9.9% |
| 2031 | 325 | 9.0% |
| 2032 | 334 | 8.9% |
| 2033 | 327 | 8.5% |
| 2034 | 334 | 8.3% |
| 2035 | 330 | 8.0% |

### Summary

| Metric | Value | IMF Threshold |
|--------|-------|---------------|
| Average GFN/GDP | 10.3% | 15% (elevated) |
| Maximum GFN/GDP | 13.4% | 20% (high risk) |
| Years > 15% | 0 | - |

**Assessment: MODERATE RISK** - GFN remains below IMF elevated risk thresholds, 
but refinancing needs are substantial and sensitive to market conditions.

### Mitigating Factors

✓ Long average maturity (14.5 years) reduces rollover pressure  
✓ Deep, liquid gilt market with strong domestic investor base  
✓ Reserve currency status provides additional flexibility  
✗ High ILG share (34%) creates inflation vulnerability  
✗ Large foreign holdings (28%) sensitive to sentiment shifts

---

## 4. Fat-Tailed Monte Carlo Simulation

We run 10,000 stochastic simulations using Student-t distributions to capture 
fat-tailed risks that normal distributions underestimate.

### Distribution Parameters

| Variable | Distribution | Degrees of Freedom |
|----------|--------------|-------------------|
| GDP Growth | Student-t | 5 |
| Inflation | Student-t | 5 |
| Interest Rates | Student-t | 7 |

### Terminal Distribution (2034-35)

| Statistic | Value |
|-----------|-------|
| Mean | 97.6% |
| Median | 96.5% |
| Std Dev | 13.5pp |
| Skewness | 0.50 |
| Excess Kurtosis | 0.70 |
| Range | [50%, 166%] |

### Threshold Breach Probabilities

| Threshold | P(Terminal) | P(Ever) |
|-----------|-------------|---------|
| >80% | 92.1% | 100.0% |
| >90% | 70.2% | 100.0% |
| >100% | 40.1% | 61.1% |
| >110% | 16.9% | 22.1% |
| >120% | 5.8% | 7.0% |
| >130% | 1.7% | 2.0% |

### Tail Risk Measures

| Measure | Value |
|---------|-------|
| VaR 95% | 121.1% |
| VaR 99% | 134.6% |
| ES 95% | 129.4% |
| ES 99% | 142.7% |

### Fat-Tail Impact

The excess kurtosis of 0.70 indicates 
substantially fatter tails than a normal distribution. This means:

- **40% probability** debt exceeds 100% in terminal year
- **61% probability** debt exceeds 100% at some point during projection
- In the **worst 1% of scenarios**, debt averages 143% of GDP

---

## 5. Scenario Stress Tests

| Scenario | Peak Debt | Terminal Debt | Key Assumptions |
|----------|-----------|---------------|-----------------|
| Baseline | 96.2% | 88.5% | OBR March 2025 |
| Adverse Rates | 98.5% | 92.0% | +200bp gilt yields |
| Recession | 105.8% | 90.5% | -3% GDP shock |
| Stagflation | 108.5% | 95.0% | +4% RPI, -2% GDP |
| Consolidation | 95.9% | 74.0% | +1% GDP primary surplus |
| Combined Adverse | 112.0% | 97.5% | Multiple shocks |

---

## 6. Risk Assessment

### Critical Risks

1. **Failed Bohn Test** - No historical evidence of debt-stabilizing fiscal response
2. **ILG Exposure** - 34% of gilts are inflation-linked; +1pp RPI = +£11bn interest
3. **Fat-Tail Risk** - 40% chance of exceeding 100% GDP; 6% chance of exceeding 120%
4. **Limited Fiscal Space** - Only 18pp buffer before estimated debt limit

### Risk Matrix

| Risk | Probability | Impact | Overall |
|------|-------------|--------|---------|
| Baseline breach of 100% | Medium (40%) | High | ⚠️ HIGH |
| Stagflation scenario | Low (15%) | Very High | ⚠️ HIGH |
| Fiscal fatigue | Medium | Medium | MODERATE |
| Market confidence loss | Low | Very High | ⚠️ HIGH |

---

## 7. Conclusions and Recommendations

### For Policymakers

1. **Credible fiscal consolidation is essential** - The failed Bohn test means 
   markets cannot rely on automatic fiscal correction
   
2. **Reduce ILG exposure** - High inflation sensitivity creates vulnerability; 
   consider rebalancing future issuance toward conventional gilts
   
3. **Build fiscal buffers** - Current 18pp fiscal space is limited; aim to 
   reduce debt/GDP to create more room for future shocks
   
4. **Monitor GFN closely** - While currently manageable, refinancing needs 
   could become stressed under adverse scenarios

### For Analysts

1. The Bohn test failure is the most significant finding - it fundamentally 
   questions whether UK fiscal policy is on a sustainable path without 
   explicit policy commitment
   
2. Fat-tailed simulations reveal materially higher risk than standard normal 
   analysis would suggest - tail risk should not be ignored
   
3. The fiscal space estimate of 18pp provides a rough upper bound, but market 
   sentiment can shift this limit rapidly

---

## Appendix: Methodology

### Data Sources
- ONS Public Sector Finances
- OBR Economic and Fiscal Outlook (March 2025)
- DMO Gilt Market Data
- Bank of England Interest Rate Statistics

### Models
- **Bohn Test**: OLS regression with HAC standard errors
- **Fiscal Space**: Ghosh et al. (2013) cubic fiscal reaction function
- **GFN**: IMF standard methodology
- **Monte Carlo**: 10,000 simulations with Student-t distributions and Gaussian copula dependence

### Software
- Python 3.x with NumPy, SciPy, Pandas
- Statistical analysis and visualization

---

*Report generated November 2025*
