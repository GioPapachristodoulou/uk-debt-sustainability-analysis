"""
UK DSA - Clean Data Loading and Econometric Analysis
"""

import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("UK DEBT SUSTAINABILITY - COMPREHENSIVE ECONOMETRIC ANALYSIS")
print("=" * 70)

# ============================================================
# 1. LOAD DEBT DATA
# ============================================================

debt_raw = pd.read_csv('/mnt/project/Public_Sector_Net_Debt.csv', skiprows=8)
debt_raw.columns = ['Date', 'Debt_bn']
debt_raw['Debt_bn'] = pd.to_numeric(debt_raw['Debt_bn'], errors='coerce')

# Get annual data (just years, not quarters)
debt_annual = debt_raw[~debt_raw['Date'].str.contains('Q|M', na=False)].copy()
debt_annual['Year'] = pd.to_numeric(debt_annual['Date'], errors='coerce')
debt_annual = debt_annual.dropna(subset=['Year', 'Debt_bn'])
debt_annual['Year'] = debt_annual['Year'].astype(int)

# Get Q1 quarterly data for longer history
debt_q1 = debt_raw[debt_raw['Date'].str.contains('Q1', na=False)].copy()
debt_q1['Year'] = debt_q1['Date'].str.extract(r'(\d{4})').astype(float)
debt_q1 = debt_q1.dropna(subset=['Year', 'Debt_bn'])
debt_q1['Year'] = debt_q1['Year'].astype(int)

# Combine - prefer annual when available, use Q1 for earlier years
debt_q1_early = debt_q1[~debt_q1['Year'].isin(debt_annual['Year'])]
debt_combined = pd.concat([debt_annual[['Year', 'Debt_bn']], debt_q1_early[['Year', 'Debt_bn']]])
debt_combined = debt_combined.sort_values('Year').drop_duplicates('Year')

print(f"\nDebt data: {debt_combined['Year'].min()}-{debt_combined['Year'].max()}, N={len(debt_combined)}")

# ============================================================
# 2. LOAD GDP DATA
# ============================================================

gdp_raw = pd.read_csv('/mnt/project/Gross_Domestic_Product_at_market_prices_NSA.csv', skiprows=8)
gdp_raw.columns = ['Date', 'GDP_m']
gdp_raw['GDP_m'] = pd.to_numeric(gdp_raw['GDP_m'], errors='coerce')
gdp_raw['Year'] = gdp_raw['Date'].str.extract(r'(\d{4})').astype(float)

# Sum to annual
gdp_annual = gdp_raw.groupby('Year')['GDP_m'].sum().reset_index()
gdp_annual['Year'] = gdp_annual['Year'].astype(int)

print(f"GDP data: {gdp_annual['Year'].min()}-{gdp_annual['Year'].max()}, N={len(gdp_annual)}")

# ============================================================
# 3. LOAD NET BORROWING DATA (for Primary Balance)
# ============================================================

psnb_raw = pd.read_csv('/mnt/project/Public_Sector_Net_Borrowing_NSA.csv', skiprows=8)
psnb_raw.columns = ['Date', 'PSNB_m']
psnb_raw['PSNB_m'] = pd.to_numeric(psnb_raw['PSNB_m'], errors='coerce')

# Get annual
psnb_annual = psnb_raw[~psnb_raw['Date'].str.contains('Q|M', na=False)].copy()
psnb_annual['Year'] = pd.to_numeric(psnb_annual['Date'], errors='coerce')
psnb_annual = psnb_annual.dropna(subset=['Year', 'PSNB_m'])
psnb_annual['Year'] = psnb_annual['Year'].astype(int)

print(f"PSNB data: {psnb_annual['Year'].min()}-{psnb_annual['Year'].max()}, N={len(psnb_annual)}")

# ============================================================
# 4. LOAD DEBT INTEREST DATA
# ============================================================

int_raw = pd.read_csv('/mnt/project/CG_interestdividends_paid_to_private_sector__RoW.csv', skiprows=8)
int_raw.columns = ['Date', 'Interest_m']
int_raw['Interest_m'] = pd.to_numeric(int_raw['Interest_m'], errors='coerce')
int_raw['Year'] = int_raw['Date'].str.extract(r'(\d{4})').astype(float)
int_annual = int_raw.groupby('Year')['Interest_m'].sum().reset_index()
int_annual['Year'] = int_annual['Year'].astype(int)

print(f"Interest data: {int_annual['Year'].min()}-{int_annual['Year'].max()}, N={len(int_annual)}")

# ============================================================
# 5. MERGE AND CALCULATE RATIOS
# ============================================================

data = debt_combined.merge(gdp_annual, on='Year', how='inner')
data = data.merge(psnb_annual[['Year', 'PSNB_m']], on='Year', how='inner')
data = data.merge(int_annual, on='Year', how='left')
data['Interest_m'] = data['Interest_m'].fillna(0)

# Calculate ratios
data['Debt_GDP'] = (data['Debt_bn'] * 1000) / data['GDP_m'] * 100

# Primary Balance = -(Net Borrowing - Interest) = -Net Borrowing + Interest
# Net Borrowing = Primary Deficit + Net Interest
# So: Primary Balance = -Primary Deficit = -(Net Borrowing - Interest)
data['PB_GDP'] = (-data['PSNB_m'] + data['Interest_m']) / data['GDP_m'] * 100

# Create lagged debt
data = data.sort_values('Year')
data['Debt_GDP_lag'] = data['Debt_GDP'].shift(1)

# Filter sample
data = data[(data['Year'] >= 1975) & (data['Year'] <= 2024)]
data = data.dropna(subset=['PB_GDP', 'Debt_GDP', 'Debt_GDP_lag'])

print(f"\nFinal merged data: {data['Year'].min()}-{data['Year'].max()}, N={len(data)}")

# ============================================================
# SUMMARY STATISTICS
# ============================================================

print("\n" + "-" * 70)
print("SUMMARY STATISTICS")
print("-" * 70)
print(f"{'Variable':<25} {'Mean':>10} {'SD':>10} {'Min':>10} {'Max':>10}")
print("-" * 70)
print(f"{'Debt/GDP (%)':<25} {data['Debt_GDP'].mean():>10.1f} {data['Debt_GDP'].std():>10.1f} {data['Debt_GDP'].min():>10.1f} {data['Debt_GDP'].max():>10.1f}")
print(f"{'Primary Balance/GDP (%)':<25} {data['PB_GDP'].mean():>10.2f} {data['PB_GDP'].std():>10.2f} {data['PB_GDP'].min():>10.2f} {data['PB_GDP'].max():>10.2f}")

# ============================================================
# BOHN REGRESSION ANALYSIS
# ============================================================

import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson

print("\n" + "=" * 70)
print("BOHN FISCAL REACTION FUNCTION TESTS")
print("=" * 70)

X = sm.add_constant(data['Debt_GDP_lag'])
y = data['PB_GDP']

# Basic OLS
model_ols = sm.OLS(y, X).fit()
dw = durbin_watson(model_ols.resid)

print("\n1. BASIC OLS REGRESSION: pb_t = α + β·d_{t-1} + ε_t")
print("-" * 70)
print(f"   Observations: N = {len(y)}")
print(f"   Constant (α): {model_ols.params[0]:.4f} (SE: {model_ols.bse[0]:.4f})")
print(f"   Debt coefficient (β): {model_ols.params[1]:.4f} (SE: {model_ols.bse[1]:.4f})")
print(f"   t-statistic: {model_ols.tvalues[1]:.3f}")
print(f"   p-value: {model_ols.pvalues[1]:.4f}")
print(f"   R²: {model_ols.rsquared:.4f}")
print(f"   Durbin-Watson: {dw:.3f}")

if model_ols.params[1] > 0:
    print(f"\n   INTERPRETATION: POSITIVE β ({model_ols.params[1]:.4f}) suggests fiscal sustainability")
    print("   Government tends to run larger primary surpluses when debt is higher")
else:
    print(f"\n   INTERPRETATION: NEGATIVE β ({model_ols.params[1]:.4f}) suggests fiscal non-sustainability")
    print("   Government does NOT systematically respond to higher debt with larger surpluses")

# Newey-West HAC
model_hac = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': 3})
print("\n2. NEWEY-WEST HAC STANDARD ERRORS (3 lags):")
print(f"   β = {model_hac.params[1]:.4f} (HAC SE: {model_hac.bse[1]:.4f})")
print(f"   t-stat = {model_hac.tvalues[1]:.3f}, p-value = {model_hac.pvalues[1]:.4f}")

# ============================================================
# UNIT ROOT TESTS
# ============================================================

from statsmodels.tsa.stattools import adfuller, kpss, coint

print("\n" + "=" * 70)
print("UNIT ROOT AND COINTEGRATION TESTS")
print("=" * 70)

def unit_root_summary(series, name):
    """Comprehensive unit root analysis"""
    clean = series.dropna()
    
    # ADF
    adf = adfuller(clean, autolag='AIC')
    
    # KPSS
    try:
        kpss_res = kpss(clean, regression='c', nlags='auto')
        kpss_stat, kpss_p = kpss_res[0], kpss_res[1]
    except:
        kpss_stat, kpss_p = np.nan, np.nan
    
    print(f"\n{name}:")
    print(f"  ADF statistic: {adf[0]:.3f} (p={adf[1]:.4f})")
    print(f"  KPSS statistic: {kpss_stat:.3f} (p~{kpss_p:.4f})")
    
    # Interpretation
    adf_stationary = adf[1] < 0.05
    kpss_nonstationary = kpss_p < 0.05
    
    if adf_stationary and not kpss_nonstationary:
        print("  → STATIONARY (I(0))")
        return 0
    elif not adf_stationary and kpss_nonstationary:
        print("  → NON-STATIONARY (I(1))")
        return 1
    else:
        print("  → INCONCLUSIVE (borderline)")
        return 0.5

order_debt = unit_root_summary(data['Debt_GDP'], "Debt/GDP")
order_pb = unit_root_summary(data['PB_GDP'], "Primary Balance/GDP")

# First differences
print("\nFirst Differences:")
order_d_debt = unit_root_summary(data['Debt_GDP'].diff(), "Δ(Debt/GDP)")
order_d_pb = unit_root_summary(data['PB_GDP'].diff(), "Δ(PB/GDP)")

# Cointegration test
print("\n" + "-" * 70)
print("ENGLE-GRANGER COINTEGRATION TEST")
coint_stat, coint_p, coint_crit = coint(data['PB_GDP'].dropna(), data['Debt_GDP_lag'].dropna())
print(f"Test statistic: {coint_stat:.3f}")
print(f"p-value: {coint_p:.4f}")
print(f"Critical values: 1%={coint_crit[0]:.3f}, 5%={coint_crit[1]:.3f}, 10%={coint_crit[2]:.3f}")

if coint_p < 0.05:
    print("\n→ COINTEGRATION DETECTED (p<0.05)")
    print("  The Bohn regression is VALID despite non-stationarity")
    print("  Primary balance and debt share a long-run equilibrium relationship")
else:
    print("\n→ NO COINTEGRATION DETECTED")
    print("  Bohn regression results should be interpreted cautiously")

# ============================================================
# STRUCTURAL BREAK TESTS
# ============================================================

print("\n" + "=" * 70)
print("STRUCTURAL BREAK ANALYSIS (CHOW TESTS)")
print("=" * 70)

def chow_test(data, break_year, y_col='PB_GDP', x_col='Debt_GDP_lag'):
    """Chow test for structural break"""
    d = data[[y_col, x_col, 'Year']].dropna()
    pre = d[d['Year'] < break_year]
    post = d[d['Year'] >= break_year]
    
    if len(pre) < 5 or len(post) < 5:
        return np.nan, np.nan, np.nan, np.nan
    
    # Regressions
    X_p = sm.add_constant(d[x_col])
    X_pre = sm.add_constant(pre[x_col])
    X_post = sm.add_constant(post[x_col])
    
    model_p = sm.OLS(d[y_col], X_p).fit()
    model_pre = sm.OLS(pre[y_col], X_pre).fit()
    model_post = sm.OLS(post[y_col], X_post).fit()
    
    RSS_p = sum(model_p.resid**2)
    RSS_pre = sum(model_pre.resid**2)
    RSS_post = sum(model_post.resid**2)
    
    k = 2
    n1, n2 = len(pre), len(post)
    
    F = ((RSS_p - RSS_pre - RSS_post) / k) / ((RSS_pre + RSS_post) / (n1 + n2 - 2*k))
    p = 1 - stats.f.cdf(F, k, n1 + n2 - 2*k)
    
    return F, p, model_pre.params[1], model_post.params[1]

print("\nKey structural break points:")
print("-" * 70)
print(f"{'Year':<10} {'Event':<30} {'F-stat':>10} {'p-value':>10} {'β_pre':>10} {'β_post':>10}")
print("-" * 70)

breaks = [
    (1997, "BoE Independence"),
    (2008, "Global Financial Crisis"),
    (2010, "Austerity Begins"),
    (2016, "Brexit Referendum"),
    (2020, "COVID-19 Pandemic")
]

sig_breaks = []
for year, event in breaks:
    F, p, b_pre, b_post = chow_test(data, year)
    if not np.isnan(F):
        sig = "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.1 else ""))
        print(f"{year:<10} {event:<30} {F:>10.2f} {p:>10.4f} {b_pre:>10.4f} {b_post:>10.4f} {sig}")
        if p < 0.05:
            sig_breaks.append(year)

if sig_breaks:
    print(f"\n→ SIGNIFICANT BREAKS DETECTED: {sig_breaks}")
    print("  Fiscal reaction has varied across policy regimes")
else:
    print("\n→ No significant structural breaks at 5% level")

# ============================================================
# SUB-SAMPLE ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("SUB-SAMPLE BOHN TEST RESULTS")
print("=" * 70)

subsamples = [
    ("Full sample", data),
    ("Pre-GFC (1993-2007)", data[(data['Year'] >= 1993) & (data['Year'] <= 2007)]),
    ("Post-GFC (2008-2024)", data[data['Year'] >= 2008]),
    ("Excluding COVID (excl 2020-21)", data[~data['Year'].isin([2020, 2021])]),
    ("Post-BoE independence (1998-2024)", data[data['Year'] >= 1998])
]

print(f"\n{'Sample':<40} {'N':>5} {'β':>10} {'SE':>10} {'t-stat':>10} {'p-val':>10}")
print("-" * 95)

for name, subset in subsamples:
    if len(subset) >= 8:
        X_s = sm.add_constant(subset['Debt_GDP_lag'])
        y_s = subset['PB_GDP']
        m_s = sm.OLS(y_s, X_s).fit()
        sig = "***" if m_s.pvalues[1] < 0.01 else ("**" if m_s.pvalues[1] < 0.05 else ("*" if m_s.pvalues[1] < 0.1 else ""))
        print(f"{name:<40} {len(subset):>5} {m_s.params[1]:>10.4f} {m_s.bse[1]:>10.4f} {m_s.tvalues[1]:>10.3f} {m_s.pvalues[1]:>10.4f} {sig}")

# ============================================================
# DISTRIBUTION AND CORRELATION ESTIMATION
# ============================================================

print("\n" + "=" * 70)
print("DISTRIBUTION AND CORRELATION ESTIMATION")
print("=" * 70)

# Load additional data for correlations
rpi_raw = pd.read_csv('/mnt/project/Retail_Prices_Index_RPI_level.csv', skiprows=8)
rpi_raw.columns = ['Date', 'RPI']
rpi_raw['RPI'] = pd.to_numeric(rpi_raw['RPI'], errors='coerce')
rpi_raw['Year'] = rpi_raw['Date'].str.extract(r'(\d{4})').astype(float)
# Get annual (last value)
rpi_annual = rpi_raw.groupby('Year')['RPI'].last().reset_index()
rpi_annual['Inflation'] = rpi_annual['RPI'].pct_change() * 100
rpi_annual['Year'] = rpi_annual['Year'].astype(int)

rate_raw = pd.read_csv('/mnt/project/Official_Bank_Rate__EndMonth.csv', skiprows=8)
rate_raw.columns = ['Date', 'Rate']
rate_raw['Rate'] = pd.to_numeric(rate_raw['Rate'], errors='coerce')
rate_raw['Year'] = rate_raw['Date'].str.extract(r'(\d{4})').astype(float)
rate_annual = rate_raw.groupby('Year')['Rate'].last().reset_index()
rate_annual['Year'] = rate_annual['Year'].astype(int)
rate_annual['Rate_change'] = rate_annual['Rate'].diff()

# Calculate GDP growth
gdp_annual['GDP_growth'] = gdp_annual['GDP_m'].pct_change() * 100

# Merge for correlation
corr_data = gdp_annual[['Year', 'GDP_growth']].merge(
    rpi_annual[['Year', 'Inflation']], on='Year'
).merge(
    rate_annual[['Year', 'Rate_change']], on='Year'
).dropna()

print(f"\nCorrelation data: {corr_data['Year'].min()}-{corr_data['Year'].max()}, N={len(corr_data)}")

corr_matrix = corr_data[['GDP_growth', 'Inflation', 'Rate_change']].corr()
print("\nEstimated Correlation Matrix:")
print(corr_matrix.round(3).to_string())

print("\nCompared to paper's assumed correlations:")
print(f"  GDP-Inflation: Paper=0.20, Estimated={corr_matrix.loc['GDP_growth', 'Inflation']:.3f}")
print(f"  GDP-Rates: Paper=-0.30, Estimated={corr_matrix.loc['GDP_growth', 'Rate_change']:.3f}")
print(f"  Inflation-Rates: Paper=0.40, Estimated={corr_matrix.loc['Inflation', 'Rate_change']:.3f}")

# MLE for t-distribution
def estimate_t_df(data, name):
    """Estimate t-distribution df via MLE"""
    from scipy.optimize import minimize_scalar
    clean = data.dropna().values
    if len(clean) < 10:
        return 5.0
    
    z = (clean - np.mean(clean)) / np.std(clean)
    
    def neg_ll(df):
        if df <= 2.1:
            return 1e10
        return -np.sum(stats.t.logpdf(z, df=df))
    
    result = minimize_scalar(neg_ll, bounds=(2.5, 50), method='bounded')
    
    jb_stat, jb_p = stats.jarque_bera(clean)
    kurt = stats.kurtosis(clean)
    
    print(f"\n{name}:")
    print(f"  N={len(clean)}, Mean={np.mean(clean):.2f}, SD={np.std(clean):.2f}")
    print(f"  Skewness={stats.skew(clean):.3f}, Excess Kurtosis={kurt:.3f}")
    print(f"  Jarque-Bera p={jb_p:.4f} {'(reject normality)' if jb_p < 0.05 else ''}")
    print(f"  MLE t-distribution df: {result.x:.2f}")
    
    return result.x

print("\nDistribution Parameter Estimation:")
df_gdp = estimate_t_df(corr_data['GDP_growth'], "GDP Growth")
df_inf = estimate_t_df(corr_data['Inflation'], "Inflation")  
df_rate = estimate_t_df(corr_data['Rate_change'], "Interest Rate Changes")

# ============================================================
# MONTE CARLO WITH FISCAL REACTION
# ============================================================

print("\n" + "=" * 70)
print("MONTE CARLO SIMULATION COMPARISON")
print("=" * 70)

def monte_carlo(n_sims=10000, n_years=10, initial_debt=96.0,
                beta_fiscal=None, df_g=5, df_i=5, df_r=7):
    """Monte Carlo with optional fiscal reaction"""
    np.random.seed(42)
    
    sigma = [3.0, 2.0, 1.5]  # volatilities
    corr = np.array([[1, 0.3, -0.2], [0.3, 1, 0.4], [-0.2, 0.4, 1]])
    
    # OBR-style baseline path
    pb_base = np.linspace(-2.0, 1.5, n_years)
    
    paths = np.zeros((n_sims, n_years + 1))
    paths[:, 0] = initial_debt
    
    for sim in range(n_sims):
        debt = initial_debt
        for t in range(n_years):
            z = np.random.multivariate_normal([0,0,0], corr)
            u = stats.norm.cdf(z)
            
            s_g = stats.t.ppf(np.clip(u[0], 0.001, 0.999), df=df_g) * sigma[0]
            s_i = stats.t.ppf(np.clip(u[1], 0.001, 0.999), df=df_i) * sigma[1]
            s_r = stats.t.ppf(np.clip(u[2], 0.001, 0.999), df=df_r) * sigma[2]
            
            gdp_g = 1.5 + s_g
            inflation = 2.0 + s_i
            nom_g = gdp_g + inflation
            
            eff_rate = 4.5 + s_r * 0.5 + inflation * 0.34
            
            if beta_fiscal is not None:
                pb = pb_base[t] + beta_fiscal * (debt - 60)
            else:
                pb = pb_base[t] + s_g * 0.35
            
            if nom_g > -20:
                debt = debt * (1 + eff_rate/100) / (1 + nom_g/100) - pb
            
            paths[sim, t+1] = np.clip(debt, 0, 300)
    
    return paths

beta_est = model_hac.params[1]

print("\nRunning simulations...")

scenarios = [
    ("Exogenous PB (original)", None),
    (f"With estimated reaction (β={beta_est:.3f})", beta_est),
    ("With weak sustainable reaction (β=0.02)", 0.02),
    ("With strong sustainable reaction (β=0.05)", 0.05)
]

print(f"\n{'Scenario':<45} {'Mean':>8} {'Med':>8} {'SD':>8} {'VaR99':>8} {'P>100%':>8}")
print("-" * 95)

for name, beta in scenarios:
    paths = monte_carlo(beta_fiscal=beta, df_g=df_gdp, df_i=df_inf, df_r=df_rate)
    term = paths[:, -1]
    print(f"{name:<45} {np.mean(term):>8.1f} {np.median(term):>8.1f} {np.std(term):>8.1f} {np.percentile(term, 99):>8.1f} {100*np.mean(term>100):>7.1f}%")

# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("SUMMARY OF KEY FINDINGS")
print("=" * 70)

beta_sign = "POSITIVE" if model_hac.params[1] > 0 else "NEGATIVE"
beta_sig = "significant" if model_hac.pvalues[1] < 0.05 else "not significant"
coint_result = "YES" if coint_p < 0.05 else "NO"

print(f"""
1. BOHN TEST RESULT:
   - Estimated β = {model_hac.params[1]:.4f} ({beta_sign}, {beta_sig} at 5%)
   - This {'SUPPORTS' if model_hac.params[1] > 0 else 'DOES NOT SUPPORT'} fiscal sustainability
   
2. UNIT ROOT / COINTEGRATION:
   - Debt/GDP: {'Non-stationary (I(1))' if order_debt >= 0.5 else 'Stationary/borderline'}
   - Primary Balance: {'Non-stationary (I(1))' if order_pb >= 0.5 else 'Stationary/borderline'}
   - Cointegration detected: {coint_result}
   - Bohn regression {'IS' if coint_p < 0.05 else 'may NOT be'} valid despite non-stationarity
   
3. STRUCTURAL BREAKS:
   - Significant breaks at: {sig_breaks if sig_breaks else 'None detected'}
   - Fiscal behavior has {'VARIED' if sig_breaks else 'remained stable'} across policy regimes
   
4. DISTRIBUTION PARAMETERS (MLE estimated):
   - GDP shocks: df = {df_gdp:.1f}
   - Inflation shocks: df = {df_inf:.1f}
   - Rate shocks: df = {df_rate:.1f}
   
5. CORRELATION MATRIX:
   - Estimated from UK historical data (see above)
   - Paper assumptions were {'reasonable' if abs(corr_matrix.loc['GDP_growth', 'Inflation'] - 0.2) < 0.2 else 'different from historical'}
""")

print("=" * 70)
