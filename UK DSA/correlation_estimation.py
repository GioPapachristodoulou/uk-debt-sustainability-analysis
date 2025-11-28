"""
Final Correlation Estimation and Summary
"""

import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("CORRELATION AND DISTRIBUTION ESTIMATION")
print("=" * 70)

# Load and process data carefully
# GDP growth
gdp_raw = pd.read_csv('/mnt/project/Gross_Domestic_Product_at_market_prices_NSA.csv', skiprows=8)
gdp_raw.columns = ['Date', 'GDP_m']
gdp_raw['GDP_m'] = pd.to_numeric(gdp_raw['GDP_m'], errors='coerce')

# Get annual totals
gdp_raw['Year'] = gdp_raw['Date'].str.extract(r'^(\d{4})')
gdp_raw = gdp_raw.dropna(subset=['Year', 'GDP_m'])
gdp_raw['Year'] = gdp_raw['Year'].astype(int)
gdp_annual = gdp_raw.groupby('Year')['GDP_m'].sum().reset_index()
gdp_annual['GDP_growth'] = gdp_annual['GDP_m'].pct_change() * 100

# RPI Inflation
rpi_raw = pd.read_csv('/mnt/project/Retail_Prices_Index_RPI_level.csv', skiprows=8)
rpi_raw.columns = ['Date', 'RPI']
rpi_raw['RPI'] = pd.to_numeric(rpi_raw['RPI'], errors='coerce')

# Get annual (December or last available)
rpi_raw['Year'] = rpi_raw['Date'].str.extract(r'(\d{4})')
rpi_raw = rpi_raw.dropna(subset=['Year', 'RPI'])
rpi_raw['Year'] = rpi_raw['Year'].astype(int)
rpi_annual = rpi_raw.groupby('Year')['RPI'].last().reset_index()
rpi_annual['Inflation'] = rpi_annual['RPI'].pct_change() * 100

# Bank Rate
rate_raw = pd.read_csv('/mnt/project/Official_Bank_Rate__EndMonth.csv', skiprows=8)
rate_raw.columns = ['Date', 'Rate']
rate_raw['Rate'] = pd.to_numeric(rate_raw['Rate'], errors='coerce')
rate_raw['Year'] = rate_raw['Date'].str.extract(r'(\d{4})')
rate_raw = rate_raw.dropna(subset=['Year', 'Rate'])
rate_raw['Year'] = rate_raw['Year'].astype(int)
rate_annual = rate_raw.groupby('Year')['Rate'].last().reset_index()
rate_annual['Rate_change'] = rate_annual['Rate'].diff()

# Merge
data = gdp_annual[['Year', 'GDP_growth']].merge(
    rpi_annual[['Year', 'Inflation']], on='Year', how='inner'
).merge(
    rate_annual[['Year', 'Rate_change']], on='Year', how='inner'
)
data = data.dropna()

print(f"\nData range: {data['Year'].min()}-{data['Year'].max()}, N={len(data)}")

# Correlation matrix
corr = data[['GDP_growth', 'Inflation', 'Rate_change']].corr()
print("\nESTIMATED CORRELATION MATRIX (from UK historical data):")
print("-" * 60)
print(corr.round(3).to_string())

print("\n\nCOMPARISON WITH PAPER'S ASSUMED VALUES:")
print("-" * 60)
print(f"{'Pair':<25} {'Paper':>12} {'Estimated':>12} {'Difference':>12}")
print("-" * 60)
print(f"{'GDP-Inflation':<25} {'0.20':>12} {corr.loc['GDP_growth', 'Inflation']:>12.3f} {corr.loc['GDP_growth', 'Inflation'] - 0.20:>12.3f}")
print(f"{'GDP-Rates':<25} {'-0.30':>12} {corr.loc['GDP_growth', 'Rate_change']:>12.3f} {corr.loc['GDP_growth', 'Rate_change'] - (-0.30):>12.3f}")
print(f"{'Inflation-Rates':<25} {'0.40':>12} {corr.loc['Inflation', 'Rate_change']:>12.3f} {corr.loc['Inflation', 'Rate_change'] - 0.40:>12.3f}")

# Distribution estimation
print("\n\nDISTRIBUTION PARAMETER ESTIMATION (MLE for t-distribution):")
print("-" * 60)

from scipy.optimize import minimize_scalar

def estimate_df(series, name):
    clean = series.dropna().values
    z = (clean - np.mean(clean)) / np.std(clean)
    
    def neg_ll(df):
        if df <= 2.1: return 1e10
        return -np.sum(stats.t.logpdf(z, df=df))
    
    result = minimize_scalar(neg_ll, bounds=(2.5, 100), method='bounded')
    df = result.x
    
    # Normality test
    jb, jb_p = stats.jarque_bera(clean)
    kurt = stats.kurtosis(clean)
    
    print(f"\n{name}:")
    print(f"  N={len(clean)}, Mean={np.mean(clean):.2f}, SD={np.std(clean):.2f}")
    print(f"  Skewness={stats.skew(clean):.3f}, Excess Kurtosis={kurt:.3f}")
    print(f"  Jarque-Bera: stat={jb:.2f}, p={jb_p:.4f}")
    print(f"  MLE t-distribution df: {df:.1f}")
    
    if df > 30:
        print(f"  → Distribution is approximately NORMAL (df>30)")
    elif df < 10:
        print(f"  → Distribution has FAT TAILS (df<10)")
    else:
        print(f"  → Distribution has moderate tails")
    
    return df

df_gdp = estimate_df(data['GDP_growth'], "GDP Growth")
df_inf = estimate_df(data['Inflation'], "Inflation")
df_rate = estimate_df(data['Rate_change'], "Interest Rate Changes")

print("\n\n" + "=" * 70)
print("PARAMETER COMPARISON: PAPER vs ESTIMATED")
print("=" * 70)
print(f"\n{'Parameter':<30} {'Paper':>15} {'Estimated':>15} {'Verdict':>20}")
print("-" * 80)
print(f"{'GDP shock df':<30} {'5':>15} {df_gdp:>15.1f} {'OK - close' if abs(df_gdp-5)<3 else 'REVISE'}")
print(f"{'Inflation shock df':<30} {'5':>15} {df_inf:>15.1f} {'OK - close' if abs(df_inf-5)<3 else 'REVISE'}")
print(f"{'Rate shock df':<30} {'7':>15} {df_rate:>15.1f} {'OK - close' if abs(df_rate-7)<3 else 'REVISE'}")
print(f"{'Corr(GDP, Inflation)':<30} {'0.20':>15} {corr.loc['GDP_growth', 'Inflation']:>15.3f} {'OK' if abs(corr.loc['GDP_growth', 'Inflation']-0.2)<0.15 else 'REVISE'}")
print(f"{'Corr(GDP, Rates)':<30} {'-0.30':>15} {corr.loc['GDP_growth', 'Rate_change']:>15.3f} {'REVISE'}")
print(f"{'Corr(Inflation, Rates)':<30} {'0.40':>15} {corr.loc['Inflation', 'Rate_change']:>15.3f} {'OK' if abs(corr.loc['Inflation', 'Rate_change']-0.4)<0.15 else 'REVISE'}")

print("\n" + "=" * 70)
