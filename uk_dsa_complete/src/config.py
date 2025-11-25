"""
UK Debt Sustainability Analysis - Comprehensive Configuration
=============================================================
Imperial College London UROP Project

This module contains:
1. Verified historical data (1997-2024) from ONS/DMO/BoE
2. OBR March 2025 baseline forecasts (2024-2030)
3. Extended projections (2030-2035)
4. Debt composition data
5. Ready reckoners for sensitivity analysis
6. Scenario definitions
7. Monte Carlo parameters

Data Sources (all verified):
- ONS Public Sector Finances (PSND, PSNB, GDP) 
- DMO Gilt Market Data (gilts in issue, yields)
- Bank of England IADS (Bank Rate, exchange rates)
- OBR Economic and Fiscal Outlook, March 2025
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional


# ==============================================================================
# SECTION 1: HISTORICAL DATA (1997-2024) - Verified from primary sources
# ==============================================================================

# Public Sector Net Debt (£bn, end of fiscal year - March)
# Source: ONS PUSF series HF6W
HISTORICAL_PSND = {
    '1997-98': 351.3, '1998-99': 350.3, '1999-00': 346.4, '2000-01': 320.2,
    '2001-02': 316.8, '2002-03': 335.6, '2003-04': 374.8, '2004-05': 424.7,
    '2005-06': 463.3, '2006-07': 498.6, '2007-08': 536.9, '2008-09': 660.7,
    '2009-10': 1027.9, '2010-11': 1168.7, '2011-12': 1261.1, '2012-13': 1366.2,
    '2013-14': 1461.1, '2014-15': 1551.9, '2015-16': 1595.0, '2016-17': 1714.5,
    '2017-18': 1757.8, '2018-19': 1776.0, '2019-20': 1815.8, '2020-21': 2155.1,
    '2021-22': 2380.9, '2022-23': 2545.5, '2023-24': 2685.8,
}

# Nominal GDP (£bn, fiscal year total Q2+Q3+Q4+Q1)
# Source: ONS series BKTL
HISTORICAL_GDP = {
    '1997-98': 855.3, '1998-99': 902.8, '1999-00': 956.9, '2000-01': 1001.3,
    '2001-02': 1040.3, '2002-03': 1098.5, '2003-04': 1168.9, '2004-05': 1233.8,
    '2005-06': 1307.7, '2006-07': 1394.3, '2007-08': 1466.9, '2008-09': 1508.4,
    '2009-10': 1540.8, '2010-11': 1585.8, '2011-12': 1657.5, '2012-13': 1719.9,
    '2013-14': 1805.2, '2014-15': 1879.5, '2015-16': 1943.8, '2016-17': 2027.1,
    '2017-18': 2114.6, '2018-19': 2189.8, '2019-20': 2263.4, '2020-21': 2110.0,
    '2021-22': 2400.4, '2022-23': 2634.3, '2023-24': 2789.0,
}

# Debt-to-GDP Ratio (%, calculated)
HISTORICAL_DEBT_TO_GDP = {
    fy: HISTORICAL_PSND[fy] / HISTORICAL_GDP[fy] * 100 
    for fy in HISTORICAL_PSND
}

# Public Sector Net Borrowing (£bn, fiscal year)
# Source: ONS series J5II
HISTORICAL_PSNB = {
    '1997-98': -3.4, '1998-99': -1.1, '1999-00': -15.8, '2000-01': -17.0,
    '2001-02': 1.3, '2002-03': 28.5, '2003-04': 38.4, '2004-05': 43.9,
    '2005-06': 37.6, '2006-07': 33.6, '2007-08': 36.6, '2008-09': 102.0,
    '2009-10': 156.4, '2010-11': 136.1, '2011-12': 119.2, '2012-13': 119.9,
    '2013-14': 99.4, '2014-15': 90.2, '2015-16': 72.7, '2016-17': 46.0,
    '2017-18': 41.6, '2018-19': 40.3, '2019-20': 56.5, '2020-21': 317.6,
    '2021-22': 129.7, '2022-23': 127.0, '2023-24': 122.1,
}

# Central Government Debt Interest (£bn, fiscal year)
# Source: ONS series JW2P
HISTORICAL_DEBT_INTEREST = {
    '2009-10': 30.9, '2010-11': 43.3, '2011-12': 48.3, '2012-13': 47.4,
    '2013-14': 47.1, '2014-15': 45.3, '2015-16': 39.1, '2016-17': 40.3,
    '2017-18': 43.6, '2018-19': 47.2, '2019-20': 47.8, '2020-21': 39.1,
    '2021-22': 64.7, '2022-23': 110.5, '2023-24': 94.4,
}

# 10-Year Zero-Coupon Gilt Yields (%, fiscal year average)
# Source: BoE series IUMMNZC
HISTORICAL_GILT_YIELD_10Y = {
    '1997-98': 6.85, '1998-99': 5.43, '1999-00': 5.35, '2000-01': 5.18,
    '2001-02': 5.01, '2002-03': 4.58, '2003-04': 4.79, '2004-05': 4.69,
    '2005-06': 4.26, '2006-07': 4.65, '2007-08': 4.88, '2008-09': 4.05,
    '2009-10': 3.89, '2010-11': 3.49, '2011-12': 2.53, '2012-13': 1.93,
    '2013-14': 2.75, '2014-15': 2.28, '2015-16': 1.80, '2016-17': 1.22,
    '2017-18': 1.33, '2018-19': 1.43, '2019-20': 0.84, '2020-21': 0.37,
    '2021-22': 1.02, '2022-23': 3.04, '2023-24': 4.09, '2024-25': 4.30,
}

# Real Yields (10-year, %)
# Source: BoE series IUMMRZC
HISTORICAL_REAL_YIELD_10Y = {
    '2009-10': 0.75, '2010-11': 0.56, '2011-12': -0.38, '2012-13': -0.89,
    '2013-14': -0.15, '2014-15': -0.53, '2015-16': -0.93, '2016-17': -1.58,
    '2017-18': -1.64, '2018-19': -1.68, '2019-20': -2.00, '2020-21': -2.52,
    '2021-22': -2.37, '2022-23': -0.12, '2023-24': 0.89, '2024-25': 1.05,
}

# Bank Rate (%, fiscal year average)
# Source: BoE series IUMBEDR
HISTORICAL_BANK_RATE = {
    '1997-98': 7.00, '1998-99': 6.50, '1999-00': 5.40, '2000-01': 6.00,
    '2001-02': 4.80, '2002-03': 4.00, '2003-04': 3.70, '2004-05': 4.40,
    '2005-06': 4.50, '2006-07': 4.80, '2007-08': 5.50, '2008-09': 3.00,
    '2009-10': 0.50, '2010-11': 0.50, '2011-12': 0.50, '2012-13': 0.50,
    '2013-14': 0.50, '2014-15': 0.50, '2015-16': 0.50, '2016-17': 0.30,
    '2017-18': 0.40, '2018-19': 0.70, '2019-20': 0.60, '2020-21': 0.10,
    '2021-22': 0.30, '2022-23': 2.80, '2023-24': 5.10, '2024-25': 4.75,
}

# RPI Inflation (%, annual)
# Source: ONS series CZBH
HISTORICAL_RPI = {
    '1997-98': 3.4, '1998-99': 2.4, '1999-00': 2.1, '2000-01': 2.5,
    '2001-02': 1.3, '2002-03': 2.2, '2003-04': 2.8, '2004-05': 3.0,
    '2005-06': 2.5, '2006-07': 4.0, '2007-08': 4.2, '2008-09': 0.0,
    '2009-10': 4.4, '2010-11': 5.0, '2011-12': 3.6, '2012-13': 3.0,
    '2013-14': 2.5, '2014-15': 1.6, '2015-16': 1.3, '2016-17': 2.5,
    '2017-18': 3.4, '2018-19': 2.6, '2019-20': 2.4, '2020-21': 1.2,
    '2021-22': 7.4, '2022-23': 11.6, '2023-24': 5.3, '2024-25': 3.6,
}

# CPI Inflation (%, annual)  
# Source: ONS series D7G7
HISTORICAL_CPI = {
    '2009-10': 3.1, '2010-11': 4.1, '2011-12': 3.4, '2012-13': 2.6,
    '2013-14': 2.0, '2014-15': 0.9, '2015-16': 0.3, '2016-17': 1.5,
    '2017-18': 2.7, '2018-19': 2.1, '2019-20': 1.7, '2020-21': 0.9,
    '2021-22': 5.4, '2022-23': 9.1, '2023-24': 4.2, '2024-25': 2.6,
}


# ==============================================================================
# SECTION 2: OBR MARCH 2025 BASELINE FORECASTS
# ==============================================================================

FORECAST_YEARS = ['2024-25', '2025-26', '2026-27', '2027-28', '2028-29', '2029-30']

# Nominal GDP (£bn)
OBR_GDP = {
    '2024-25': 2864.0, '2025-26': 3001.7, '2026-27': 3113.0,
    '2027-28': 3222.8, '2028-29': 3336.5, '2029-30': 3456.8,
}

# Real GDP Growth (%)
OBR_REAL_GDP_GROWTH = {
    '2024-25': 0.9, '2025-26': 1.0, '2026-27': 1.9,
    '2027-28': 1.8, '2028-29': 1.7, '2029-30': 1.8,
}

# Nominal GDP Growth (%)
OBR_NOMINAL_GDP_GROWTH = {
    '2024-25': 4.2, '2025-26': 4.8, '2026-27': 3.7,
    '2027-28': 3.5, '2028-29': 3.5, '2029-30': 3.6,
}

# Public Sector Net Debt (£bn)
OBR_PSND = {
    '2024-25': 2746.3, '2025-26': 2854.2, '2026-27': 2981.3,
    '2027-28': 3098.1, '2028-29': 3211.5, '2029-30': 3322.4,
}

# PSND as % of GDP
OBR_DEBT_TO_GDP = {
    '2024-25': 95.9, '2025-26': 95.1, '2026-27': 95.8,
    '2027-28': 96.1, '2028-29': 96.3, '2029-30': 96.1,
}

# Public Sector Net Borrowing (£bn)
OBR_PSNB = {
    '2024-25': 137.3, '2025-26': 117.7, '2026-27': 97.2,
    '2027-28': 80.2, '2028-29': 77.4, '2029-30': 74.0,
}

# PSNB as % of GDP
OBR_DEFICIT_TO_GDP = {
    '2024-25': 4.8, '2025-26': 3.9, '2026-27': 3.1,
    '2027-28': 2.5, '2028-29': 2.3, '2029-30': 2.1,
}

# Debt Interest (£bn, net of APF)
OBR_DEBT_INTEREST = {
    '2024-25': 105.2, '2025-26': 111.2, '2026-27': 111.4,
    '2027-28': 117.9, '2028-29': 124.2, '2029-30': 131.6,
}

# Debt Interest as % of GDP
OBR_DEBT_INTEREST_GDP = {
    '2024-25': 3.67, '2025-26': 3.70, '2026-27': 3.58,
    '2027-28': 3.66, '2028-29': 3.72, '2029-30': 3.81,
}

# Total Receipts (£bn)
OBR_RECEIPTS = {
    '2024-25': 1194.6, '2025-26': 1272.7, '2026-27': 1324.5,
    '2027-28': 1377.5, '2028-29': 1431.7, '2029-30': 1489.9,
}

# Total Managed Expenditure (£bn)
OBR_TME = {
    '2024-25': 1331.9, '2025-26': 1390.4, '2026-27': 1421.7,
    '2027-28': 1457.7, '2028-29': 1509.1, '2029-30': 1563.9,
}

# RPI Inflation (%, annual average)
OBR_RPI = {
    '2024-25': 3.6, '2025-26': 4.0, '2026-27': 3.1,
    '2027-28': 2.9, '2028-29': 2.8, '2029-30': 2.8,
}

# CPI Inflation (%, annual average)
OBR_CPI = {
    '2024-25': 2.6, '2025-26': 3.2, '2026-27': 2.4,
    '2027-28': 2.1, '2028-29': 2.0, '2029-30': 2.0,
}

# Gilt Yields Assumption (10-year, %)
OBR_GILT_YIELD = {
    '2024-25': 4.4, '2025-26': 4.5, '2026-27': 4.4,
    '2027-28': 4.3, '2028-29': 4.2, '2029-30': 4.1,
}

# Bank Rate Assumption (%)
OBR_BANK_RATE = {
    '2024-25': 4.50, '2025-26': 4.00, '2026-27': 3.50,
    '2027-28': 3.25, '2028-29': 3.00, '2029-30': 3.00,
}


# ==============================================================================
# SECTION 3: DEBT COMPOSITION (November 2025, from DMO)
# ==============================================================================

# Total gilts outstanding (£bn nominal, including ILG uplift)
DEBT_COMPOSITION = {
    # Conventional Gilts
    'conventional_ultra_short': 190.0,   # <3 years
    'conventional_short': 235.0,         # 3-7 years
    'conventional_medium': 420.0,        # 7-15 years
    'conventional_long': 515.0,          # >15 years
    'conventional_total': 1360.0,
    
    # Index-Linked Gilts (with inflation uplift)
    'ilg_short': 98.0,                   # <10 years
    'ilg_medium': 195.0,                 # 10-25 years
    'ilg_long': 393.0,                   # >25 years
    'ilg_total': 686.0,
    
    # Other Debt
    'ns_and_i': 234.1,
    'treasury_bills': 45.0,
    'other': 50.0,
    
    # Totals
    'total_gilts': 2046.0,
    'total_marketable': 2091.0,
    'psnd_end_sept_2025': 2916.1,
}

# Key ratios
ILG_SHARE = DEBT_COMPOSITION['ilg_total'] / DEBT_COMPOSITION['total_gilts']  # ~33.5%
CONVENTIONAL_SHARE = 1 - ILG_SHARE

# Average maturity (years, approximate)
AVERAGE_MATURITY = {
    'conventional': 14.5,
    'ilg': 17.8,
    'total': 15.6,
}

# Redemption profile (£bn nominal due each year)
REDEMPTIONS = {
    '2025-26': 95.0,
    '2026-27': 110.0,
    '2027-28': 120.0,
    '2028-29': 105.0,
    '2029-30': 115.0,
}


# ==============================================================================
# SECTION 4: OBR READY RECKONERS (Debt Interest Sensitivities)
# ==============================================================================

# Impact on debt interest (£bn) of 1 percentage point sustained increase
# Source: OBR March 2025, Table 5.1

READY_RECKONERS = {
    # +1pp in gilt rates from start of 2024-25
    'gilt_rates_1pp': {
        '2024-25': 0.39,
        '2025-26': 3.50,
        '2026-27': 5.97,
        '2027-28': 8.24,
        '2028-29': 10.31,
        '2029-30': 12.37,
    },
    
    # +1pp in short-term interest rates (Bank Rate, T-bills, NS&I)
    'short_rates_1pp': {
        '2024-25': 2.38,
        '2025-26': 3.26,
        '2026-27': 3.72,
        '2027-28': 4.11,
        '2028-29': 4.41,
        '2029-30': 4.66,
    },
    
    # +1pp in RPI inflation (affects index-linked gilts)
    'inflation_1pp': {
        '2024-25': 2.07,
        '2025-26': 5.87,
        '2026-27': 7.18,
        '2027-28': 8.28,
        '2028-29': 9.45,
        '2029-30': 10.99,
    },
    
    # +£10bn in cash borrowing requirement
    'cgncr_10bn': {
        '2024-25': 0.10,
        '2025-26': 0.52,
        '2026-27': 0.88,
        '2027-28': 1.31,
        '2028-29': 1.69,
        '2029-30': 2.09,
    },
}


# ==============================================================================
# SECTION 5: SCENARIO DEFINITIONS
# ==============================================================================

@dataclass
class Scenario:
    """Definition of a stress scenario."""
    name: str
    description: str
    gilt_yield_shock: float = 0.0       # Percentage points
    short_rate_shock: float = 0.0       # Percentage points
    inflation_shock: float = 0.0        # Percentage points (RPI)
    growth_shock: float = 0.0           # Percentage points (real GDP)
    primary_balance_shock: float = 0.0  # % of GDP
    duration: int = 5                   # Years shock persists
    probability: float = 0.0            # Subjective probability


SCENARIOS = {
    'baseline': Scenario(
        name='Baseline',
        description='OBR March 2025 central forecast',
        probability=0.50,
    ),
    
    'adverse_rates': Scenario(
        name='Adverse Interest Rates',
        description='Sustained 200bp increase in gilt yields',
        gilt_yield_shock=2.0,
        short_rate_shock=1.5,
        growth_shock=-0.3,
        duration=5,
        probability=0.15,
    ),
    
    'high_inflation': Scenario(
        name='High Inflation',
        description='Persistent above-target inflation (+3pp RPI)',
        gilt_yield_shock=0.5,
        short_rate_shock=1.0,
        inflation_shock=3.0,
        growth_shock=-0.5,
        duration=3,
        probability=0.10,
    ),
    
    'recession': Scenario(
        name='Recession',
        description='Significant economic downturn',
        gilt_yield_shock=0.5,
        short_rate_shock=-1.5,
        inflation_shock=-1.0,
        growth_shock=-3.0,
        primary_balance_shock=2.0,  # Automatic stabilizers
        duration=2,
        probability=0.10,
    ),
    
    'stagflation': Scenario(
        name='Stagflation',
        description='High inflation combined with recession',
        gilt_yield_shock=2.5,
        short_rate_shock=2.0,
        inflation_shock=4.0,
        growth_shock=-2.0,
        primary_balance_shock=1.5,
        duration=3,
        probability=0.05,
    ),
    
    'fiscal_consolidation': Scenario(
        name='Fiscal Consolidation',
        description='Proactive 1% GDP primary surplus improvement',
        gilt_yield_shock=-0.5,
        primary_balance_shock=-1.0,  # Negative = surplus improvement
        growth_shock=0.2,
        duration=5,
        probability=0.10,
    ),
    
    'combined_adverse': Scenario(
        name='Combined Adverse',
        description='Multiple simultaneous adverse shocks',
        gilt_yield_shock=2.0,
        short_rate_shock=1.5,
        inflation_shock=2.5,
        growth_shock=-1.5,
        primary_balance_shock=1.5,
        duration=4,
        probability=0.03,
    ),
    
    'mini_budget_redux': Scenario(
        name='Mini Budget Redux',
        description='September 2022-style market stress',
        gilt_yield_shock=3.0,
        short_rate_shock=2.5,
        inflation_shock=2.0,
        growth_shock=-1.0,
        duration=2,
        probability=0.02,
    ),
}


# ==============================================================================
# SECTION 6: MONTE CARLO PARAMETERS
# ==============================================================================

@dataclass
class MonteCarloConfig:
    """Configuration for Monte Carlo simulation."""
    n_simulations: int = 10000
    horizon_years: int = 10
    random_seed: int = 42
    
    # GDP growth shock parameters
    real_gdp_growth_mean: float = 0.015      # 1.5% average
    real_gdp_growth_std: float = 0.020       # 2.0% std dev
    real_gdp_ar1: float = 0.3                # Persistence
    
    # Inflation parameters
    cpi_inflation_mean: float = 0.020        # 2.0% target
    cpi_inflation_std: float = 0.015         # 1.5% std dev
    cpi_ar1: float = 0.5                     # Persistence
    
    rpi_inflation_mean: float = 0.030        # ~1pp wedge over CPI
    rpi_inflation_std: float = 0.018         # 1.8% std dev
    rpi_ar1: float = 0.5
    
    # Interest rate parameters
    gilt_yield_mean: float = 0.045           # 4.5% average
    gilt_yield_std: float = 0.012            # 1.2% std dev
    gilt_ar1: float = 0.85                   # High persistence
    gilt_floor: float = 0.005                # 0.5% floor
    
    # Correlation matrix [GDP growth, CPI, RPI, Gilt yields]
    correlation_matrix: np.ndarray = field(default_factory=lambda: np.array([
        [1.00, 0.20, 0.20, -0.30],   # GDP growth
        [0.20, 1.00, 0.85, 0.40],    # CPI
        [0.20, 0.85, 1.00, 0.45],    # RPI
        [-0.30, 0.40, 0.45, 1.00],   # Gilt yields
    ]))
    
    # Fan chart percentiles
    percentiles: List[int] = field(default_factory=lambda: [5, 10, 25, 50, 75, 90, 95])


MONTE_CARLO_CONFIG = MonteCarloConfig()


# ==============================================================================
# SECTION 7: FISCAL RULES AND SUSTAINABILITY THRESHOLDS
# ==============================================================================

FISCAL_RULES = {
    # Current UK fiscal rules (as of March 2025)
    'debt_target': 100.0,           # Debt/GDP must be falling by year 5
    'deficit_ceiling': 3.0,         # Current budget in balance by year 5
    'welfare_cap': True,            # Welfare spending cap
    
    # Sustainability thresholds (for analysis)
    'debt_warning': 100.0,          # Amber warning threshold
    'debt_critical': 120.0,         # Critical threshold
    'interest_gdp_warning': 4.0,    # Interest/GDP warning
    'interest_gdp_critical': 5.0,   # Interest/GDP critical
    'interest_revenue_warning': 10.0,  # Interest/Revenue warning
    'interest_revenue_critical': 12.0, # Interest/Revenue critical
    
    # Historical benchmarks
    'post_ww2_peak': 270.0,         # 1946 peak
    'pre_gfc_level': 37.0,          # 2007-08 level
    'pre_covid_level': 80.0,        # 2019-20 level
    'covid_peak': 102.0,            # 2020-21 peak
}


# ==============================================================================
# SECTION 8: EXTENDED PROJECTIONS (2030-2035)
# ==============================================================================

# Simple extrapolation beyond OBR horizon
EXTENDED_YEARS = ['2030-31', '2031-32', '2032-33', '2033-34', '2034-35']

EXTENDED_PROJECTIONS = {
    # Assumptions for extension period
    'real_gdp_growth': 1.5,         # % per annum
    'gdp_deflator': 2.0,            # % per annum (implies ~3.5% nominal)
    'primary_balance_gdp': -0.5,    # % of GDP (slight deficit)
    'effective_interest_rate': 4.0, # % on existing debt
    'new_debt_rate': 4.5,           # % on marginal borrowing
}


# ==============================================================================
# SECTION 9: OUTPUT CONFIGURATION
# ==============================================================================

OUTPUT_CONFIG = {
    'figure_dpi': 300,
    'figure_format': 'png',
    'table_format': ['xlsx', 'tex', 'csv'],
    
    # Colors for charts
    'colors': {
        'baseline': '#2c3e50',
        'historical': '#7f8c8d',
        'forecast': '#3498db',
        'adverse': '#e74c3c',
        'favorable': '#27ae60',
        'confidence_band': '#85c1e9',
    },
    
    # Key dates for annotations
    'key_events': {
        2008: 'Global Financial Crisis',
        2016: 'Brexit Referendum',
        2020: 'COVID-19 Pandemic',
        2022: 'Mini Budget',
    },
}


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def get_combined_series(metric: str, start_year: str = '1997-98') -> Dict[str, float]:
    """
    Get combined historical + forecast series for a metric.
    """
    historical_map = {
        'psnd': HISTORICAL_PSND,
        'gdp': HISTORICAL_GDP,
        'debt_to_gdp': HISTORICAL_DEBT_TO_GDP,
        'psnb': HISTORICAL_PSNB,
        'gilt_yield': HISTORICAL_GILT_YIELD_10Y,
        'bank_rate': HISTORICAL_BANK_RATE,
        'rpi': HISTORICAL_RPI,
    }
    
    forecast_map = {
        'psnd': OBR_PSND,
        'gdp': OBR_GDP,
        'debt_to_gdp': OBR_DEBT_TO_GDP,
        'psnb': OBR_PSNB,
        'gilt_yield': OBR_GILT_YIELD,
        'bank_rate': OBR_BANK_RATE,
        'rpi': OBR_RPI,
    }
    
    if metric not in historical_map:
        raise ValueError(f"Unknown metric: {metric}")
    
    combined = {}
    for fy, val in historical_map[metric].items():
        if fy >= start_year:
            combined[fy] = val
    
    for fy, val in forecast_map[metric].items():
        if fy not in combined:
            combined[fy] = val
    
    return dict(sorted(combined.items()))


def fiscal_year_to_numeric(fy: str) -> int:
    """Convert fiscal year string to starting calendar year."""
    return int(fy.split('-')[0])


def numeric_to_fiscal_year(year: int) -> str:
    """Convert calendar year to fiscal year string."""
    return f"{year}-{str(year + 1)[2:]}"
