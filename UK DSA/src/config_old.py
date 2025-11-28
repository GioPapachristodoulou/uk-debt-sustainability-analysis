"""
UK Debt Sustainability Analysis - Configuration
===============================================
Imperial College London UROP Project
March 2025 OBR Forecast Baseline

This file contains all key parameters, OBR forecasts, and scenario definitions
for the comprehensive UK debt sustainability analysis.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple

# ==============================================================================
# OBR MARCH 2025 BASELINE FORECASTS
# ==============================================================================

# Fiscal Years (UK convention: April-March)
FORECAST_YEARS = [
    '2023-24', '2024-25', '2025-26', '2026-27', '2027-28', '2028-29', '2029-30'
]

# Nominal GDP (£ billion, financial year basis)
OBR_GDP = {
    '2023-24': 2749.2,
    '2024-25': 2864.0,
    '2025-26': 3001.7,
    '2026-27': 3113.0,
    '2027-28': 3222.8,
    '2028-29': 3336.5,
    '2029-30': 3456.8
}

# Public Sector Net Debt (£ billion, end-year)
OBR_PSND = {
    '2023-24': 2625.0,  # Outturn
    '2024-25': 2746.3,
    '2025-26': 2854.2,
    '2026-27': 2981.3,
    '2027-28': 3098.1,
    '2028-29': 3211.5,
    '2029-30': 3322.4
}

# PSND as % of GDP
OBR_DEBT_TO_GDP = {
    '2023-24': 95.48,  # Outturn
    '2024-25': 95.86,
    '2025-26': 95.12,
    '2026-27': 95.77,
    '2027-28': 96.14,
    '2028-29': 96.26,
    '2029-30': 96.11
}

# Public Sector Net Borrowing (£ billion)
OBR_PSNB = {
    '2023-24': 131.3,  # Outturn
    '2024-25': 137.3,
    '2025-26': 117.7,
    '2026-27': 97.2,
    '2027-28': 80.2,
    '2028-29': 77.4,
    '2029-30': 74.0
}

# Current Receipts (£ billion)
OBR_RECEIPTS = {
    '2024-25': 1141.2,
    '2025-26': 1229.5,
    '2026-27': 1292.3,
    '2027-28': 1350.7,
    '2028-29': 1394.0,
    '2029-30': 1445.0
}

# Total Managed Expenditure (£ billion)
OBR_TME = {
    '2024-25': 1278.6,
    '2025-26': 1347.2,
    '2026-27': 1389.5,
    '2027-28': 1430.8,
    '2028-29': 1471.4,
    '2029-30': 1519.0
}

# Debt Interest (net of APF, £ billion)
OBR_DEBT_INTEREST = {
    '2023-24': 106.7,
    '2024-25': 105.2,
    '2025-26': 111.2,
    '2026-27': 111.4,
    '2027-28': 117.9,
    '2028-29': 124.2,
    '2029-30': 131.6
}

# ==============================================================================
# DEBT COMPOSITION (End 2024-25 forecast)
# ==============================================================================

DEBT_COMPOSITION = {
    'conventional_gilts_private': 1202.4,  # £bn
    'conventional_gilts_apf': 578.6,       # £bn (QE holdings)
    'index_linked_gilts': 613.8,           # £bn
    'ns_and_i': 234.1,                     # £bn
    'other_debt': 189.6,                   # £bn
    'total_gross_debt': 2818.5             # £bn
}

# Index-linked gilts share of total
ILG_SHARE = DEBT_COMPOSITION['index_linked_gilts'] / DEBT_COMPOSITION['total_gross_debt']  # ~21.8%

# ==============================================================================
# INTEREST RATE ASSUMPTIONS
# ==============================================================================

# Effective interest rates by instrument (2024-25)
EFFECTIVE_RATES = {
    'conventional_gilts': 2.46,    # %
    'index_linked_gilts': 3.60,    # % (includes uplift)
    'ns_and_i': 3.75,              # %
    'other_debt': 5.88,            # %
    'blended_rate': 3.73           # % overall
}

# Gilt yield assumptions (10-year nominal)
GILT_YIELDS = {
    '2024-25': 4.4,
    '2025-26': 4.5,
    '2026-27': 4.5,
    '2027-28': 4.5,
    '2028-29': 4.5,
    '2029-30': 4.5
}

# Bank Rate assumptions
BANK_RATE = {
    '2024-25': 4.75,
    '2025-26': 4.25,
    '2026-27': 3.75,
    '2027-28': 3.50,
    '2028-29': 3.50,
    '2029-30': 3.50
}

# ==============================================================================
# INFLATION ASSUMPTIONS
# ==============================================================================

# CPI Inflation (%)
CPI_INFLATION = {
    '2024-25': 2.6,
    '2025-26': 3.2,
    '2026-27': 2.4,
    '2027-28': 2.1,
    '2028-29': 2.0,
    '2029-30': 2.0
}

# RPI Inflation (%) - critical for index-linked gilts
RPI_INFLATION = {
    '2024-25': 3.6,
    '2025-26': 4.0,
    '2026-27': 3.1,
    '2027-28': 2.9,
    '2028-29': 2.8,
    '2029-30': 2.8
}

# ==============================================================================
# GROWTH ASSUMPTIONS
# ==============================================================================

# Real GDP Growth (%)
REAL_GDP_GROWTH = {
    '2024': 0.9,
    '2025': 1.0,
    '2026': 1.9,
    '2027': 1.8,
    '2028': 1.7,
    '2029': 1.8
}

# Nominal GDP Growth (%)
NOMINAL_GDP_GROWTH = {
    '2024-25': 4.2,
    '2025-26': 4.8,
    '2026-27': 3.7,
    '2027-28': 3.5,
    '2028-29': 3.5,
    '2029-30': 3.6
}

# ==============================================================================
# DEBT INTEREST READY RECKONERS (OBR Table 5.1)
# Impact per 1 percentage point increase, £ billion by 2029-30
# ==============================================================================

READY_RECKONERS = {
    'gilt_rates_1pp': {
        '2024-25': 0.59,
        '2025-26': 4.16,
        '2026-27': 6.57,
        '2027-28': 8.67,
        '2028-29': 10.69,
        '2029-30': 12.37
    },
    'short_rates_1pp': {
        '2024-25': 6.87,
        '2025-26': 8.05,
        '2026-27': 7.23,
        '2027-28': 6.22,
        '2028-29': 5.38,
        '2029-30': 4.66
    },
    'inflation_1pp': {
        '2024-25': 0.13,
        '2025-26': 7.76,
        '2026-27': 8.59,
        '2027-28': 9.77,
        '2028-29': 10.35,
        '2029-30': 10.99
    },
    'cgncr_10bn': {
        '2024-25': 0.0,
        '2025-26': 0.24,
        '2026-27': 0.67,
        '2027-28': 1.14,
        '2028-29': 1.60,
        '2029-30': 2.09
    }
}

# ==============================================================================
# SCENARIO DEFINITIONS
# ==============================================================================

@dataclass
class Scenario:
    """Scenario definition for stress testing"""
    name: str
    description: str
    gilt_yield_shock: float      # pp change in gilt yields
    short_rate_shock: float      # pp change in Bank Rate
    inflation_shock: float       # pp change in RPI
    growth_shock: float          # pp change in real GDP growth
    primary_balance_shock: float # £bn change in primary balance
    duration: int                # years shock persists

# Define stress scenarios
SCENARIOS = {
    'baseline': Scenario(
        name='Baseline',
        description='OBR March 2025 central forecast',
        gilt_yield_shock=0.0,
        short_rate_shock=0.0,
        inflation_shock=0.0,
        growth_shock=0.0,
        primary_balance_shock=0.0,
        duration=0
    ),
    'adverse_rates': Scenario(
        name='Adverse Interest Rates',
        description='+200bp gilt yields, +150bp Bank Rate',
        gilt_yield_shock=2.0,
        short_rate_shock=1.5,
        inflation_shock=0.0,
        growth_shock=-0.3,  # Higher rates slow growth
        primary_balance_shock=0.0,
        duration=5
    ),
    'high_inflation': Scenario(
        name='High Inflation',
        description='+3pp RPI inflation for 2 years',
        gilt_yield_shock=1.0,
        short_rate_shock=1.5,
        inflation_shock=3.0,
        growth_shock=-0.5,
        primary_balance_shock=0.0,
        duration=2
    ),
    'recession': Scenario(
        name='Recession',
        description='-2% real GDP growth in Year 1, slow recovery',
        gilt_yield_shock=-0.5,  # Flight to safety
        short_rate_shock=-1.0,  # Policy response
        inflation_shock=-0.5,
        growth_shock=-3.0,      # vs baseline growth
        primary_balance_shock=-40.0,  # Automatic stabilizers
        duration=1
    ),
    'stagflation': Scenario(
        name='Stagflation',
        description='High inflation + low growth',
        gilt_yield_shock=1.5,
        short_rate_shock=2.0,
        inflation_shock=4.0,
        growth_shock=-2.0,
        primary_balance_shock=-20.0,
        duration=3
    ),
    'fiscal_consolidation': Scenario(
        name='Fiscal Consolidation',
        description='+1% GDP primary surplus annually',
        gilt_yield_shock=-0.25,
        short_rate_shock=0.0,
        inflation_shock=0.0,
        growth_shock=-0.2,  # Short-term drag
        primary_balance_shock=30.0,  # ~1% GDP
        duration=5
    ),
    'combined_adverse': Scenario(
        name='Combined Adverse',
        description='Higher rates + inflation + lower growth',
        gilt_yield_shock=2.0,
        short_rate_shock=1.5,
        inflation_shock=2.0,
        growth_shock=-1.0,
        primary_balance_shock=-15.0,
        duration=3
    ),
    'mini_budget': Scenario(
        name='Mini Budget Crisis',
        description='September 2022-style market stress',
        gilt_yield_shock=3.0,
        short_rate_shock=2.0,
        inflation_shock=1.0,
        growth_shock=-1.0,
        primary_balance_shock=-25.0,
        duration=1
    )
}

# ==============================================================================
# MONTE CARLO PARAMETERS
# ==============================================================================

MONTE_CARLO_CONFIG = {
    'n_simulations': 10000,
    'horizon_years': 10,  # 2025-2035
    'random_seed': 42,
    
    # Stochastic parameters (annual, calibrated from historical data)
    'real_gdp_growth': {
        'mean': 0.015,    # 1.5% p.a.
        'std': 0.020,     # 2.0% volatility
        'persistence': 0.3  # AR(1) coefficient
    },
    'cpi_inflation': {
        'mean': 0.020,    # 2.0% target
        'std': 0.015,     # 1.5% volatility
        'persistence': 0.5
    },
    'rpi_inflation': {
        'mean': 0.030,    # 3.0% (RPI-CPI wedge)
        'std': 0.018,
        'persistence': 0.5
    },
    'gilt_10y': {
        'mean': 0.045,    # 4.5%
        'std': 0.010,     # 1.0% volatility
        'persistence': 0.8
    },
    
    # Correlation matrix (GDP_growth, CPI, RPI, Gilt_yield)
    'correlation_matrix': np.array([
        [1.00, -0.20, -0.15, 0.10],   # GDP growth
        [-0.20, 1.00, 0.95, 0.30],    # CPI
        [-0.15, 0.95, 1.00, 0.35],    # RPI
        [0.10, 0.30, 0.35, 1.00]      # Gilt yield
    ]),
    
    # Percentiles for fan charts
    'percentiles': [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
}

# ==============================================================================
# FISCAL RULE THRESHOLDS
# ==============================================================================

FISCAL_RULES = {
    # Government's fiscal rules (Autumn 2024)
    'debt_falling_year5': True,       # PSND/GDP falling in year 5
    'current_budget_balance_year5': True,  # Current budget in balance in year 5
    'investment_rule': 3.0,           # Investment spending cap (% GDP)
    'debt_interest_rule': None,       # No explicit rule but monitored
    
    # Risk thresholds for analysis
    'debt_warning': 100.0,            # % GDP
    'debt_critical': 120.0,           # % GDP
    'interest_warning': 4.0,          # % GDP
    'interest_critical': 6.0          # % GDP
}

# ==============================================================================
# HISTORICAL DATA PERIODS
# ==============================================================================

HISTORICAL_PERIODS = {
    'full_sample': (1997, 2024),
    'pre_crisis': (1997, 2007),
    'post_crisis': (2010, 2019),
    'covid_era': (2020, 2024),
    'recent': (2019, 2024)
}

# ==============================================================================
# OUTPUT CONFIGURATION
# ==============================================================================

OUTPUT_CONFIG = {
    'figure_dpi': 300,
    'figure_format': 'png',
    'table_format': 'latex',
    'decimal_places': 2,
    'font_size': 11,
    'figure_width': 10,
    'figure_height': 6
}

# Colors for charts
CHART_COLORS = {
    'baseline': '#1f77b4',
    'adverse': '#d62728',
    'favorable': '#2ca02c',
    'historical': '#7f7f7f',
    'confidence_95': '#cce5ff',
    'confidence_75': '#99ccff',
    'confidence_50': '#66b3ff',
    'median': '#0066cc'
}
