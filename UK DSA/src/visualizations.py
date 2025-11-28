"""
UK Debt Sustainability Analysis - Publication Figures
=====================================================
Creates all charts for the comprehensive DSA report.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import FuncFormatter
import warnings
warnings.filterwarnings('ignore')

# Set publication style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['font.family'] = 'serif'

# Color palette
COLORS = {
    'baseline': '#2c3e50',
    'historical': '#7f8c8d',
    'forecast': '#3498db',
    'adverse': '#e74c3c',
    'favorable': '#27ae60',
    'ci_90': '#85c1e9',
    'ci_50': '#5dade2',
    'ci_80': '#aed6f1',
}

def create_historical_debt_chart(output_path='historical_debt.png'):
    """
    Figure 1: Historical debt-to-GDP with key events annotated.
    """
    # Data from config
    historical_data = {
        1997: 41.1, 1998: 38.8, 1999: 36.2, 2000: 32.0, 2001: 30.5,
        2002: 30.6, 2003: 32.1, 2004: 34.4, 2005: 35.4, 2006: 35.8,
        2007: 36.6, 2008: 43.8, 2009: 66.7, 2010: 73.7, 2011: 76.1,
        2012: 79.4, 2013: 80.9, 2014: 82.6, 2015: 82.1, 2016: 84.5,
        2017: 83.1, 2018: 81.1, 2019: 80.2, 2020: 102.1, 2021: 99.2,
        2022: 96.7, 2023: 96.3, 2024: 95.9,
    }
    
    forecast_data = {
        2024: 95.9, 2025: 95.1, 2026: 95.8, 2027: 96.1, 2028: 96.3, 2029: 96.1,
        2030: 94.9, 2031: 93.7, 2032: 92.4, 2033: 91.1, 2034: 89.7,
    }
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Historical data
    years_hist = list(historical_data.keys())
    debt_hist = list(historical_data.values())
    ax.plot(years_hist, debt_hist, color=COLORS['historical'], linewidth=2.5, label='Historical')
    
    # Forecast data
    years_fc = list(forecast_data.keys())
    debt_fc = list(forecast_data.values())
    ax.plot(years_fc, debt_fc, color=COLORS['forecast'], linewidth=2.5, linestyle='--', label='OBR Forecast')
    
    # Fill historical area
    ax.fill_between(years_hist, 0, debt_hist, color=COLORS['historical'], alpha=0.15)
    
    # Key events
    events = {
        2008: ('Global Financial\nCrisis', 48),
        2016: ('Brexit\nReferendum', 88),
        2020: ('COVID-19\nPandemic', 105),
        2022: ('Mini Budget\nCrisis', 100),
    }
    
    for year, (text, y_pos) in events.items():
        ax.axvline(year, color='gray', linestyle=':', alpha=0.5)
        ax.annotate(text, xy=(year, historical_data.get(year, forecast_data.get(year))),
                   xytext=(year, y_pos),
                   fontsize=10, ha='center',
                   arrowprops=dict(arrowstyle='->', color='gray', alpha=0.7))
    
    # Reference lines
    ax.axhline(100, color='red', linestyle='--', alpha=0.5, label='100% Threshold')
    ax.axhline(60, color='orange', linestyle=':', alpha=0.5, label='Maastricht Criterion')
    
    ax.set_xlabel('Fiscal Year (ending March)')
    ax.set_ylabel('Public Sector Net Debt (% of GDP)')
    ax.set_title('UK Public Debt: Historical Trajectory and OBR Forecast\n1997-2035', fontweight='bold')
    ax.legend(loc='upper left')
    ax.set_xlim(1997, 2035)
    ax.set_ylim(0, 120)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def create_fan_chart(output_path='fan_chart.png'):
    """
    Figure 2: Monte Carlo fan chart with confidence intervals.
    """
    # Data from Monte Carlo
    years = np.arange(2024, 2035)
    
    # Percentile data (from simulation)
    median = [96.6, 95.9, 95.6, 94.7, 93.9, 92.6, 91.5, 90.3, 89.0, 87.8, 86.5]
    p5 = [92.7, 90.5, 88.9, 87.1, 85.3, 83.5, 81.8, 80.1, 78.4, 76.8, 75.2]
    p25 = [95.0, 93.7, 92.8, 91.5, 90.3, 88.9, 87.4, 86.0, 84.6, 83.1, 81.8]
    p75 = [98.3, 98.3, 98.5, 98.0, 97.5, 96.6, 95.6, 94.7, 93.6, 92.5, 91.4]
    p95 = [100.7, 101.9, 102.7, 102.9, 102.9, 102.4, 101.9, 101.3, 100.5, 99.8, 99.0]
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # 90% confidence band
    ax.fill_between(years, p5, p95, color=COLORS['ci_90'], alpha=0.3, label='90% CI')
    
    # 50% confidence band
    ax.fill_between(years, p25, p75, color=COLORS['ci_50'], alpha=0.4, label='50% CI')
    
    # Median path
    ax.plot(years, median, color=COLORS['baseline'], linewidth=3, label='Median')
    
    # OBR baseline (for comparison)
    obr_baseline = [95.9, 95.1, 95.8, 96.1, 96.3, 96.1, 94.9, 93.7, 92.4, 91.1, 89.7]
    ax.plot(years, obr_baseline, color='black', linewidth=2, linestyle='--', label='OBR Baseline')
    
    # Reference lines
    ax.axhline(100, color='red', linestyle='--', alpha=0.6, linewidth=1.5)
    ax.text(2034.5, 101, '100%', va='bottom', ha='left', fontsize=10, color='red')
    
    ax.set_xlabel('Fiscal Year')
    ax.set_ylabel('Public Sector Net Debt (% of GDP)')
    ax.set_title('UK Debt Projection: Monte Carlo Fan Chart (10,000 Simulations)\n2024-2035', fontweight='bold')
    ax.legend(loc='upper right')
    ax.set_xlim(2024, 2035)
    ax.set_ylim(70, 110)
    
    # Add probability annotation
    ax.annotate('P(Debt > 100%) ≈ 31.5% at some point\nP(Debt > 100% in 2034-35) ≈ 3.8%',
               xy=(2025, 73), fontsize=10,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def create_scenario_comparison(output_path='scenario_comparison.png'):
    """
    Figure 3: Scenario stress test comparison.
    """
    scenarios = {
        'Baseline': [95.9, 95.1, 95.8, 96.1, 96.3, 96.1, 94.9, 93.7, 92.4, 91.1, 89.7],
        'Adverse Rates': [96.5, 95.5, 95.8, 96.4, 97.0, 95.5, 94.3, 93.1, 91.9, 90.5, 89.2],
        'Recession': [102.3, 103.5, 104.2, 104.9, 103.5, 97.9, 96.2, 94.6, 93.1, 92.1, 91.2],
        'Stagflation': [100.9, 104.5, 107.2, 107.8, 103.8, 101.9, 99.8, 97.9, 96.2, 95.4, 94.6],
        'Consolidation': [94.8, 92.1, 90.5, 89.2, 88.1, 87.9, 86.5, 85.2, 84.0, 83.4, 82.8],
        'Combined Adverse': [100.0, 104.8, 107.5, 108.1, 105.2, 103.6, 101.4, 99.4, 97.6, 96.8, 96.1],
    }
    
    years = list(range(2024, 2035))
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    styles = {
        'Baseline': {'color': COLORS['baseline'], 'linestyle': '-', 'linewidth': 3},
        'Adverse Rates': {'color': '#e74c3c', 'linestyle': '--', 'linewidth': 2},
        'Recession': {'color': '#9b59b6', 'linestyle': '-.', 'linewidth': 2},
        'Stagflation': {'color': '#e67e22', 'linestyle': ':', 'linewidth': 2.5},
        'Consolidation': {'color': COLORS['favorable'], 'linestyle': '--', 'linewidth': 2},
        'Combined Adverse': {'color': '#c0392b', 'linestyle': '-', 'linewidth': 2},
    }
    
    for name, data in scenarios.items():
        ax.plot(years, data, label=name, **styles[name])
    
    # Reference line
    ax.axhline(100, color='red', linestyle='--', alpha=0.3)
    ax.fill_between(years, 100, 120, color='red', alpha=0.05)
    
    ax.set_xlabel('Fiscal Year')
    ax.set_ylabel('Public Sector Net Debt (% of GDP)')
    ax.set_title('UK Debt Scenarios: Stress Test Comparison\n2024-2035', fontweight='bold')
    ax.legend(loc='upper right')
    ax.set_xlim(2024, 2034)
    ax.set_ylim(75, 115)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def create_debt_decomposition(output_path='debt_decomposition.png'):
    """
    Figure 4: Debt dynamics decomposition.
    """
    years = ['2020-21', '2021-22', '2022-23', '2023-24', '2024-25', '2025-26', '2026-27', '2027-28', '2028-29', '2029-30']
    
    # Decomposition components (stylized from OBR)
    interest_growth = [4.5, -1.2, 0.8, 1.5, 1.8, 1.2, 0.8, 0.6, 0.4, 0.2]
    primary_balance = [13.5, 3.2, 2.8, 2.5, 1.1, 0.2, -0.5, -1.2, -1.4, -1.7]
    sfa = [3.8, 1.5, -0.5, -0.2, 0.1, 0.0, -0.1, -0.1, 0.0, 0.0]
    
    x = np.arange(len(years))
    width = 0.6
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Stacked bars
    ax.bar(x, interest_growth, width, label='Interest-Growth Effect', color='#3498db')
    ax.bar(x, primary_balance, width, bottom=interest_growth, label='Primary Balance', color='#e74c3c')
    
    bottom = [i + p for i, p in zip(interest_growth, primary_balance)]
    ax.bar(x, sfa, width, bottom=bottom, label='Stock-Flow Adjustment', color='#95a5a6')
    
    # Total change line
    total = [i + p + s for i, p, s in zip(interest_growth, primary_balance, sfa)]
    ax.plot(x, total, 'ko-', markersize=8, linewidth=2, label='Total Change')
    
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(years, rotation=45, ha='right')
    ax.set_ylabel('Change in Debt/GDP (percentage points)')
    ax.set_title('UK Debt Dynamics Decomposition\n2020-2030', fontweight='bold')
    ax.legend(loc='upper right')
    ax.set_ylim(-5, 25)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def create_r_g_differential(output_path='r_g_differential.png'):
    """
    Figure 5: Interest-growth differential over time.
    """
    # Historical data
    years_hist = list(range(1998, 2025))
    r_g_hist = [2.1, 1.5, 0.8, 0.5, 0.2, -0.5, -0.3, 0.1, 0.8, 1.2,
               -1.5, -0.2, -1.8, -2.5, -2.8, -1.5, -1.2, -2.8, -3.5, -4.2,
               -5.5, -4.2, -3.5, 2.5, 1.8, 0.5, 1.2][:len(years_hist)]
    
    # Forecast
    years_fc = list(range(2024, 2035))
    r_g_fc = [1.2, 0.8, 0.7, 0.8, 0.7, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    ax.fill_between(years_hist, 0, r_g_hist, where=np.array(r_g_hist)>0, 
                   color='#e74c3c', alpha=0.3, label='Debt-increasing (r > g)')
    ax.fill_between(years_hist, 0, r_g_hist, where=np.array(r_g_hist)<=0,
                   color='#27ae60', alpha=0.3, label='Debt-reducing (r < g)')
    
    ax.plot(years_hist, r_g_hist, color=COLORS['historical'], linewidth=2)
    ax.plot(years_fc, r_g_fc, color=COLORS['forecast'], linewidth=2, linestyle='--')
    
    ax.axhline(0, color='black', linewidth=1)
    ax.axvline(2024, color='gray', linestyle=':', alpha=0.5)
    ax.text(2024.5, 3, 'Forecast →', fontsize=10, va='bottom')
    
    ax.set_xlabel('Fiscal Year')
    ax.set_ylabel('Interest-Growth Differential (r - g, percentage points)')
    ax.set_title('UK Interest Rate-Growth Differential\n1998-2035', fontweight='bold')
    ax.legend(loc='upper left')
    ax.set_xlim(1998, 2034)
    ax.set_ylim(-6, 4)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def create_interest_burden(output_path='interest_burden.png'):
    """
    Figure 6: Debt interest burden.
    """
    years = list(range(2010, 2035))
    
    # Interest as % of GDP
    interest_gdp = [2.7, 3.0, 2.9, 2.6, 2.4, 2.0, 2.0, 2.1, 2.2, 2.1,
                   1.9, 2.7, 4.2, 3.4, 3.7, 3.7, 3.6, 3.7, 3.7, 3.8,
                   3.8, 3.8, 3.8, 3.7, 3.7]
    
    # Interest as % of revenue (approximation)
    interest_rev = [i / 0.38 for i in interest_gdp[:14]] + [i / 0.42 for i in interest_gdp[14:]]
    
    fig, ax1 = plt.subplots(figsize=(14, 8))
    
    ax1.bar(years, interest_gdp, color=COLORS['baseline'], alpha=0.7, label='Interest / GDP (%)')
    ax1.set_ylabel('Debt Interest (% of GDP)', color=COLORS['baseline'])
    ax1.tick_params(axis='y', labelcolor=COLORS['baseline'])
    ax1.set_ylim(0, 5)
    
    ax2 = ax1.twinx()
    ax2.plot(years, interest_rev, color='#e74c3c', linewidth=2.5, marker='o', markersize=5,
            label='Interest / Revenue (%)')
    ax2.set_ylabel('Debt Interest (% of Revenue)', color='#e74c3c')
    ax2.tick_params(axis='y', labelcolor='#e74c3c')
    ax2.set_ylim(0, 12)
    
    # Warning thresholds
    ax2.axhline(10, color='#e74c3c', linestyle='--', alpha=0.5)
    ax2.text(2034.5, 10.2, 'Warning (10%)', va='bottom', fontsize=9, color='#e74c3c')
    
    ax1.axvline(2024, color='gray', linestyle=':', alpha=0.5)
    
    ax1.set_xlabel('Fiscal Year')
    ax1.set_title('UK Debt Interest Burden\n2010-2035', fontweight='bold')
    
    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def create_debt_composition_chart(output_path='debt_composition.png'):
    """
    Figure 7: Debt composition breakdown.
    """
    categories = ['Conventional\nGilts', 'Index-Linked\nGilts', 'NS&I', 'Treasury\nBills', 'Other']
    values = [1360, 686, 234, 45, 50]
    colors = ['#3498db', '#e74c3c', '#27ae60', '#f39c12', '#95a5a6']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    
    # Pie chart
    wedges, texts, autotexts = ax1.pie(values, labels=categories, colors=colors,
                                       autopct='%1.1f%%', startangle=90,
                                       explode=[0, 0.05, 0, 0, 0])
    ax1.set_title('Gilt Stock Composition\n(£bn, November 2025)', fontweight='bold')
    
    # Bar chart of maturity profile
    maturity_cats = ['<3 years', '3-7 years', '7-15 years', '15-25 years', '>25 years']
    conventional = [190, 235, 420, 300, 215]
    ilg = [30, 68, 98, 180, 310]
    
    x = np.arange(len(maturity_cats))
    width = 0.35
    
    ax2.bar(x - width/2, conventional, width, label='Conventional', color='#3498db')
    ax2.bar(x + width/2, ilg, width, label='Index-Linked', color='#e74c3c')
    
    ax2.set_xticks(x)
    ax2.set_xticklabels(maturity_cats)
    ax2.set_ylabel('Outstanding (£bn)')
    ax2.set_title('Gilt Maturity Profile', fontweight='bold')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def create_ilg_sensitivity(output_path='ilg_sensitivity.png'):
    """
    Figure 8: Index-linked gilt sensitivity to inflation.
    """
    rpi_scenarios = np.arange(1, 8, 0.5)
    
    # Impact on debt interest (£bn) using ready reckoner
    base_interest = 131.6  # 2029-30 baseline
    extra_interest = (rpi_scenarios - 2.8) * 11.0  # ~£11bn per 1pp RPI
    total_interest = base_interest + extra_interest
    
    # Interest as % of GDP
    gdp = 3456.8  # 2029-30
    interest_gdp = total_interest / gdp * 100
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    ax.fill_between(rpi_scenarios, 3.0, interest_gdp, where=rpi_scenarios>=2.8,
                   color='#e74c3c', alpha=0.3)
    ax.fill_between(rpi_scenarios, interest_gdp, 3.0, where=rpi_scenarios<2.8,
                   color='#27ae60', alpha=0.3)
    
    ax.plot(rpi_scenarios, interest_gdp, color=COLORS['baseline'], linewidth=3)
    
    ax.axvline(2.8, color='gray', linestyle='--', alpha=0.5)
    ax.text(2.9, 4.3, 'OBR Baseline\n(2.8%)', fontsize=10)
    
    ax.axhline(4.0, color='orange', linestyle=':', alpha=0.7)
    ax.text(7.2, 4.05, 'Warning', fontsize=9, color='orange')
    
    ax.set_xlabel('RPI Inflation (%)')
    ax.set_ylabel('Debt Interest (% of GDP)')
    ax.set_title('Index-Linked Gilt Sensitivity: Impact of Inflation on Interest Costs\n(2029-30 Estimate)', fontweight='bold')
    ax.set_xlim(1, 7.5)
    ax.set_ylim(3.0, 5.0)
    
    ax.annotate(f'ILG Share: ~34% of gilt stock\n+1pp RPI ≈ +£11bn interest',
               xy=(5.5, 3.2), fontsize=10,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def create_all_figures(output_dir='/mnt/user-data/outputs'):
    """Generate all publication figures."""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    print("Generating publication figures...")
    print("=" * 50)
    
    create_historical_debt_chart(f'{output_dir}/fig1_historical_debt.png')
    create_fan_chart(f'{output_dir}/fig2_fan_chart.png')
    create_scenario_comparison(f'{output_dir}/fig3_scenarios.png')
    create_debt_decomposition(f'{output_dir}/fig4_decomposition.png')
    create_r_g_differential(f'{output_dir}/fig5_r_g_differential.png')
    create_interest_burden(f'{output_dir}/fig6_interest_burden.png')
    create_debt_composition_chart(f'{output_dir}/fig7_debt_composition.png')
    create_ilg_sensitivity(f'{output_dir}/fig8_ilg_sensitivity.png')
    
    print("=" * 50)
    print("All figures generated successfully!")


if __name__ == '__main__':
    create_all_figures()
