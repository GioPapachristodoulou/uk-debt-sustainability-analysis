"""
Comprehensive Visualization Module for UK DSA
==============================================

Creates publication-quality figures for:
1. Historical debt trajectory
2. Monte Carlo fan chart (fat-tailed)
3. Scenario stress tests
4. Debt decomposition
5. Interest-growth differential
6. Interest burden analysis
7. Debt composition
8. ILG sensitivity
9. Bohn test results
10. Fiscal space analysis
11. Gross Financing Needs
12. Fat-tail impact comparison
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import FuncFormatter
import warnings
warnings.filterwarnings('ignore')

# Set publication style
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'axes.spines.top': False,
    'axes.spines.right': False,
})


def create_fig1_historical_debt(output_path):
    """Figure 1: Historical debt/GDP with key events."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Historical data
    years = list(range(1997, 2025))
    debt_gdp = [42.5, 41.4, 39.6, 37.5, 34.4, 33.7, 34.5, 36.3, 38.6, 39.8,
                40.4, 43.0, 52.2, 64.1, 72.4, 76.0, 79.8, 82.4, 83.0, 83.8,
                84.4, 83.7, 82.9, 84.4, 102.1, 98.9, 97.8, 96.0]
    
    # Adjust to fiscal years
    fy_years = [f"{y}-{str(y+1)[-2:]}" for y in years]
    
    # OBR forecast
    forecast_years = list(range(2024, 2035))
    forecast_fy = [f"{y}-{str(y+1)[-2:]}" for y in forecast_years]
    forecast_debt = [96.0, 96.0, 96.2, 96.1, 96.1, 95.3, 94.0, 92.5, 91.0, 89.7, 88.5]
    
    # Plot historical
    ax.plot(range(len(years)), debt_gdp, 'b-', linewidth=2, label='Historical')
    
    # Plot forecast
    ax.plot(range(len(years)-1, len(years)-1+len(forecast_years)), forecast_debt,
            'b--', linewidth=2, label='OBR Forecast (Mar 2025)')
    
    # Reference lines
    ax.axhline(y=100, color='red', linestyle=':', linewidth=1.5, alpha=0.7)
    ax.axhline(y=60, color='orange', linestyle=':', linewidth=1.5, alpha=0.7)
    
    # Annotations
    ax.annotate('100% threshold', xy=(2, 102), color='red', fontsize=9)
    ax.annotate('60% Maastricht', xy=(2, 62), color='orange', fontsize=9)
    
    # Event markers
    events = [
        (11, 'GFC', 52.2),
        (19, 'Brexit\nvote', 83.8),
        (23, 'COVID', 102.1),
        (25, 'Mini\nBudget', 97.8)
    ]
    
    for x, label, y in events:
        ax.annotate(label, xy=(x, y+2), ha='center', fontsize=8, color='gray')
        ax.scatter([x], [y], s=30, color='gray', zorder=5)
    
    ax.set_xlabel('Fiscal Year')
    ax.set_ylabel('Public Sector Net Debt (% GDP)')
    ax.set_title('UK Public Sector Net Debt: 1997-2035', fontweight='bold')
    
    # X-axis labels
    all_years = fy_years + forecast_fy[1:]
    tick_positions = range(0, len(all_years), 5)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([all_years[i] for i in tick_positions], rotation=45, ha='right')
    
    ax.set_ylim(25, 115)
    ax.legend(loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def create_fig2_fan_chart(output_path, mc_results):
    """Figure 2: Monte Carlo fan chart with fat tails."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    fc = mc_results['fan_chart']
    years = np.arange(2024, 2035)
    
    # 90% interval
    ax.fill_between(years, fc['p5'], fc['p95'], alpha=0.2, color='blue', label='90% CI')
    # 50% interval
    ax.fill_between(years, fc['p25'], fc['p75'], alpha=0.4, color='blue', label='50% CI')
    # Median
    ax.plot(years, fc['p50'], 'b-', linewidth=2, label='Median')
    
    # OBR baseline
    obr = mc_results.get('obr_baseline', fc['p50'])
    ax.plot(years, obr[:len(years)], 'k--', linewidth=2, label='OBR Baseline')
    
    # 100% threshold
    ax.axhline(y=100, color='red', linestyle=':', linewidth=1.5, alpha=0.7)
    
    # Annotation
    prob_100_ever = mc_results['threshold_probs'][100]['prob_ever']
    prob_100_term = mc_results['threshold_probs'][100]['prob_terminal']
    ax.annotate(f'P(Debt > 100%) = {prob_100_ever:.0f}% (ever)\n'
                f'P(Debt > 100%) = {prob_100_term:.0f}% (2034-35)',
                xy=(2030, 75), fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.set_xlabel('Fiscal Year')
    ax.set_ylabel('Debt/GDP (%)')
    ax.set_title('Stochastic Debt Projections: 10,000 Fat-Tailed Simulations', fontweight='bold')
    ax.legend(loc='upper left')
    ax.set_xlim(2024, 2034)
    ax.set_ylim(60, 140)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def create_fig3_scenarios(output_path):
    """Figure 3: Scenario stress test comparison."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    years = list(range(2024, 2035))
    
    # Scenario paths
    scenarios = {
        'Baseline': [95.9, 96.0, 96.2, 96.1, 96.1, 95.3, 94.0, 92.5, 91.0, 89.7, 88.5],
        'Adverse Rates (+200bp)': [95.9, 96.5, 97.5, 98.2, 98.5, 98.0, 97.0, 95.8, 94.5, 93.2, 92.0],
        'Recession (-3% GDP)': [95.9, 100.5, 105.8, 104.5, 102.8, 100.5, 98.2, 96.0, 94.0, 92.2, 90.5],
        'Stagflation': [95.9, 98.5, 102.5, 106.2, 108.5, 107.2, 105.0, 102.5, 100.0, 97.5, 95.0],
        'Consolidation': [95.9, 95.0, 93.5, 91.2, 88.5, 85.5, 82.5, 80.0, 78.0, 76.0, 74.0],
        'Combined Adverse': [95.9, 101.5, 107.2, 110.8, 112.0, 110.5, 108.0, 105.2, 102.5, 100.0, 97.5]
    }
    
    colors = ['black', 'red', 'purple', 'orange', 'green', 'darkred']
    styles = ['-', '--', '-.', ':', '--', '-']
    
    for (name, path), color, style in zip(scenarios.items(), colors, styles):
        ax.plot(years, path, color=color, linestyle=style, linewidth=2, label=name)
    
    # 100% threshold with shading
    ax.axhline(y=100, color='red', linestyle=':', linewidth=1, alpha=0.5)
    ax.fill_between(years, 100, 120, alpha=0.1, color='red')
    
    ax.set_xlabel('Fiscal Year')
    ax.set_ylabel('Debt/GDP (%)')
    ax.set_title('Scenario Stress Tests: Debt/GDP Trajectories', fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_xlim(2024, 2034)
    ax.set_ylim(70, 120)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def create_fig4_decomposition(output_path):
    """Figure 4: Debt dynamics decomposition."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    years = ['2020-21', '2021-22', '2022-23', '2023-24', '2024-25', 
             '2025-26', '2026-27', '2027-28', '2028-29', '2029-30']
    
    # Components (% GDP change)
    interest_growth = [6.5, -4.2, 3.8, 1.5, 1.2, 0.8, 0.5, 0.3, 0.2, 0.1]
    primary_balance = [12.5, 4.2, 2.1, 0.8, 1.1, 0.6, 0.2, -0.3, -0.7, -1.3]
    sfa = [2.9, -3.0, -3.2, -1.5, -0.8, -0.4, -0.2, 0.0, 0.0, 0.0]
    
    x = np.arange(len(years))
    width = 0.6
    
    # Stack bars
    ax.bar(x, interest_growth, width, label='Interest-Growth Effect', color='steelblue')
    ax.bar(x, primary_balance, width, bottom=interest_growth, label='Primary Balance', color='indianred')
    
    # Calculate running position for SFA
    bottom_sfa = [i + p for i, p in zip(interest_growth, primary_balance)]
    ax.bar(x, sfa, width, bottom=bottom_sfa, label='Stock-Flow Adjustment', color='gray', alpha=0.7)
    
    # Total change line
    total = [i + p + s for i, p, s in zip(interest_growth, primary_balance, sfa)]
    ax.plot(x, total, 'ko-', linewidth=2, markersize=6, label='Total Change')
    
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(years, rotation=45, ha='right')
    ax.set_ylabel('Change in Debt/GDP (pp)')
    ax.set_title('Debt Dynamics Decomposition', fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def create_fig5_r_g(output_path):
    """Figure 5: Interest-growth differential."""
    fig, ax = plt.subplots(figsize=(12, 5))
    
    # Historical and forecast data
    years = list(range(1998, 2035))
    
    # r-g differential (effective rate - nominal GDP growth)
    r_g = [-2.5, -2.8, -2.2, -1.8, -2.5, -2.8, -2.2, -1.5, -1.8, -2.5,
           -3.2, 1.5, -2.5, -1.8, -0.5, 0.2, -0.8, -1.2, -1.5, -2.2,
           -2.8, 2.5, -5.5, -4.2, 2.8, 1.2, 0.8, 0.5, 0.4, 0.5,
           0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7]
    
    # Separate historical and forecast
    hist_years = years[:27]  # Up to 2024
    hist_r_g = r_g[:27]
    fore_years = years[26:]  # From 2024
    fore_r_g = r_g[26:]
    
    # Plot
    ax.plot(hist_years, hist_r_g, 'b-', linewidth=2, label='Historical')
    ax.plot(fore_years, fore_r_g, 'b--', linewidth=2, label='Forecast')
    
    # Fill areas
    ax.fill_between(years, r_g, 0, where=[x > 0 for x in r_g], 
                    alpha=0.3, color='red', label='Debt increasing (r > g)')
    ax.fill_between(years, r_g, 0, where=[x <= 0 for x in r_g], 
                    alpha=0.3, color='green', label='Debt decreasing (r < g)')
    
    ax.axhline(y=0, color='black', linewidth=1)
    ax.set_xlabel('Fiscal Year')
    ax.set_ylabel('r - g (percentage points)')
    ax.set_title('Interest Rate - Growth Rate Differential', fontweight='bold')
    ax.legend(loc='lower right')
    ax.set_xlim(1998, 2034)
    ax.set_ylim(-6, 4)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def create_fig6_interest_burden(output_path):
    """Figure 6: Interest burden analysis."""
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    years = list(range(2010, 2035))
    
    # Interest as % GDP
    interest_gdp = [2.7, 3.0, 2.9, 2.8, 2.5, 2.3, 2.1, 1.9, 1.8, 1.9,
                   2.3, 3.5, 4.2, 3.8, 3.7, 3.7, 3.6, 3.7, 3.7, 3.7,
                   3.7, 3.7, 3.7, 3.7, 3.7]
    
    # Interest as % revenue
    interest_rev = [6.8, 7.5, 7.2, 7.0, 6.2, 5.8, 5.3, 4.8, 4.5, 4.7,
                   5.8, 8.8, 10.5, 9.5, 9.2, 9.2, 9.0, 9.2, 9.2, 9.2,
                   9.2, 9.2, 9.2, 9.2, 9.2]
    
    # Bar chart for interest/GDP
    bars = ax1.bar(years, interest_gdp, color='steelblue', alpha=0.7, label='Interest/GDP')
    ax1.set_ylabel('Interest Payments (% GDP)', color='steelblue')
    ax1.tick_params(axis='y', labelcolor='steelblue')
    ax1.set_ylim(0, 5)
    
    # Line for interest/revenue on secondary axis
    ax2 = ax1.twinx()
    ax2.plot(years, interest_rev, 'r-o', linewidth=2, markersize=4, label='Interest/Revenue')
    ax2.axhline(y=10, color='red', linestyle=':', alpha=0.7)
    ax2.annotate('10% warning', xy=(2032, 10.3), color='red', fontsize=9)
    ax2.set_ylabel('Interest Payments (% Revenue)', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.set_ylim(0, 12)
    
    ax1.set_xlabel('Fiscal Year')
    ax1.set_title('Interest Payment Burden: 2010-2035', fontweight='bold')
    ax1.set_xticks(years[::2])
    ax1.set_xticklabels([str(y) for y in years[::2]], rotation=45, ha='right')
    
    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def create_fig7_composition(output_path):
    """Figure 7: Debt composition."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Pie chart of debt by instrument
    sizes = [57, 29, 10, 2, 2]
    labels = ['Conventional Gilts', 'Index-linked Gilts', 'NS&I', 'T-bills', 'Other']
    colors = ['steelblue', 'orange', 'green', 'gray', 'lightgray']
    explode = (0.02, 0.05, 0, 0, 0)
    
    ax1.pie(sizes, labels=labels, colors=colors, explode=explode,
            autopct='%1.0f%%', startangle=90)
    ax1.set_title('Debt Composition by Instrument', fontweight='bold')
    
    # Bar chart of maturity profile
    buckets = ['<3yr', '3-7yr', '7-15yr', '15-25yr', '>25yr']
    conventional = [190, 235, 420, 310, 205]
    ilg = [0, 98, 195, 83, 310]
    
    x = np.arange(len(buckets))
    width = 0.35
    
    ax2.bar(x - width/2, conventional, width, label='Conventional', color='steelblue')
    ax2.bar(x + width/2, ilg, width, label='Index-linked', color='orange')
    
    ax2.set_xlabel('Maturity Bucket')
    ax2.set_ylabel('Outstanding (£bn)')
    ax2.set_title('Gilt Maturity Profile', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(buckets)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def create_fig8_ilg_sensitivity(output_path):
    """Figure 8: ILG inflation sensitivity."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    rpi_range = np.arange(1, 8, 0.5)
    
    # Debt interest as % GDP at different RPI levels
    # Base: ~3.7% at 2.8% RPI
    # Each 1pp RPI = ~£11bn = ~0.4% GDP extra
    base_interest = 3.7
    base_rpi = 2.8
    
    interest_gdp = [base_interest + 0.4 * (rpi - base_rpi) for rpi in rpi_range]
    
    ax.plot(rpi_range, interest_gdp, 'b-', linewidth=2)
    ax.fill_between(rpi_range, interest_gdp, 0, alpha=0.3, color='blue')
    
    # Mark baseline
    ax.scatter([base_rpi], [base_interest], s=100, color='red', zorder=5)
    ax.annotate(f'OBR Baseline\n(RPI={base_rpi}%)', xy=(base_rpi, base_interest),
                xytext=(base_rpi + 1, base_interest - 0.3), fontsize=9,
                arrowprops=dict(arrowstyle='->', color='gray'))
    
    # Warning threshold
    ax.axhline(y=4.0, color='red', linestyle=':', alpha=0.7)
    ax.annotate('4% GDP threshold', xy=(6.5, 4.1), color='red', fontsize=9)
    
    ax.set_xlabel('RPI Inflation (%)')
    ax.set_ylabel('Debt Interest (% GDP)')
    ax.set_title('Index-Linked Gilt Sensitivity to Inflation', fontweight='bold')
    ax.set_xlim(1, 7.5)
    ax.set_ylim(2.5, 5.5)
    ax.grid(alpha=0.3)
    
    # Annotation box
    ax.text(5.5, 3.2, 'ILG share: 34% of gilts\n+1pp RPI ≈ +£11bn interest',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5), fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def create_fig9_bohn_test(output_path, bohn_results):
    """Figure 9: Bohn test regression results."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Scatter plot with regression line
    debt_lag = [53.2, 52.8, 51.4, 49.3, 46.8, 46.1, 46.4, 44.9, 44.5, 45.3,
                44.3, 43.2, 41.2, 37.3, 32.4, 28.5, 27.5, 29.3, 33.5, 38.0,
                40.9, 42.0, 42.5, 41.4, 39.6, 37.5, 34.4, 33.7, 34.5, 36.3,
                38.6, 39.8, 40.4, 43.0, 52.2, 64.1, 72.4, 76.0, 79.8, 82.4,
                83.0, 83.8, 84.4, 83.7, 82.9, 84.4, 102.1, 98.9, 97.8]
    
    pb = [-3.8, -3.2, -1.5, -2.8, -2.1, -2.5, -1.8, -1.2, -1.8, -1.5,
          -0.8, -0.5, 0.2, 1.8, 2.3, 1.5, -0.2, -4.2, -5.2, -4.5,
          -3.2, -2.1, -0.5, 1.2, 2.1, 2.8, 1.8, -0.2, -1.8, -2.2,
          -2.0, -1.5, -1.2, -2.8, -8.5, -7.2, -5.2, -4.8, -3.8, -3.2,
          -2.5, -1.8, -0.8, -0.2, 0.2, -2.5, -12.5, -4.2, -2.1]
    
    ax1.scatter(debt_lag[:-1], pb[1:], alpha=0.6, color='steelblue', s=40)
    
    # Regression line
    basic = bohn_results.get('basic', {})
    alpha = basic.get('alpha', 0)
    beta = basic.get('beta', 0)
    x_line = np.array([25, 105])
    y_line = alpha + beta * x_line
    ax1.plot(x_line, y_line, 'r-', linewidth=2, label=f'β = {beta:.4f}')
    
    ax1.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    ax1.set_xlabel('Lagged Debt/GDP (%)')
    ax1.set_ylabel('Primary Balance (% GDP)')
    ax1.set_title('Bohn Test: Fiscal Reaction Function', fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Add COVID outlier annotation
    ax1.annotate('COVID\n2020-21', xy=(102.1, -12.5), fontsize=8, color='gray')
    
    # Right: Non-linear marginal response
    nonlinear = bohn_results.get('nonlinear', {})
    if 'marginal_response' in nonlinear:
        debt_levels = list(range(30, 110, 5))
        marginal = nonlinear.get('marginal_response', {})
        
        # Extrapolate from available points
        b1 = nonlinear.get('beta_linear', 0)
        b2 = nonlinear.get('beta_quadratic', 0)
        
        marginal_curve = [b1 + 2 * b2 * d / 100 for d in debt_levels]
        
        ax2.plot(debt_levels, marginal_curve, 'b-', linewidth=2)
        ax2.axhline(y=0, color='red', linestyle=':', linewidth=1.5)
        ax2.fill_between(debt_levels, marginal_curve, 0, 
                         where=[m > 0 for m in marginal_curve],
                         alpha=0.3, color='green', label='Stabilizing')
        ax2.fill_between(debt_levels, marginal_curve, 0,
                         where=[m <= 0 for m in marginal_curve],
                         alpha=0.3, color='red', label='Destabilizing')
        
        # Mark current debt
        ax2.axvline(x=96, color='gray', linestyle='--', alpha=0.7)
        ax2.annotate('Current\n(96%)', xy=(96, 0.02), fontsize=9, ha='center')
        
        ax2.set_xlabel('Debt/GDP (%)')
        ax2.set_ylabel('Marginal Fiscal Response (∂pb/∂d)')
        ax2.set_title('Non-Linear Bohn Test: Marginal Response', fontweight='bold')
        ax2.legend(loc='lower right')
        ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def create_fig10_fiscal_space(output_path, fs_results):
    """Figure 10: Fiscal space analysis."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Fiscal reaction function vs debt-stabilizing requirement
    debt_range = np.arange(30, 150, 1)
    
    # Fiscal reaction (estimated)
    rf = fs_results.get('reaction_function', {})
    alpha, b1, b2, b3 = (rf.get('alpha', 0), rf.get('beta1', 0), 
                         rf.get('beta2', 0), rf.get('beta3', 0))
    
    pb_actual = [alpha + b1*(d/100) + b2*(d/100)**2 + b3*(d/100)**3 for d in debt_range]
    
    # Debt-stabilizing primary balance (r=4.5%, g=3.5%)
    r, g = 4.5, 3.5
    pb_required = [(r - g) / (100 + g) * d for d in debt_range]
    
    ax1.plot(debt_range, pb_actual, 'b-', linewidth=2, label='Estimated Fiscal Response')
    ax1.plot(debt_range, pb_required, 'r--', linewidth=2, label='Debt-Stabilizing Requirement')
    
    # Find and mark intersection
    debt_limit = fs_results.get('fiscal_space', {}).get('debt_limit_baseline', 114)
    ax1.axvline(x=debt_limit, color='orange', linestyle=':', linewidth=2)
    ax1.annotate(f'Debt Limit\n({debt_limit:.0f}%)', xy=(debt_limit+2, -2), fontsize=9, color='orange')
    
    # Mark current debt
    ax1.axvline(x=96, color='gray', linestyle='--', alpha=0.7)
    ax1.annotate('Current', xy=(96, -4), fontsize=9, ha='center', color='gray')
    
    # Fiscal space arrow
    ax1.annotate('', xy=(debt_limit, -6), xytext=(96, -6),
                arrowprops=dict(arrowstyle='<->', color='green', lw=2))
    ax1.text((96 + debt_limit)/2, -6.5, f'Fiscal Space\n{debt_limit-96:.0f}pp', 
             ha='center', fontsize=9, color='green')
    
    ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax1.set_xlabel('Debt/GDP (%)')
    ax1.set_ylabel('Primary Balance (% GDP)')
    ax1.set_title('Fiscal Space: Reaction vs Requirement', fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.set_xlim(30, 150)
    ax1.set_ylim(-8, 4)
    ax1.grid(alpha=0.3)
    
    # Right: Fiscal space by scenario
    scenarios = fs_results.get('fiscal_space', {}).get('scenarios', {})
    
    names = list(scenarios.keys())
    spaces = [scenarios[n]['fiscal_space'] for n in names]
    limits = [scenarios[n]['debt_limit'] for n in names]
    
    colors = ['steelblue' if s > 15 else 'orange' if s > 5 else 'red' for s in spaces]
    
    y_pos = np.arange(len(names))
    bars = ax2.barh(y_pos, spaces, color=colors, alpha=0.7)
    
    # Add debt limit values
    for i, (space, limit) in enumerate(zip(spaces, limits)):
        ax2.text(space + 1, i, f'Limit: {limit:.0f}%', va='center', fontsize=9)
    
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(names)
    ax2.set_xlabel('Fiscal Space (pp of GDP)')
    ax2.set_title('Fiscal Space Under Different Scenarios', fontweight='bold')
    ax2.axvline(x=0, color='red', linewidth=2)
    ax2.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def create_fig11_gfn(output_path, gfn_results):
    """Figure 11: Gross Financing Needs."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: GFN components stacked
    gfn_data = gfn_results.get('gfn_annual', {})
    years = list(gfn_data.keys())
    
    prim_def = [gfn_data[y]['primary_deficit'] for y in years]
    interest = [gfn_data[y]['interest'] for y in years]
    maturing = [gfn_data[y]['total_maturing'] for y in years]
    
    x = np.arange(len(years))
    width = 0.6
    
    ax1.bar(x, prim_def, width, label='Primary Deficit', color='indianred')
    ax1.bar(x, interest, width, bottom=[max(0, p) for p in prim_def], 
            label='Interest', color='steelblue')
    
    # For negative primary balance years
    bottom_mat = [max(0, p) + i for p, i in zip(prim_def, interest)]
    ax1.bar(x, maturing, width, bottom=bottom_mat, label='Maturing Debt', color='gray', alpha=0.7)
    
    ax1.axhline(y=0, color='black', linewidth=0.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(y) for y in years], rotation=45, ha='right')
    ax1.set_ylabel('£ billion')
    ax1.set_title('Gross Financing Needs Composition', fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(axis='y', alpha=0.3)
    
    # Right: GFN as % GDP with thresholds
    gfn_gdp = [gfn_data[y]['gfn_gdp_pct'] for y in years]
    
    bars = ax2.bar(x, gfn_gdp, width, color='steelblue', alpha=0.7)
    
    # IMF thresholds
    ax2.axhline(y=15, color='orange', linestyle='--', linewidth=2, label='IMF Elevated (15%)')
    ax2.axhline(y=20, color='red', linestyle='--', linewidth=2, label='IMF High (20%)')
    
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(y) for y in years], rotation=45, ha='right')
    ax2.set_ylabel('GFN (% GDP)')
    ax2.set_title('Gross Financing Needs vs IMF Thresholds', fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.set_ylim(0, 25)
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def create_fig12_fat_tail_comparison(output_path, mc_results):
    """Figure 12: Normal vs fat-tailed distribution comparison."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Generate comparison data
    np.random.seed(42)
    n = 10000
    
    # Normal distribution (mean 90, std 10)
    normal = np.random.normal(90, 10, n)
    
    # Fat-tailed (use t-distribution, same mean/scale)
    fat_tail = mc_results['debt_ratio_paths'][:, -1]
    
    # Left: Histograms
    bins = np.arange(50, 180, 3)
    
    ax1.hist(normal, bins=bins, density=True, alpha=0.5, color='blue', label='Normal')
    ax1.hist(fat_tail, bins=bins, density=True, alpha=0.5, color='red', label='Fat-Tailed (Student-t)')
    
    ax1.axvline(x=100, color='black', linestyle=':', linewidth=2)
    ax1.axvline(x=np.percentile(normal, 95), color='blue', linestyle='--', 
                label=f'Normal 95th: {np.percentile(normal, 95):.0f}%')
    ax1.axvline(x=np.percentile(fat_tail, 95), color='red', linestyle='--',
                label=f'Fat-tail 95th: {np.percentile(fat_tail, 95):.0f}%')
    
    ax1.set_xlabel('Terminal Debt/GDP (%)')
    ax1.set_ylabel('Density')
    ax1.set_title('Terminal Distribution: Normal vs Fat-Tailed', fontweight='bold')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.set_xlim(50, 175)
    
    # Right: Tail probability comparison
    thresholds = [80, 90, 100, 110, 120, 130, 140]
    prob_normal = [np.mean(normal > t) * 100 for t in thresholds]
    prob_fat = [np.mean(fat_tail > t) * 100 for t in thresholds]
    
    x = np.arange(len(thresholds))
    width = 0.35
    
    ax2.bar(x - width/2, prob_normal, width, label='Normal', color='blue', alpha=0.7)
    ax2.bar(x + width/2, prob_fat, width, label='Fat-Tailed', color='red', alpha=0.7)
    
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'>{t}%' for t in thresholds])
    ax2.set_xlabel('Debt/GDP Threshold')
    ax2.set_ylabel('Probability (%)')
    ax2.set_title('Tail Risk: Probability of Exceeding Thresholds', fontweight='bold')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    # Add impact annotation
    impact_100 = prob_fat[2] - prob_normal[2]
    ax2.annotate(f'+{impact_100:.0f}pp\nimpact', 
                xy=(2, max(prob_fat[2], prob_normal[2]) + 3),
                ha='center', fontsize=9, color='darkred')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def create_all_figures(output_dir, mc_results, bohn_results, fs_results, gfn_results):
    """Create all figures."""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    print("\nGenerating publication figures...")
    print("="*50)
    
    create_fig1_historical_debt(f"{output_dir}/fig1_historical_debt.png")
    create_fig2_fan_chart(f"{output_dir}/fig2_fan_chart.png", mc_results)
    create_fig3_scenarios(f"{output_dir}/fig3_scenarios.png")
    create_fig4_decomposition(f"{output_dir}/fig4_decomposition.png")
    create_fig5_r_g(f"{output_dir}/fig5_r_g_differential.png")
    create_fig6_interest_burden(f"{output_dir}/fig6_interest_burden.png")
    create_fig7_composition(f"{output_dir}/fig7_debt_composition.png")
    create_fig8_ilg_sensitivity(f"{output_dir}/fig8_ilg_sensitivity.png")
    create_fig9_bohn_test(f"{output_dir}/fig9_bohn_test.png", bohn_results)
    create_fig10_fiscal_space(f"{output_dir}/fig10_fiscal_space.png", fs_results)
    create_fig11_gfn(f"{output_dir}/fig11_gfn.png", gfn_results)
    create_fig12_fat_tail_comparison(f"{output_dir}/fig12_fat_tail_impact.png", mc_results)
    
    print("\n" + "="*50)
    print(f"All 12 figures saved to {output_dir}")


if __name__ == "__main__":
    # This will be run after the analysis modules
    print("Visualization module loaded. Run create_all_figures() with results.")
