"""
Updated Monte Carlo Simulation - OBR Baseline Approach
======================================================
Uses November 2025 OBR projected primary balance path as baseline,
with stochastic shocks around that path.

This represents the "policy success" scenario where government achieves
stated consolidation, but faces macroeconomic uncertainty.
"""

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

def monte_carlo_obr_baseline(
    initial_debt=96.0,
    n_simulations=10000,
    n_years=10,
    # OBR November 2025 projections
    obr_primary_balance=[-1.5, -0.6, -0.1, 0.5, 1.3, 1.4, 1.4, 1.4, 1.4, 1.4],  # % GDP
    obr_growth=[1.5, 1.4, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5],  # % real
    # Interest rate assumptions
    r_base=4.5,  # Base effective interest rate
    ilg_share=0.34,  # Index-linked gilt share
    # Shock parameters (from MLE estimation)
    sigma_growth=2.0,
    sigma_rates=1.0,
    sigma_pb=1.5,
    sigma_inflation=1.0,
    df_growth=3,  # Degrees of freedom (bounded from MLE estimate)
    df_rates=4,
    df_inflation=5,
    # Correlation matrix (estimated from UK data)
    correlation_matrix=None,
    seed=42
):
    """
    Monte Carlo with OBR baseline primary balance path.
    
    Key difference from fiscal reaction version: 
    - Uses projected consolidation path as baseline
    - Shocks represent deviations from planned fiscal policy
    - Does not assume negative fiscal reaction continues
    """
    np.random.seed(seed)
    
    if correlation_matrix is None:
        # Estimated from UK data
        correlation_matrix = np.array([
            [1.0, 0.4, 0.2],   # Growth
            [0.4, 1.0, 0.4],   # Rates  
            [0.2, 0.4, 1.0]    # Inflation
        ])
    
    # Ensure correlation matrix is positive definite
    eigvals = np.linalg.eigvals(correlation_matrix)
    if np.any(eigvals <= 0):
        correlation_matrix = np.eye(3)  # Fall back to independent
    
    L = np.linalg.cholesky(correlation_matrix)
    
    # Initialize paths
    debt_paths = np.zeros((n_simulations, n_years + 1))
    debt_paths[:, 0] = initial_debt
    
    # Store annual results
    annual_stats = []
    
    for t in range(n_years):
        # OBR baseline values for this year
        pb_baseline = obr_primary_balance[min(t, len(obr_primary_balance)-1)]
        g_baseline = obr_growth[min(t, len(obr_growth)-1)]
        
        # Generate correlated shocks
        z = np.random.normal(0, 1, (n_simulations, 3))
        z_corr = (L @ z.T).T
        
        # Transform to fat-tailed (t) distributions
        u = stats.norm.cdf(z_corr)
        
        shock_growth = stats.t.ppf(u[:, 0], df=df_growth) * sigma_growth
        shock_rates = stats.t.ppf(u[:, 1], df=df_rates) * sigma_rates
        shock_inflation = stats.t.ppf(u[:, 2], df=df_inflation) * sigma_inflation
        
        # Realized values
        g_real = g_baseline + shock_growth
        inflation = 2.0 + shock_inflation  # Base inflation 2%
        g_nominal = g_real + inflation
        
        # Interest rate with ILG sensitivity
        r_conv = r_base + shock_rates * 0.7  # Conventional component
        r_ilg = r_base - 1.5 + inflation + shock_rates * 0.3  # ILG component (real rate + inflation)
        r_effective = r_conv * (1 - ilg_share) + r_ilg * ilg_share
        r_effective = np.maximum(r_effective, 0.5)  # Floor at 0.5%
        
        # Primary balance shock (reduced variance - government adjusts)
        shock_pb = np.random.standard_t(df=5, size=n_simulations) * sigma_pb * 0.5
        pb = pb_baseline + shock_pb
        
        # Debt dynamics
        # d_t = d_{t-1} * (1 + r) / (1 + g) - pb
        debt_prev = debt_paths[:, t]
        growth_factor = (1 + r_effective/100) / (1 + g_nominal/100)
        debt_new = debt_prev * growth_factor - pb
        
        # Ensure non-negative debt
        debt_paths[:, t + 1] = np.maximum(debt_new, 0)
        
        # Annual statistics
        annual_stats.append({
            'year': t + 1,
            'debt_mean': np.mean(debt_paths[:, t + 1]),
            'debt_median': np.median(debt_paths[:, t + 1]),
            'debt_std': np.std(debt_paths[:, t + 1]),
            'debt_5th': np.percentile(debt_paths[:, t + 1], 5),
            'debt_95th': np.percentile(debt_paths[:, t + 1], 95),
            'prob_above_100': np.mean(debt_paths[:, t + 1] > 100) * 100,
            'prob_above_120': np.mean(debt_paths[:, t + 1] > 120) * 100
        })
    
    # Terminal distribution statistics
    terminal_debt = debt_paths[:, -1]
    
    results = {
        'debt_paths': debt_paths,
        'annual_stats': pd.DataFrame(annual_stats),
        'terminal_stats': {
            'mean': np.mean(terminal_debt),
            'median': np.median(terminal_debt),
            'std': np.std(terminal_debt),
            'skewness': stats.skew(terminal_debt),
            'kurtosis': stats.kurtosis(terminal_debt),
            'percentiles': {
                '1st': np.percentile(terminal_debt, 1),
                '5th': np.percentile(terminal_debt, 5),
                '10th': np.percentile(terminal_debt, 10),
                '25th': np.percentile(terminal_debt, 25),
                '50th': np.percentile(terminal_debt, 50),
                '75th': np.percentile(terminal_debt, 75),
                '90th': np.percentile(terminal_debt, 90),
                '95th': np.percentile(terminal_debt, 95),
                '99th': np.percentile(terminal_debt, 99)
            },
            'VaR_95': np.percentile(terminal_debt, 95),
            'VaR_99': np.percentile(terminal_debt, 99),
            'ES_95': np.mean(terminal_debt[terminal_debt > np.percentile(terminal_debt, 95)]),
            'ES_99': np.mean(terminal_debt[terminal_debt > np.percentile(terminal_debt, 99)])
        },
        'breach_probabilities': {
            'terminal_100': np.mean(terminal_debt > 100) * 100,
            'terminal_110': np.mean(terminal_debt > 110) * 100,
            'terminal_120': np.mean(terminal_debt > 120) * 100,
            'ever_100': np.mean(np.max(debt_paths, axis=1) > 100) * 100,
            'ever_110': np.mean(np.max(debt_paths, axis=1) > 110) * 100,
            'ever_120': np.mean(np.max(debt_paths, axis=1) > 120) * 100
        },
        'comparison_normal': compare_with_normal(debt_paths, initial_debt, obr_primary_balance, obr_growth, r_base, n_simulations, n_years)
    }
    
    return results


def compare_with_normal(debt_paths, initial_debt, obr_pb, obr_growth, r_base, n_sims, n_years):
    """Compare fat-tailed results with normal distribution assumption."""
    np.random.seed(43)  # Different seed for comparison
    
    debt_normal = np.zeros((n_sims, n_years + 1))
    debt_normal[:, 0] = initial_debt
    
    for t in range(n_years):
        pb = obr_pb[min(t, len(obr_pb)-1)] + np.random.normal(0, 1.5*0.5, n_sims)
        g = obr_growth[min(t, len(obr_growth)-1)] + np.random.normal(0, 2.0, n_sims)
        r = r_base + np.random.normal(0, 1.0, n_sims)
        
        inflation = 2.0 + np.random.normal(0, 1.0, n_sims)
        g_nom = g + inflation
        
        debt_prev = debt_normal[:, t]
        growth_factor = (1 + r/100) / (1 + g_nom/100)
        debt_normal[:, t + 1] = np.maximum(debt_prev * growth_factor - pb, 0)
    
    terminal_normal = debt_normal[:, -1]
    terminal_fat = debt_paths[:, -1]
    
    return {
        'normal_mean': np.mean(terminal_normal),
        'normal_std': np.std(terminal_normal),
        'normal_VaR95': np.percentile(terminal_normal, 95),
        'normal_VaR99': np.percentile(terminal_normal, 99),
        'normal_prob_100': np.mean(terminal_normal > 100) * 100,
        'fat_mean': np.mean(terminal_fat),
        'fat_std': np.std(terminal_fat),
        'fat_VaR95': np.percentile(terminal_fat, 95),
        'fat_VaR99': np.percentile(terminal_fat, 99),
        'fat_prob_100': np.mean(terminal_fat > 100) * 100
    }


def run_scenario_analysis():
    """Run Monte Carlo under different scenarios."""
    
    scenarios = {
        'Baseline (OBR)': {
            'obr_primary_balance': [-1.5, -0.6, -0.1, 0.5, 1.3, 1.4, 1.4, 1.4, 1.4, 1.4],
            'obr_growth': [1.5, 1.4, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5],
            'r_base': 4.5
        },
        'Higher rates (+200bp)': {
            'obr_primary_balance': [-1.5, -0.6, -0.1, 0.5, 1.3, 1.4, 1.4, 1.4, 1.4, 1.4],
            'obr_growth': [1.5, 1.4, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5],
            'r_base': 6.5
        },
        'Lower growth': {
            'obr_primary_balance': [-1.5, -0.6, -0.1, 0.5, 1.3, 1.4, 1.4, 1.4, 1.4, 1.4],
            'obr_growth': [1.0, 0.9, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            'r_base': 4.5
        },
        'Fiscal slippage': {
            'obr_primary_balance': [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 0.5, 0.5, 0.5, 0.5],
            'obr_growth': [1.5, 1.4, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5],
            'r_base': 4.5
        },
        'Stagflation': {
            'obr_primary_balance': [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 0.5, 0.5, 0.5, 0.5],
            'obr_growth': [0.5, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            'r_base': 5.5
        },
        'Consolidation success': {
            'obr_primary_balance': [-1.0, 0.0, 0.5, 1.0, 1.5, 2.0, 2.0, 2.0, 2.0, 2.0],
            'obr_growth': [1.5, 1.4, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5],
            'r_base': 4.0
        }
    }
    
    results = {}
    for name, params in scenarios.items():
        print(f"Running {name}...")
        results[name] = monte_carlo_obr_baseline(
            initial_debt=96.0,
            n_simulations=10000,
            n_years=10,
            **params,
            seed=42
        )
    
    return results


def print_results_table(results):
    """Print formatted results table."""
    
    print("\n" + "="*80)
    print("REVISED MONTE CARLO RESULTS - OBR BASELINE WITH FAT-TAILED SHOCKS")
    print("="*80)
    
    print("\n{:<25} {:>10} {:>10} {:>10} {:>10} {:>10}".format(
        "Scenario", "Mean", "Median", "VaR95", "VaR99", "P(>100%)"
    ))
    print("-"*80)
    
    for name, res in results.items():
        ts = res['terminal_stats']
        bp = res['breach_probabilities']
        print("{:<25} {:>10.1f} {:>10.1f} {:>10.1f} {:>10.1f} {:>10.1f}%".format(
            name,
            ts['mean'],
            ts['median'],
            ts['VaR_95'],
            ts['VaR_99'],
            bp['terminal_100']
        ))
    
    print("\n" + "-"*80)
    print("FAT-TAIL IMPACT (Baseline scenario):")
    baseline = results['Baseline (OBR)']
    comp = baseline['comparison_normal']
    print(f"  Normal distribution:  VaR95 = {comp['normal_VaR95']:.1f}%, P(>100%) = {comp['normal_prob_100']:.1f}%")
    print(f"  Fat-tailed (t-dist):  VaR95 = {comp['fat_VaR95']:.1f}%, P(>100%) = {comp['fat_prob_100']:.1f}%")
    print(f"  Difference:           VaR95 = +{comp['fat_VaR95']-comp['normal_VaR95']:.1f}pp, P(>100%) = +{comp['fat_prob_100']-comp['normal_prob_100']:.1f}pp")


def generate_fan_chart(results, scenario='Baseline (OBR)'):
    """Generate fan chart visualization."""
    
    res = results[scenario]
    debt_paths = res['debt_paths']
    
    years = np.arange(2025, 2036)  # 2025-26 to 2034-35
    
    # Calculate percentiles
    p5 = np.percentile(debt_paths, 5, axis=0)
    p10 = np.percentile(debt_paths, 10, axis=0)
    p25 = np.percentile(debt_paths, 25, axis=0)
    p50 = np.percentile(debt_paths, 50, axis=0)
    p75 = np.percentile(debt_paths, 75, axis=0)
    p90 = np.percentile(debt_paths, 90, axis=0)
    p95 = np.percentile(debt_paths, 95, axis=0)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Fan chart
    ax.fill_between(years, p5, p95, alpha=0.2, color='blue', label='90% CI')
    ax.fill_between(years, p10, p90, alpha=0.3, color='blue', label='80% CI')
    ax.fill_between(years, p25, p75, alpha=0.4, color='blue', label='50% CI')
    ax.plot(years, p50, 'b-', linewidth=2, label='Median')
    
    # OBR central projection
    obr_debt = [95.0, 95.3, 96.3, 97.0, 96.8, 96.1, 95.5, 95.0, 94.5, 94.0, 93.5]
    ax.plot(years, obr_debt, 'k--', linewidth=2, label='OBR Central')
    
    # Reference lines
    ax.axhline(y=100, color='red', linestyle='--', alpha=0.7, label='100% threshold')
    ax.axhline(y=120, color='red', linestyle=':', alpha=0.5, label='120% threshold')
    
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Public Sector Net Debt (% GDP)', fontsize=12)
    ax.set_title(f'UK Debt Fan Chart - {scenario}\n(Fat-tailed Monte Carlo, 10,000 simulations)', fontsize=14)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(2025, 2035)
    ax.set_ylim(70, 140)
    
    plt.tight_layout()
    plt.savefig('/home/claude/uk_dsa/outputs/revised_fan_chart.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Fan chart saved to /home/claude/uk_dsa/outputs/revised_fan_chart.png")


if __name__ == "__main__":
    print("Running revised Monte Carlo simulations...")
    
    results = run_scenario_analysis()
    print_results_table(results)
    
    print("\nGenerating fan chart...")
    generate_fan_chart(results)
    
    # Save detailed results
    baseline = results['Baseline (OBR)']
    
    print("\n" + "="*80)
    print("DETAILED BASELINE RESULTS")
    print("="*80)
    
    print("\nTerminal Distribution (2034-35):")
    ts = baseline['terminal_stats']
    print(f"  Mean:      {ts['mean']:.1f}%")
    print(f"  Median:    {ts['median']:.1f}%")
    print(f"  Std Dev:   {ts['std']:.1f}%")
    print(f"  Skewness:  {ts['skewness']:.2f}")
    print(f"  Kurtosis:  {ts['kurtosis']:.2f}")
    
    print("\nPercentiles:")
    for pct, val in ts['percentiles'].items():
        print(f"  {pct}: {val:.1f}%")
    
    print("\nRisk Measures:")
    print(f"  VaR 95%:   {ts['VaR_95']:.1f}%")
    print(f"  VaR 99%:   {ts['VaR_99']:.1f}%")
    print(f"  ES 95%:    {ts['ES_95']:.1f}%")
    print(f"  ES 99%:    {ts['ES_99']:.1f}%")
    
    print("\nBreach Probabilities:")
    bp = baseline['breach_probabilities']
    print(f"  P(>100% terminal): {bp['terminal_100']:.1f}%")
    print(f"  P(>100% ever):     {bp['ever_100']:.1f}%")
    print(f"  P(>110% terminal): {bp['terminal_110']:.1f}%")
    print(f"  P(>120% terminal): {bp['terminal_120']:.1f}%")
    
    print("\nAnnual Evolution:")
    print(baseline['annual_stats'].to_string(index=False))
