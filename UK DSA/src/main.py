"""
UK Debt Sustainability Analysis - Main Analysis Script
======================================================
Imperial College London UROP Project

This script runs the complete debt sustainability analysis:
1. Load and process historical data
2. Set up OBR baseline projections
3. Run deterministic scenarios
4. Execute Monte Carlo simulations
5. Generate publication-quality outputs
"""

import numpy as np
import pandas as pd
import warnings
from pathlib import Path
from datetime import datetime

# Import project modules
from config import (
    OBR_GDP, OBR_PSND, OBR_DEBT_TO_GDP, OBR_PSNB, OBR_DEBT_INTEREST,
    OBR_RECEIPTS, OBR_TME, DEBT_COMPOSITION, READY_RECKONERS,
    RPI_INFLATION, CPI_INFLATION, GILT_YIELDS, NOMINAL_GDP_GROWTH,
    SCENARIOS, MONTE_CARLO_CONFIG, FISCAL_RULES, FORECAST_YEARS,
    ILG_SHARE
)
from debt_dynamics import DebtDynamicsModel, FiscalReactionFunction, compute_r_g_differential
from monte_carlo import MonteCarloEngine, MonteCarloConfig, ScenarioSimulator
from visualization import DSAVisualizer, create_all_figures

warnings.filterwarnings('ignore')


def setup_directories():
    """Create output directories if they don't exist."""
    dirs = ['outputs/figures', 'outputs/tables', 'outputs/data']
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
    return Path('outputs')


def create_obr_baseline_dataframe():
    """
    Create DataFrame with OBR March 2025 baseline forecasts.
    """
    years = FORECAST_YEARS[1:]  # Exclude 2023-24 (outturn)
    
    df = pd.DataFrame({
        'year': years,
        'gdp': [OBR_GDP[y] for y in years],
        'debt': [OBR_PSND[y] for y in years],
        'debt_to_gdp': [OBR_DEBT_TO_GDP[y] for y in years],
        'psnb': [OBR_PSNB[y] for y in years],
        'debt_interest': [OBR_DEBT_INTEREST[y] for y in years],
        'receipts': [OBR_RECEIPTS[y] for y in years],
        'tme': [OBR_TME[y] for y in years],
        'rpi_inflation': [RPI_INFLATION[y] / 100 for y in years],
        'cpi_inflation': [CPI_INFLATION[y] / 100 for y in years],
        'gilt_yield': [GILT_YIELDS[y] / 100 for y in years],
        'nominal_gdp_growth': [NOMINAL_GDP_GROWTH[y] / 100 for y in years],
    })
    
    # Compute derived metrics
    df['primary_balance'] = df['psnb'] - df['debt_interest']
    df['primary_balance_gdp'] = df['primary_balance'] / df['gdp'] * 100
    df['debt_interest_gdp'] = df['debt_interest'] / df['gdp'] * 100
    
    return df


def run_debt_dynamics_analysis(baseline_df: pd.DataFrame):
    """
    Run debt dynamics decomposition on OBR baseline.
    """
    model = DebtDynamicsModel()
    
    results = []
    
    # Initial conditions from 2023-24
    debt_prev = OBR_PSND['2023-24']
    gdp_prev = OBR_GDP['2023-24']
    
    for i, row in baseline_df.iterrows():
        # Stock-flow adjustment (approximate as residual)
        if i == 0:
            sfa = 0
        else:
            expected_debt = debt_prev + row['psnb']
            actual_debt = row['debt']
            sfa = actual_debt - expected_debt
        
        result = model.compute_debt_dynamics(
            debt_t_minus_1=debt_prev,
            gdp_t_minus_1=gdp_prev,
            gdp_t=row['gdp'],
            interest_paid=row['debt_interest'],
            primary_balance=row['primary_balance'],
            stock_flow_adj=sfa
        )
        result['year'] = row['year']
        results.append(result)
        
        # Update for next iteration
        debt_prev = row['debt']
        gdp_prev = row['gdp']
    
    return pd.DataFrame(results)


def run_scenario_analysis(baseline_df: pd.DataFrame):
    """
    Run deterministic scenario analysis using OBR ready reckoners.
    """
    scenarios_results = {}
    
    # Baseline
    scenarios_results['Baseline'] = baseline_df[['year', 'debt_to_gdp']].copy()
    scenarios_results['Baseline']['debt_ratio'] = baseline_df['debt_to_gdp']
    
    # Adverse interest rate scenario (+200bp)
    adverse_rates = baseline_df.copy()
    for y in adverse_rates['year']:
        if y in READY_RECKONERS['gilt_rates_1pp']:
            shock_impact = READY_RECKONERS['gilt_rates_1pp'][y] * 2  # 200bp = 2 × 100bp
            adverse_rates.loc[adverse_rates['year'] == y, 'debt_interest'] += shock_impact
    
    # Recalculate debt path
    adverse_debt = [OBR_PSND['2023-24']]
    for i, row in adverse_rates.iterrows():
        new_debt = adverse_debt[-1] + row['psnb'] + (adverse_rates.loc[i, 'debt_interest'] - baseline_df.loc[i, 'debt_interest'])
        adverse_debt.append(new_debt)
    adverse_rates['debt_shocked'] = adverse_debt[1:]
    adverse_rates['debt_ratio'] = adverse_rates['debt_shocked'] / adverse_rates['gdp'] * 100
    scenarios_results['Adverse Rates (+200bp)'] = adverse_rates[['year', 'debt_ratio']]
    
    # High inflation scenario (+3pp RPI)
    high_inflation = baseline_df.copy()
    for y in high_inflation['year']:
        if y in READY_RECKONERS['inflation_1pp']:
            shock_impact = READY_RECKONERS['inflation_1pp'][y] * 3  # 3pp
            high_inflation.loc[high_inflation['year'] == y, 'debt_interest'] += shock_impact
    
    inflation_debt = [OBR_PSND['2023-24']]
    for i, row in high_inflation.iterrows():
        new_debt = inflation_debt[-1] + row['psnb'] + (high_inflation.loc[i, 'debt_interest'] - baseline_df.loc[i, 'debt_interest'])
        inflation_debt.append(new_debt)
    high_inflation['debt_shocked'] = inflation_debt[1:]
    high_inflation['debt_ratio'] = high_inflation['debt_shocked'] / high_inflation['gdp'] * 100
    scenarios_results['High Inflation (+3pp RPI)'] = high_inflation[['year', 'debt_ratio']]
    
    # Combined adverse
    combined = baseline_df.copy()
    for y in combined['year']:
        impact = 0
        if y in READY_RECKONERS['gilt_rates_1pp']:
            impact += READY_RECKONERS['gilt_rates_1pp'][y] * 2
        if y in READY_RECKONERS['inflation_1pp']:
            impact += READY_RECKONERS['inflation_1pp'][y] * 2
        if y in READY_RECKONERS['short_rates_1pp']:
            impact += READY_RECKONERS['short_rates_1pp'][y] * 1.5
        combined.loc[combined['year'] == y, 'debt_interest'] += impact
    
    combined_debt = [OBR_PSND['2023-24']]
    for i, row in combined.iterrows():
        # Also add growth shock effect (higher borrowing)
        growth_shock = row['gdp'] * 0.01  # ~1% GDP additional borrowing
        new_debt = combined_debt[-1] + row['psnb'] + growth_shock + (combined.loc[i, 'debt_interest'] - baseline_df.loc[i, 'debt_interest'])
        combined_debt.append(new_debt)
    combined['debt_shocked'] = combined_debt[1:]
    combined['gdp_shocked'] = combined['gdp'] * 0.97  # Lower GDP
    combined['debt_ratio'] = combined['debt_shocked'] / combined['gdp_shocked'] * 100
    scenarios_results['Combined Adverse'] = combined[['year', 'debt_ratio']]
    
    # Fiscal consolidation (+1% GDP primary surplus)
    consolidation = baseline_df.copy()
    consolidation_debt = [OBR_PSND['2023-24']]
    cumulative_savings = 0
    for i, row in consolidation.iterrows():
        annual_saving = row['gdp'] * 0.01  # 1% GDP savings
        cumulative_savings += annual_saving
        new_debt = consolidation_debt[-1] + row['psnb'] - annual_saving
        consolidation_debt.append(new_debt)
    consolidation['debt_shocked'] = consolidation_debt[1:]
    consolidation['debt_ratio'] = consolidation['debt_shocked'] / consolidation['gdp'] * 100
    scenarios_results['Fiscal Consolidation'] = consolidation[['year', 'debt_ratio']]
    
    return scenarios_results


def run_monte_carlo_simulation():
    """
    Run Monte Carlo simulation for stochastic DSA.
    """
    # Configure simulation
    config = MonteCarloConfig(
        n_simulations=10000,
        horizon_years=10,
        random_seed=42,
        real_gdp_growth_mean=0.015,
        real_gdp_growth_std=0.020,
        cpi_inflation_mean=0.020,
        cpi_inflation_std=0.015,
        rpi_inflation_mean=0.030,
        rpi_inflation_std=0.018,
        gilt_yield_mean=0.045,
        gilt_yield_std=0.010,
    )
    
    engine = MonteCarloEngine(config)
    
    # Initial conditions from 2024-25
    results = engine.run_full_simulation(
        initial_debt=OBR_PSND['2024-25'],
        initial_gdp=OBR_GDP['2024-25'],
        ilg_share=ILG_SHARE,
        baseline_primary_balance_gdp=-0.02  # Approximate from PSNB
    )
    
    return results


def generate_summary_tables(
    baseline_df: pd.DataFrame,
    decomposition_df: pd.DataFrame,
    mc_results: dict,
    scenarios: dict
):
    """
    Generate summary tables for the paper.
    """
    tables = {}
    
    # Table 1: OBR Baseline Fiscal Projections
    table1 = baseline_df[[
        'year', 'gdp', 'debt_to_gdp', 'psnb', 'debt_interest',
        'primary_balance_gdp', 'debt_interest_gdp'
    ]].copy()
    table1.columns = [
        'Fiscal Year', 'Nominal GDP (£bn)', 'Debt/GDP (%)', 
        'PSNB (£bn)', 'Debt Interest (£bn)',
        'Primary Balance (% GDP)', 'Debt Interest (% GDP)'
    ]
    tables['baseline_projections'] = table1.round(1)
    
    # Table 2: Debt Dynamics Decomposition
    table2 = decomposition_df[[
        'year', 'change_in_debt_ratio', 'interest_growth_effect',
        'primary_balance_effect', 'sfa_effect', 'r_minus_g'
    ]].copy()
    table2.columns = [
        'Fiscal Year', 'Change in Debt/GDP (pp)', 'r-g Effect (pp)',
        'Primary Balance (pp)', 'Stock-Flow Adj. (pp)', 'r-g Differential (%)'
    ]
    tables['debt_decomposition'] = table2.round(2)
    
    # Table 3: Monte Carlo Summary Statistics
    summary = mc_results['summary']
    table3 = pd.DataFrame({
        'Metric': [
            'Mean Final Debt/GDP',
            'Median Final Debt/GDP',
            'Standard Deviation',
            '5th Percentile',
            '95th Percentile',
            'Prob(Debt > 100%)',
            'Prob(Debt > 120%)'
        ],
        'Value': [
            f"{summary['final_debt_ratio_mean']:.1f}%",
            f"{summary['final_debt_ratio_median']:.1f}%",
            f"{summary['final_debt_ratio_std']:.1f}pp",
            f"{summary['final_debt_ratio_p5']:.1f}%",
            f"{summary['final_debt_ratio_p95']:.1f}%",
            f"{summary['prob_above_100_final']*100:.1f}%",
            f"{summary['prob_above_120_final']*100:.1f}%"
        ]
    })
    tables['monte_carlo_summary'] = table3
    
    # Table 4: Scenario Comparison
    scenario_df = pd.DataFrame()
    for name, data in scenarios.items():
        scenario_df[name] = data['debt_ratio'].values
    scenario_df.index = scenarios['Baseline']['year'].values
    tables['scenario_comparison'] = scenario_df.round(1)
    
    # Table 5: Ready Reckoner Sensitivities
    table5 = pd.DataFrame({
        'Shock': ['+1pp Gilt Yields', '+1pp Short Rates', '+1pp RPI Inflation'],
        '2025-26': [
            f"£{READY_RECKONERS['gilt_rates_1pp']['2025-26']:.1f}bn",
            f"£{READY_RECKONERS['short_rates_1pp']['2025-26']:.1f}bn",
            f"£{READY_RECKONERS['inflation_1pp']['2025-26']:.1f}bn"
        ],
        '2027-28': [
            f"£{READY_RECKONERS['gilt_rates_1pp']['2027-28']:.1f}bn",
            f"£{READY_RECKONERS['short_rates_1pp']['2027-28']:.1f}bn",
            f"£{READY_RECKONERS['inflation_1pp']['2027-28']:.1f}bn"
        ],
        '2029-30': [
            f"£{READY_RECKONERS['gilt_rates_1pp']['2029-30']:.1f}bn",
            f"£{READY_RECKONERS['short_rates_1pp']['2029-30']:.1f}bn",
            f"£{READY_RECKONERS['inflation_1pp']['2029-30']:.1f}bn"
        ]
    })
    tables['sensitivity_analysis'] = table5
    
    return tables


def export_results(tables: dict, output_dir: Path):
    """
    Export tables to various formats.
    """
    # Excel workbook
    excel_path = output_dir / 'tables' / 'dsa_results.xlsx'
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        for name, df in tables.items():
            df.to_excel(writer, sheet_name=name[:31], index=False)
    
    # LaTeX tables
    latex_dir = output_dir / 'tables' / 'latex'
    latex_dir.mkdir(exist_ok=True)
    
    for name, df in tables.items():
        latex_path = latex_dir / f'{name}.tex'
        df.to_latex(latex_path, index=False, escape=False)
    
    print(f"Results exported to {output_dir / 'tables'}")


def main():
    """
    Main execution function.
    """
    print("=" * 60)
    print("UK DEBT SUSTAINABILITY ANALYSIS")
    print("Imperial College London UROP Project")
    print(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)
    
    # Setup
    output_dir = setup_directories()
    
    # Step 1: Create OBR baseline
    print("\n[1/5] Creating OBR baseline projections...")
    baseline_df = create_obr_baseline_dataframe()
    print(f"    Baseline covers: {baseline_df['year'].iloc[0]} to {baseline_df['year'].iloc[-1]}")
    print(f"    Terminal debt/GDP: {baseline_df['debt_to_gdp'].iloc[-1]:.1f}%")
    
    # Step 2: Debt dynamics decomposition
    print("\n[2/5] Running debt dynamics decomposition...")
    decomposition_df = run_debt_dynamics_analysis(baseline_df)
    avg_r_g = decomposition_df['r_minus_g'].mean()
    print(f"    Average r-g differential: {avg_r_g:.2f}pp")
    
    # Step 3: Scenario analysis
    print("\n[3/5] Running scenario analysis...")
    scenarios = run_scenario_analysis(baseline_df)
    print(f"    Scenarios analyzed: {len(scenarios)}")
    for name, data in scenarios.items():
        terminal = data['debt_ratio'].iloc[-1]
        print(f"      - {name}: {terminal:.1f}%")
    
    # Step 4: Monte Carlo simulation
    print("\n[4/5] Running Monte Carlo simulation (10,000 paths)...")
    mc_results = run_monte_carlo_simulation()
    summary = mc_results['summary']
    print(f"    Median terminal debt/GDP: {summary['final_debt_ratio_median']:.1f}%")
    print(f"    90% confidence interval: [{summary['final_debt_ratio_p5']:.1f}%, {summary['final_debt_ratio_p95']:.1f}%]")
    print(f"    Prob(debt > 100%): {summary['prob_above_100_final']*100:.1f}%")
    
    # Step 5: Generate outputs
    print("\n[5/5] Generating outputs...")
    
    # Tables
    tables = generate_summary_tables(baseline_df, decomposition_df, mc_results, scenarios)
    export_results(tables, output_dir)
    
    # Figures
    viz = DSAVisualizer()
    
    # Fan chart
    viz.plot_fan_chart(
        mc_results['fan_chart_data'],
        baseline_path=baseline_df['debt_to_gdp'].values,
        title='UK Public Sector Net Debt: Monte Carlo Projections (10,000 paths)',
        save_path=str(output_dir / 'figures' / 'fan_chart.png')
    )
    
    # Debt decomposition
    viz.plot_debt_decomposition(
        decomposition_df,
        title='UK Debt Dynamics Decomposition (OBR March 2025)',
        save_path=str(output_dir / 'figures' / 'debt_decomposition.png')
    )
    
    # Scenario comparison
    viz.plot_scenario_comparison(
        scenarios,
        metric='debt_ratio',
        title='UK Debt-to-GDP Under Alternative Scenarios',
        save_path=str(output_dir / 'figures' / 'scenario_comparison.png')
    )
    
    # r-g differential
    viz.plot_r_g_differential(
        decomposition_df,
        title='Interest Rate-Growth Differential (r-g): OBR Forecast',
        save_path=str(output_dir / 'figures' / 'r_g_differential.png')
    )
    
    # Probability of breach
    viz.plot_probability_breach(
        mc_results['risk_metrics'],
        title='Probability of Exceeding Debt Thresholds (Monte Carlo)',
        save_path=str(output_dir / 'figures' / 'probability_breach.png')
    )
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"\nOutputs saved to: {output_dir}")
    print("\nKey files:")
    print("  - figures/fan_chart.png")
    print("  - figures/scenario_comparison.png")
    print("  - figures/debt_decomposition.png")
    print("  - tables/dsa_results.xlsx")
    print("  - tables/latex/*.tex")
    
    return {
        'baseline': baseline_df,
        'decomposition': decomposition_df,
        'scenarios': scenarios,
        'monte_carlo': mc_results,
        'tables': tables
    }


if __name__ == '__main__':
    results = main()
