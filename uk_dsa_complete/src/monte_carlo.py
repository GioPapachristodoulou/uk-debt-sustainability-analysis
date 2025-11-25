"""
UK Debt Sustainability Analysis - Monte Carlo Simulation Engine
===============================================================
Imperial College London UROP Project

This module implements stochastic debt sustainability analysis using
Monte Carlo simulation with correlated macroeconomic shocks.

Key features:
- Correlated multivariate normal shocks (GDP, inflation, yields)
- AR(1) persistence in shocks
- Fan chart generation with confidence intervals
- Probability assessment of fiscal risks
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings


@dataclass
class MonteCarloConfig:
    """Configuration for Monte Carlo simulation"""
    n_simulations: int = 10000
    horizon_years: int = 10
    random_seed: int = 42
    
    # Stochastic process parameters
    real_gdp_growth_mean: float = 0.015
    real_gdp_growth_std: float = 0.020
    real_gdp_growth_persistence: float = 0.3
    
    cpi_inflation_mean: float = 0.020
    cpi_inflation_std: float = 0.015
    cpi_inflation_persistence: float = 0.5
    
    rpi_inflation_mean: float = 0.030
    rpi_inflation_std: float = 0.018
    rpi_inflation_persistence: float = 0.5
    
    gilt_yield_mean: float = 0.045
    gilt_yield_std: float = 0.010
    gilt_yield_persistence: float = 0.8
    
    # Percentiles for fan charts
    percentiles: List[float] = None
    
    def __post_init__(self):
        if self.percentiles is None:
            self.percentiles = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]


class MonteCarloEngine:
    """
    Monte Carlo simulation engine for debt sustainability analysis.
    
    Generates stochastic paths for macroeconomic variables and
    computes resulting debt trajectories.
    """
    
    def __init__(self, config: MonteCarloConfig):
        """
        Initialize Monte Carlo engine.
        
        Args:
            config: MonteCarloConfig object with simulation parameters
        """
        self.config = config
        self.rng = np.random.default_rng(config.random_seed)
        
        # Build correlation matrix
        # Variables: [real_gdp_growth, cpi_inflation, rpi_inflation, gilt_yield]
        self.correlation_matrix = np.array([
            [1.00, -0.20, -0.15, 0.10],   # Real GDP growth
            [-0.20, 1.00, 0.95, 0.30],    # CPI inflation
            [-0.15, 0.95, 1.00, 0.35],    # RPI inflation
            [0.10, 0.30, 0.35, 1.00]      # Gilt yield
        ])
        
        # Compute Cholesky decomposition for correlated draws
        self.cholesky = np.linalg.cholesky(self.correlation_matrix)
        
    def generate_correlated_shocks(
        self,
        n_sims: int,
        n_periods: int
    ) -> np.ndarray:
        """
        Generate correlated standard normal shocks.
        
        Args:
            n_sims: Number of simulations
            n_periods: Number of time periods
            
        Returns:
            Array of shape (n_sims, n_periods, 4) with correlated shocks
        """
        # Independent standard normals
        independent = self.rng.standard_normal((n_sims, n_periods, 4))
        
        # Apply Cholesky transformation to induce correlation
        correlated = np.zeros_like(independent)
        for t in range(n_periods):
            correlated[:, t, :] = independent[:, t, :] @ self.cholesky.T
            
        return correlated
    
    def simulate_macro_paths(self) -> Dict[str, np.ndarray]:
        """
        Simulate paths for macroeconomic variables.
        
        Returns:
            Dictionary with arrays of shape (n_sims, horizon_years)
            for each variable
        """
        n_sims = self.config.n_simulations
        horizon = self.config.horizon_years
        
        # Generate correlated shocks
        shocks = self.generate_correlated_shocks(n_sims, horizon)
        
        # Initialize paths with long-run means
        real_gdp_growth = np.zeros((n_sims, horizon))
        cpi_inflation = np.zeros((n_sims, horizon))
        rpi_inflation = np.zeros((n_sims, horizon))
        gilt_yield = np.zeros((n_sims, horizon))
        
        # Parameters
        cfg = self.config
        
        # AR(1) simulation for each variable
        for t in range(horizon):
            if t == 0:
                # Start from unconditional means
                real_gdp_growth[:, t] = (
                    cfg.real_gdp_growth_mean + 
                    cfg.real_gdp_growth_std * shocks[:, t, 0]
                )
                cpi_inflation[:, t] = (
                    cfg.cpi_inflation_mean + 
                    cfg.cpi_inflation_std * shocks[:, t, 1]
                )
                rpi_inflation[:, t] = (
                    cfg.rpi_inflation_mean + 
                    cfg.rpi_inflation_std * shocks[:, t, 2]
                )
                gilt_yield[:, t] = (
                    cfg.gilt_yield_mean + 
                    cfg.gilt_yield_std * shocks[:, t, 3]
                )
            else:
                # AR(1) dynamics with mean reversion
                real_gdp_growth[:, t] = (
                    cfg.real_gdp_growth_mean * (1 - cfg.real_gdp_growth_persistence) +
                    cfg.real_gdp_growth_persistence * real_gdp_growth[:, t-1] +
                    cfg.real_gdp_growth_std * np.sqrt(1 - cfg.real_gdp_growth_persistence**2) * shocks[:, t, 0]
                )
                cpi_inflation[:, t] = (
                    cfg.cpi_inflation_mean * (1 - cfg.cpi_inflation_persistence) +
                    cfg.cpi_inflation_persistence * cpi_inflation[:, t-1] +
                    cfg.cpi_inflation_std * np.sqrt(1 - cfg.cpi_inflation_persistence**2) * shocks[:, t, 1]
                )
                rpi_inflation[:, t] = (
                    cfg.rpi_inflation_mean * (1 - cfg.rpi_inflation_persistence) +
                    cfg.rpi_inflation_persistence * rpi_inflation[:, t-1] +
                    cfg.rpi_inflation_std * np.sqrt(1 - cfg.rpi_inflation_persistence**2) * shocks[:, t, 2]
                )
                gilt_yield[:, t] = (
                    cfg.gilt_yield_mean * (1 - cfg.gilt_yield_persistence) +
                    cfg.gilt_yield_persistence * gilt_yield[:, t-1] +
                    cfg.gilt_yield_std * np.sqrt(1 - cfg.gilt_yield_persistence**2) * shocks[:, t, 3]
                )
        
        # Ensure non-negative yields
        gilt_yield = np.maximum(gilt_yield, 0.005)  # Floor at 0.5%
        
        return {
            'real_gdp_growth': real_gdp_growth,
            'cpi_inflation': cpi_inflation,
            'rpi_inflation': rpi_inflation,
            'gilt_yield': gilt_yield
        }
    
    def simulate_debt_paths(
        self,
        initial_debt: float,
        initial_gdp: float,
        ilg_share: float = 0.22,
        baseline_primary_balance_gdp: float = -0.02
    ) -> Dict[str, np.ndarray]:
        """
        Simulate debt-to-GDP ratio paths.
        
        Args:
            initial_debt: Initial debt level (£bn)
            initial_gdp: Initial GDP level (£bn)
            ilg_share: Share of debt that is index-linked
            baseline_primary_balance_gdp: Primary balance as fraction of GDP
            
        Returns:
            Dictionary with debt paths and statistics
        """
        # Generate macro paths
        macro_paths = self.simulate_macro_paths()
        
        n_sims = self.config.n_simulations
        horizon = self.config.horizon_years
        
        # Initialize arrays
        debt = np.zeros((n_sims, horizon + 1))
        gdp = np.zeros((n_sims, horizon + 1))
        debt_ratio = np.zeros((n_sims, horizon + 1))
        interest_payments = np.zeros((n_sims, horizon))
        
        # Initial values
        debt[:, 0] = initial_debt
        gdp[:, 0] = initial_gdp
        debt_ratio[:, 0] = initial_debt / initial_gdp
        
        for t in range(horizon):
            # Nominal GDP growth = real growth + inflation
            nominal_growth = (
                macro_paths['real_gdp_growth'][:, t] + 
                macro_paths['cpi_inflation'][:, t]
            )
            
            # Update GDP
            gdp[:, t+1] = gdp[:, t] * (1 + nominal_growth)
            
            # Effective interest rate
            # Conventional debt: gilt yield
            # Index-linked debt: real yield + RPI inflation
            conventional_rate = macro_paths['gilt_yield'][:, t]
            ilg_rate = (
                macro_paths['gilt_yield'][:, t] - 
                macro_paths['cpi_inflation'][:, t] + 
                macro_paths['rpi_inflation'][:, t]
            )
            effective_rate = (
                (1 - ilg_share) * conventional_rate + 
                ilg_share * ilg_rate
            )
            
            # Interest payments
            interest_payments[:, t] = debt[:, t] * effective_rate
            
            # Primary balance (as £bn)
            primary_balance = baseline_primary_balance_gdp * gdp[:, t]
            
            # Simple fiscal reaction: tighten if debt ratio high
            reaction_coefficient = 0.01  # 1% adjustment per 10pp above 90%
            debt_feedback = np.maximum(debt_ratio[:, t] - 0.90, 0) * reaction_coefficient * gdp[:, t]
            primary_balance = primary_balance + debt_feedback
            
            # Debt accumulation
            debt[:, t+1] = (
                debt[:, t] + 
                interest_payments[:, t] + 
                primary_balance
            )
            
            # Debt ratio
            debt_ratio[:, t+1] = debt[:, t+1] / gdp[:, t+1]
        
        return {
            'debt': debt,
            'gdp': gdp,
            'debt_ratio': debt_ratio * 100,  # As percentage
            'interest_payments': interest_payments,
            'macro_paths': macro_paths
        }
    
    def compute_fan_chart_data(
        self,
        debt_ratio_paths: np.ndarray
    ) -> pd.DataFrame:
        """
        Compute percentiles for fan chart visualization.
        
        Args:
            debt_ratio_paths: Array of shape (n_sims, horizon+1)
            
        Returns:
            DataFrame with percentile values for each year
        """
        percentiles = self.config.percentiles
        horizon = debt_ratio_paths.shape[1]
        
        results = {}
        for p in percentiles:
            results[f'p{int(p*100)}'] = np.percentile(
                debt_ratio_paths, p * 100, axis=0
            )
        
        results['mean'] = np.mean(debt_ratio_paths, axis=0)
        results['year'] = np.arange(horizon)
        
        return pd.DataFrame(results)
    
    def compute_risk_metrics(
        self,
        debt_ratio_paths: np.ndarray,
        thresholds: List[float] = [100, 110, 120]
    ) -> Dict[str, np.ndarray]:
        """
        Compute probability of exceeding debt thresholds.
        
        Args:
            debt_ratio_paths: Array of shape (n_sims, horizon+1)
            thresholds: List of debt-to-GDP thresholds to assess
            
        Returns:
            Dictionary with probabilities for each threshold and year
        """
        results = {}
        horizon = debt_ratio_paths.shape[1]
        
        for threshold in thresholds:
            prob = np.mean(debt_ratio_paths > threshold, axis=0)
            results[f'prob_above_{threshold}'] = prob
        
        # Also compute probability of debt rising
        debt_change = np.diff(debt_ratio_paths, axis=1)
        results['prob_debt_rising'] = np.mean(debt_change > 0, axis=0)
        
        return results
    
    def run_full_simulation(
        self,
        initial_debt: float,
        initial_gdp: float,
        ilg_share: float = 0.22,
        baseline_primary_balance_gdp: float = -0.02
    ) -> Dict:
        """
        Run complete Monte Carlo simulation and return all results.
        
        Args:
            initial_debt: Initial debt level (£bn)
            initial_gdp: Initial GDP level (£bn)
            ilg_share: Share of debt that is index-linked
            baseline_primary_balance_gdp: Primary balance as fraction of GDP
            
        Returns:
            Comprehensive dictionary with all simulation results
        """
        # Run simulation
        sim_results = self.simulate_debt_paths(
            initial_debt=initial_debt,
            initial_gdp=initial_gdp,
            ilg_share=ilg_share,
            baseline_primary_balance_gdp=baseline_primary_balance_gdp
        )
        
        # Compute fan chart data
        fan_chart = self.compute_fan_chart_data(sim_results['debt_ratio'])
        
        # Compute risk metrics
        risk_metrics = self.compute_risk_metrics(sim_results['debt_ratio'])
        
        # Summary statistics
        final_debt_ratio = sim_results['debt_ratio'][:, -1]
        summary = {
            'final_debt_ratio_mean': np.mean(final_debt_ratio),
            'final_debt_ratio_median': np.median(final_debt_ratio),
            'final_debt_ratio_std': np.std(final_debt_ratio),
            'final_debt_ratio_p5': np.percentile(final_debt_ratio, 5),
            'final_debt_ratio_p95': np.percentile(final_debt_ratio, 95),
            'prob_above_100_final': np.mean(final_debt_ratio > 100),
            'prob_above_120_final': np.mean(final_debt_ratio > 120)
        }
        
        return {
            'simulation_results': sim_results,
            'fan_chart_data': fan_chart,
            'risk_metrics': risk_metrics,
            'summary': summary
        }


class ScenarioSimulator:
    """
    Scenario-based simulation for deterministic stress tests.
    """
    
    def __init__(self, baseline_params: Dict):
        """
        Initialize with baseline parameters.
        
        Args:
            baseline_params: Dictionary with baseline projections
        """
        self.baseline = baseline_params
    
    def apply_shock(
        self,
        shock_type: str,
        shock_size: float,
        duration: int,
        start_year: int = 1
    ) -> Dict[str, np.ndarray]:
        """
        Apply a specific shock to baseline projections.
        
        Args:
            shock_type: Type of shock ('gilt_yield', 'inflation', 'growth', 'primary_balance')
            shock_size: Size of shock (in appropriate units)
            duration: Number of years shock persists
            start_year: Year shock begins (0-indexed)
            
        Returns:
            Dictionary with shocked paths
        """
        horizon = len(self.baseline['gdp'])
        
        # Create shock profile
        shock_profile = np.zeros(horizon)
        end_year = min(start_year + duration, horizon)
        shock_profile[start_year:end_year] = shock_size
        
        # Apply to appropriate variable
        shocked = {k: v.copy() for k, v in self.baseline.items()}
        
        if shock_type == 'gilt_yield':
            shocked['gilt_yield'] = shocked['gilt_yield'] + shock_profile
        elif shock_type == 'inflation':
            shocked['rpi_inflation'] = shocked['rpi_inflation'] + shock_profile
            shocked['cpi_inflation'] = shocked['cpi_inflation'] + shock_profile * 0.8
        elif shock_type == 'growth':
            shocked['real_gdp_growth'] = shocked['real_gdp_growth'] + shock_profile
        elif shock_type == 'primary_balance':
            shocked['primary_balance'] = shocked['primary_balance'] + shock_profile
            
        return shocked
    
    def run_scenario(
        self,
        scenario_params: Dict,
        initial_debt: float,
        initial_gdp: float
    ) -> pd.DataFrame:
        """
        Run a complete scenario with multiple shocks.
        
        Args:
            scenario_params: Dictionary with shock parameters
            initial_debt: Initial debt level
            initial_gdp: Initial GDP level
            
        Returns:
            DataFrame with scenario results
        """
        # Apply all shocks
        shocked = self.baseline.copy()
        
        for shock_type, shock_size in scenario_params.items():
            if shock_size != 0:
                duration = scenario_params.get('duration', 5)
                temp_shocked = self.apply_shock(
                    shock_type, 
                    shock_size, 
                    duration
                )
                shocked.update(temp_shocked)
        
        # Project debt path
        horizon = len(shocked['gdp'])
        debt = np.zeros(horizon + 1)
        gdp = np.zeros(horizon + 1)
        debt_ratio = np.zeros(horizon + 1)
        
        debt[0] = initial_debt
        gdp[0] = initial_gdp
        debt_ratio[0] = initial_debt / initial_gdp * 100
        
        for t in range(horizon):
            # GDP evolution
            nominal_growth = (
                shocked['real_gdp_growth'][t] + 
                shocked['cpi_inflation'][t]
            )
            gdp[t+1] = gdp[t] * (1 + nominal_growth)
            
            # Interest (simplified)
            interest = debt[t] * shocked['gilt_yield'][t]
            
            # Primary balance
            primary = shocked['primary_balance'][t]
            
            # Debt accumulation
            debt[t+1] = debt[t] + interest + primary
            debt_ratio[t+1] = debt[t+1] / gdp[t+1] * 100
        
        return pd.DataFrame({
            'year': np.arange(horizon + 1),
            'debt': debt,
            'gdp': gdp,
            'debt_ratio': debt_ratio
        })


def run_comparative_monte_carlo(
    configs: List[MonteCarloConfig],
    initial_conditions: Dict,
    scenario_names: List[str]
) -> Dict[str, pd.DataFrame]:
    """
    Run Monte Carlo simulations under different configurations.
    
    Useful for comparing results under different assumptions
    about volatility, correlations, or starting conditions.
    
    Args:
        configs: List of MonteCarloConfig objects
        initial_conditions: Dict with initial_debt, initial_gdp, etc.
        scenario_names: Names for each configuration
        
    Returns:
        Dictionary of DataFrames with results for each scenario
    """
    results = {}
    
    for config, name in zip(configs, scenario_names):
        engine = MonteCarloEngine(config)
        sim_results = engine.run_full_simulation(**initial_conditions)
        results[name] = sim_results['fan_chart_data']
        
    return results
