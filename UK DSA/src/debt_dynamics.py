"""
UK Debt Sustainability Analysis - Core Debt Dynamics Model
==========================================================
Imperial College London UROP Project

This module implements the standard debt dynamics equation and related
sustainability metrics following IMF/ECB/OBR methodology.

Debt Dynamics Equation:
    ΔD/Y = (r - g) × D(-1)/Y(-1) + pb + sfa

Where:
    D/Y = debt-to-GDP ratio
    r = effective interest rate on debt
    g = nominal GDP growth rate
    pb = primary balance (deficit positive)
    sfa = stock-flow adjustment
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class DebtDynamicsResult:
    """Container for debt dynamics decomposition results"""
    year: str
    debt_to_gdp: float
    change_in_debt_ratio: float
    interest_growth_differential: float
    primary_balance_effect: float
    stock_flow_adjustment: float
    effective_interest_rate: float
    nominal_gdp_growth: float
    primary_balance_to_gdp: float


class DebtDynamicsModel:
    """
    Core debt dynamics model for UK DSA.
    
    Implements the standard debt accumulation equation with extensions for:
    - Index-linked gilt sensitivity to inflation
    - Stock-flow adjustments (exchange rate, revaluation, etc.)
    - Fiscal reaction functions
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the model with configuration parameters.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        
    def compute_debt_dynamics(
        self,
        debt_t_minus_1: float,
        gdp_t_minus_1: float,
        gdp_t: float,
        interest_paid: float,
        primary_balance: float,
        stock_flow_adj: float = 0.0
    ) -> Dict[str, float]:
        """
        Compute one-period debt dynamics decomposition.
        
        Args:
            debt_t_minus_1: Outstanding debt at end of previous period (£bn)
            gdp_t_minus_1: GDP in previous period (£bn)
            gdp_t: GDP in current period (£bn)
            interest_paid: Interest payments in current period (£bn)
            primary_balance: Primary balance (deficit positive) (£bn)
            stock_flow_adj: Stock-flow adjustment (£bn)
            
        Returns:
            Dictionary with debt dynamics decomposition
        """
        # Debt ratios
        debt_ratio_t_minus_1 = debt_t_minus_1 / gdp_t_minus_1
        
        # Growth rates
        nominal_gdp_growth = (gdp_t - gdp_t_minus_1) / gdp_t_minus_1
        
        # Effective interest rate (interest paid / previous debt stock)
        effective_rate = interest_paid / debt_t_minus_1 if debt_t_minus_1 > 0 else 0
        
        # Interest-growth differential (r - g)
        r_minus_g = effective_rate - nominal_gdp_growth
        
        # Debt dynamics components
        interest_growth_effect = r_minus_g * debt_ratio_t_minus_1
        primary_balance_effect = primary_balance / gdp_t
        sfa_effect = stock_flow_adj / gdp_t
        
        # New debt level
        debt_t = debt_t_minus_1 + interest_paid + primary_balance + stock_flow_adj
        debt_ratio_t = debt_t / gdp_t
        
        # Change in debt ratio
        change_in_ratio = debt_ratio_t - debt_ratio_t_minus_1
        
        return {
            'debt_t': debt_t,
            'debt_ratio_t': debt_ratio_t * 100,  # As percentage
            'debt_ratio_t_minus_1': debt_ratio_t_minus_1 * 100,
            'change_in_debt_ratio': change_in_ratio * 100,
            'interest_growth_effect': interest_growth_effect * 100,
            'primary_balance_effect': primary_balance_effect * 100,
            'sfa_effect': sfa_effect * 100,
            'effective_interest_rate': effective_rate * 100,
            'nominal_gdp_growth': nominal_gdp_growth * 100,
            'r_minus_g': r_minus_g * 100
        }
    
    def compute_historical_decomposition(
        self,
        debt_series: pd.Series,
        gdp_series: pd.Series,
        interest_series: pd.Series,
        borrowing_series: pd.Series
    ) -> pd.DataFrame:
        """
        Compute historical debt dynamics decomposition.
        
        Args:
            debt_series: Time series of debt levels (£bn)
            gdp_series: Time series of nominal GDP (£bn)
            interest_series: Time series of interest payments (£bn)
            borrowing_series: Time series of net borrowing (£bn)
            
        Returns:
            DataFrame with annual decomposition
        """
        results = []
        
        for i in range(1, len(debt_series)):
            year = debt_series.index[i]
            
            # Primary balance = borrowing - interest
            primary_balance = borrowing_series.iloc[i] - interest_series.iloc[i]
            
            # Stock-flow adjustment (residual)
            debt_change = debt_series.iloc[i] - debt_series.iloc[i-1]
            implied_change = interest_series.iloc[i] + primary_balance
            sfa = debt_change - implied_change
            
            result = self.compute_debt_dynamics(
                debt_t_minus_1=debt_series.iloc[i-1],
                gdp_t_minus_1=gdp_series.iloc[i-1],
                gdp_t=gdp_series.iloc[i],
                interest_paid=interest_series.iloc[i],
                primary_balance=primary_balance,
                stock_flow_adj=sfa
            )
            result['year'] = year
            results.append(result)
            
        return pd.DataFrame(results)
    
    def debt_stabilizing_primary_balance(
        self,
        debt_ratio: float,
        effective_rate: float,
        nominal_growth: float
    ) -> float:
        """
        Calculate the primary balance required to stabilize debt-to-GDP ratio.
        
        pb* = (r - g) × d
        
        Args:
            debt_ratio: Current debt-to-GDP ratio (as decimal, e.g., 0.95)
            effective_rate: Effective interest rate (as decimal)
            nominal_growth: Nominal GDP growth rate (as decimal)
            
        Returns:
            Required primary balance as % of GDP (surplus positive)
        """
        r_minus_g = effective_rate - nominal_growth
        pb_star = r_minus_g * debt_ratio
        return pb_star * 100  # As percentage
    
    def project_debt_path(
        self,
        initial_debt: float,
        initial_gdp: float,
        years: int,
        interest_rate_path: List[float],
        gdp_growth_path: List[float],
        primary_balance_path: List[float],
        sfa_path: Optional[List[float]] = None
    ) -> pd.DataFrame:
        """
        Project debt path over multiple years.
        
        Args:
            initial_debt: Starting debt level (£bn)
            initial_gdp: Starting GDP level (£bn)
            years: Number of years to project
            interest_rate_path: List of effective interest rates (as decimals)
            gdp_growth_path: List of nominal GDP growth rates (as decimals)
            primary_balance_path: List of primary balances (£bn)
            sfa_path: Optional list of stock-flow adjustments (£bn)
            
        Returns:
            DataFrame with projected debt path and decomposition
        """
        if sfa_path is None:
            sfa_path = [0.0] * years
            
        debt = initial_debt
        gdp = initial_gdp
        results = []
        
        for i in range(years):
            # GDP in current year
            gdp_new = gdp * (1 + gdp_growth_path[i])
            
            # Interest payment
            interest = debt * interest_rate_path[i]
            
            result = self.compute_debt_dynamics(
                debt_t_minus_1=debt,
                gdp_t_minus_1=gdp,
                gdp_t=gdp_new,
                interest_paid=interest,
                primary_balance=primary_balance_path[i],
                stock_flow_adj=sfa_path[i]
            )
            result['year'] = i + 1
            results.append(result)
            
            # Update for next iteration
            debt = result['debt_t']
            gdp = gdp_new
            
        return pd.DataFrame(results)
    
    def compute_ilg_sensitivity(
        self,
        ilg_stock: float,
        rpi_shock: float,
        years: int = 1
    ) -> float:
        """
        Compute the impact of RPI inflation on index-linked gilt costs.
        
        The uplift on ILGs accrues based on RPI changes and adds to
        measured debt interest.
        
        Args:
            ilg_stock: Outstanding index-linked gilt stock (£bn)
            rpi_shock: Change in RPI inflation (percentage points)
            years: Number of years shock persists
            
        Returns:
            Additional debt interest cost (£bn)
        """
        # Simplified: uplift approximately = ILG stock × RPI change
        # In reality, this depends on maturity profile and lag structure
        uplift = ilg_stock * (rpi_shock / 100) * years
        return uplift
    
    def apply_interest_rate_shock(
        self,
        baseline_interest: Dict[str, float],
        shock_bp: float,
        ready_reckoner: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Apply interest rate shock using OBR ready reckoners.
        
        Args:
            baseline_interest: Baseline debt interest by year
            shock_bp: Shock size in basis points
            ready_reckoner: OBR ready reckoner (impact per 100bp)
            
        Returns:
            Shocked debt interest path
        """
        shocked = {}
        shock_pp = shock_bp / 100  # Convert to percentage points
        
        for year, baseline in baseline_interest.items():
            if year in ready_reckoner:
                impact = ready_reckoner[year] * shock_pp
                shocked[year] = baseline + impact
            else:
                shocked[year] = baseline
                
        return shocked


class FiscalReactionFunction:
    """
    Models the government's fiscal response to debt levels.
    
    Implements Bohn (1998) style reaction functions:
        pb_t = α + β × d_{t-1} + γ × gap_t + ε_t
    
    Where:
    - pb = primary balance (% GDP)
    - d = debt ratio (% GDP)
    - gap = output gap (% potential GDP)
    """
    
    def __init__(self, alpha: float = -0.02, beta: float = 0.03, gamma: float = 0.5):
        """
        Initialize fiscal reaction function.
        
        Args:
            alpha: Intercept (negative = structural deficit tendency)
            beta: Response to debt (positive = stabilizing)
            gamma: Response to output gap (positive = countercyclical)
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
    
    def predict_primary_balance(
        self,
        debt_ratio: float,
        output_gap: float = 0.0
    ) -> float:
        """
        Predict primary balance given debt ratio and output gap.
        
        Args:
            debt_ratio: Lagged debt-to-GDP ratio (as decimal)
            output_gap: Output gap as % of potential GDP
            
        Returns:
            Predicted primary balance (% GDP, surplus positive)
        """
        pb = self.alpha + self.beta * debt_ratio + self.gamma * output_gap
        return pb
    
    def estimate_from_data(
        self,
        primary_balance: pd.Series,
        debt_ratio: pd.Series,
        output_gap: Optional[pd.Series] = None
    ) -> Dict[str, float]:
        """
        Estimate reaction function parameters from historical data.
        
        Uses OLS regression.
        
        Args:
            primary_balance: Time series of primary balance (% GDP)
            debt_ratio: Time series of debt ratio (% GDP)
            output_gap: Optional time series of output gap
            
        Returns:
            Dictionary with estimated parameters
        """
        import statsmodels.api as sm
        
        # Align and lag debt ratio
        df = pd.DataFrame({
            'pb': primary_balance,
            'd_lag': debt_ratio.shift(1)
        }).dropna()
        
        if output_gap is not None:
            df['gap'] = output_gap
            X = df[['d_lag', 'gap']]
        else:
            X = df[['d_lag']]
            
        X = sm.add_constant(X)
        y = df['pb']
        
        model = sm.OLS(y, X).fit()
        
        results = {
            'alpha': model.params['const'],
            'beta': model.params['d_lag'],
            'r_squared': model.rsquared,
            'beta_se': model.bse['d_lag'],
            'beta_pvalue': model.pvalues['d_lag']
        }
        
        if output_gap is not None:
            results['gamma'] = model.params['gap']
            
        return results


def compute_r_g_differential(
    interest_payments: float,
    debt_stock: float,
    gdp_current: float,
    gdp_previous: float
) -> Tuple[float, float, float]:
    """
    Compute the interest-growth (r-g) differential.
    
    This is the key determinant of debt dynamics:
    - r > g: Debt grows faster than economy (unstable)
    - r < g: Debt burden shrinks naturally (favorable)
    
    Args:
        interest_payments: Interest payments in period (£bn)
        debt_stock: Debt at start of period (£bn)
        gdp_current: GDP in current period (£bn)
        gdp_previous: GDP in previous period (£bn)
        
    Returns:
        Tuple of (effective_rate, nominal_growth, r_minus_g) as percentages
    """
    r = (interest_payments / debt_stock) * 100 if debt_stock > 0 else 0
    g = ((gdp_current - gdp_previous) / gdp_previous) * 100
    r_minus_g = r - g
    
    return r, g, r_minus_g


def calculate_adjustment_need(
    current_debt_ratio: float,
    target_debt_ratio: float,
    r_minus_g: float,
    years: int
) -> float:
    """
    Calculate required fiscal adjustment to reach debt target.
    
    Solves for primary balance needed to reduce debt ratio from
    current level to target over specified period.
    
    Args:
        current_debt_ratio: Current debt-to-GDP ratio (as decimal)
        target_debt_ratio: Target debt-to-GDP ratio (as decimal)
        r_minus_g: Interest-growth differential (as decimal)
        years: Adjustment period
        
    Returns:
        Required primary surplus (% GDP) per year
    """
    # Simplified linear adjustment path
    debt_reduction_needed = current_debt_ratio - target_debt_ratio
    
    # Average debt ratio during adjustment
    avg_debt_ratio = (current_debt_ratio + target_debt_ratio) / 2
    
    # Required primary balance = (r-g) × avg_debt + annual reduction
    pb_required = r_minus_g * avg_debt_ratio + (debt_reduction_needed / years)
    
    return pb_required * 100  # As percentage
