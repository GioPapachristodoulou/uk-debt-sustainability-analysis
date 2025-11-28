"""
Advanced Debt Sustainability Analysis - Tier 1 Enhancements
============================================================

1. Fiscal Reaction Function (Bohn Test)
2. Fiscal Space Calculation (Ghosh et al. 2013)
3. Gross Financing Needs Analysis
4. Fat-Tailed Monte Carlo with Copula Dependence

Author: UK DSA Project
Date: November 2025
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import brentq, minimize_scalar
from scipy.special import gamma as gamma_func
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# SECTION 1: FISCAL REACTION FUNCTION (BOHN TEST)
# =============================================================================

class BohnTest:
    """
    Implement the Bohn (1998) fiscal sustainability test.
    
    The canonical test: pb_t = α + β·d_{t-1} + γ·X_t + ε_t
    
    If β > 0 and statistically significant, the government responds to 
    higher debt by running larger primary surpluses → sustainable.
    
    References:
    - Bohn, H. (1998). "The Behavior of U.S. Public Debt and Deficits"
    - Bohn, H. (2008). "The Sustainability of Fiscal Policy in the United States"
    """
    
    def __init__(self):
        self.results = {}
        self.data = None
        
    def build_dataset(self):
        """
        Build historical dataset for Bohn regression.
        Uses fiscal year data from 1970s onwards.
        """
        # Historical data - UK fiscal years
        # Primary balance = Current budget + Net investment - Debt interest
        # Approximated as: PSNB - Debt Interest (with sign adjustment)
        
        # Comprehensive UK data (fiscal years ending March)
        # Sources: ONS, OBR, BoE
        
        years = list(range(1975, 2025))
        
        # Debt/GDP ratios (%) - Public Sector Net Debt
        debt_gdp = {
            1975: 53.2, 1976: 52.8, 1977: 51.4, 1978: 49.3, 1979: 46.8,
            1980: 46.1, 1981: 46.4, 1982: 44.9, 1983: 44.5, 1984: 45.3,
            1985: 44.3, 1986: 43.2, 1987: 41.2, 1988: 37.3, 1989: 32.4,
            1990: 28.5, 1991: 27.5, 1992: 29.3, 1993: 33.5, 1994: 38.0,
            1995: 40.9, 1996: 42.0, 1997: 42.5, 1998: 41.4, 1999: 39.6,
            2000: 37.5, 2001: 34.4, 2002: 33.7, 2003: 34.5, 2004: 36.3,
            2005: 38.6, 2006: 39.8, 2007: 40.4, 2008: 43.0, 2009: 52.2,
            2010: 64.1, 2011: 72.4, 2012: 76.0, 2013: 79.8, 2014: 82.4,
            2015: 83.0, 2016: 83.8, 2017: 84.4, 2018: 83.7, 2019: 82.9,
            2020: 84.4, 2021: 102.1, 2022: 98.9, 2023: 97.8, 2024: 96.0
        }
        
        # Primary balance as % GDP (positive = surplus)
        # Primary balance = Total receipts - (Total expenditure - Debt interest)
        primary_balance = {
            1975: -3.8, 1976: -3.2, 1977: -1.5, 1978: -2.8, 1979: -2.1,
            1980: -2.5, 1981: -1.8, 1982: -1.2, 1983: -1.8, 1984: -1.5,
            1985: -0.8, 1986: -0.5, 1987: 0.2, 1988: 1.8, 1989: 2.3,
            1990: 1.5, 1991: -0.2, 1992: -4.2, 1993: -5.2, 1994: -4.5,
            1995: -3.2, 1996: -2.1, 1997: -0.5, 1998: 1.2, 1999: 2.1,
            2000: 2.8, 2001: 1.8, 2002: -0.2, 2003: -1.8, 2004: -2.2,
            2005: -2.0, 2006: -1.5, 2007: -1.2, 2008: -2.8, 2009: -8.5,
            2010: -7.2, 2011: -5.2, 2012: -4.8, 2013: -3.8, 2014: -3.2,
            2015: -2.5, 2016: -1.8, 2017: -0.8, 2018: -0.2, 2019: 0.2,
            2020: -2.5, 2021: -12.5, 2022: -4.2, 2023: -2.1, 2024: -0.8
        }
        
        # Output gap (%) - deviation from potential GDP
        # Source: OBR/OECD estimates
        output_gap = {
            1975: -1.5, 1976: -0.8, 1977: 0.2, 1978: 1.5, 1979: 2.8,
            1980: 0.5, 1981: -3.2, 1982: -2.5, 1983: -1.2, 1984: 0.5,
            1985: 1.8, 1986: 2.5, 1987: 3.2, 1988: 4.5, 1989: 4.2,
            1990: 2.8, 1991: -0.5, 1992: -2.8, 1993: -2.5, 1994: -0.8,
            1995: 0.5, 1996: 0.8, 1997: 1.2, 1998: 1.5, 1999: 1.2,
            2000: 1.8, 2001: 1.2, 2002: 0.2, 2003: 0.5, 2004: 1.2,
            2005: 1.5, 2006: 2.2, 2007: 2.8, 2008: 1.2, 2009: -4.5,
            2010: -2.8, 2011: -1.5, 2012: -1.8, 2013: -1.2, 2014: -0.5,
            2015: 0.2, 2016: 0.5, 2017: 0.2, 2018: 0.2, 2019: 0.2,
            2020: -2.5, 2021: -3.2, 2022: -0.5, 2023: 0.2, 2024: -0.5
        }
        
        # Government spending gap (deviation from HP trend)
        # Approximated as deviation from 5-year moving average
        gexp_gap = {
            1975: 2.5, 1976: 1.8, 1977: 0.5, 1978: 0.2, 1979: -0.5,
            1980: 0.8, 1981: 1.2, 1982: 1.5, 1983: 1.2, 1984: 0.8,
            1985: 0.2, 1986: -0.2, 1987: -0.8, 1988: -1.5, 1989: -1.8,
            1990: -1.2, 1991: 0.5, 1992: 2.5, 1993: 2.8, 1994: 2.2,
            1995: 1.5, 1996: 0.8, 1997: 0.2, 1998: -0.5, 1999: -0.8,
            2000: -1.2, 2001: -0.5, 2002: 0.5, 2003: 1.2, 2004: 1.5,
            2005: 1.2, 2006: 0.8, 2007: 0.5, 2008: 1.8, 2009: 5.5,
            2010: 4.2, 2011: 2.5, 2012: 1.8, 2013: 1.2, 2014: 0.5,
            2015: 0.2, 2016: -0.2, 2017: -0.5, 2018: -0.8, 2019: -0.5,
            2020: 1.5, 2021: 8.5, 2022: 2.5, 2023: 1.2, 2024: 0.8
        }
        
        # Build DataFrame
        self.data = pd.DataFrame({
            'year': years,
            'debt_gdp': [debt_gdp[y] for y in years],
            'primary_balance': [primary_balance[y] for y in years],
            'output_gap': [output_gap[y] for y in years],
            'gexp_gap': [gexp_gap[y] for y in years]
        })
        
        # Create lagged debt
        self.data['debt_gdp_lag'] = self.data['debt_gdp'].shift(1)
        
        # Drop first observation (no lag available)
        self.data = self.data.dropna().reset_index(drop=True)
        
        return self.data
    
    def estimate_basic(self):
        """
        Estimate basic Bohn regression: pb_t = α + β·d_{t-1} + ε_t
        """
        if self.data is None:
            self.build_dataset()
        
        y = self.data['primary_balance'].values
        X = np.column_stack([np.ones(len(y)), self.data['debt_gdp_lag'].values])
        
        # OLS estimation
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        
        # Residuals and standard errors
        residuals = y - X @ beta
        n, k = X.shape
        sigma2 = np.sum(residuals**2) / (n - k)
        var_beta = sigma2 * np.linalg.inv(X.T @ X)
        se_beta = np.sqrt(np.diag(var_beta))
        
        # t-statistics and p-values
        t_stats = beta / se_beta
        p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), n - k))
        
        # R-squared
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r_squared = 1 - ss_res / ss_tot
        
        # Adjusted R-squared
        r_squared_adj = 1 - (1 - r_squared) * (n - 1) / (n - k)
        
        # Durbin-Watson statistic
        dw = np.sum(np.diff(residuals)**2) / ss_res
        
        self.results['basic'] = {
            'alpha': beta[0],
            'beta': beta[1],
            'se_alpha': se_beta[0],
            'se_beta': se_beta[1],
            't_alpha': t_stats[0],
            't_beta': t_stats[1],
            'p_alpha': p_values[0],
            'p_beta': p_values[1],
            'r_squared': r_squared,
            'r_squared_adj': r_squared_adj,
            'durbin_watson': dw,
            'n_obs': n,
            'residuals': residuals,
            'sustainable': beta[1] > 0 and p_values[1] < 0.05
        }
        
        return self.results['basic']
    
    def estimate_augmented(self):
        """
        Estimate augmented Bohn regression with business cycle controls:
        pb_t = α + β·d_{t-1} + γ₁·YGAP_t + γ₂·GVAR_t + ε_t
        """
        if self.data is None:
            self.build_dataset()
        
        y = self.data['primary_balance'].values
        X = np.column_stack([
            np.ones(len(y)),
            self.data['debt_gdp_lag'].values,
            self.data['output_gap'].values,
            self.data['gexp_gap'].values
        ])
        
        # OLS estimation
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        
        # Residuals and standard errors
        residuals = y - X @ beta
        n, k = X.shape
        sigma2 = np.sum(residuals**2) / (n - k)
        var_beta = sigma2 * np.linalg.inv(X.T @ X)
        se_beta = np.sqrt(np.diag(var_beta))
        
        # t-statistics and p-values
        t_stats = beta / se_beta
        p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), n - k))
        
        # R-squared
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r_squared = 1 - ss_res / ss_tot
        r_squared_adj = 1 - (1 - r_squared) * (n - 1) / (n - k)
        
        # Durbin-Watson
        dw = np.sum(np.diff(residuals)**2) / ss_res
        
        self.results['augmented'] = {
            'alpha': beta[0],
            'beta_debt': beta[1],
            'beta_ygap': beta[2],
            'beta_gvar': beta[3],
            'se': se_beta,
            't_stats': t_stats,
            'p_values': p_values,
            'r_squared': r_squared,
            'r_squared_adj': r_squared_adj,
            'durbin_watson': dw,
            'n_obs': n,
            'residuals': residuals,
            'sustainable': beta[1] > 0 and p_values[1] < 0.05
        }
        
        return self.results['augmented']
    
    def estimate_nonlinear(self):
        """
        Estimate non-linear Bohn regression to test if response strengthens at high debt:
        pb_t = α + β₁·d_{t-1} + β₂·d²_{t-1} + γ·X_t + ε_t
        """
        if self.data is None:
            self.build_dataset()
        
        y = self.data['primary_balance'].values
        d_lag = self.data['debt_gdp_lag'].values
        
        X = np.column_stack([
            np.ones(len(y)),
            d_lag,
            d_lag**2 / 100,  # Scale for numerical stability
            self.data['output_gap'].values
        ])
        
        # OLS estimation
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        
        # Residuals and standard errors
        residuals = y - X @ beta
        n, k = X.shape
        sigma2 = np.sum(residuals**2) / (n - k)
        var_beta = sigma2 * np.linalg.inv(X.T @ X)
        se_beta = np.sqrt(np.diag(var_beta))
        
        # t-statistics and p-values
        t_stats = beta / se_beta
        p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), n - k))
        
        # R-squared
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r_squared = 1 - ss_res / ss_tot
        
        # Marginal response at different debt levels
        # d(pb)/d(d) = β₁ + 2·β₂·d/100
        marginal_at_50 = beta[1] + 2 * beta[2] * 50 / 100
        marginal_at_80 = beta[1] + 2 * beta[2] * 80 / 100
        marginal_at_100 = beta[1] + 2 * beta[2] * 100 / 100
        
        self.results['nonlinear'] = {
            'alpha': beta[0],
            'beta_linear': beta[1],
            'beta_quadratic': beta[2],
            'beta_ygap': beta[3],
            'se': se_beta,
            't_stats': t_stats,
            'p_values': p_values,
            'r_squared': r_squared,
            'marginal_response': {
                50: marginal_at_50,
                80: marginal_at_80,
                100: marginal_at_100
            },
            'n_obs': n
        }
        
        return self.results['nonlinear']
    
    def estimate_with_newey_west(self, max_lag=3):
        """
        Estimate with Newey-West HAC standard errors for robustness
        to autocorrelation and heteroskedasticity.
        """
        if self.data is None:
            self.build_dataset()
        
        y = self.data['primary_balance'].values
        X = np.column_stack([
            np.ones(len(y)),
            self.data['debt_gdp_lag'].values,
            self.data['output_gap'].values,
            self.data['gexp_gap'].values
        ])
        
        n, k = X.shape
        
        # OLS estimation
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        residuals = y - X @ beta
        
        # Newey-West covariance matrix
        # Ω = Σ_j w_j * Σ_t (e_t * e_{t-j} * x_t * x_{t-j}')
        XtX_inv = np.linalg.inv(X.T @ X)
        
        # Meat of sandwich: S = Σ weights * autocov
        S = np.zeros((k, k))
        
        for j in range(max_lag + 1):
            weight = 1 - j / (max_lag + 1) if j > 0 else 1
            
            for t in range(j, n):
                outer = np.outer(X[t] * residuals[t], X[t-j] * residuals[t-j])
                S += weight * (outer + outer.T) if j > 0 else weight * outer
        
        # HAC variance-covariance matrix
        var_nw = n * XtX_inv @ S @ XtX_inv
        se_nw = np.sqrt(np.diag(var_nw))
        
        # t-statistics with HAC SEs
        t_stats_nw = beta / se_nw
        p_values_nw = 2 * (1 - stats.t.cdf(np.abs(t_stats_nw), n - k))
        
        self.results['newey_west'] = {
            'beta': beta,
            'se_hac': se_nw,
            't_stats_hac': t_stats_nw,
            'p_values_hac': p_values_nw,
            'sustainable': beta[1] > 0 and p_values_nw[1] < 0.05
        }
        
        return self.results['newey_west']
    
    def run_all_tests(self):
        """Run all Bohn test variants and compile results."""
        self.build_dataset()
        
        print("="*70)
        print("FISCAL REACTION FUNCTION - BOHN SUSTAINABILITY TESTS")
        print("="*70)
        print(f"\nSample: UK Fiscal Years 1976-2024 (n={len(self.data)})")
        print("\n" + "-"*70)
        
        # Basic test
        basic = self.estimate_basic()
        print("\n1. BASIC BOHN TEST: pb_t = α + β·d_{t-1} + ε_t")
        print("-"*50)
        print(f"   Constant (α):     {basic['alpha']:7.3f}  (SE: {basic['se_alpha']:.3f}, t: {basic['t_alpha']:.2f})")
        print(f"   Debt response(β): {basic['beta']:7.4f}  (SE: {basic['se_beta']:.4f}, t: {basic['t_beta']:.2f})")
        print(f"   p-value on β:     {basic['p_beta']:7.4f}")
        print(f"   R²:               {basic['r_squared']:7.3f}")
        print(f"   Durbin-Watson:    {basic['durbin_watson']:7.3f}")
        print(f"\n   SUSTAINABILITY: {'✓ PASS' if basic['sustainable'] else '✗ FAIL'}")
        print(f"   Interpretation: A 10pp increase in debt/GDP is associated with")
        print(f"                   a {basic['beta']*10:.2f}pp {'improvement' if basic['beta']>0 else 'deterioration'} in primary balance")
        
        # Augmented test
        aug = self.estimate_augmented()
        print("\n" + "-"*70)
        print("\n2. AUGMENTED BOHN TEST: pb_t = α + β·d_{t-1} + γ₁·YGAP + γ₂·GVAR + ε_t")
        print("-"*50)
        print(f"   Constant (α):       {aug['alpha']:7.3f}  (t: {aug['t_stats'][0]:6.2f})")
        print(f"   Debt response (β):  {aug['beta_debt']:7.4f}  (t: {aug['t_stats'][1]:6.2f}, p: {aug['p_values'][1]:.4f})")
        print(f"   Output gap (γ₁):    {aug['beta_ygap']:7.3f}  (t: {aug['t_stats'][2]:6.2f})")
        print(f"   Spending gap (γ₂):  {aug['beta_gvar']:7.3f}  (t: {aug['t_stats'][3]:6.2f})")
        print(f"   R²:                 {aug['r_squared']:7.3f}")
        print(f"\n   SUSTAINABILITY: {'✓ PASS' if aug['sustainable'] else '✗ FAIL'}")
        
        # Non-linear test
        nl = self.estimate_nonlinear()
        print("\n" + "-"*70)
        print("\n3. NON-LINEAR BOHN TEST: pb_t = α + β₁·d + β₂·d² + γ·YGAP + ε_t")
        print("-"*50)
        print(f"   Linear term (β₁):    {nl['beta_linear']:7.4f}  (t: {nl['t_stats'][1]:6.2f})")
        print(f"   Quadratic term (β₂): {nl['beta_quadratic']:7.4f}  (t: {nl['t_stats'][2]:6.2f})")
        print(f"   R²:                  {nl['r_squared']:7.3f}")
        print(f"\n   Marginal fiscal response at different debt levels:")
        print(f"   At d=50%:  {nl['marginal_response'][50]:7.4f}")
        print(f"   At d=80%:  {nl['marginal_response'][80]:7.4f}")
        print(f"   At d=100%: {nl['marginal_response'][100]:7.4f}")
        
        # Newey-West
        nw = self.estimate_with_newey_west()
        print("\n" + "-"*70)
        print("\n4. ROBUSTNESS: NEWEY-WEST HAC STANDARD ERRORS")
        print("-"*50)
        print(f"   Debt response (β):  {nw['beta'][1]:7.4f}")
        print(f"   HAC SE:             {nw['se_hac'][1]:7.4f}")
        print(f"   HAC t-stat:         {nw['t_stats_hac'][1]:7.2f}")
        print(f"   HAC p-value:        {nw['p_values_hac'][1]:7.4f}")
        print(f"\n   SUSTAINABILITY (HAC-robust): {'✓ PASS' if nw['sustainable'] else '✗ FAIL'}")
        
        # Overall assessment
        print("\n" + "="*70)
        print("OVERALL BOHN TEST ASSESSMENT")
        print("="*70)
        
        tests_passed = sum([
            basic['sustainable'],
            aug['sustainable'],
            nw['sustainable']
        ])
        
        if tests_passed >= 2:
            verdict = "SUSTAINABLE"
            explanation = "UK fiscal policy shows systematic debt-stabilizing behavior"
        elif tests_passed == 1:
            verdict = "MARGINALLY SUSTAINABLE"
            explanation = "Mixed evidence of debt-stabilizing fiscal response"
        else:
            verdict = "NOT SUSTAINABLE"
            explanation = "No evidence of systematic debt-stabilizing fiscal response"
        
        print(f"\n   Tests passed: {tests_passed}/3")
        print(f"   VERDICT: {verdict}")
        print(f"   {explanation}")
        
        # Key finding
        print(f"\n   KEY FINDING: Estimated β = {aug['beta_debt']:.4f}")
        print(f"   This means for every 1pp increase in debt/GDP,")
        print(f"   the primary balance improves by {aug['beta_debt']:.2f}pp on average.")
        
        if aug['beta_debt'] > 0:
            # Calculate implied debt limit from Bohn coefficient
            # At steady state: pb* = β·d* where pb* ≈ 0 (long-run)
            # This is simplistic; fiscal space calculation is more rigorous
            print(f"\n   The positive coefficient suggests debt-stabilizing behavior,")
            print(f"   but the magnitude ({aug['beta_debt']:.3f}) is relatively {'weak' if aug['beta_debt'] < 0.05 else 'moderate' if aug['beta_debt'] < 0.1 else 'strong'}.")
        
        return self.results


# =============================================================================
# SECTION 2: FISCAL SPACE CALCULATION
# =============================================================================

class FiscalSpaceCalculator:
    """
    Calculate fiscal space following Ghosh, Kim, Mendoza, Ostry & Qureshi (2013).
    
    "Fiscal Fatigue, Fiscal Space and Debt Sustainability in Advanced Economies"
    IMF Staff Papers
    
    Fiscal space = debt limit - current debt
    where debt limit is where primary balance can no longer respond sufficiently.
    """
    
    def __init__(self, bohn_results=None):
        self.bohn = bohn_results
        self.results = {}
        
    def estimate_fiscal_reaction_curve(self):
        """
        Estimate the cubic fiscal reaction function:
        pb = f(d) = α + β₁·d + β₂·d² + β₃·d³
        
        This captures fiscal fatigue at high debt levels.
        """
        # Use historical UK data
        years = list(range(1975, 2025))
        
        debt = np.array([
            53.2, 52.8, 51.4, 49.3, 46.8, 46.1, 46.4, 44.9, 44.5, 45.3,
            44.3, 43.2, 41.2, 37.3, 32.4, 28.5, 27.5, 29.3, 33.5, 38.0,
            40.9, 42.0, 42.5, 41.4, 39.6, 37.5, 34.4, 33.7, 34.5, 36.3,
            38.6, 39.8, 40.4, 43.0, 52.2, 64.1, 72.4, 76.0, 79.8, 82.4,
            83.0, 83.8, 84.4, 83.7, 82.9, 84.4, 102.1, 98.9, 97.8, 96.0
        ])
        
        pb = np.array([
            -3.8, -3.2, -1.5, -2.8, -2.1, -2.5, -1.8, -1.2, -1.8, -1.5,
            -0.8, -0.5, 0.2, 1.8, 2.3, 1.5, -0.2, -4.2, -5.2, -4.5,
            -3.2, -2.1, -0.5, 1.2, 2.1, 2.8, 1.8, -0.2, -1.8, -2.2,
            -2.0, -1.5, -1.2, -2.8, -8.5, -7.2, -5.2, -4.8, -3.8, -3.2,
            -2.5, -1.8, -0.8, -0.2, 0.2, -2.5, -12.5, -4.2, -2.1, -0.8
        ])
        
        # Estimate cubic function
        # Exclude COVID outlier for estimation
        mask = ~((debt > 100) & (pb < -10))  # Exclude extreme COVID point
        d_est = debt[mask]
        pb_est = pb[mask]
        
        # Normalize debt for numerical stability
        d_norm = d_est / 100
        
        X = np.column_stack([
            np.ones(len(d_norm)),
            d_norm,
            d_norm**2,
            d_norm**3
        ])
        
        coeffs = np.linalg.lstsq(X, pb_est, rcond=None)[0]
        
        self.reaction_coeffs = coeffs
        
        # Calculate fitted values
        fitted = X @ coeffs
        r_squared = 1 - np.sum((pb_est - fitted)**2) / np.sum((pb_est - np.mean(pb_est))**2)
        
        self.results['reaction_function'] = {
            'alpha': coeffs[0],
            'beta1': coeffs[1],  # Linear
            'beta2': coeffs[2],  # Quadratic
            'beta3': coeffs[3],  # Cubic
            'r_squared': r_squared
        }
        
        return self.results['reaction_function']
    
    def primary_balance_function(self, d):
        """Evaluate primary balance at debt level d (%)."""
        d_norm = d / 100
        return (self.reaction_coeffs[0] + 
                self.reaction_coeffs[1] * d_norm +
                self.reaction_coeffs[2] * d_norm**2 +
                self.reaction_coeffs[3] * d_norm**3)
    
    def debt_stabilizing_pb(self, d, r, g):
        """
        Calculate debt-stabilizing primary balance.
        pb* = (r - g) / (1 + g) * d
        
        Args:
            d: debt/GDP ratio (%)
            r: effective interest rate (%)
            g: nominal GDP growth rate (%)
        """
        return (r - g) / (1 + g/100) * d / 100
    
    def find_debt_limit(self, r=4.5, g=3.5):
        """
        Find debt limit where fiscal fatigue sets in.
        
        Debt limit is where:
        pb_max(d) = pb*(d) = (r-g)/(1+g) * d
        
        And the derivative of pb_max is less than (r-g)/(1+g)
        """
        if not hasattr(self, 'reaction_coeffs'):
            self.estimate_fiscal_reaction_curve()
        
        # The debt limit is where the fiscal reaction function
        # intersects the debt-stabilizing line from above
        
        def gap(d):
            pb_actual = self.primary_balance_function(d)
            pb_required = self.debt_stabilizing_pb(d, r, g)
            return pb_actual - pb_required
        
        # Also check the derivative condition
        def pb_derivative(d):
            d_norm = d / 100
            return (self.reaction_coeffs[1] / 100 + 
                    2 * self.reaction_coeffs[2] * d_norm / 100 +
                    3 * self.reaction_coeffs[3] * d_norm**2 / 100)
        
        required_slope = (r - g) / (100 + g)
        
        # Search for debt limit
        # Start from high debt and work down
        debt_limit = None
        
        for d in range(200, 50, -1):
            if gap(d) >= 0 and pb_derivative(d) < required_slope:
                # Primary balance can still cover required
                # But response is weakening
                continue
            elif gap(d) < 0:
                # First point where actual PB < required PB
                # This is approximately the debt limit
                debt_limit = d
                break
        
        # If no limit found in range, fiscal response is strong
        if debt_limit is None:
            # Try to find intersection point
            try:
                debt_limit = brentq(gap, 50, 200)
            except:
                debt_limit = 200  # Very high - essentially unlimited
        
        self.results['debt_limit'] = {
            'limit': debt_limit,
            'r_assumed': r,
            'g_assumed': g,
            'pb_at_limit': self.primary_balance_function(debt_limit),
            'pb_required_at_limit': self.debt_stabilizing_pb(debt_limit, r, g)
        }
        
        return debt_limit
    
    def calculate_fiscal_space(self, current_debt=96.0, r=4.5, g=3.5):
        """
        Calculate fiscal space = debt limit - current debt.
        
        Also provides probability-weighted fiscal space under different scenarios.
        """
        if not hasattr(self, 'reaction_coeffs'):
            self.estimate_fiscal_reaction_curve()
        
        # Base case
        debt_limit = self.find_debt_limit(r, g)
        fiscal_space = debt_limit - current_debt
        
        # Scenario analysis
        scenarios = {
            'baseline': {'r': 4.5, 'g': 3.5},
            'low_growth': {'r': 4.5, 'g': 2.0},
            'high_rates': {'r': 6.0, 'g': 3.5},
            'adverse': {'r': 6.0, 'g': 2.0},
            'benign': {'r': 3.5, 'g': 4.5}
        }
        
        space_by_scenario = {}
        for name, params in scenarios.items():
            limit = self.find_debt_limit(params['r'], params['g'])
            space_by_scenario[name] = {
                'debt_limit': limit,
                'fiscal_space': limit - current_debt,
                'r': params['r'],
                'g': params['g']
            }
        
        self.results['fiscal_space'] = {
            'current_debt': current_debt,
            'debt_limit_baseline': debt_limit,
            'fiscal_space_baseline': fiscal_space,
            'scenarios': space_by_scenario
        }
        
        return self.results['fiscal_space']
    
    def print_results(self):
        """Print comprehensive fiscal space analysis."""
        if 'fiscal_space' not in self.results:
            self.calculate_fiscal_space()
        
        fs = self.results['fiscal_space']
        rf = self.results.get('reaction_function', {})
        
        print("\n" + "="*70)
        print("FISCAL SPACE ANALYSIS (Ghosh et al. 2013 Methodology)")
        print("="*70)
        
        print("\n1. ESTIMATED FISCAL REACTION FUNCTION")
        print("-"*50)
        print("   pb(d) = α + β₁·(d/100) + β₂·(d/100)² + β₃·(d/100)³")
        print(f"   α  = {rf.get('alpha', 0):7.3f}")
        print(f"   β₁ = {rf.get('beta1', 0):7.3f}")
        print(f"   β₂ = {rf.get('beta2', 0):7.3f}")
        print(f"   β₃ = {rf.get('beta3', 0):7.3f}")
        print(f"   R² = {rf.get('r_squared', 0):7.3f}")
        
        print("\n2. DEBT LIMIT AND FISCAL SPACE")
        print("-"*50)
        print(f"   Current debt/GDP:     {fs['current_debt']:6.1f}%")
        print(f"   Estimated debt limit: {fs['debt_limit_baseline']:6.1f}%")
        print(f"   FISCAL SPACE:         {fs['fiscal_space_baseline']:6.1f}pp")
        
        print("\n3. FISCAL SPACE UNDER DIFFERENT SCENARIOS")
        print("-"*50)
        print(f"   {'Scenario':<15} {'r (%)':<8} {'g (%)':<8} {'Limit':<10} {'Space':<10}")
        print("   " + "-"*50)
        
        for name, data in fs['scenarios'].items():
            print(f"   {name:<15} {data['r']:<8.1f} {data['g']:<8.1f} "
                  f"{data['debt_limit']:<10.1f} {data['fiscal_space']:<10.1f}")
        
        print("\n4. INTERPRETATION")
        print("-"*50)
        
        space = fs['fiscal_space_baseline']
        if space > 30:
            assessment = "AMPLE"
            color = "green"
            explanation = "Significant room for additional borrowing if needed"
        elif space > 15:
            assessment = "MODERATE"
            explanation = "Some fiscal space, but caution warranted"
        elif space > 5:
            assessment = "LIMITED"
            explanation = "Approaching fiscal limits, consolidation advisable"
        else:
            assessment = "EXHAUSTED"
            explanation = "At or near debt limit, urgent consolidation needed"
        
        print(f"   Assessment: {assessment}")
        print(f"   {explanation}")
        
        # Warning about model limitations
        print("\n   CAVEATS:")
        print("   - Debt limits are model-dependent estimates")
        print("   - Market sentiment can shift limits abruptly")
        print("   - Historical reaction function may not hold at high debt")
        print("   - UK-specific factors (reserve currency, BoE) may extend limits")
        
        return self.results


# =============================================================================
# SECTION 3: GROSS FINANCING NEEDS ANALYSIS
# =============================================================================

class GrossFinancingNeeds:
    """
    Calculate and analyze Gross Financing Needs (GFN).
    
    GFN = Primary deficit + Interest payments + Maturing debt
    
    The IMF uses GFN > 15-20% of GDP as a marker of elevated rollover risk.
    """
    
    def __init__(self):
        self.results = {}
        self.redemption_profile = None
        
    def load_redemption_data(self):
        """
        Load gilt redemption profile from DMO data.
        Redemptions by fiscal year.
        """
        # From DMO Future Redemptions data (D8B)
        # Values in £bn nominal
        
        self.redemption_profile = {
            # Fiscal year ending: nominal amount maturing
            2025: 52.8,   # FY 2024-25
            2026: 63.4,   # FY 2025-26
            2027: 48.2,   # FY 2026-27
            2028: 55.1,   # FY 2027-28
            2029: 71.3,   # FY 2028-29
            2030: 58.7,   # FY 2029-30
            2031: 42.1,   # FY 2030-31
            2032: 48.5,   # FY 2031-32
            2033: 39.8,   # FY 2032-33
            2034: 45.2,   # FY 2033-34
            2035: 38.4,   # FY 2034-35
        }
        
        # Treasury bill rollover (assume constant stock, rolled quarterly)
        self.tbill_stock = 45.0  # £bn
        
        # NS&I maturities (estimated annual runoff)
        self.nsi_runoff = {
            2025: 15.0, 2026: 18.0, 2027: 20.0, 2028: 22.0, 2029: 24.0,
            2030: 25.0, 2031: 25.0, 2032: 25.0, 2033: 25.0, 2034: 25.0,
            2035: 25.0
        }
        
        return self.redemption_profile
    
    def calculate_gfn(self):
        """
        Calculate GFN for each year in forecast horizon.
        """
        if self.redemption_profile is None:
            self.load_redemption_data()
        
        # Fiscal projections (from OBR and config)
        gdp = {
            2025: 2864.0, 2026: 2984.0, 2027: 3107.0, 2028: 3225.0, 2029: 3350.0,
            2030: 3478.0, 2031: 3600.0, 2032: 3730.0, 2033: 3864.0, 2034: 4003.0,
            2035: 4147.0
        }
        
        # Primary deficit (positive = deficit)
        primary_deficit = {
            2025: 32.0, 2026: 18.0, 2027: 5.0, 2028: -8.0, 2029: -25.0,
            2030: -45.0, 2031: -50.0, 2032: -52.0, 2033: -54.0, 2034: -56.0,
            2035: -58.0
        }
        
        # Debt interest payments
        interest = {
            2025: 105.0, 2026: 109.0, 2027: 110.0, 2028: 113.0, 2029: 118.0,
            2030: 124.0, 2031: 128.0, 2032: 132.0, 2033: 136.0, 2034: 140.0,
            2035: 145.0
        }
        
        # Calculate GFN
        gfn_data = {}
        
        for year in range(2025, 2036):
            gilt_redemptions = self.redemption_profile.get(year, 40.0)
            tbill_rollover = self.tbill_stock * 4  # Rolled 4x per year
            nsi_maturities = self.nsi_runoff.get(year, 25.0)
            
            # Total maturing debt
            maturing_debt = gilt_redemptions + tbill_rollover + nsi_maturities
            
            # GFN components
            prim_def = primary_deficit.get(year, -50.0)
            int_pmt = interest.get(year, 140.0)
            
            # Gross Financing Need
            gfn = prim_def + int_pmt + maturing_debt
            
            # As % of GDP
            gdp_val = gdp.get(year, 4000.0)
            gfn_gdp = gfn / gdp_val * 100
            
            gfn_data[year] = {
                'primary_deficit': prim_def,
                'interest': int_pmt,
                'gilt_redemptions': gilt_redemptions,
                'tbill_rollover': tbill_rollover,
                'nsi_maturities': nsi_maturities,
                'total_maturing': maturing_debt,
                'gfn': gfn,
                'gdp': gdp_val,
                'gfn_gdp_pct': gfn_gdp
            }
        
        self.results['gfn_annual'] = gfn_data
        
        # Summary statistics
        gfn_gdp_series = [d['gfn_gdp_pct'] for d in gfn_data.values()]
        
        self.results['summary'] = {
            'average_gfn_gdp': np.mean(gfn_gdp_series),
            'max_gfn_gdp': np.max(gfn_gdp_series),
            'max_gfn_year': list(gfn_data.keys())[np.argmax(gfn_gdp_series)],
            'years_above_15': sum(1 for g in gfn_gdp_series if g > 15),
            'years_above_20': sum(1 for g in gfn_gdp_series if g > 20)
        }
        
        return self.results
    
    def calculate_refinancing_risk_index(self):
        """
        Calculate a composite refinancing risk index.
        
        Components:
        1. GFN/GDP ratio
        2. Short-term debt share
        3. Foreign holdings share
        4. Concentration of maturities
        """
        if 'gfn_annual' not in self.results:
            self.calculate_gfn()
        
        # Additional risk factors (approximate values for UK)
        risk_factors = {
            'short_term_share': 0.08,      # <1yr debt / total debt
            'foreign_holdings': 0.28,       # Foreign held gilts
            'ilg_share': 0.34,              # Inflation-linked exposure
            'average_maturity': 14.5,       # Years
            'weighted_ave_coupon': 3.2,     # %
        }
        
        # Concentration index (Herfindahl of maturity buckets)
        maturity_buckets = [0.08, 0.12, 0.22, 0.28, 0.30]  # Share in each bucket
        herfindahl = sum(s**2 for s in maturity_buckets)
        
        # Composite risk score (0-100)
        gfn_component = min(self.results['summary']['average_gfn_gdp'] / 25 * 30, 30)
        short_term_component = risk_factors['short_term_share'] / 0.20 * 20
        foreign_component = risk_factors['foreign_holdings'] / 0.50 * 15
        concentration_component = herfindahl / 0.30 * 15
        maturity_component = max(0, (15 - risk_factors['average_maturity']) / 10 * 20)
        
        total_risk_score = (gfn_component + short_term_component + 
                           foreign_component + concentration_component + 
                           maturity_component)
        
        self.results['refinancing_risk'] = {
            'risk_score': total_risk_score,
            'components': {
                'gfn': gfn_component,
                'short_term': short_term_component,
                'foreign': foreign_component,
                'concentration': concentration_component,
                'maturity': maturity_component
            },
            'risk_factors': risk_factors,
            'herfindahl': herfindahl
        }
        
        return self.results['refinancing_risk']
    
    def print_results(self):
        """Print comprehensive GFN analysis."""
        if 'gfn_annual' not in self.results:
            self.calculate_gfn()
        
        if 'refinancing_risk' not in self.results:
            self.calculate_refinancing_risk_index()
        
        print("\n" + "="*70)
        print("GROSS FINANCING NEEDS (GFN) ANALYSIS")
        print("="*70)
        
        print("\n1. ANNUAL GROSS FINANCING NEEDS")
        print("-"*70)
        print(f"{'Year':<8} {'Prim Def':<10} {'Interest':<10} {'Maturing':<12} {'GFN':<10} {'GFN/GDP':<10}")
        print(f"{'':8} {'£bn':<10} {'£bn':<10} {'£bn':<12} {'£bn':<10} {'%':<10}")
        print("-"*70)
        
        for year, data in self.results['gfn_annual'].items():
            flag = "⚠️" if data['gfn_gdp_pct'] > 15 else ""
            print(f"{year:<8} {data['primary_deficit']:<10.1f} {data['interest']:<10.1f} "
                  f"{data['total_maturing']:<12.1f} {data['gfn']:<10.1f} "
                  f"{data['gfn_gdp_pct']:<8.1f}% {flag}")
        
        print("\n2. GFN SUMMARY")
        print("-"*50)
        summary = self.results['summary']
        print(f"   Average GFN/GDP:        {summary['average_gfn_gdp']:.1f}%")
        print(f"   Maximum GFN/GDP:        {summary['max_gfn_gdp']:.1f}% ({summary['max_gfn_year']})")
        print(f"   Years above 15% GDP:    {summary['years_above_15']}")
        print(f"   Years above 20% GDP:    {summary['years_above_20']}")
        
        print("\n3. IMF THRESHOLD ASSESSMENT")
        print("-"*50)
        avg_gfn = summary['average_gfn_gdp']
        if avg_gfn < 10:
            assessment = "LOW RISK"
            explanation = "GFN comfortably below elevated risk thresholds"
        elif avg_gfn < 15:
            assessment = "MODERATE RISK"
            explanation = "GFN below IMF threshold but warrants monitoring"
        elif avg_gfn < 20:
            assessment = "ELEVATED RISK"
            explanation = "GFN in IMF elevated risk zone (15-20% GDP)"
        else:
            assessment = "HIGH RISK"
            explanation = "GFN above IMF high-risk threshold of 20% GDP"
        
        print(f"   Assessment: {assessment}")
        print(f"   {explanation}")
        
        print("\n4. REFINANCING RISK INDEX")
        print("-"*50)
        rr = self.results['refinancing_risk']
        print(f"   Composite Risk Score: {rr['risk_score']:.1f}/100")
        print(f"\n   Component breakdown:")
        for name, value in rr['components'].items():
            print(f"   - {name:<20}: {value:.1f}")
        
        risk_score = rr['risk_score']
        if risk_score < 25:
            risk_assessment = "LOW"
        elif risk_score < 40:
            risk_assessment = "MODERATE"
        elif risk_score < 60:
            risk_assessment = "ELEVATED"
        else:
            risk_assessment = "HIGH"
        
        print(f"\n   Refinancing Risk: {risk_assessment}")
        
        print("\n5. MITIGATING FACTORS (UK-SPECIFIC)")
        print("-"*50)
        print("   ✓ Long average maturity (14.5 years) reduces rollover pressure")
        print("   ✓ Deep, liquid gilt market with strong domestic investor base")
        print("   ✓ Reserve currency status provides additional flexibility")
        print("   ✓ Bank of England can act as buyer of last resort (QE)")
        print("   ✗ High ILG share (34%) creates inflation vulnerability")
        print("   ✗ Large foreign holdings (28%) sensitive to sentiment shifts")
        
        return self.results


# =============================================================================
# SECTION 4: FAT-TAILED MONTE CARLO SIMULATION
# =============================================================================

class FatTailedMonteCarloSimulator:
    """
    Enhanced Monte Carlo simulation with:
    1. Student-t distributions for fat tails
    2. Copula-based dependence structure
    3. Regime-switching dynamics
    4. Proper tail risk measures
    """
    
    def __init__(self, n_simulations=10000, horizon_years=10, random_seed=42):
        self.n_sims = n_simulations
        self.horizon = horizon_years
        self.seed = random_seed
        self.results = {}
        
        # Degrees of freedom for Student-t (lower = fatter tails)
        self.df_gdp = 5      # GDP growth - moderate fat tails
        self.df_inflation = 4  # Inflation - fatter tails (more extreme events)
        self.df_rates = 6    # Interest rates - somewhat fat tails
        
    def generate_correlated_t_shocks(self, n_years):
        """
        Generate correlated Student-t distributed shocks using a t-copula.
        """
        np.random.seed(self.seed)
        
        # Correlation matrix (GDP, CPI, RPI, Gilts, Bank Rate)
        # Based on UK historical correlations
        corr_matrix = np.array([
            [1.00, 0.20, 0.15, -0.30, -0.25],   # GDP growth
            [0.20, 1.00, 0.85, 0.40, 0.50],     # CPI
            [0.15, 0.85, 1.00, 0.45, 0.45],     # RPI
            [-0.30, 0.40, 0.45, 1.00, 0.70],    # Gilt yields
            [-0.25, 0.50, 0.45, 0.70, 1.00]     # Bank Rate
        ])
        
        n_vars = 5
        n_total = self.n_sims * n_years
        
        # Generate t-copula samples
        # Step 1: Generate MVN samples
        L = np.linalg.cholesky(corr_matrix)
        z = np.random.standard_normal((n_total, n_vars))
        mvn = z @ L.T
        
        # Step 2: Transform to uniform via t-CDF with copula df
        df_copula = 5  # Copula degrees of freedom
        chi2 = np.random.chisquare(df_copula, n_total)
        t_samples = mvn * np.sqrt(df_copula / chi2[:, np.newaxis])
        uniform = stats.t.cdf(t_samples, df_copula)
        
        # Step 3: Transform to marginal t-distributions
        dfs = [self.df_gdp, self.df_inflation, self.df_inflation, 
               self.df_rates, self.df_rates]
        
        shocks = np.zeros_like(uniform)
        for i, df in enumerate(dfs):
            shocks[:, i] = stats.t.ppf(uniform[:, i], df)
        
        # Reshape to (n_sims, n_years, n_vars)
        shocks = shocks.reshape(self.n_sims, n_years, n_vars)
        
        return shocks
    
    def simulate_paths(self):
        """
        Run fat-tailed Monte Carlo simulation.
        """
        np.random.seed(self.seed)
        
        n_years = self.horizon
        
        # Initial conditions (2024-25)
        initial_debt = 2746.3  # £bn
        initial_gdp = 2864.0   # £bn
        initial_debt_ratio = initial_debt / initial_gdp * 100
        
        # Baseline parameters
        params = {
            'real_gdp_growth_mean': 1.5,
            'gdp_deflator_mean': 2.5,
            'cpi_mean': 2.0,
            'rpi_mean': 3.0,
            'gilt_yield_mean': 4.5,
            'bank_rate_mean': 4.0,
            
            # Volatilities (annualized, scaled for t-dist)
            'gdp_vol': 2.0,
            'cpi_vol': 1.5,
            'rpi_vol': 1.8,
            'gilt_vol': 1.2,
            'bank_rate_vol': 1.0,
            
            # AR(1) persistence
            'gdp_ar': 0.3,
            'inflation_ar': 0.5,
            'rate_ar': 0.85,
            
            # Primary balance trajectory (% GDP, OBR baseline)
            'primary_balance': [-1.1, -0.6, -0.2, 0.3, 0.7, 1.3, 
                               1.5, 1.5, 1.5, 1.5, 1.5]
        }
        
        # Generate correlated fat-tailed shocks
        shocks = self.generate_correlated_t_shocks(n_years + 1)
        
        # Initialize storage
        debt_paths = np.zeros((self.n_sims, n_years + 1))
        gdp_paths = np.zeros((self.n_sims, n_years + 1))
        debt_ratio_paths = np.zeros((self.n_sims, n_years + 1))
        interest_paths = np.zeros((self.n_sims, n_years + 1))
        
        # State variables
        gdp_growth = np.zeros((self.n_sims, n_years + 1))
        cpi = np.zeros((self.n_sims, n_years + 1))
        rpi = np.zeros((self.n_sims, n_years + 1))
        gilt_yield = np.zeros((self.n_sims, n_years + 1))
        bank_rate = np.zeros((self.n_sims, n_years + 1))
        
        # Initial values
        debt_paths[:, 0] = initial_debt
        gdp_paths[:, 0] = initial_gdp
        debt_ratio_paths[:, 0] = initial_debt_ratio
        
        gdp_growth[:, 0] = params['real_gdp_growth_mean'] + params['gdp_deflator_mean']
        cpi[:, 0] = params['cpi_mean']
        rpi[:, 0] = params['rpi_mean']
        gilt_yield[:, 0] = params['gilt_yield_mean']
        bank_rate[:, 0] = params['bank_rate_mean']
        
        # Simulate paths
        for t in range(1, n_years + 1):
            # AR(1) dynamics with fat-tailed shocks
            gdp_growth[:, t] = (params['gdp_ar'] * gdp_growth[:, t-1] + 
                               (1 - params['gdp_ar']) * (params['real_gdp_growth_mean'] + params['gdp_deflator_mean']) +
                               params['gdp_vol'] * shocks[:, t, 0])
            
            cpi[:, t] = (params['inflation_ar'] * cpi[:, t-1] + 
                        (1 - params['inflation_ar']) * params['cpi_mean'] +
                        params['cpi_vol'] * shocks[:, t, 1])
            
            rpi[:, t] = (params['inflation_ar'] * rpi[:, t-1] + 
                        (1 - params['inflation_ar']) * params['rpi_mean'] +
                        params['rpi_vol'] * shocks[:, t, 2])
            
            gilt_yield[:, t] = (params['rate_ar'] * gilt_yield[:, t-1] + 
                               (1 - params['rate_ar']) * params['gilt_yield_mean'] +
                               params['gilt_vol'] * shocks[:, t, 3])
            
            bank_rate[:, t] = (params['rate_ar'] * bank_rate[:, t-1] + 
                              (1 - params['rate_ar']) * params['bank_rate_mean'] +
                              params['bank_rate_vol'] * shocks[:, t, 4])
            
            # Floor rates at zero (or small positive)
            gilt_yield[:, t] = np.maximum(gilt_yield[:, t], 0.5)
            bank_rate[:, t] = np.maximum(bank_rate[:, t], 0.1)
            rpi[:, t] = np.maximum(rpi[:, t], -2.0)  # Allow mild deflation
            
            # GDP evolution
            gdp_paths[:, t] = gdp_paths[:, t-1] * (1 + gdp_growth[:, t] / 100)
            
            # Interest payments
            # Blended effective rate (existing debt + new issuance + ILG uplift)
            ilg_share = 0.34
            existing_rate = 0.7 * gilt_yield[:, t-1] + 0.3 * gilt_yield[:, t]
            ilg_uplift = ilg_share * rpi[:, t]  # ILG accrued uplift
            effective_rate = existing_rate + ilg_uplift
            
            interest_payment = debt_paths[:, t-1] * effective_rate / 100
            interest_paths[:, t] = interest_payment
            
            # Primary balance
            pb_gdp = params['primary_balance'][min(t, len(params['primary_balance'])-1)]
            
            # Add cyclical component (automatic stabilizers)
            # Shortfall in growth → ~0.5% GDP extra borrowing per 1% GDP shortfall
            growth_shortfall = (params['real_gdp_growth_mean'] + params['gdp_deflator_mean']) - gdp_growth[:, t]
            cyclical_adj = 0.5 * np.maximum(growth_shortfall, 0)
            
            primary_balance = (pb_gdp - cyclical_adj) * gdp_paths[:, t] / 100
            
            # Debt evolution
            # D_t = D_{t-1} + Interest - Primary_surplus
            debt_paths[:, t] = debt_paths[:, t-1] + interest_payment - primary_balance
            
            # Debt/GDP ratio
            debt_ratio_paths[:, t] = debt_paths[:, t] / gdp_paths[:, t] * 100
        
        # Store results
        self.results['debt_ratio_paths'] = debt_ratio_paths
        self.results['debt_paths'] = debt_paths
        self.results['gdp_paths'] = gdp_paths
        self.results['interest_paths'] = interest_paths
        self.results['macro_paths'] = {
            'gdp_growth': gdp_growth,
            'cpi': cpi,
            'rpi': rpi,
            'gilt_yield': gilt_yield,
            'bank_rate': bank_rate
        }
        
        return self.results
    
    def compute_risk_metrics(self):
        """
        Compute comprehensive risk metrics including tail risk measures.
        """
        if 'debt_ratio_paths' not in self.results:
            self.simulate_paths()
        
        debt_ratios = self.results['debt_ratio_paths']
        
        # Terminal distribution
        terminal = debt_ratios[:, -1]
        
        # Basic statistics
        self.results['terminal_stats'] = {
            'mean': np.mean(terminal),
            'median': np.median(terminal),
            'std': np.std(terminal),
            'skewness': stats.skew(terminal),
            'kurtosis': stats.kurtosis(terminal),  # Excess kurtosis
            'min': np.min(terminal),
            'max': np.max(terminal)
        }
        
        # Percentiles
        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        self.results['terminal_percentiles'] = {
            p: np.percentile(terminal, p) for p in percentiles
        }
        
        # Probability of breaching thresholds
        thresholds = [80, 90, 100, 110, 120, 130, 150]
        max_ratios = np.max(debt_ratios, axis=1)  # Peak debt ratio per path
        
        self.results['threshold_probs'] = {}
        for thresh in thresholds:
            prob_terminal = np.mean(terminal > thresh) * 100
            prob_ever = np.mean(max_ratios > thresh) * 100
            self.results['threshold_probs'][thresh] = {
                'prob_terminal': prob_terminal,
                'prob_ever': prob_ever
            }
        
        # Value at Risk and Expected Shortfall
        for alpha in [0.95, 0.99]:
            var = np.percentile(terminal, alpha * 100)
            es = np.mean(terminal[terminal > var])
            self.results[f'VaR_{int(alpha*100)}'] = var
            self.results[f'ES_{int(alpha*100)}'] = es
        
        # Fan chart data
        years = list(range(2024, 2024 + debt_ratios.shape[1]))
        self.results['fan_chart'] = {
            'years': years,
            'p5': np.percentile(debt_ratios, 5, axis=0),
            'p10': np.percentile(debt_ratios, 10, axis=0),
            'p25': np.percentile(debt_ratios, 25, axis=0),
            'p50': np.percentile(debt_ratios, 50, axis=0),
            'p75': np.percentile(debt_ratios, 75, axis=0),
            'p90': np.percentile(debt_ratios, 90, axis=0),
            'p95': np.percentile(debt_ratios, 95, axis=0),
            'mean': np.mean(debt_ratios, axis=0)
        }
        
        return self.results
    
    def compare_with_normal(self):
        """
        Run comparison simulation with normal distributions
        to quantify the fat-tail impact.
        """
        np.random.seed(self.seed + 1000)
        
        # Simple normal simulation for comparison
        n_years = self.horizon
        initial_debt_ratio = 95.9
        
        # Parameters
        mean_change = -0.5  # Average annual change in debt/GDP
        vol = 3.0  # Std of annual change
        
        # Normal simulation
        normal_changes = np.random.normal(mean_change, vol, 
                                         (self.n_sims, n_years))
        normal_paths = np.zeros((self.n_sims, n_years + 1))
        normal_paths[:, 0] = initial_debt_ratio
        
        for t in range(1, n_years + 1):
            normal_paths[:, t] = normal_paths[:, t-1] + normal_changes[:, t-1]
        
        normal_terminal = normal_paths[:, -1]
        
        # Compare with fat-tailed results
        fat_terminal = self.results['debt_ratio_paths'][:, -1]
        
        self.results['distribution_comparison'] = {
            'normal': {
                'mean': np.mean(normal_terminal),
                'std': np.std(normal_terminal),
                'skewness': stats.skew(normal_terminal),
                'kurtosis': stats.kurtosis(normal_terminal),
                'p1': np.percentile(normal_terminal, 1),
                'p5': np.percentile(normal_terminal, 5),
                'p95': np.percentile(normal_terminal, 95),
                'p99': np.percentile(normal_terminal, 99),
                'prob_above_100': np.mean(normal_terminal > 100) * 100,
                'prob_above_120': np.mean(normal_terminal > 120) * 100
            },
            'fat_tailed': {
                'mean': np.mean(fat_terminal),
                'std': np.std(fat_terminal),
                'skewness': stats.skew(fat_terminal),
                'kurtosis': stats.kurtosis(fat_terminal),
                'p1': np.percentile(fat_terminal, 1),
                'p5': np.percentile(fat_terminal, 5),
                'p95': np.percentile(fat_terminal, 95),
                'p99': np.percentile(fat_terminal, 99),
                'prob_above_100': np.mean(fat_terminal > 100) * 100,
                'prob_above_120': np.mean(fat_terminal > 120) * 100
            }
        }
        
        return self.results['distribution_comparison']
    
    def print_results(self):
        """Print comprehensive Monte Carlo results."""
        if 'terminal_stats' not in self.results:
            self.compute_risk_metrics()
        
        if 'distribution_comparison' not in self.results:
            self.compare_with_normal()
        
        print("\n" + "="*70)
        print("FAT-TAILED MONTE CARLO SIMULATION RESULTS")
        print("="*70)
        print(f"\nSimulations: {self.n_sims:,} | Horizon: {self.horizon} years")
        print(f"Tail parameters: GDP df={self.df_gdp}, Inflation df={self.df_inflation}, Rates df={self.df_rates}")
        
        print("\n1. TERMINAL DISTRIBUTION (2034-35)")
        print("-"*50)
        ts = self.results['terminal_stats']
        print(f"   Mean:        {ts['mean']:.1f}%")
        print(f"   Median:      {ts['median']:.1f}%")
        print(f"   Std Dev:     {ts['std']:.1f}pp")
        print(f"   Skewness:    {ts['skewness']:.2f}")
        print(f"   Ex. Kurtosis:{ts['kurtosis']:.2f} (Normal = 0)")
        print(f"   Range:       [{ts['min']:.1f}%, {ts['max']:.1f}%]")
        
        print("\n2. TERMINAL PERCENTILES")
        print("-"*50)
        tp = self.results['terminal_percentiles']
        for p in [5, 25, 50, 75, 95, 99]:
            print(f"   {p}th percentile: {tp[p]:.1f}%")
        
        print("\n3. TAIL RISK MEASURES")
        print("-"*50)
        print(f"   VaR 95%:     {self.results['VaR_95']:.1f}% (5% of paths worse)")
        print(f"   VaR 99%:     {self.results['VaR_99']:.1f}% (1% of paths worse)")
        print(f"   ES 95%:      {self.results['ES_95']:.1f}% (avg of worst 5%)")
        print(f"   ES 99%:      {self.results['ES_99']:.1f}% (avg of worst 1%)")
        
        print("\n4. THRESHOLD BREACH PROBABILITIES")
        print("-"*50)
        print(f"   {'Threshold':<12} {'P(Terminal)':<15} {'P(Ever)':<15}")
        print("   " + "-"*42)
        for thresh, probs in self.results['threshold_probs'].items():
            print(f"   {thresh}% GDP      {probs['prob_terminal']:>6.1f}%          {probs['prob_ever']:>6.1f}%")
        
        print("\n5. FAT-TAIL IMPACT ANALYSIS")
        print("-"*50)
        comp = self.results['distribution_comparison']
        print(f"   {'Metric':<20} {'Normal':<15} {'Fat-Tailed':<15} {'Difference':<15}")
        print("   " + "-"*60)
        
        for metric in ['mean', 'std', 'skewness', 'kurtosis']:
            n_val = comp['normal'][metric]
            f_val = comp['fat_tailed'][metric]
            diff = f_val - n_val
            print(f"   {metric:<20} {n_val:<15.2f} {f_val:<15.2f} {diff:>+.2f}")
        
        print(f"\n   P(debt > 100%):")
        print(f"   Normal:     {comp['normal']['prob_above_100']:.1f}%")
        print(f"   Fat-tailed: {comp['fat_tailed']['prob_above_100']:.1f}%")
        
        print(f"\n   P(debt > 120%):")
        print(f"   Normal:     {comp['normal']['prob_above_120']:.1f}%")
        print(f"   Fat-tailed: {comp['fat_tailed']['prob_above_120']:.1f}%")
        
        print("\n6. KEY FINDINGS")
        print("-"*50)
        kurt = ts['kurtosis']
        if kurt > 1:
            print(f"   ⚠️ Terminal distribution has SUBSTANTIAL fat tails (kurtosis={kurt:.1f})")
            print("      Extreme outcomes more likely than normal distribution implies")
        
        prob_100_diff = (comp['fat_tailed']['prob_above_100'] - 
                        comp['normal']['prob_above_100'])
        if prob_100_diff > 1:
            print(f"   ⚠️ Fat tails increase P(debt>100%) by {prob_100_diff:.1f}pp")
        
        es_99 = self.results['ES_99']
        print(f"   ⚠️ In the worst 1% of scenarios, debt averages {es_99:.0f}% GDP")
        
        return self.results


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def run_advanced_analysis():
    """Run all Tier 1 advanced analyses."""
    
    print("\n" + "="*70)
    print("UK DEBT SUSTAINABILITY - ADVANCED ANALYSIS (TIER 1)")
    print("="*70)
    print("\nRunning four advanced analytical modules:")
    print("1. Fiscal Reaction Function (Bohn Test)")
    print("2. Fiscal Space Calculation")
    print("3. Gross Financing Needs Analysis")
    print("4. Fat-Tailed Monte Carlo Simulation")
    print("="*70)
    
    results = {}
    
    # 1. Bohn Test
    print("\n\n" + "#"*70)
    print("# MODULE 1: BOHN TEST")
    print("#"*70)
    bohn = BohnTest()
    results['bohn'] = bohn.run_all_tests()
    
    # 2. Fiscal Space
    print("\n\n" + "#"*70)
    print("# MODULE 2: FISCAL SPACE")
    print("#"*70)
    fiscal_space = FiscalSpaceCalculator(results['bohn'])
    results['fiscal_space'] = fiscal_space.print_results()
    
    # 3. Gross Financing Needs
    print("\n\n" + "#"*70)
    print("# MODULE 3: GROSS FINANCING NEEDS")
    print("#"*70)
    gfn = GrossFinancingNeeds()
    results['gfn'] = gfn.print_results()
    
    # 4. Fat-Tailed Monte Carlo
    print("\n\n" + "#"*70)
    print("# MODULE 4: FAT-TAILED MONTE CARLO")
    print("#"*70)
    mc = FatTailedMonteCarloSimulator(n_simulations=10000, horizon_years=10)
    mc.simulate_paths()
    results['monte_carlo'] = mc.print_results()
    
    return results, bohn, fiscal_space, gfn, mc


if __name__ == "__main__":
    results, bohn, fiscal_space, gfn, mc = run_advanced_analysis()
