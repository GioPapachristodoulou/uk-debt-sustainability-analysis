"""
Recalibrated Fat-Tailed Monte Carlo Simulation
===============================================

Calibrated to match OBR baseline as median while incorporating
realistic fat-tailed risk for extreme scenarios.
"""

import numpy as np
import pandas as pd
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class CalibratedFatTailMC:
    """
    Monte Carlo simulation calibrated to:
    1. Match OBR baseline as median path
    2. Use t-distributions for fat tails
    3. Produce realistic tail risk measures
    """
    
    def __init__(self, n_simulations=10000, horizon_years=10, random_seed=42):
        self.n_sims = n_simulations
        self.horizon = horizon_years
        self.seed = random_seed
        self.results = {}
        
        # Degrees of freedom for Student-t (calibrated to historical crises)
        self.df_growth = 5      # GDP growth
        self.df_inflation = 5   # Inflation
        self.df_rates = 7       # Interest rates
        
    def simulate(self):
        """
        Run calibrated simulation.
        """
        np.random.seed(self.seed)
        
        # OBR baseline path (fiscal years 2024-25 to 2034-35)
        obr_debt_ratio = np.array([95.9, 96.0, 96.2, 96.1, 96.1, 95.3, 
                                    94.0, 92.5, 91.0, 89.7, 88.5])
        
        n_years = self.horizon
        
        # Initial conditions
        initial_debt = 2746.3  # £bn
        initial_gdp = 2864.0   # £bn
        
        # Baseline annual GDP growth (nominal, from OBR)
        baseline_gdp_growth = np.array([4.2, 3.8, 3.5, 3.5, 3.5, 3.5,
                                        3.5, 3.5, 3.5, 3.5, 3.5])
        
        # Baseline effective interest rate on debt
        baseline_eff_rate = np.array([3.8, 4.0, 4.0, 4.1, 4.2, 4.2,
                                      4.2, 4.2, 4.2, 4.2, 4.2])
        
        # Baseline primary balance (% GDP, positive = surplus)
        baseline_pb = np.array([-1.1, -0.6, -0.2, 0.3, 0.7, 1.3,
                                1.5, 1.5, 1.5, 1.5, 1.5])
        
        # Volatility parameters (calibrated to historical UK data)
        vol_growth = 1.5       # Std dev of nominal GDP growth shocks
        vol_rate = 0.8         # Std dev of effective rate shocks  
        vol_pb = 0.6           # Std dev of primary balance shocks
        
        # Correlation matrix (growth, rate, pb)
        corr = np.array([
            [1.0, -0.2, 0.3],   # Growth: negative corr with rates, positive with pb
            [-0.2, 1.0, -0.1],  # Rate: negative corr with growth and pb
            [0.3, -0.1, 1.0]    # PB: positive corr with growth
        ])
        
        L = np.linalg.cholesky(corr)
        
        # Storage
        debt_ratio_paths = np.zeros((self.n_sims, n_years + 1))
        debt_paths = np.zeros((self.n_sims, n_years + 1))
        gdp_paths = np.zeros((self.n_sims, n_years + 1))
        
        # Track macro variables
        growth_paths = np.zeros((self.n_sims, n_years + 1))
        rate_paths = np.zeros((self.n_sims, n_years + 1))
        pb_paths = np.zeros((self.n_sims, n_years + 1))
        
        # Initial values
        debt_paths[:, 0] = initial_debt
        gdp_paths[:, 0] = initial_gdp
        debt_ratio_paths[:, 0] = initial_debt / initial_gdp * 100
        
        # AR(1) persistence
        ar_growth = 0.3
        ar_rate = 0.7
        ar_pb = 0.5
        
        # Generate all shocks upfront
        for t in range(1, n_years + 1):
            # Generate correlated uniform via Gaussian copula
            z = np.random.standard_normal((self.n_sims, 3))
            z_corr = z @ L.T
            u = stats.norm.cdf(z_corr)
            
            # Transform to t-distributed shocks
            shock_growth = stats.t.ppf(u[:, 0], self.df_growth)
            shock_rate = stats.t.ppf(u[:, 1], self.df_rates)
            shock_pb = stats.t.ppf(u[:, 2], self.df_growth)  # Use same df as growth
            
            # Scale shocks (t-distribution has variance df/(df-2))
            scale_growth = np.sqrt(self.df_growth / (self.df_growth - 2))
            scale_rate = np.sqrt(self.df_rates / (self.df_rates - 2))
            
            shock_growth = shock_growth / scale_growth * vol_growth
            shock_rate = shock_rate / scale_rate * vol_rate
            shock_pb = shock_pb / scale_growth * vol_pb
            
            # Apply AR(1) dynamics around baseline
            if t == 1:
                growth_paths[:, t] = baseline_gdp_growth[t] + shock_growth
                rate_paths[:, t] = baseline_eff_rate[t] + shock_rate
                pb_paths[:, t] = baseline_pb[t] + shock_pb
            else:
                # Mean reversion to baseline
                growth_paths[:, t] = (baseline_gdp_growth[t] + 
                                      ar_growth * (growth_paths[:, t-1] - baseline_gdp_growth[t-1]) +
                                      shock_growth)
                rate_paths[:, t] = (baseline_eff_rate[t] + 
                                   ar_rate * (rate_paths[:, t-1] - baseline_eff_rate[t-1]) +
                                   shock_rate)
                pb_paths[:, t] = (baseline_pb[t] + 
                                 ar_pb * (pb_paths[:, t-1] - baseline_pb[t-1]) +
                                 shock_pb)
            
            # Apply floors
            rate_paths[:, t] = np.maximum(rate_paths[:, t], 0.5)
            growth_paths[:, t] = np.maximum(growth_paths[:, t], -10)  # Allow deep recession
            
            # Automatic stabilizers: growth shortfall increases borrowing
            growth_shortfall = baseline_gdp_growth[t] - growth_paths[:, t]
            stabilizer_effect = 0.4 * np.maximum(growth_shortfall, 0)
            pb_paths[:, t] = pb_paths[:, t] - stabilizer_effect
            
            # Evolve GDP
            gdp_paths[:, t] = gdp_paths[:, t-1] * (1 + growth_paths[:, t] / 100)
            
            # Interest payments
            interest = debt_paths[:, t-1] * rate_paths[:, t] / 100
            
            # Primary balance in £bn
            primary_balance = pb_paths[:, t] * gdp_paths[:, t] / 100
            
            # Debt evolution: D_t = D_{t-1} + Interest - Primary_balance
            debt_paths[:, t] = debt_paths[:, t-1] + interest - primary_balance
            
            # Debt/GDP ratio
            debt_ratio_paths[:, t] = debt_paths[:, t] / gdp_paths[:, t] * 100
        
        self.results['debt_ratio_paths'] = debt_ratio_paths
        self.results['debt_paths'] = debt_paths
        self.results['gdp_paths'] = gdp_paths
        self.results['growth_paths'] = growth_paths
        self.results['rate_paths'] = rate_paths
        self.results['pb_paths'] = pb_paths
        self.results['obr_baseline'] = obr_debt_ratio
        
        return self.results
    
    def compute_statistics(self):
        """Compute comprehensive statistics."""
        if 'debt_ratio_paths' not in self.results:
            self.simulate()
        
        debt_ratios = self.results['debt_ratio_paths']
        terminal = debt_ratios[:, -1]
        max_ratios = np.max(debt_ratios, axis=1)
        
        # Terminal statistics
        self.results['terminal_stats'] = {
            'mean': np.mean(terminal),
            'median': np.median(terminal),
            'std': np.std(terminal),
            'skewness': stats.skew(terminal),
            'kurtosis': stats.kurtosis(terminal),
            'min': np.min(terminal),
            'max': np.max(terminal)
        }
        
        # Percentiles by year
        years = list(range(2024, 2035))
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
        
        # Terminal percentiles
        self.results['terminal_percentiles'] = {
            p: np.percentile(terminal, p) for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]
        }
        
        # Threshold probabilities
        thresholds = [80, 90, 100, 110, 120, 130]
        self.results['threshold_probs'] = {}
        for thresh in thresholds:
            self.results['threshold_probs'][thresh] = {
                'prob_terminal': np.mean(terminal > thresh) * 100,
                'prob_ever': np.mean(max_ratios > thresh) * 100
            }
        
        # VaR and ES
        for alpha in [0.95, 0.99]:
            var = np.percentile(terminal, alpha * 100)
            es = np.mean(terminal[terminal > var]) if np.any(terminal > var) else var
            self.results[f'VaR_{int(alpha*100)}'] = var
            self.results[f'ES_{int(alpha*100)}'] = es
        
        return self.results
    
    def print_results(self):
        """Print results."""
        if 'terminal_stats' not in self.results:
            self.compute_statistics()
        
        ts = self.results['terminal_stats']
        tp = self.results['terminal_percentiles']
        fc = self.results['fan_chart']
        
        print("\n" + "="*70)
        print("CALIBRATED FAT-TAILED MONTE CARLO RESULTS")
        print("="*70)
        print(f"Simulations: {self.n_sims:,} | Horizon: {self.horizon} years")
        print(f"Student-t df: Growth={self.df_growth}, Rates={self.df_rates}")
        
        print("\n1. TERMINAL YEAR DISTRIBUTION (2034-35)")
        print("-"*50)
        print(f"   Mean:           {ts['mean']:.1f}%")
        print(f"   Median:         {ts['median']:.1f}%")
        print(f"   Std Dev:        {ts['std']:.1f}pp")
        print(f"   Skewness:       {ts['skewness']:.2f}")
        print(f"   Excess Kurtosis:{ts['kurtosis']:.2f}")
        print(f"   Range:          [{ts['min']:.1f}%, {ts['max']:.1f}%]")
        
        print("\n2. TERMINAL PERCENTILES")
        print("-"*50)
        for p in [5, 10, 25, 50, 75, 90, 95, 99]:
            print(f"   {p:>2}th percentile: {tp[p]:>6.1f}%")
        
        print("\n3. TAIL RISK MEASURES")
        print("-"*50)
        print(f"   VaR 95%:        {self.results['VaR_95']:.1f}%")
        print(f"   VaR 99%:        {self.results['VaR_99']:.1f}%")
        print(f"   ES 95%:         {self.results['ES_95']:.1f}%")
        print(f"   ES 99%:         {self.results['ES_99']:.1f}%")
        
        print("\n4. THRESHOLD BREACH PROBABILITIES")
        print("-"*50)
        print(f"   {'Threshold':<12} {'P(Terminal)':<15} {'P(Ever)':<15}")
        print("   " + "-"*42)
        for thresh, probs in self.results['threshold_probs'].items():
            print(f"   {thresh}% GDP      {probs['prob_terminal']:>6.1f}%          {probs['prob_ever']:>6.1f}%")
        
        print("\n5. FAN CHART DATA (Debt/GDP %)")
        print("-"*50)
        print(f"   {'Year':<8} {'5th':<8} {'25th':<8} {'Median':<8} {'75th':<8} {'95th':<8}")
        print("   " + "-"*48)
        for i, year in enumerate(fc['years']):
            print(f"   {year:<8} {fc['p5'][i]:<8.1f} {fc['p25'][i]:<8.1f} "
                  f"{fc['p50'][i]:<8.1f} {fc['p75'][i]:<8.1f} {fc['p95'][i]:<8.1f}")
        
        return self.results


def run_calibrated_mc():
    """Run the calibrated Monte Carlo."""
    mc = CalibratedFatTailMC(n_simulations=10000, horizon_years=10)
    mc.simulate()
    results = mc.print_results()
    return mc, results


if __name__ == "__main__":
    mc, results = run_calibrated_mc()
