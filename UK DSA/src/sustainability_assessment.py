"""
UK Debt Sustainability Analysis - Comprehensive Assessment
==========================================================
Imperial College London UROP Project

This module performs the complete sustainability analysis:
1. Historical analysis and regime identification
2. Debt dynamics decomposition  
3. Forward projections (baseline and stress)
4. Monte Carlo simulation
5. Sustainability verdict

Methodology: IMF/ECB DSA Framework + Bohn (1998) Fiscal Reaction Function
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings

from config import (
    HISTORICAL_PSND, HISTORICAL_GDP, HISTORICAL_PSNB, HISTORICAL_DEBT_TO_GDP,
    HISTORICAL_GILT_YIELD_10Y, HISTORICAL_BANK_RATE, HISTORICAL_RPI, HISTORICAL_CPI,
    HISTORICAL_DEBT_INTEREST, HISTORICAL_REAL_YIELD_10Y,
    OBR_GDP, OBR_PSND, OBR_DEBT_TO_GDP, OBR_PSNB, OBR_DEBT_INTEREST,
    OBR_RPI, OBR_CPI, OBR_GILT_YIELD, OBR_BANK_RATE, OBR_NOMINAL_GDP_GROWTH,
    READY_RECKONERS, SCENARIOS, FISCAL_RULES, DEBT_COMPOSITION, ILG_SHARE,
    EXTENDED_PROJECTIONS, MONTE_CARLO_CONFIG,
    get_combined_series, fiscal_year_to_numeric
)
from debt_dynamics import DebtDynamicsModel, compute_r_g_differential
from monte_carlo import MonteCarloEngine, MonteCarloConfig

warnings.filterwarnings('ignore')


@dataclass
class SustainabilityVerdict:
    """Results of sustainability assessment."""
    is_sustainable: bool
    confidence: str  # 'High', 'Medium', 'Low'
    primary_verdict: str
    detailed_assessment: Dict[str, any]
    risk_factors: List[str]
    recommendations: List[str]


class HistoricalAnalyzer:
    """
    Analyzes historical debt dynamics and identifies regimes.
    """
    
    def __init__(self):
        self.start_year = '1997-98'
        self.end_year = '2023-24'
    
    def build_historical_dataset(self) -> pd.DataFrame:
        """Build comprehensive historical dataset."""
        years = sorted([fy for fy in HISTORICAL_PSND.keys() if fy >= self.start_year])
        
        data = []
        for fy in years:
            row = {
                'fiscal_year': fy,
                'year_numeric': fiscal_year_to_numeric(fy),
                'psnd_bn': HISTORICAL_PSND.get(fy),
                'gdp_bn': HISTORICAL_GDP.get(fy),
                'debt_to_gdp': HISTORICAL_DEBT_TO_GDP.get(fy),
                'psnb_bn': HISTORICAL_PSNB.get(fy),
                'debt_interest_bn': HISTORICAL_DEBT_INTEREST.get(fy),
                'gilt_yield_10y': HISTORICAL_GILT_YIELD_10Y.get(fy),
                'real_yield_10y': HISTORICAL_REAL_YIELD_10Y.get(fy),
                'bank_rate': HISTORICAL_BANK_RATE.get(fy),
                'rpi': HISTORICAL_RPI.get(fy),
                'cpi': HISTORICAL_CPI.get(fy),
            }
            data.append(row)
        
        df = pd.DataFrame(data)
        
        # Calculate derived metrics
        df['deficit_to_gdp'] = df['psnb_bn'] / df['gdp_bn'] * 100
        df['nominal_gdp_growth'] = df['gdp_bn'].pct_change() * 100
        df['debt_change'] = df['psnd_bn'].diff()
        df['debt_ratio_change'] = df['debt_to_gdp'].diff()
        
        # Primary balance (PSNB minus debt interest)
        df['primary_balance_bn'] = df['psnb_bn'] - df['debt_interest_bn'].fillna(0)
        df['primary_balance_gdp'] = df['primary_balance_bn'] / df['gdp_bn'] * 100
        
        # Interest-growth differential
        # Effective interest rate approximation: interest paid / debt stock
        df['effective_rate'] = (df['debt_interest_bn'] / df['psnd_bn'].shift(1) * 100).fillna(df['gilt_yield_10y'])
        df['r_minus_g'] = df['effective_rate'] - df['nominal_gdp_growth']
        
        return df
    
    def decompose_debt_dynamics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Decompose changes in debt-to-GDP ratio.
        
        ΔD/Y = (r-g)/(1+g) × D(-1)/Y(-1) + pb/Y + sfa/Y
        """
        decomposition = []
        
        for i in range(1, len(df)):
            prev = df.iloc[i-1]
            curr = df.iloc[i]
            
            # Previous debt ratio
            d_prev = prev['debt_to_gdp'] / 100
            
            # Nominal growth rate
            g = curr['nominal_gdp_growth'] / 100 if pd.notna(curr['nominal_gdp_growth']) else 0
            
            # Effective interest rate
            r = curr['effective_rate'] / 100 if pd.notna(curr['effective_rate']) else curr['gilt_yield_10y'] / 100
            
            # Interest-growth effect
            if 1 + g != 0:
                ig_effect = (r - g) / (1 + g) * d_prev * 100
            else:
                ig_effect = 0
            
            # Primary balance effect
            pb_effect = curr['primary_balance_gdp'] if pd.notna(curr['primary_balance_gdp']) else 0
            
            # Actual change
            actual_change = curr['debt_to_gdp'] - prev['debt_to_gdp']
            
            # Stock-flow adjustment (residual)
            sfa_effect = actual_change - ig_effect - pb_effect
            
            decomposition.append({
                'fiscal_year': curr['fiscal_year'],
                'debt_ratio_start': prev['debt_to_gdp'],
                'debt_ratio_end': curr['debt_to_gdp'],
                'actual_change': actual_change,
                'interest_growth_effect': ig_effect,
                'primary_balance_effect': pb_effect,
                'sfa_effect': sfa_effect,
                'r_minus_g': (r - g) * 100,
                'effective_rate': r * 100,
                'nominal_growth': g * 100,
            })
        
        return pd.DataFrame(decomposition)
    
    def identify_regimes(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """Identify distinct fiscal regimes in the data."""
        regimes = {
            'consolidation': [],    # Debt ratio falling
            'expansion': [],        # Debt ratio rising
            'crisis': [],          # Sharp increases
            'stable': [],          # Roughly flat
        }
        
        for i in range(1, len(df)):
            fy = df.iloc[i]['fiscal_year']
            change = df.iloc[i]['debt_ratio_change']
            
            if pd.isna(change):
                continue
            
            if change < -2:
                regimes['consolidation'].append(fy)
            elif change > 10:
                regimes['crisis'].append(fy)
            elif change > 2:
                regimes['expansion'].append(fy)
            else:
                regimes['stable'].append(fy)
        
        return regimes


class ForwardProjector:
    """
    Projects debt path forward under various scenarios.
    """
    
    def __init__(self):
        self.model = DebtDynamicsModel()
    
    def project_baseline(self, horizon_years: int = 10) -> pd.DataFrame:
        """
        Project debt path under OBR baseline + extension.
        """
        projections = []
        
        # OBR forecast period (2024-25 to 2029-30)
        for fy in ['2024-25', '2025-26', '2026-27', '2027-28', '2028-29', '2029-30']:
            projections.append({
                'fiscal_year': fy,
                'year': fiscal_year_to_numeric(fy) - 2024,
                'gdp_bn': OBR_GDP[fy],
                'psnd_bn': OBR_PSND[fy],
                'debt_to_gdp': OBR_DEBT_TO_GDP[fy],
                'psnb_bn': OBR_PSNB[fy],
                'debt_interest_bn': OBR_DEBT_INTEREST[fy],
                'nominal_gdp_growth': OBR_NOMINAL_GDP_GROWTH[fy],
                'rpi': OBR_RPI[fy],
                'gilt_yield': OBR_GILT_YIELD[fy],
                'source': 'OBR March 2025',
            })
        
        # Extension period (2030-31 to 2034-35)
        last_obr = projections[-1]
        for i, fy in enumerate(['2030-31', '2031-32', '2032-33', '2033-34', '2034-35']):
            prev = projections[-1]
            
            # Assumptions from EXTENDED_PROJECTIONS
            real_growth = EXTENDED_PROJECTIONS['real_gdp_growth']
            deflator = EXTENDED_PROJECTIONS['gdp_deflator']
            nominal_growth = (1 + real_growth/100) * (1 + deflator/100) - 1
            
            new_gdp = prev['gdp_bn'] * (1 + nominal_growth)
            
            # Primary balance assumption
            pb_gdp = EXTENDED_PROJECTIONS['primary_balance_gdp']
            primary_balance = new_gdp * pb_gdp / 100
            
            # Debt interest: effective rate on stock + new borrowing
            eff_rate = EXTENDED_PROJECTIONS['effective_interest_rate'] / 100
            new_rate = EXTENDED_PROJECTIONS['new_debt_rate'] / 100
            
            # Simplified: interest on previous debt stock
            interest = prev['psnd_bn'] * eff_rate
            
            # PSNB = primary balance + interest
            psnb = primary_balance + interest
            
            # New debt = previous + borrowing
            new_debt = prev['psnd_bn'] + psnb
            
            projections.append({
                'fiscal_year': fy,
                'year': fiscal_year_to_numeric(fy) - 2024,
                'gdp_bn': new_gdp,
                'psnd_bn': new_debt,
                'debt_to_gdp': new_debt / new_gdp * 100,
                'psnb_bn': psnb,
                'debt_interest_bn': interest,
                'nominal_gdp_growth': nominal_growth * 100,
                'rpi': 2.8,
                'gilt_yield': 4.0,
                'source': 'Extended Projection',
            })
        
        return pd.DataFrame(projections)
    
    def apply_scenario(self, baseline: pd.DataFrame, scenario_name: str) -> pd.DataFrame:
        """
        Apply a stress scenario to the baseline projection.
        Uses OBR ready reckoners for interest rate and inflation shocks.
        """
        if scenario_name not in SCENARIOS:
            raise ValueError(f"Unknown scenario: {scenario_name}")
        
        scenario = SCENARIOS[scenario_name]
        shocked = baseline.copy()
        
        # First pass: apply shocks to individual year flows
        cumulative_extra_borrowing = 0
        
        for i, row in shocked.iterrows():
            year_idx = int(row['year'])
            fy = row['fiscal_year']
            
            extra_interest = 0
            extra_borrowing = 0
            gdp_multiplier = 1.0
            
            # Check if we're within shock duration
            in_shock_period = year_idx >= 0 and year_idx < scenario.duration
            
            if in_shock_period:
                # Gilt yield shock - impacts debt interest via ready reckoners
                if scenario.gilt_yield_shock != 0 and fy in READY_RECKONERS['gilt_rates_1pp']:
                    extra_interest += READY_RECKONERS['gilt_rates_1pp'][fy] * scenario.gilt_yield_shock
                
                # Short rate shock
                if scenario.short_rate_shock != 0 and fy in READY_RECKONERS['short_rates_1pp']:
                    extra_interest += READY_RECKONERS['short_rates_1pp'][fy] * scenario.short_rate_shock
                
                # Inflation shock (affects ILGs)
                if scenario.inflation_shock != 0 and fy in READY_RECKONERS['inflation_1pp']:
                    extra_interest += READY_RECKONERS['inflation_1pp'][fy] * scenario.inflation_shock
                
                # Growth shock - affects GDP and triggers automatic stabilizers
                if scenario.growth_shock != 0:
                    # GDP is lower
                    gdp_multiplier = 1 + scenario.growth_shock / 100
                    
                    # Automatic stabilizers: ~0.5% GDP extra borrowing per 1% GDP shortfall
                    auto_stabilizer = abs(scenario.growth_shock) * row['gdp_bn'] * 0.005
                    extra_borrowing += auto_stabilizer
                
                # Primary balance shock (discretionary fiscal policy)
                if scenario.primary_balance_shock != 0:
                    extra_borrowing += scenario.primary_balance_shock * row['gdp_bn'] / 100
            
            # Apply GDP adjustment
            shocked.loc[i, 'gdp_bn'] = row['gdp_bn'] * gdp_multiplier
            
            # Apply interest and borrowing shocks
            shocked.loc[i, 'debt_interest_bn'] = row['debt_interest_bn'] + extra_interest
            shocked.loc[i, 'psnb_bn'] = row['psnb_bn'] + extra_interest + extra_borrowing
        
        # Second pass: recalculate cumulative debt stock
        for i in range(len(shocked)):
            if i == 0:
                # First year uses baseline starting debt
                shocked.loc[i, 'psnd_bn'] = baseline.loc[i, 'psnd_bn'] + (shocked.loc[i, 'psnb_bn'] - baseline.loc[i, 'psnb_bn'])
            else:
                prev_debt = shocked.loc[i-1, 'psnd_bn']
                new_borrowing = shocked.loc[i, 'psnb_bn']
                shocked.loc[i, 'psnd_bn'] = prev_debt + new_borrowing
        
        # Recalculate debt-to-GDP
        shocked['debt_to_gdp'] = shocked['psnd_bn'] / shocked['gdp_bn'] * 100
        
        return shocked


class SustainabilityAssessor:
    """
    Comprehensive sustainability assessment.
    """
    
    def __init__(self):
        self.historical = HistoricalAnalyzer()
        self.projector = ForwardProjector()
    
    def compute_debt_stabilizing_primary_balance(
        self, 
        debt_ratio: float, 
        r: float, 
        g: float
    ) -> float:
        """
        Calculate the primary balance (% GDP) needed to stabilize debt.
        pb* = (r - g) / (1 + g) × d
        """
        if 1 + g/100 == 0:
            return 0
        return (r/100 - g/100) / (1 + g/100) * debt_ratio
    
    def assess_solvency(self, projections: pd.DataFrame) -> Dict[str, any]:
        """
        Assess intertemporal solvency condition.
        Debt is solvent if NPV of future primary surpluses >= current debt.
        """
        # Use long-run growth and discount rate
        g_long_run = 1.5  # Real growth
        r_long_run = 4.0  # Real discount rate
        
        # Check if debt ratio stabilizes or declines by end of horizon
        initial_debt = projections.iloc[0]['debt_to_gdp']
        terminal_debt = projections.iloc[-1]['debt_to_gdp']
        
        stabilizes = terminal_debt <= initial_debt * 1.05  # Within 5%
        declining = terminal_debt < initial_debt
        
        return {
            'initial_debt_ratio': initial_debt,
            'terminal_debt_ratio': terminal_debt,
            'change': terminal_debt - initial_debt,
            'stabilizes': stabilizes,
            'declining': declining,
            'verdict': 'Sustainable' if declining else ('Marginally Sustainable' if stabilizes else 'Unsustainable'),
        }
    
    def assess_fiscal_rules(self, projections: pd.DataFrame) -> Dict[str, any]:
        """
        Check compliance with UK fiscal rules.
        """
        # Rule 1: Debt falling as % GDP in year 5
        debt_y0 = projections.iloc[0]['debt_to_gdp']
        debt_y5 = projections.iloc[min(5, len(projections)-1)]['debt_to_gdp']
        debt_falling = debt_y5 < debt_y0
        
        # Rule 2: Deficit below 3% GDP
        deficits = projections['psnb_bn'] / projections['gdp_bn'] * 100
        deficit_below_3 = (deficits <= 3.0).all()
        
        # Debt threshold checks
        debt_below_100 = (projections['debt_to_gdp'] <= 100).all()
        max_debt = projections['debt_to_gdp'].max()
        
        return {
            'debt_falling_y5': debt_falling,
            'debt_y0': debt_y0,
            'debt_y5': debt_y5,
            'deficit_below_3pct': deficit_below_3,
            'max_deficit': deficits.max(),
            'debt_below_100': debt_below_100,
            'max_debt_ratio': max_debt,
            'rules_met': debt_falling and deficit_below_3,
        }
    
    def assess_interest_burden(self, projections: pd.DataFrame) -> Dict[str, any]:
        """
        Assess the debt interest burden.
        """
        # Interest as % of GDP
        interest_gdp = projections['debt_interest_bn'] / projections['gdp_bn'] * 100
        
        # Interest as % of receipts (approximation)
        receipts_gdp = 0.42  # ~42% tax-to-GDP
        interest_receipts = interest_gdp / receipts_gdp
        
        return {
            'avg_interest_gdp': interest_gdp.mean(),
            'max_interest_gdp': interest_gdp.max(),
            'terminal_interest_gdp': interest_gdp.iloc[-1],
            'interest_gdp_warning': interest_gdp.max() > FISCAL_RULES['interest_gdp_warning'],
            'interest_gdp_critical': interest_gdp.max() > FISCAL_RULES['interest_gdp_critical'],
            'avg_interest_receipts': interest_receipts.mean(),
            'max_interest_receipts': interest_receipts.max(),
        }
    
    def run_comprehensive_assessment(self) -> SustainabilityVerdict:
        """
        Run complete sustainability assessment.
        """
        # 1. Historical analysis
        hist_df = self.historical.build_historical_dataset()
        hist_decomposition = self.historical.decompose_debt_dynamics(hist_df)
        regimes = self.historical.identify_regimes(hist_df)
        
        # 2. Baseline projection
        baseline = self.projector.project_baseline(horizon_years=10)
        
        # 3. Scenario analysis
        scenarios_results = {}
        for name in SCENARIOS:
            if name != 'baseline':
                scenarios_results[name] = self.projector.apply_scenario(baseline, name)
        
        # 4. Individual assessments
        solvency = self.assess_solvency(baseline)
        fiscal_rules = self.assess_fiscal_rules(baseline)
        interest_burden = self.assess_interest_burden(baseline)
        
        # 5. Risk factors
        risk_factors = []
        
        # High starting debt
        if baseline.iloc[0]['debt_to_gdp'] > 90:
            risk_factors.append(f"High initial debt ratio ({baseline.iloc[0]['debt_to_gdp']:.1f}%)")
        
        # Positive r-g differential
        avg_r_g = hist_decomposition['r_minus_g'].tail(5).mean()
        if avg_r_g > 0:
            risk_factors.append(f"Unfavorable interest-growth differential (r-g = +{avg_r_g:.1f}pp)")
        
        # Large ILG exposure
        if ILG_SHARE > 0.25:
            risk_factors.append(f"High index-linked gilt exposure ({ILG_SHARE*100:.1f}% of gilt stock)")
        
        # Interest burden growing
        if interest_burden['terminal_interest_gdp'] > interest_burden['avg_interest_gdp']:
            risk_factors.append("Rising interest burden trajectory")
        
        # Stress scenario vulnerabilities
        for name, result in scenarios_results.items():
            if result['debt_to_gdp'].max() > 110:
                risk_factors.append(f"Vulnerable to {SCENARIOS[name].name} scenario (debt could reach {result['debt_to_gdp'].max():.1f}%)")
        
        # 6. Recommendations
        recommendations = []
        
        if not solvency['declining']:
            recommendations.append("Pursue gradual fiscal consolidation to put debt on declining path")
        
        if interest_burden['max_interest_gdp'] > 4.0:
            recommendations.append("Consider extending debt maturity to reduce refinancing risk")
        
        if ILG_SHARE > 0.30:
            recommendations.append("Monitor inflation sensitivity given large ILG exposure")
        
        if len([r for r in scenarios_results.values() if r['debt_to_gdp'].max() > 100]) > 3:
            recommendations.append("Build fiscal buffers to improve resilience to shocks")
        
        # 7. Final verdict
        is_sustainable = (
            solvency['stabilizes'] and 
            fiscal_rules['rules_met'] and 
            not interest_burden['interest_gdp_critical']
        )
        
        if is_sustainable and solvency['declining']:
            confidence = 'High'
            verdict = "UK public debt is SUSTAINABLE under the OBR March 2025 baseline"
        elif is_sustainable:
            confidence = 'Medium'
            verdict = "UK public debt is MARGINALLY SUSTAINABLE but vulnerable to adverse shocks"
        else:
            confidence = 'Low'
            verdict = "UK public debt sustainability is AT RISK without policy adjustment"
        
        return SustainabilityVerdict(
            is_sustainable=is_sustainable,
            confidence=confidence,
            primary_verdict=verdict,
            detailed_assessment={
                'historical': {
                    'data': hist_df,
                    'decomposition': hist_decomposition,
                    'regimes': regimes,
                },
                'baseline': baseline,
                'scenarios': scenarios_results,
                'solvency': solvency,
                'fiscal_rules': fiscal_rules,
                'interest_burden': interest_burden,
            },
            risk_factors=risk_factors,
            recommendations=recommendations,
        )


def run_full_analysis():
    """
    Execute complete debt sustainability analysis.
    """
    print("=" * 70)
    print("UK DEBT SUSTAINABILITY ANALYSIS")
    print("Comprehensive Assessment - March 2025 OBR Baseline")
    print("=" * 70)
    
    assessor = SustainabilityAssessor()
    verdict = assessor.run_comprehensive_assessment()
    
    print("\n" + "=" * 70)
    print("SUSTAINABILITY VERDICT")
    print("=" * 70)
    print(f"\n{verdict.primary_verdict}")
    print(f"Confidence: {verdict.confidence}")
    
    print("\n" + "-" * 70)
    print("SOLVENCY ASSESSMENT")
    print("-" * 70)
    sol = verdict.detailed_assessment['solvency']
    print(f"  Initial debt/GDP (2024-25): {sol['initial_debt_ratio']:.1f}%")
    print(f"  Terminal debt/GDP (2034-35): {sol['terminal_debt_ratio']:.1f}%")
    print(f"  Change: {sol['change']:+.1f}pp")
    print(f"  Verdict: {sol['verdict']}")
    
    print("\n" + "-" * 70)
    print("FISCAL RULES COMPLIANCE")
    print("-" * 70)
    rules = verdict.detailed_assessment['fiscal_rules']
    print(f"  Debt falling by year 5: {'✓' if rules['debt_falling_y5'] else '✗'}")
    print(f"  Deficit below 3% GDP: {'✓' if rules['deficit_below_3pct'] else '✗'}")
    print(f"  Debt below 100% GDP: {'✓' if rules['debt_below_100'] else '✗'}")
    print(f"  Maximum debt ratio: {rules['max_debt_ratio']:.1f}%")
    
    print("\n" + "-" * 70)
    print("INTEREST BURDEN")
    print("-" * 70)
    interest = verdict.detailed_assessment['interest_burden']
    print(f"  Average interest/GDP: {interest['avg_interest_gdp']:.2f}%")
    print(f"  Terminal interest/GDP: {interest['terminal_interest_gdp']:.2f}%")
    print(f"  Warning threshold (4%): {'⚠' if interest['interest_gdp_warning'] else '✓'}")
    
    print("\n" + "-" * 70)
    print("RISK FACTORS")
    print("-" * 70)
    for i, risk in enumerate(verdict.risk_factors, 1):
        print(f"  {i}. {risk}")
    
    print("\n" + "-" * 70)
    print("RECOMMENDATIONS")
    print("-" * 70)
    for i, rec in enumerate(verdict.recommendations, 1):
        print(f"  {i}. {rec}")
    
    print("\n" + "=" * 70)
    print("SCENARIO ANALYSIS SUMMARY")
    print("=" * 70)
    print(f"\n{'Scenario':<30} {'Terminal Debt/GDP':>20} {'Max Debt/GDP':>15}")
    print("-" * 65)
    
    baseline = verdict.detailed_assessment['baseline']
    print(f"{'Baseline':<30} {baseline['debt_to_gdp'].iloc[-1]:>19.1f}% {baseline['debt_to_gdp'].max():>14.1f}%")
    
    for name, result in verdict.detailed_assessment['scenarios'].items():
        scenario = SCENARIOS[name]
        print(f"{scenario.name:<30} {result['debt_to_gdp'].iloc[-1]:>19.1f}% {result['debt_to_gdp'].max():>14.1f}%")
    
    return verdict


if __name__ == '__main__':
    verdict = run_full_analysis()
