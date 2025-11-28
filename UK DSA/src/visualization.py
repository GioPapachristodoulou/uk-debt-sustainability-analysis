"""
UK Debt Sustainability Analysis - Visualization Module
======================================================
Imperial College London UROP Project

This module creates publication-quality figures for the DSA paper:
- Fan charts for debt projections
- Debt dynamics decomposition charts
- Scenario comparison plots
- Historical context charts
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import seaborn as sns
from typing import Dict, List, Tuple, Optional


# Style configuration for publication-quality figures
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Color schemes
COLORS = {
    'baseline': '#1f77b4',
    'adverse': '#d62728',
    'favorable': '#2ca02c',
    'historical': '#7f7f7f',
    'outturn': '#2c3e50',
    'forecast': '#3498db',
    'october_forecast': '#95a5a6',
}

FAN_COLORS = {
    'p95': '#cce5ff',
    'p90': '#99ccff', 
    'p75': '#66b3ff',
    'p50': '#3399ff',
    'median': '#0066cc',
}


class DSAVisualizer:
    """
    Visualization class for debt sustainability analysis.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (10, 6)):
        """
        Initialize visualizer.
        
        Args:
            figsize: Default figure size
        """
        self.figsize = figsize
        
    def plot_fan_chart(
        self,
        fan_chart_data: pd.DataFrame,
        baseline_path: Optional[pd.Series] = None,
        title: str = 'UK Public Sector Net Debt Projections',
        ylabel: str = 'Debt-to-GDP Ratio (%)',
        start_year: int = 2024,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create a fan chart showing probabilistic debt projections.
        
        Args:
            fan_chart_data: DataFrame with percentile columns (p5, p10, etc.)
            baseline_path: Optional baseline projection to overlay
            title: Chart title
            ylabel: Y-axis label
            start_year: Starting year for x-axis
            save_path: Optional path to save figure
            
        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        years = start_year + fan_chart_data['year'].values
        
        # Fill between percentile bands (outer to inner)
        # 90% confidence interval
        ax.fill_between(
            years, 
            fan_chart_data['p5'], 
            fan_chart_data['p95'],
            alpha=0.2, 
            color='#3498db', 
            label='90% CI'
        )
        
        # 80% confidence interval
        ax.fill_between(
            years, 
            fan_chart_data['p10'], 
            fan_chart_data['p90'],
            alpha=0.3, 
            color='#3498db', 
            label='80% CI'
        )
        
        # 50% confidence interval
        ax.fill_between(
            years, 
            fan_chart_data['p25'], 
            fan_chart_data['p75'],
            alpha=0.4, 
            color='#3498db', 
            label='50% CI'
        )
        
        # Median line
        ax.plot(
            years, 
            fan_chart_data['p50'], 
            color='#2c3e50', 
            linewidth=2.5, 
            label='Median'
        )
        
        # Mean line (dashed)
        ax.plot(
            years, 
            fan_chart_data['mean'], 
            color='#e74c3c', 
            linewidth=1.5, 
            linestyle='--',
            label='Mean'
        )
        
        # Baseline if provided
        if baseline_path is not None:
            ax.plot(
                years[:len(baseline_path)], 
                baseline_path, 
                color='#27ae60', 
                linewidth=2, 
                linestyle=':',
                label='OBR Baseline'
            )
        
        # Reference lines
        ax.axhline(y=100, color='#7f8c8d', linestyle='--', linewidth=1, alpha=0.7)
        ax.axhline(y=90, color='#7f8c8d', linestyle=':', linewidth=1, alpha=0.5)
        
        # Formatting
        ax.set_xlabel('Fiscal Year')
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight='bold', pad=20)
        
        # Legend
        ax.legend(loc='upper left', framealpha=0.9)
        
        # Grid
        ax.grid(True, alpha=0.3, linestyle='-')
        ax.set_axisbelow(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            
        return fig
    
    def plot_debt_decomposition(
        self,
        decomposition_data: pd.DataFrame,
        title: str = 'Debt Dynamics Decomposition',
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create a stacked bar chart showing debt dynamics decomposition.
        
        Components:
        - Interest-growth differential effect
        - Primary balance contribution
        - Stock-flow adjustment
        
        Args:
            decomposition_data: DataFrame with decomposition components
            title: Chart title
            save_path: Optional path to save figure
            
        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        years = decomposition_data['year']
        width = 0.7
        
        # Component values
        r_g_effect = decomposition_data['interest_growth_effect'].values
        pb_effect = decomposition_data['primary_balance_effect'].values
        sfa_effect = decomposition_data['sfa_effect'].values
        
        # Stack bars
        # Positive contributions
        pos_r_g = np.maximum(r_g_effect, 0)
        pos_pb = np.maximum(pb_effect, 0)
        pos_sfa = np.maximum(sfa_effect, 0)
        
        # Negative contributions
        neg_r_g = np.minimum(r_g_effect, 0)
        neg_pb = np.minimum(pb_effect, 0)
        neg_sfa = np.minimum(sfa_effect, 0)
        
        x = np.arange(len(years))
        
        # Positive bars
        bars1 = ax.bar(x, pos_r_g, width, label='Interest-Growth (r-g)', 
                       color='#e74c3c', alpha=0.8)
        bars2 = ax.bar(x, pos_pb, width, bottom=pos_r_g, label='Primary Balance',
                       color='#3498db', alpha=0.8)
        bars3 = ax.bar(x, pos_sfa, width, bottom=pos_r_g + pos_pb, label='Stock-Flow Adj.',
                       color='#9b59b6', alpha=0.8)
        
        # Negative bars
        ax.bar(x, neg_r_g, width, color='#e74c3c', alpha=0.8)
        ax.bar(x, neg_pb, width, bottom=neg_r_g, color='#3498db', alpha=0.8)
        ax.bar(x, neg_sfa, width, bottom=neg_r_g + neg_pb, color='#9b59b6', alpha=0.8)
        
        # Net change line
        net_change = decomposition_data['change_in_debt_ratio'].values
        ax.plot(x, net_change, 'ko-', linewidth=2, markersize=6, 
                label='Net Change', zorder=5)
        
        # Formatting
        ax.set_xlabel('Fiscal Year')
        ax.set_ylabel('Contribution to Change in Debt Ratio (pp)')
        ax.set_title(title, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(years, rotation=45, ha='right')
        
        # Zero line
        ax.axhline(y=0, color='black', linewidth=0.8)
        
        ax.legend(loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            
        return fig
    
    def plot_scenario_comparison(
        self,
        scenarios: Dict[str, pd.DataFrame],
        metric: str = 'debt_ratio',
        title: str = 'Debt-to-GDP Under Alternative Scenarios',
        ylabel: str = 'Debt-to-GDP Ratio (%)',
        start_year: int = 2024,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot multiple scenarios on same axes for comparison.
        
        Args:
            scenarios: Dictionary mapping scenario names to DataFrames
            metric: Column name to plot
            title: Chart title
            ylabel: Y-axis label
            start_year: Starting year
            save_path: Optional path to save figure
            
        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(scenarios)))
        linestyles = ['-', '--', '-.', ':', '-', '--', '-.', ':']
        
        for i, (name, data) in enumerate(scenarios.items()):
            if 'year' in data.columns:
                years = data['year'].values
                # Handle both numeric and string years
                if isinstance(years[0], str):
                    # Already fiscal year strings
                    x_values = np.arange(len(years))
                    x_labels = years
                else:
                    x_values = start_year + years
                    x_labels = None
            x_plot = np.arange(len(data))
            ax.plot(
                x_plot, 
                data[metric], 
                color=colors[i],
                linewidth=2.5 if name.lower() == 'baseline' else 1.8,
                linestyle=linestyles[i % len(linestyles)],
                label=name,
                marker='o' if len(x_plot) < 15 else None,
                markersize=4
            )
        
        # Reference lines
        ax.axhline(y=100, color='#7f8c8d', linestyle='--', linewidth=1, alpha=0.7,
                   label='100% Threshold')
        
        ax.set_xlabel('Fiscal Year')
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight='bold', pad=20)
        ax.legend(loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            
        return fig
    
    def plot_r_g_differential(
        self,
        data: pd.DataFrame,
        title: str = 'Interest Rate-Growth Differential (r-g)',
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot the interest rate-growth differential over time.
        
        The r-g differential is key to debt sustainability:
        - Positive: debt grows faster than GDP (unfavorable)
        - Negative: GDP grows faster than debt service (favorable)
        
        Args:
            data: DataFrame with 'year', 'effective_rate', 'nominal_growth' columns
            title: Chart title
            save_path: Optional path to save figure
            
        Returns:
            matplotlib Figure object
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(self.figsize[0], self.figsize[1]*1.3),
                                        sharex=True, gridspec_kw={'height_ratios': [2, 1]})
        
        years = data['year']
        
        # Top panel: r and g separately
        ax1.plot(years, data['effective_interest_rate'], 'o-', 
                 color='#e74c3c', linewidth=2, label='Effective Interest Rate (r)')
        ax1.plot(years, data['nominal_gdp_growth'], 's-', 
                 color='#3498db', linewidth=2, label='Nominal GDP Growth (g)')
        
        ax1.set_ylabel('Percent (%)')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        ax1.set_title(title, fontweight='bold', pad=15)
        ax1.axhline(y=0, color='black', linewidth=0.8)
        
        # Bottom panel: r-g differential
        r_minus_g = data['r_minus_g']
        colors = ['#e74c3c' if x > 0 else '#27ae60' for x in r_minus_g]
        ax2.bar(years, r_minus_g, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
        
        ax2.set_xlabel('Fiscal Year')
        ax2.set_ylabel('r - g (pp)')
        ax2.axhline(y=0, color='black', linewidth=1)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Rotate x-axis labels
        plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            
        return fig
    
    def plot_probability_breach(
        self,
        risk_metrics: Dict[str, np.ndarray],
        thresholds: List[int] = [100, 110, 120],
        start_year: int = 2024,
        title: str = 'Probability of Exceeding Debt Thresholds',
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot probability of debt exceeding various thresholds over time.
        
        Args:
            risk_metrics: Dictionary with probability arrays
            thresholds: Debt-to-GDP thresholds
            start_year: Starting year
            title: Chart title
            save_path: Optional path to save figure
            
        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        colors = ['#27ae60', '#f39c12', '#e74c3c']
        
        for i, threshold in enumerate(thresholds):
            key = f'prob_above_{threshold}'
            if key in risk_metrics:
                prob = risk_metrics[key] * 100  # Convert to percentage
                years = start_year + np.arange(len(prob))
                ax.plot(years, prob, 'o-', color=colors[i], linewidth=2,
                       label=f'Prob(Debt > {threshold}%)', markersize=5)
        
        ax.set_xlabel('Fiscal Year')
        ax.set_ylabel('Probability (%)')
        ax.set_title(title, fontweight='bold', pad=20)
        ax.set_ylim(0, 100)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # Add reference lines
        ax.axhline(y=50, color='#7f8c8d', linestyle='--', linewidth=1, alpha=0.5)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            
        return fig
    
    def plot_historical_debt(
        self,
        historical_data: pd.DataFrame,
        forecast_data: Optional[pd.DataFrame] = None,
        title: str = 'UK Public Sector Net Debt: Historical and Forecast',
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot historical debt-to-GDP with optional forecast overlay.
        
        Args:
            historical_data: DataFrame with 'year' and 'debt_to_gdp' columns
            forecast_data: Optional forecast data
            title: Chart title
            save_path: Optional path to save figure
            
        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Historical series
        ax.plot(historical_data['year'], historical_data['debt_to_gdp'], 
                color=COLORS['historical'], linewidth=2.5, label='Historical')
        
        # Fill below historical line
        ax.fill_between(historical_data['year'], 0, historical_data['debt_to_gdp'],
                        alpha=0.2, color=COLORS['historical'])
        
        # Forecast if provided
        if forecast_data is not None:
            # Connect to historical
            last_hist_year = historical_data['year'].iloc[-1]
            last_hist_value = historical_data['debt_to_gdp'].iloc[-1]
            
            forecast_years = list(forecast_data['year'])
            forecast_values = list(forecast_data['debt_to_gdp'])
            
            ax.plot([last_hist_year] + forecast_years, 
                    [last_hist_value] + forecast_values,
                    color=COLORS['forecast'], linewidth=2.5, linestyle='--',
                    label='OBR Forecast')
        
        # Key events annotations
        events = {
            2008: 'Financial Crisis',
            2020: 'COVID-19'
        }
        for year, event in events.items():
            if year in historical_data['year'].values:
                idx = historical_data[historical_data['year'] == year].index[0]
                value = historical_data.loc[idx, 'debt_to_gdp']
                ax.annotate(event, xy=(year, value), 
                           xytext=(year + 2, value + 10),
                           fontsize=9, arrowprops=dict(arrowstyle='->', color='gray'),
                           color='gray')
        
        ax.set_xlabel('Fiscal Year')
        ax.set_ylabel('Debt-to-GDP Ratio (%)')
        ax.set_title(title, fontweight='bold', pad=20)
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, None)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            
        return fig


def create_all_figures(
    results: Dict,
    output_dir: str = 'outputs/figures'
) -> None:
    """
    Generate all publication figures from simulation results.
    
    Args:
        results: Dictionary containing all analysis results
        output_dir: Directory to save figures
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    viz = DSAVisualizer()
    
    # Fan chart
    if 'fan_chart_data' in results:
        viz.plot_fan_chart(
            results['fan_chart_data'],
            title='UK Debt-to-GDP Ratio: Monte Carlo Projections',
            save_path=f'{output_dir}/fan_chart.png'
        )
    
    # Debt decomposition
    if 'decomposition_data' in results:
        viz.plot_debt_decomposition(
            results['decomposition_data'],
            save_path=f'{output_dir}/debt_decomposition.png'
        )
    
    # Scenario comparison
    if 'scenarios' in results:
        viz.plot_scenario_comparison(
            results['scenarios'],
            save_path=f'{output_dir}/scenario_comparison.png'
        )
    
    # Risk metrics
    if 'risk_metrics' in results:
        viz.plot_probability_breach(
            results['risk_metrics'],
            save_path=f'{output_dir}/probability_breach.png'
        )
    
    print(f"All figures saved to {output_dir}/")
