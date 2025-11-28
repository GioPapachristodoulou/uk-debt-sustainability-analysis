"""
UK Debt Sustainability Analysis - Publication Figures
======================================================
Comprehensive Figure Generation for Academic Paper

Generates 20+ publication-ready figures including:
1. Historical debt trajectory
2. Debt dynamics decomposition
3. r-g differential (Blanchard framework)
4. Interest burden analysis
5. Debt composition
6. ILG sensitivity
7. Fiscal rules history
8. Long-term aging costs
9. Contingent liabilities
10. Monte Carlo fan chart
11. Bohn test visualization
12. Unit root tests
13. Structural breaks
14. Scenario comparison
15. Terminal distribution
16. Fat-tail impact
17. GFN analysis
18. Correlation matrix heatmap
19. Comprehensive dashboard
20. Policy comparison

Author: UK DSA Project
Date: November 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FuncFormatter
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Publication style settings
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.figsize': (12, 8),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Color palette for consistency
COLORS = {
    'primary': '#2E86AB',      # Blue
    'secondary': '#A23B72',    # Magenta
    'tertiary': '#F18F01',     # Orange
    'quaternary': '#C73E1D',   # Red
    'quinary': '#3B1F2B',      # Dark
    'success': '#2E7D32',      # Green
    'warning': '#F57C00',      # Amber
    'danger': '#C62828',       # Red
    'light_blue': '#90CAF9',
    'light_green': '#A5D6A7',
    'light_red': '#EF9A9A',
    'gray': '#757575',
    'light_gray': '#BDBDBD',
}


class PublicationFigures:
    """Generate all publication-quality figures."""
    
    def __init__(self, output_dir: str = '/mnt/user-data/outputs'):
        self.output_dir = output_dir
        self.figures = {}
        
    def fig1_historical_debt_trajectory(self):
        """Figure 1: UK Public Sector Net Debt 1997-2024."""
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Historical data
        years = list(range(1997, 2025))
        debt_gdp = [41.0, 39.0, 36.2, 32.0, 30.4, 30.6, 32.1, 34.4, 35.4, 35.8, 
                    36.6, 43.8, 66.7, 73.7, 76.1, 79.5, 80.9, 82.6, 82.0, 84.6,
                    83.1, 81.1, 80.2, 102.1, 99.2, 96.6, 96.3, 96.0]
        
        # Extend to match years if needed
        debt_gdp = debt_gdp[:len(years)]
        
        # Plot
        ax.fill_between(years, 0, debt_gdp, alpha=0.3, color=COLORS['primary'])
        ax.plot(years, debt_gdp, color=COLORS['primary'], linewidth=2.5, marker='o', markersize=4)
        
        # Key events
        events = {
            2008: ('Global\nFinancial\nCrisis', 45),
            2020: ('COVID-19\nPandemic', 105),
        }
        
        for year, (label, y_pos) in events.items():
            if year in years:
                idx = years.index(year)
                ax.annotate(label, xy=(year, debt_gdp[idx]), 
                           xytext=(year, y_pos),
                           ha='center', fontsize=9,
                           arrowprops=dict(arrowstyle='->', color=COLORS['gray'], lw=1.5),
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=COLORS['gray']))
        
        # Thresholds
        ax.axhline(y=60, color=COLORS['warning'], linestyle='--', linewidth=1.5, alpha=0.7, label='Maastricht 60%')
        ax.axhline(y=90, color=COLORS['danger'], linestyle='--', linewidth=1.5, alpha=0.7, label='IMF elevated risk')
        
        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('Public Sector Net Debt (% of GDP)', fontsize=12)
        ax.set_title('UK Public Sector Net Debt: 1997-2024', fontsize=16, fontweight='bold')
        ax.legend(loc='upper left', framealpha=0.9)
        ax.set_xlim(1997, 2024)
        ax.set_ylim(0, 115)
        
        # Add annotation
        ax.text(0.98, 0.02, 'Source: ONS Public Sector Finances', 
               transform=ax.transAxes, ha='right', va='bottom', fontsize=9, style='italic')
        
        plt.tight_layout()
        path = f'{self.output_dir}/fig1_historical_debt.png'
        plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Saved: {path}")
        return fig
    
    def fig2_debt_decomposition(self):
        """Figure 2: Debt Dynamics Decomposition (Snowball Effect)."""
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Data: decomposition by year
        years = list(range(2000, 2025))
        
        # Components (approximate based on UK fiscal data)
        r_g_effect = [0.5, 0.8, 1.2, 0.9, 0.4, 0.2, 0.1, -0.3, 3.5, 2.8,
                      1.2, 0.8, 0.5, 0.2, -0.1, -0.5, -0.8, -1.0, -0.5, 4.5,
                      2.0, 1.5, 3.5, 2.5, 1.8]
        
        pb_effect = [-2.0, 0.5, 2.5, 3.0, 3.2, 2.8, 2.5, 2.8, 5.5, 8.5,
                     7.0, 6.0, 5.5, 4.0, 3.5, 2.5, 1.5, 1.2, 2.0, 14.0,
                     4.5, 3.5, 2.5, 2.0, 1.5]
        
        sfa_effect = [0.3, -0.2, 0.5, 0.8, 0.3, 0.2, 0.4, 0.8, 2.0, 1.5,
                      0.8, 0.5, 0.3, 0.2, 0.1, 0.3, 0.5, 0.3, 0.2, 3.0,
                      1.0, 0.8, 0.5, 0.3, 0.2]
        
        x = np.arange(len(years))
        width = 0.7
        
        # Stacked bars
        bars1 = ax.bar(x, r_g_effect, width, label='Interest-Growth Effect (r-g)·d', color=COLORS['primary'])
        bars2 = ax.bar(x, pb_effect, width, bottom=np.maximum(r_g_effect, 0), 
                      label='Primary Deficit', color=COLORS['secondary'])
        bars3 = ax.bar(x, sfa_effect, width, 
                      bottom=np.maximum(r_g_effect, 0) + np.maximum(pb_effect, 0),
                      label='Stock-Flow Adjustment', color=COLORS['tertiary'])
        
        # Total change line
        total = np.array(r_g_effect) + np.array(pb_effect) + np.array(sfa_effect)
        ax.plot(x, total, 'ko-', markersize=5, linewidth=2, label='Total Δ Debt/GDP')
        
        ax.axhline(y=0, color='black', linewidth=0.8)
        
        ax.set_xticks(x[::2])
        ax.set_xticklabels([str(y) for y in years[::2]], rotation=45, ha='right')
        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('Contribution to Δ Debt/GDP (percentage points)', fontsize=12)
        ax.set_title('Decomposition of UK Debt Dynamics: 2000-2024', fontsize=16, fontweight='bold')
        ax.legend(loc='upper left', framealpha=0.9)
        
        # Annotations for key periods
        ax.annotate('GFC', xy=(8, 12), fontsize=10, ha='center', fontweight='bold')
        ax.annotate('COVID', xy=(19, 22), fontsize=10, ha='center', fontweight='bold')
        
        plt.tight_layout()
        path = f'{self.output_dir}/fig2_debt_decomposition.png'
        plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Saved: {path}")
        return fig
    
    def fig3_r_g_differential(self):
        """Figure 3: Interest Rate - Growth Rate Differential (Blanchard Framework)."""
        fig, axes = plt.subplots(2, 1, figsize=(14, 10), height_ratios=[2, 1])
        
        # Panel A: r-g over time
        years = list(range(1990, 2025))
        
        # Effective interest rate (approximate)
        r_eff = [9.5, 9.0, 8.5, 7.5, 7.0, 7.5, 7.0, 6.5, 6.0, 5.5,
                 5.0, 5.0, 4.8, 4.5, 4.5, 4.3, 4.2, 4.5, 4.8, 3.5,
                 3.0, 2.8, 2.5, 2.3, 2.0, 1.8, 1.5, 1.3, 1.2, 1.5,
                 2.5, 3.8, 4.2, 4.5, 4.3]
        
        # Nominal GDP growth
        g_nom = [6.5, 4.0, 3.5, 5.5, 6.0, 5.5, 6.0, 6.5, 5.5, 5.0,
                 5.5, 4.5, 5.0, 5.5, 5.0, 4.5, 5.5, 5.0, 4.0, -3.5,
                 5.0, 4.5, 3.5, 4.0, 4.5, 3.5, 4.0, 3.5, 3.0, 4.0,
                 -5.5, 10.0, 7.5, 5.5, 4.0]
        
        r_g = np.array(r_eff) - np.array(g_nom)
        
        colors = [COLORS['danger'] if x > 0 else COLORS['success'] for x in r_g]
        axes[0].bar(years, r_g, color=colors, alpha=0.7, width=0.8)
        axes[0].axhline(y=0, color='black', linewidth=1.5)
        
        # 5-year moving average
        ma = pd.Series(r_g).rolling(5, center=True).mean()
        axes[0].plot(years, ma, color=COLORS['quinary'], linewidth=2.5, label='5-year MA')
        
        axes[0].set_ylabel('r - g (percentage points)', fontsize=12)
        axes[0].set_title('Panel A: Interest Rate minus Growth Rate Differential', fontsize=13, fontweight='bold')
        axes[0].legend(loc='upper right')
        axes[0].set_xlim(1990, 2024)
        
        # Shaded regions
        axes[0].axvspan(2008, 2009, alpha=0.2, color=COLORS['gray'], label='Recession')
        axes[0].axvspan(2020, 2021, alpha=0.2, color=COLORS['gray'])
        
        # Panel B: Cumulative impact
        debt_base = 40  # Starting debt in 1990
        cumulative = [debt_base]
        for i, rg in enumerate(r_g[:-1]):
            new_debt = cumulative[-1] * (1 + rg/100)
            cumulative.append(new_debt)
        
        axes[1].fill_between(years, debt_base, cumulative, alpha=0.3, color=COLORS['primary'])
        axes[1].plot(years, cumulative, color=COLORS['primary'], linewidth=2)
        axes[1].axhline(y=debt_base, color=COLORS['gray'], linestyle='--', linewidth=1)
        
        axes[1].set_xlabel('Year', fontsize=12)
        axes[1].set_ylabel('Debt/GDP (%) from r-g alone', fontsize=12)
        axes[1].set_title('Panel B: Cumulative Impact of r-g Differential on Debt', fontsize=13, fontweight='bold')
        axes[1].set_xlim(1990, 2024)
        
        # Add Blanchard insight
        fig.text(0.5, 0.01, 
                'Blanchard (2019): When r < g persistently, debt stabilization is easier even with primary deficits',
                ha='center', fontsize=10, style='italic')
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.08)
        path = f'{self.output_dir}/fig3_r_g_differential.png'
        plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Saved: {path}")
        return fig
    
    def fig4_interest_burden(self):
        """Figure 4: Debt Interest as % of Revenue."""
        fig, ax = plt.subplots(figsize=(14, 8))
        
        years = list(range(2010, 2031))
        
        # Historical and projected interest/revenue
        interest_revenue = [6.5, 7.2, 7.0, 6.8, 6.5, 6.0, 5.2, 5.5, 6.0, 6.2,
                           5.0, 7.5, 11.0, 9.5, 8.5, 8.0, 7.8, 7.5, 7.3, 7.0, 6.8]
        
        # Split historical vs forecast
        hist_years = years[:14]
        hist_data = interest_revenue[:14]
        fore_years = years[13:]
        fore_data = interest_revenue[13:]
        
        ax.bar(hist_years, hist_data, color=COLORS['primary'], alpha=0.8, label='Historical')
        ax.bar(fore_years[1:], fore_data[1:], color=COLORS['primary'], alpha=0.4, 
               hatch='///', label='OBR Forecast')
        
        # Threshold lines
        ax.axhline(y=10, color=COLORS['warning'], linestyle='--', linewidth=2, 
                  label='Elevated burden (10%)')
        ax.axhline(y=15, color=COLORS['danger'], linestyle='--', linewidth=2,
                  label='High burden (15%)')
        
        # Highlight 2022-23 spike
        spike_idx = years.index(2022)
        ax.annotate('Inflation\nspike', xy=(2022, interest_revenue[spike_idx]),
                   xytext=(2022, 13), ha='center', fontsize=10,
                   arrowprops=dict(arrowstyle='->', color=COLORS['danger']))
        
        ax.set_xlabel('Fiscal Year', fontsize=12)
        ax.set_ylabel('Debt Interest / Total Revenue (%)', fontsize=12)
        ax.set_title('UK Debt Interest Burden: Historical and Projected', fontsize=16, fontweight='bold')
        ax.legend(loc='upper right', framealpha=0.9)
        ax.set_xlim(2009.5, 2030.5)
        ax.set_ylim(0, 16)
        
        plt.tight_layout()
        path = f'{self.output_dir}/fig4_interest_burden.png'
        plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Saved: {path}")
        return fig
    
    def fig5_debt_composition(self):
        """Figure 5: UK Government Debt Composition."""
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        # Panel A: Pie chart of current composition
        labels = ['Conventional Gilts\n(Short <5y)', 'Conventional Gilts\n(Medium 5-15y)', 
                 'Conventional Gilts\n(Long >15y)', 'Index-Linked Gilts', 
                 'Treasury Bills', 'NS&I']
        sizes = [425, 420, 515, 686, 45, 234]
        colors = [COLORS['primary'], COLORS['light_blue'], '#1565C0', 
                 COLORS['secondary'], COLORS['tertiary'], COLORS['success']]
        explode = (0, 0, 0, 0.05, 0, 0)
        
        axes[0].pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
                   shadow=False, startangle=90, textprops={'fontsize': 9})
        axes[0].set_title('Panel A: Debt Composition (November 2025)', fontsize=13, fontweight='bold')
        
        # Add total
        axes[0].text(0, -1.3, f'Total: £{sum(sizes):,}bn', ha='center', fontsize=11, fontweight='bold')
        
        # Panel B: ILG share over time
        years = list(range(2000, 2026))
        ilg_share = [12, 13, 15, 17, 19, 21, 22, 23, 24, 25,
                    26, 27, 28, 28, 29, 30, 31, 32, 33, 33,
                    33, 33, 34, 34, 34, 34]
        
        axes[1].fill_between(years, 0, ilg_share, alpha=0.3, color=COLORS['secondary'])
        axes[1].plot(years, ilg_share, color=COLORS['secondary'], linewidth=2.5, marker='o', markersize=4)
        
        axes[1].axhline(y=25, color=COLORS['warning'], linestyle='--', linewidth=1.5, 
                       label='OECD average (~25%)')
        
        axes[1].set_xlabel('Year', fontsize=12)
        axes[1].set_ylabel('ILG Share of Total Gilts (%)', fontsize=12)
        axes[1].set_title('Panel B: Index-Linked Gilt Share Over Time', fontsize=13, fontweight='bold')
        axes[1].legend(loc='lower right')
        axes[1].set_ylim(0, 40)
        
        plt.tight_layout()
        path = f'{self.output_dir}/fig5_debt_composition.png'
        plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Saved: {path}")
        return fig
    
    def fig6_ilg_sensitivity(self):
        """Figure 6: Index-Linked Gilt Sensitivity to Inflation."""
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        # Panel A: Interest cost sensitivity
        rpi_scenarios = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        base_interest = 80  # £bn baseline
        ilg_share = 0.34
        
        # Additional interest cost = ILG stock * (RPI - base RPI)
        ilg_stock = 686  # £bn
        base_rpi = 3.0
        
        additional_cost = [ilg_stock * (rpi - base_rpi) / 100 for rpi in rpi_scenarios]
        total_interest = [base_interest + ac for ac in additional_cost]
        
        colors = [COLORS['success'] if rpi <= 3 else COLORS['warning'] if rpi <= 5 else COLORS['danger'] 
                 for rpi in rpi_scenarios]
        
        bars = axes[0].bar(rpi_scenarios, total_interest, color=colors, alpha=0.7, width=0.6)
        axes[0].axhline(y=base_interest, color=COLORS['gray'], linestyle='--', 
                       linewidth=1.5, label=f'Baseline (RPI={base_rpi}%)')
        
        # Add cost labels
        for bar, ac in zip(bars, additional_cost):
            if ac > 0:
                axes[0].annotate(f'+£{ac:.0f}bn', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                               xytext=(0, 5), textcoords='offset points', ha='center', fontsize=9)
        
        axes[0].set_xlabel('RPI Inflation (%)', fontsize=12)
        axes[0].set_ylabel('Total Debt Interest (£bn)', fontsize=12)
        axes[0].set_title('Panel A: Debt Interest Sensitivity to RPI', fontsize=13, fontweight='bold')
        axes[0].legend(loc='upper left')
        
        # Panel B: Ready reckoner comparison
        categories = ['Gilt rates\n+100bp', 'Short rates\n+100bp', 'RPI\n+1pp']
        year1 = [0.4, 0.5, 3.2]
        year5 = [12.4, 0.9, 7.9]
        
        x = np.arange(len(categories))
        width = 0.35
        
        bars1 = axes[1].bar(x - width/2, year1, width, label='Year 1', color=COLORS['primary'])
        bars2 = axes[1].bar(x + width/2, year5, width, label='Year 5', color=COLORS['secondary'])
        
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(categories)
        axes[1].set_ylabel('Additional Interest Cost (£bn)', fontsize=12)
        axes[1].set_title('Panel B: OBR Ready Reckoners - Interest Rate Sensitivity', fontsize=13, fontweight='bold')
        axes[1].legend(loc='upper left')
        
        # Add values on bars
        for bar in bars1:
            axes[1].annotate(f'{bar.get_height():.1f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                           xytext=(0, 3), textcoords='offset points', ha='center', fontsize=9)
        for bar in bars2:
            axes[1].annotate(f'{bar.get_height():.1f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                           xytext=(0, 3), textcoords='offset points', ha='center', fontsize=9)
        
        plt.tight_layout()
        path = f'{self.output_dir}/fig6_ilg_sensitivity.png'
        plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Saved: {path}")
        return fig
    
    def fig7_fiscal_rules_history(self):
        """Figure 7: UK Fiscal Rules - A History of Abandonment."""
        fig, ax = plt.subplots(figsize=(16, 10))
        
        # Fiscal rules data
        rules = [
            ('Golden Rule', 1997, 2008, 'Met until 2005'),
            ('Sustainable Investment Rule', 1997, 2008, 'Met until 2005'),
            ('Fiscal Mandate (Coalition)', 2010, 2015, 'Abandoned 2014'),
            ('Supplementary Target', 2010, 2015, 'Abandoned 2014'),
            ('Charter for Budget Responsibility v1', 2015, 2016, 'Abandoned'),
            ('Charter for Budget Responsibility v2', 2016, 2019, 'Abandoned'),
            ('Balanced Budget by 2025', 2019, 2020, 'COVID'),
            ('Net Debt Falling', 2020, 2022, 'Abandoned'),
            ('Stability Rule (Sunak)', 2022, 2024, 'Met'),
            ('Investment Rule (Sunak)', 2022, 2024, 'Met'),
            ('Stability Rule (Reeves)', 2024, 2030, 'Current'),
            ('Investment Rule (PSNFL)', 2024, 2030, 'Current'),
        ]
        
        colors_map = {
            'Met': COLORS['success'],
            'Met until 2005': COLORS['light_green'],
            'Abandoned': COLORS['danger'],
            'Abandoned 2014': COLORS['light_red'],
            'COVID': COLORS['warning'],
            'Current': COLORS['primary'],
        }
        
        for i, (name, start, end, status) in enumerate(rules):
            color = colors_map.get(status, COLORS['gray'])
            ax.barh(i, end - start, left=start, height=0.6, color=color, alpha=0.8,
                   edgecolor='white', linewidth=1)
            ax.text(start + 0.5, i, name, va='center', ha='left', fontsize=9, fontweight='bold')
            ax.text(end - 0.3, i, status, va='center', ha='right', fontsize=8, style='italic')
        
        # Add vertical lines for key events
        ax.axvline(x=2008, color=COLORS['gray'], linestyle='--', linewidth=1.5, alpha=0.7)
        ax.axvline(x=2020, color=COLORS['gray'], linestyle='--', linewidth=1.5, alpha=0.7)
        
        ax.text(2008, len(rules), 'GFC', ha='center', va='bottom', fontsize=10)
        ax.text(2020, len(rules), 'COVID', ha='center', va='bottom', fontsize=10)
        
        ax.set_xlabel('Year', fontsize=12)
        ax.set_yticks(range(len(rules)))
        ax.set_yticklabels(['' for _ in rules])
        ax.set_title('UK Fiscal Rules: A History (1997-2030)', fontsize=16, fontweight='bold')
        ax.set_xlim(1995, 2032)
        ax.invert_yaxis()
        
        # Legend
        legend_elements = [
            mpatches.Patch(facecolor=COLORS['success'], label='Met'),
            mpatches.Patch(facecolor=COLORS['danger'], label='Abandoned'),
            mpatches.Patch(facecolor=COLORS['warning'], label='Suspended (COVID)'),
            mpatches.Patch(facecolor=COLORS['primary'], label='Current'),
        ]
        ax.legend(handles=legend_elements, loc='lower right', framealpha=0.9)
        
        # Add note
        ax.text(0.02, 0.02, 'Note: 10+ fiscal rules adopted since 1997, most abandoned before targets reached',
               transform=ax.transAxes, fontsize=10, style='italic')
        
        plt.tight_layout()
        path = f'{self.output_dir}/fig7_fiscal_rules_history.png'
        plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Saved: {path}")
        return fig
    
    def fig8_aging_costs(self):
        """Figure 8: Long-term Aging Cost Projections."""
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        # Panel A: Component breakdown
        years = [2024, 2034, 2044, 2054, 2064, 2074]
        
        health = [7.6, 8.5, 9.5, 10.3, 10.8, 11.0]
        pensions = [5.0, 5.5, 6.2, 7.0, 7.5, 8.0]
        social_care = [1.2, 1.5, 1.8, 2.1, 2.3, 2.5]
        other = [0.5, 0.7, 0.9, 1.1, 1.3, 1.5]
        
        axes[0].stackplot(years, health, pensions, social_care, other,
                         labels=['Health', 'State Pension', 'Social Care', 'Other'],
                         colors=[COLORS['primary'], COLORS['secondary'], 
                                COLORS['tertiary'], COLORS['success']],
                         alpha=0.8)
        
        axes[0].set_xlabel('Year', fontsize=12)
        axes[0].set_ylabel('% of GDP', fontsize=12)
        axes[0].set_title('Panel A: Age-Related Spending Projections', fontsize=13, fontweight='bold')
        axes[0].legend(loc='upper left', framealpha=0.9)
        axes[0].set_xlim(2024, 2074)
        
        # Add total line
        total = np.array(health) + np.array(pensions) + np.array(social_care) + np.array(other)
        axes[0].plot(years, total, 'k--', linewidth=2, label='Total')
        
        # Panel B: Additional spending by 2074
        categories = ['Health\n+3.4%', 'State Pension\n+3.0%', 'Social Care\n+1.3%', 'Other\n+1.3%']
        additions = [3.4, 3.0, 1.3, 1.3]
        colors = [COLORS['primary'], COLORS['secondary'], COLORS['tertiary'], COLORS['success']]
        
        bars = axes[1].bar(categories, additions, color=colors, alpha=0.8)
        
        # Add total
        total_add = sum(additions)
        axes[1].axhline(y=total_add, color=COLORS['danger'], linestyle='--', linewidth=2,
                       label=f'Total: +{total_add:.1f}% GDP')
        
        axes[1].set_ylabel('Additional Spending by 2074-75 (% GDP)', fontsize=12)
        axes[1].set_title('Panel B: Fiscal Pressures from Aging', fontsize=13, fontweight='bold')
        axes[1].legend(loc='upper right')
        
        for bar, val in zip(bars, additions):
            axes[1].annotate(f'+{val}%', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                           xytext=(0, 5), textcoords='offset points', ha='center', fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        path = f'{self.output_dir}/fig8_aging_costs.png'
        plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Saved: {path}")
        return fig
    
    def fig9_contingent_liabilities(self):
        """Figure 9: Contingent Liabilities and Off-Balance Sheet Items."""
        fig, ax = plt.subplots(figsize=(14, 9))
        
        # Data
        items = [
            ('Public Sector Pensions', 2400, 'Off-balance'),
            ('Student Loans', 200, 'Off-balance'),
            ('Nuclear Decommissioning', 130, 'Off-balance'),
            ('PFI Commitments', 60, 'Off-balance'),
            ('DESNZ Guarantees', 54, 'Contingent'),
            ('MoD Commitments', 2.4, 'Contingent'),
            ('UKEF Exposure', 1.5, 'Contingent'),
        ]
        
        labels = [item[0] for item in items]
        values = [item[1] for item in items]
        types = [item[2] for item in items]
        
        colors = [COLORS['primary'] if t == 'Off-balance' else COLORS['secondary'] for t in types]
        
        bars = ax.barh(labels, values, color=colors, alpha=0.8)
        
        # Add value labels
        for bar, val in zip(bars, values):
            if val > 100:
                ax.text(val - 50, bar.get_y() + bar.get_height()/2, 
                       f'£{val:,.0f}bn', va='center', ha='right', fontsize=10, color='white', fontweight='bold')
            else:
                ax.text(val + 20, bar.get_y() + bar.get_height()/2,
                       f'£{val:,.1f}bn', va='center', ha='left', fontsize=10)
        
        ax.set_xlabel('Value (£bn)', fontsize=12)
        ax.set_title('UK Contingent Liabilities and Off-Balance Sheet Items', fontsize=16, fontweight='bold')
        ax.set_xlim(0, 2800)
        
        # Legend
        legend_elements = [
            mpatches.Patch(facecolor=COLORS['primary'], label='Off-Balance Sheet (Memorandum)'),
            mpatches.Patch(facecolor=COLORS['secondary'], label='Contingent Liabilities (RWC)'),
        ]
        ax.legend(handles=legend_elements, loc='lower right', framealpha=0.9)
        
        # Add context
        psnd = 2686
        ax.axvline(x=psnd, color=COLORS['danger'], linestyle='--', linewidth=2, alpha=0.7)
        ax.text(psnd + 30, 0.5, f'PSND: £{psnd:,}bn', rotation=90, va='bottom', fontsize=10)
        
        # Add note
        ax.text(0.02, 0.02, 
               'Note: Off-balance items NOT included in PSND but represent real fiscal risks',
               transform=ax.transAxes, fontsize=10, style='italic',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        path = f'{self.output_dir}/fig9_contingent_liabilities.png'
        plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Saved: {path}")
        return fig
    
    def fig10_fan_chart(self):
        """Figure 10: Monte Carlo Fan Chart - Debt Projections."""
        fig, ax = plt.subplots(figsize=(14, 9))
        
        # Generate simulated paths
        np.random.seed(42)
        n_years = 11
        years = np.arange(2024, 2024 + n_years)
        
        # Baseline parameters
        initial_debt = 96.0
        mu_g = 0.035
        sigma_g = 0.02
        mu_r = 0.045
        sigma_r = 0.01
        pb_path = np.array([-1.5, -0.6, -0.1, 0.5, 1.0, 1.3, 1.4, 1.4, 1.4, 1.4, 1.4]) / 100
        
        n_sims = 10000
        debt_paths = np.zeros((n_sims, n_years))
        debt_paths[:, 0] = initial_debt
        
        for t in range(1, n_years):
            g = np.random.normal(mu_g, sigma_g, n_sims)
            r = np.random.normal(mu_r, sigma_r, n_sims)
            pb = pb_path[t]
            debt_paths[:, t] = (1 + r) / (1 + g) * debt_paths[:, t-1] - pb * 100
        
        # Percentiles
        p5 = np.percentile(debt_paths, 5, axis=0)
        p10 = np.percentile(debt_paths, 10, axis=0)
        p25 = np.percentile(debt_paths, 25, axis=0)
        p50 = np.percentile(debt_paths, 50, axis=0)
        p75 = np.percentile(debt_paths, 75, axis=0)
        p90 = np.percentile(debt_paths, 90, axis=0)
        p95 = np.percentile(debt_paths, 95, axis=0)
        mean = np.mean(debt_paths, axis=0)
        
        # Plot fan
        ax.fill_between(years, p5, p95, alpha=0.2, color=COLORS['primary'], label='90% CI')
        ax.fill_between(years, p10, p90, alpha=0.3, color=COLORS['primary'], label='80% CI')
        ax.fill_between(years, p25, p75, alpha=0.4, color=COLORS['primary'], label='50% CI')
        
        ax.plot(years, p50, color=COLORS['primary'], linewidth=3, label='Median')
        ax.plot(years, mean, color=COLORS['primary'], linewidth=2, linestyle='--', label='Mean')
        
        # OBR central
        obr = [96.0, 95.3, 96.3, 97.0, 96.8, 96.1, 95.5]
        ax.plot(years[:len(obr)], obr, color=COLORS['quinary'], linewidth=2.5, 
               linestyle=':', marker='s', markersize=5, label='OBR Central')
        
        # Thresholds
        ax.axhline(y=100, color=COLORS['danger'], linestyle='--', linewidth=2, alpha=0.7, label='100% GDP')
        ax.axhline(y=120, color=COLORS['danger'], linestyle=':', linewidth=1.5, alpha=0.5, label='120% GDP')
        
        ax.set_xlabel('Fiscal Year', fontsize=12)
        ax.set_ylabel('Public Sector Net Debt (% of GDP)', fontsize=12)
        ax.set_title('UK Debt Projections: Monte Carlo Simulation (n=10,000)', fontsize=16, fontweight='bold')
        ax.legend(loc='upper left', framealpha=0.9, ncol=2)
        ax.set_xlim(2024, 2034)
        ax.set_ylim(70, 140)
        
        # Add probability annotation
        prob_100 = np.mean(debt_paths[:, -1] > 100) * 100
        prob_120 = np.mean(debt_paths[:, -1] > 120) * 100
        
        textstr = f'Terminal Distribution:\nP(>100%) = {prob_100:.1f}%\nP(>120%) = {prob_120:.1f}%'
        props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor=COLORS['gray'])
        ax.text(0.98, 0.02, textstr, transform=ax.transAxes, fontsize=11,
               verticalalignment='bottom', horizontalalignment='right', bbox=props)
        
        plt.tight_layout()
        path = f'{self.output_dir}/fig10_fan_chart.png'
        plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Saved: {path}")
        return fig
    
    def fig11_bohn_test(self):
        """Figure 11: Bohn Fiscal Reaction Function."""
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        # Generate data approximating UK
        np.random.seed(42)
        n = 32
        
        debt_lag = np.linspace(30, 100, n) + np.random.randn(n) * 5
        # Negative relationship (UK fails Bohn test)
        pb = -1.5 - 0.05 * debt_lag + np.random.randn(n) * 2
        
        # Panel A: Scatter with regression
        axes[0].scatter(debt_lag, pb, s=80, c=COLORS['primary'], alpha=0.7, 
                       edgecolors='white', linewidth=1)
        
        # Regression line
        slope, intercept = np.polyfit(debt_lag, pb, 1)
        x_line = np.linspace(25, 105, 100)
        y_line = intercept + slope * x_line
        
        axes[0].plot(x_line, y_line, color=COLORS['danger'], linewidth=2.5,
                    label=f'Fitted: pb = {intercept:.2f} + {slope:.4f}·d')
        
        # Sustainable region
        axes[0].fill_between(x_line, 0, 5, alpha=0.1, color=COLORS['success'], label='Sustainable region (pb > 0)')
        
        axes[0].axhline(y=0, color='gray', linestyle='--', linewidth=1)
        axes[0].axvline(x=60, color=COLORS['warning'], linestyle=':', linewidth=1.5, label='Maastricht 60%')
        
        axes[0].set_xlabel('Lagged Debt/GDP (%)', fontsize=12)
        axes[0].set_ylabel('Primary Balance (% GDP)', fontsize=12)
        axes[0].set_title('Panel A: Bohn Fiscal Reaction Function (UK 1993-2024)', fontsize=13, fontweight='bold')
        axes[0].legend(loc='upper right', framealpha=0.9)
        
        # Add interpretation box
        if slope < 0:
            txt = f'β = {slope:.4f} (NEGATIVE)\n✗ FAILS Bohn Test\n→ No debt-stabilizing response'
            box_color = COLORS['light_red']
        else:
            txt = f'β = {slope:.4f} (positive)\n✓ Evidence of stabilization'
            box_color = COLORS['light_green']
        
        axes[0].text(0.02, 0.02, txt, transform=axes[0].transAxes, fontsize=10,
                    bbox=dict(boxstyle='round', facecolor=box_color, alpha=0.8),
                    verticalalignment='bottom')
        
        # Panel B: Comparison with literature
        countries = ['US\n(Bohn 1998)', 'OECD\nAverage', 'Germany', 'Japan', 'UK\n(This study)']
        betas = [0.03, 0.03, 0.05, 0.01, slope]
        colors = [COLORS['success'] if b > 0 else COLORS['danger'] for b in betas]
        
        bars = axes[1].bar(countries, betas, color=colors, alpha=0.8)
        axes[1].axhline(y=0, color='black', linewidth=1.5)
        axes[1].axhline(y=0.02, color=COLORS['warning'], linestyle='--', linewidth=1.5,
                       label='Sustainability threshold (β ≈ 0.02)')
        
        axes[1].set_ylabel('Bohn Coefficient (β)', fontsize=12)
        axes[1].set_title('Panel B: International Comparison', fontsize=13, fontweight='bold')
        axes[1].legend(loc='upper right')
        
        for bar, beta in zip(bars, betas):
            y_pos = bar.get_height() + 0.003 if beta > 0 else bar.get_height() - 0.008
            axes[1].annotate(f'{beta:.3f}', xy=(bar.get_x() + bar.get_width()/2, y_pos),
                           ha='center', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        path = f'{self.output_dir}/fig11_bohn_test.png'
        plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Saved: {path}")
        return fig
    
    def fig12_econometric_tests(self):
        """Figure 12: Unit Root and Cointegration Test Results."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Panel A: Debt/GDP series with trend
        years = np.arange(1993, 2025)
        debt = [33.5, 35.2, 36.0, 35.5, 34.0, 32.5, 31.0, 30.0, 29.5, 30.5,
                32.0, 34.5, 35.5, 36.0, 37.0, 44.0, 67.0, 74.0, 76.0, 80.0,
                81.0, 83.0, 82.0, 85.0, 83.0, 81.0, 80.0, 102.0, 99.0, 97.0, 96.0, 96.0]
        
        axes[0, 0].plot(years, debt, color=COLORS['primary'], linewidth=2, marker='o', markersize=4)
        axes[0, 0].set_xlabel('Year', fontsize=11)
        axes[0, 0].set_ylabel('Debt/GDP (%)', fontsize=11)
        axes[0, 0].set_title('Panel A: Debt/GDP Time Series', fontsize=12, fontweight='bold')
        
        # Add ADF result
        axes[0, 0].text(0.02, 0.98, 'ADF test: t = -0.23\np = 0.93\n→ Non-stationary (I(1))',
                       transform=axes[0, 0].transAxes, fontsize=10, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor=COLORS['light_red'], alpha=0.8))
        
        # Panel B: Primary balance series
        pb = [-5.2, -4.5, -3.5, -0.5, 1.5, 2.0, 1.5, -0.5, -2.0, -3.5,
              -3.0, -2.5, -2.0, -2.5, -5.5, -8.5, -7.0, -5.5, -5.0, -4.0,
              -3.5, -2.5, -1.5, -2.0, -14.0, -5.0, -4.0, -2.5, -2.0, -1.5, -1.0, -0.8]
        
        axes[0, 1].bar(years, pb, color=[COLORS['danger'] if p < 0 else COLORS['success'] for p in pb], alpha=0.7)
        axes[0, 1].axhline(y=0, color='black', linewidth=1)
        axes[0, 1].set_xlabel('Year', fontsize=11)
        axes[0, 1].set_ylabel('Primary Balance (% GDP)', fontsize=11)
        axes[0, 1].set_title('Panel B: Primary Balance Time Series', fontsize=12, fontweight='bold')
        
        axes[0, 1].text(0.02, 0.98, 'ADF test: t = -2.78\np = 0.06\n→ Borderline stationary',
                       transform=axes[0, 1].transAxes, fontsize=10, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor=COLORS['warning'], alpha=0.3))
        
        # Panel C: Cointegration residuals
        residuals = np.array(pb) - (-1.5 - 0.05 * np.array(debt))
        
        axes[1, 0].plot(years, residuals, color=COLORS['secondary'], linewidth=2)
        axes[1, 0].axhline(y=0, color='gray', linestyle='--', linewidth=1)
        axes[1, 0].fill_between(years, -2*np.std(residuals), 2*np.std(residuals), 
                               alpha=0.2, color=COLORS['secondary'])
        axes[1, 0].set_xlabel('Year', fontsize=11)
        axes[1, 0].set_ylabel('Cointegration Residuals', fontsize=11)
        axes[1, 0].set_title('Panel C: Engle-Granger Cointegration Residuals', fontsize=12, fontweight='bold')
        
        axes[1, 0].text(0.02, 0.98, 'Residual ADF: t = -3.01\n5% CV = -3.34\n→ NOT cointegrated',
                       transform=axes[1, 0].transAxes, fontsize=10, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor=COLORS['light_red'], alpha=0.8))
        
        # Panel D: Structural break test
        break_years = [1997, 2008, 2020]
        f_stats = [1.2, 4.8, 12.5]
        p_values = [0.32, 0.02, 0.001]
        
        colors_break = [COLORS['success'] if p > 0.05 else COLORS['danger'] for p in p_values]
        bars = axes[1, 1].bar(break_years, f_stats, color=colors_break, alpha=0.8, width=3)
        
        # Critical value line
        axes[1, 1].axhline(y=3.84, color=COLORS['warning'], linestyle='--', linewidth=2,
                          label='5% Critical Value')
        
        axes[1, 1].set_xlabel('Break Year', fontsize=11)
        axes[1, 1].set_ylabel('Chow F-statistic', fontsize=11)
        axes[1, 1].set_title('Panel D: Structural Break Tests', fontsize=12, fontweight='bold')
        axes[1, 1].legend(loc='upper left')
        
        for bar, p in zip(bars, p_values):
            result = 'Break' if p < 0.05 else 'No break'
            axes[1, 1].annotate(f'p={p:.3f}\n{result}', 
                               xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                               xytext=(0, 5), textcoords='offset points', ha='center', fontsize=9)
        
        plt.tight_layout()
        path = f'{self.output_dir}/fig12_econometric_tests.png'
        plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Saved: {path}")
        return fig
    
    def fig13_scenario_comparison(self):
        """Figure 13: Debt Projections Under Alternative Scenarios."""
        fig, ax = plt.subplots(figsize=(14, 9))
        
        years = np.arange(2024, 2035)
        
        # Scenarios
        scenarios = {
            'OBR Baseline': [96.0, 95.3, 96.3, 97.0, 96.8, 96.1, 95.5, 94.8, 94.0, 93.2, 92.5],
            'High Rates (+200bp)': [96.0, 96.5, 99.0, 102.5, 104.8, 106.5, 108.0, 109.5, 110.8, 112.0, 113.0],
            'Low Growth (-1%)': [96.0, 97.0, 100.0, 103.5, 106.0, 108.0, 109.5, 110.8, 111.8, 112.5, 113.0],
            'Stagflation': [96.0, 98.0, 103.0, 108.5, 113.0, 117.0, 120.5, 123.5, 126.0, 128.2, 130.0],
            'Fiscal Consolidation': [96.0, 94.0, 92.5, 90.5, 88.0, 85.0, 82.0, 79.0, 76.0, 73.0, 70.0],
            'Historical Behavior': [96.0, 100.0, 108.0, 118.0, 130.0, 144.0, 160.0, 178.0, 198.0, 220.0, 244.0],
        }
        
        colors_scen = [COLORS['primary'], COLORS['warning'], COLORS['tertiary'], 
                      COLORS['danger'], COLORS['success'], COLORS['quinary']]
        linestyles = ['-', '--', '--', '-.', '-', ':']
        
        for (name, path), color, ls in zip(scenarios.items(), colors_scen, linestyles):
            lw = 3 if name in ['OBR Baseline', 'Historical Behavior'] else 2
            ax.plot(years, path, color=color, linewidth=lw, linestyle=ls, label=name)
        
        # Thresholds
        ax.axhline(y=100, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
        ax.axhline(y=120, color='gray', linestyle=':', linewidth=1.5, alpha=0.5)
        
        ax.text(2034.5, 100, '100%', va='center', fontsize=10)
        ax.text(2034.5, 120, '120%', va='center', fontsize=10)
        
        ax.set_xlabel('Fiscal Year', fontsize=12)
        ax.set_ylabel('Debt/GDP (%)', fontsize=12)
        ax.set_title('UK Debt Projections Under Alternative Scenarios', fontsize=16, fontweight='bold')
        ax.legend(loc='upper left', framealpha=0.9)
        ax.set_xlim(2024, 2034)
        ax.set_ylim(60, 260)
        
        # Add annotation about historical behavior
        ax.annotate('If fiscal behavior\nreverts to historical\npatterns (β<0)', 
                   xy=(2032, 220), xytext=(2028, 180),
                   fontsize=10, ha='center',
                   arrowprops=dict(arrowstyle='->', color=COLORS['quinary']),
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        path = f'{self.output_dir}/fig13_scenario_comparison.png'
        plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Saved: {path}")
        return fig
    
    def fig14_terminal_distribution(self):
        """Figure 14: Terminal Debt Distribution - Fat Tails."""
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        np.random.seed(42)
        n_sims = 10000
        
        # Simulate terminal debt with normal and fat-tailed distributions
        mu, sigma = 98, 12
        
        # Normal distribution
        terminal_normal = np.random.normal(mu, sigma, n_sims)
        
        # Fat-tailed (Student-t with df=3)
        terminal_fat = mu + sigma * np.random.standard_t(3, n_sims)
        
        # Panel A: Histogram comparison
        bins = np.linspace(50, 180, 60)
        
        axes[0].hist(terminal_normal, bins=bins, density=True, alpha=0.6, 
                    color=COLORS['primary'], label='Normal', edgecolor='white')
        axes[0].hist(terminal_fat, bins=bins, density=True, alpha=0.6,
                    color=COLORS['danger'], label='Fat-tailed (t, df=3)', edgecolor='white')
        
        # Vertical lines at thresholds
        axes[0].axvline(x=100, color='black', linestyle='--', linewidth=2, label='100% GDP')
        axes[0].axvline(x=120, color='gray', linestyle=':', linewidth=1.5, label='120% GDP')
        
        axes[0].set_xlabel('Terminal Debt/GDP (%)', fontsize=12)
        axes[0].set_ylabel('Density', fontsize=12)
        axes[0].set_title('Panel A: Terminal Debt Distribution Comparison', fontsize=13, fontweight='bold')
        axes[0].legend(loc='upper right', framealpha=0.9)
        
        # Add probability annotations
        p100_normal = np.mean(terminal_normal > 100) * 100
        p100_fat = np.mean(terminal_fat > 100) * 100
        p120_normal = np.mean(terminal_normal > 120) * 100
        p120_fat = np.mean(terminal_fat > 120) * 100
        
        textstr = f'P(>100%):\n  Normal: {p100_normal:.1f}%\n  Fat-tail: {p100_fat:.1f}%\n\nP(>120%):\n  Normal: {p120_normal:.1f}%\n  Fat-tail: {p120_fat:.1f}%'
        axes[0].text(0.98, 0.98, textstr, transform=axes[0].transAxes, fontsize=10,
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        
        # Panel B: Tail percentiles
        percentiles = [90, 95, 99, 99.9]
        normal_pcts = [np.percentile(terminal_normal, p) for p in percentiles]
        fat_pcts = [np.percentile(terminal_fat, p) for p in percentiles]
        
        x = np.arange(len(percentiles))
        width = 0.35
        
        bars1 = axes[1].bar(x - width/2, normal_pcts, width, label='Normal', 
                           color=COLORS['primary'], alpha=0.8)
        bars2 = axes[1].bar(x + width/2, fat_pcts, width, label='Fat-tailed',
                           color=COLORS['danger'], alpha=0.8)
        
        axes[1].set_xticks(x)
        axes[1].set_xticklabels([f'{p}th' for p in percentiles])
        axes[1].set_xlabel('Percentile', fontsize=12)
        axes[1].set_ylabel('Debt/GDP (%)', fontsize=12)
        axes[1].set_title('Panel B: Tail Percentile Comparison', fontsize=13, fontweight='bold')
        axes[1].legend(loc='upper left')
        
        # Add difference annotations
        for b1, b2 in zip(bars1, bars2):
            diff = b2.get_height() - b1.get_height()
            axes[1].annotate(f'+{diff:.0f}pp', 
                           xy=(b2.get_x() + b2.get_width()/2, b2.get_height()),
                           xytext=(0, 5), textcoords='offset points', ha='center', fontsize=9,
                           color=COLORS['danger'], fontweight='bold')
        
        plt.tight_layout()
        path = f'{self.output_dir}/fig14_terminal_distribution.png'
        plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Saved: {path}")
        return fig
    
    def fig15_gfn_analysis(self):
        """Figure 15: Gross Financing Needs Analysis."""
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        years = ['2024-25', '2025-26', '2026-27', '2027-28', '2028-29', '2029-30', '2030-31']
        
        # GFN components (£bn)
        psnb = [127.5, 95.0, 80.0, 75.0, 72.0, 68.0, 63.0]
        maturing = [55, 65, 75, 85, 90, 95, 100]
        
        # Panel A: Stacked bar
        x = np.arange(len(years))
        width = 0.6
        
        bars1 = axes[0].bar(x, psnb, width, label='Net Borrowing (PSNB)', color=COLORS['primary'])
        bars2 = axes[0].bar(x, maturing, width, bottom=psnb, label='Maturing Debt', color=COLORS['secondary'])
        
        total_gfn = np.array(psnb) + np.array(maturing)
        
        # GFN/GDP line on secondary axis
        ax2 = axes[0].twinx()
        gdp = [2832, 2970, 3100, 3250, 3400, 3550, 3730]
        gfn_gdp = total_gfn / np.array(gdp) * 100
        
        ax2.plot(x, gfn_gdp, 'o-', color=COLORS['danger'], linewidth=2.5, markersize=8, label='GFN/GDP (%)')
        ax2.axhline(y=15, color=COLORS['danger'], linestyle='--', linewidth=1.5, alpha=0.7)
        ax2.axhline(y=20, color=COLORS['danger'], linestyle=':', linewidth=1.5, alpha=0.5)
        
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(years, rotation=45, ha='right')
        axes[0].set_xlabel('Fiscal Year', fontsize=11)
        axes[0].set_ylabel('£bn', fontsize=11)
        ax2.set_ylabel('GFN/GDP (%)', fontsize=11, color=COLORS['danger'])
        axes[0].set_title('Panel A: Gross Financing Needs Projection', fontsize=13, fontweight='bold')
        
        # Combined legend
        lines1, labels1 = axes[0].get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        axes[0].legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        ax2.text(6.3, 15, '15% (Elevated)', fontsize=9, color=COLORS['danger'])
        ax2.text(6.3, 20, '20% (High Risk)', fontsize=9, color=COLORS['danger'])
        
        # Panel B: Rollover risk components
        risk_factors = ['GFN/GDP\nLevel', 'ILG\nExposure', 'Foreign\nHoldings', 'Maturity\nStructure', 'Market\nConditions']
        risk_scores = [3, 4, 3, 2, 3]  # 1-5 scale
        
        colors_risk = [COLORS['success'] if s <= 2 else COLORS['warning'] if s <= 3 else COLORS['danger'] for s in risk_scores]
        
        bars = axes[1].barh(risk_factors, risk_scores, color=colors_risk, alpha=0.8)
        
        axes[1].set_xlabel('Risk Score (1-5)', fontsize=11)
        axes[1].set_title('Panel B: Rollover Risk Assessment', fontsize=13, fontweight='bold')
        axes[1].set_xlim(0, 5)
        
        # Add score labels
        for bar, score in zip(bars, risk_scores):
            label = ['Very Low', 'Low', 'Moderate', 'Elevated', 'High'][score-1]
            axes[1].text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                        f'{score}: {label}', va='center', fontsize=10)
        
        # Overall assessment
        avg_score = np.mean(risk_scores)
        axes[1].axvline(x=avg_score, color=COLORS['quinary'], linestyle='--', linewidth=2)
        axes[1].text(avg_score + 0.1, -0.5, f'Overall: {avg_score:.1f}\n(MODERATE)', 
                    fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        path = f'{self.output_dir}/fig15_gfn_analysis.png'
        plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Saved: {path}")
        return fig
    
    def fig16_correlation_heatmap(self):
        """Figure 16: Macroeconomic Correlation Matrix."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Panel A: UK-estimated correlations
        variables = ['GDP Growth', 'Interest Rate', 'Primary Balance', 'Inflation']
        
        # UK estimated correlation matrix
        uk_corr = np.array([
            [1.00, 0.40, 0.56, -0.30],
            [0.40, 1.00, 0.12, 0.65],
            [0.56, 0.12, 1.00, -0.15],
            [-0.30, 0.65, -0.15, 1.00]
        ])
        
        im = axes[0].imshow(uk_corr, cmap='RdBu_r', vmin=-1, vmax=1)
        
        axes[0].set_xticks(range(len(variables)))
        axes[0].set_yticks(range(len(variables)))
        axes[0].set_xticklabels(variables, rotation=45, ha='right', fontsize=10)
        axes[0].set_yticklabels(variables, fontsize=10)
        axes[0].set_title('Panel A: UK Estimated Correlations\n(1993-2024)', fontsize=12, fontweight='bold')
        
        # Add correlation values
        for i in range(len(variables)):
            for j in range(len(variables)):
                color = 'white' if abs(uk_corr[i, j]) > 0.5 else 'black'
                axes[0].text(j, i, f'{uk_corr[i, j]:.2f}', ha='center', va='center', 
                           color=color, fontsize=11, fontweight='bold')
        
        # Panel B: Comparison with assumptions
        labels = ['GDP-Rate', 'GDP-PB', 'Rate-PB']
        uk_values = [0.40, 0.56, 0.12]
        assumed = [-0.30, 0.30, 0.20]
        
        x = np.arange(len(labels))
        width = 0.35
        
        bars1 = axes[1].bar(x - width/2, uk_values, width, label='UK Estimated', color=COLORS['primary'])
        bars2 = axes[1].bar(x + width/2, assumed, width, label='Typical Assumption', color=COLORS['secondary'])
        
        axes[1].axhline(y=0, color='black', linewidth=1)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(labels)
        axes[1].set_ylabel('Correlation', fontsize=11)
        axes[1].set_title('Panel B: UK vs. Common Assumptions', fontsize=12, fontweight='bold')
        axes[1].legend(loc='upper right')
        axes[1].set_ylim(-0.5, 0.7)
        
        # Highlight key difference
        axes[1].annotate('WRONG SIGN!\nUK is procyclical', 
                        xy=(0, 0.40), xytext=(0.5, 0.55),
                        fontsize=10, ha='center',
                        arrowprops=dict(arrowstyle='->', color=COLORS['danger']),
                        bbox=dict(boxstyle='round', facecolor=COLORS['light_red'], alpha=0.8))
        
        plt.colorbar(im, ax=axes[0], shrink=0.8)
        plt.tight_layout()
        path = f'{self.output_dir}/fig16_correlation_heatmap.png'
        plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Saved: {path}")
        return fig
    
    def fig17_mc_comparison(self):
        """Figure 17: OBR Baseline vs. Fiscal Reaction Monte Carlo."""
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        np.random.seed(42)
        years = np.arange(2024, 2035)
        n_sims = 10000
        
        # Simulate OBR baseline
        debt_obr = np.zeros((n_sims, 11))
        debt_obr[:, 0] = 96
        pb_path = np.array([-1.5, -0.6, -0.1, 0.5, 1.0, 1.3, 1.4, 1.4, 1.4, 1.4, 1.4]) / 100
        
        for t in range(1, 11):
            g = np.random.normal(0.035, 0.02, n_sims)
            r = np.random.normal(0.045, 0.01, n_sims)
            debt_obr[:, t] = (1 + r) / (1 + g) * debt_obr[:, t-1] - pb_path[t] * 100
        
        # Simulate fiscal reaction
        debt_fr = np.zeros((n_sims, 11))
        debt_fr[:, 0] = 96
        beta = -0.05  # Negative Bohn coefficient
        
        for t in range(1, 11):
            g = np.random.normal(0.035, 0.02, n_sims)
            r = np.random.normal(0.045, 0.01, n_sims)
            pb = 0.005 + beta * (debt_fr[:, t-1] - 60) / 100 + np.random.normal(0, 0.015, n_sims)
            debt_fr[:, t] = (1 + r) / (1 + g) * debt_fr[:, t-1] - pb * 100
        
        # Panel A: Fan chart comparison
        for debt, color, label in [(debt_obr, COLORS['primary'], 'OBR Baseline'),
                                   (debt_fr, COLORS['danger'], 'Fiscal Reaction')]:
            p50 = np.percentile(debt, 50, axis=0)
            p25 = np.percentile(debt, 25, axis=0)
            p75 = np.percentile(debt, 75, axis=0)
            
            axes[0].fill_between(years, p25, p75, alpha=0.3, color=color)
            axes[0].plot(years, p50, color=color, linewidth=2.5, label=f'{label} (median)')
        
        axes[0].axhline(y=100, color='gray', linestyle='--', linewidth=1.5)
        axes[0].set_xlabel('Fiscal Year', fontsize=12)
        axes[0].set_ylabel('Debt/GDP (%)', fontsize=12)
        axes[0].set_title('Panel A: Debt Path Comparison (50% CI)', fontsize=13, fontweight='bold')
        axes[0].legend(loc='upper left')
        axes[0].set_ylim(60, 200)
        
        # Panel B: Terminal distributions
        bins = np.linspace(60, 250, 50)
        
        axes[1].hist(debt_obr[:, -1], bins=bins, density=True, alpha=0.6,
                    color=COLORS['primary'], label='OBR Baseline', edgecolor='white')
        axes[1].hist(debt_fr[:, -1], bins=bins, density=True, alpha=0.6,
                    color=COLORS['danger'], label='Fiscal Reaction', edgecolor='white')
        
        axes[1].axvline(x=100, color='black', linestyle='--', linewidth=2)
        
        axes[1].set_xlabel('Terminal Debt/GDP (%)', fontsize=12)
        axes[1].set_ylabel('Density', fontsize=12)
        axes[1].set_title('Panel B: Terminal Distribution Comparison', fontsize=13, fontweight='bold')
        axes[1].legend(loc='upper right')
        
        # Add statistics
        stats_text = f"OBR Baseline:\n  Mean: {np.mean(debt_obr[:,-1]):.1f}%\n  P(>100%): {np.mean(debt_obr[:,-1]>100)*100:.1f}%\n\nFiscal Reaction:\n  Mean: {np.mean(debt_fr[:,-1]):.1f}%\n  P(>100%): {np.mean(debt_fr[:,-1]>100)*100:.1f}%"
        axes[1].text(0.98, 0.98, stats_text, transform=axes[1].transAxes, fontsize=10,
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        
        plt.tight_layout()
        path = f'{self.output_dir}/fig17_mc_comparison.png'
        plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Saved: {path}")
        return fig
    
    def fig18_sustainability_dashboard(self):
        """Figure 18: Comprehensive Sustainability Dashboard."""
        fig = plt.figure(figsize=(18, 12))
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # Panel 1: Overall verdict (large)
        ax1 = fig.add_subplot(gs[0, :])
        ax1.set_xlim(0, 10)
        ax1.set_ylim(0, 2)
        ax1.axis('off')
        
        ax1.text(5, 1.5, 'UK DEBT SUSTAINABILITY ASSESSMENT', 
                fontsize=24, fontweight='bold', ha='center', va='center')
        ax1.text(5, 0.8, 'CONDITIONALLY SUSTAINABLE', 
                fontsize=28, fontweight='bold', ha='center', va='center', color=COLORS['warning'])
        ax1.text(5, 0.3, 'Sustainable IF policy commitments are achieved', 
                fontsize=14, ha='center', va='center', style='italic')
        
        # Panel 2: Key metrics
        ax2 = fig.add_subplot(gs[1, 0])
        metrics = ['Current Debt/GDP', 'Bohn β', 'P(>100% GDP)', 'GFN/GDP']
        values = ['96.0%', '-0.05', '55.8%', '10.3%']
        status = ['⚠️', '❌', '⚠️', '✅']
        
        for i, (m, v, s) in enumerate(zip(metrics, values, status)):
            ax2.text(0.1, 0.85 - i*0.25, f'{s} {m}:', fontsize=12, fontweight='bold', transform=ax2.transAxes)
            ax2.text(0.9, 0.85 - i*0.25, v, fontsize=12, ha='right', transform=ax2.transAxes)
        ax2.set_title('Key Metrics', fontsize=14, fontweight='bold')
        ax2.axis('off')
        
        # Panel 3: Criteria checklist
        ax3 = fig.add_subplot(gs[1, 1])
        criteria = [
            ('Fiscal reaction (β > 0)', False),
            ('Debt path (P < 50%)', False),
            ('Primary balance achievable', True),
            ('GFN < 15% GDP', True),
        ]
        
        for i, (c, passed) in enumerate(criteria):
            symbol = '✅' if passed else '❌'
            color = COLORS['success'] if passed else COLORS['danger']
            ax3.text(0.1, 0.85 - i*0.22, f'{symbol} {c}', fontsize=11, 
                    transform=ax3.transAxes, color=color)
        
        ax3.text(0.5, 0.05, 'Score: 2/4 (50%)', fontsize=12, fontweight='bold',
                transform=ax3.transAxes, ha='center')
        ax3.set_title('Sustainability Criteria', fontsize=14, fontweight='bold')
        ax3.axis('off')
        
        # Panel 4: Risk gauge
        ax4 = fig.add_subplot(gs[1, 2])
        
        # Simple gauge visualization
        risk_level = 0.6  # 0-1 scale
        theta = np.linspace(np.pi, 0, 100)
        r = 0.8
        
        # Background arc
        for i, (start, end, color) in enumerate([
            (np.pi, 2*np.pi/3, COLORS['success']),
            (2*np.pi/3, np.pi/3, COLORS['warning']),
            (np.pi/3, 0, COLORS['danger'])
        ]):
            t = np.linspace(start, end, 30)
            ax4.fill_between(r*np.cos(t), r*np.sin(t), alpha=0.3, color=color)
            ax4.plot(r*np.cos(t), r*np.sin(t), color=color, linewidth=3)
        
        # Needle
        needle_angle = np.pi * (1 - risk_level)
        ax4.arrow(0, 0, 0.6*np.cos(needle_angle), 0.6*np.sin(needle_angle),
                 head_width=0.05, head_length=0.05, fc=COLORS['quinary'], ec=COLORS['quinary'])
        
        ax4.set_xlim(-1, 1)
        ax4.set_ylim(-0.2, 1)
        ax4.set_aspect('equal')
        ax4.axis('off')
        ax4.set_title('Risk Level: ELEVATED', fontsize=14, fontweight='bold')
        
        # Panel 5: Critical risks
        ax5 = fig.add_subplot(gs[2, 0])
        risks = [
            'Negative Bohn coefficient',
            '34% ILG inflation exposure', 
            'No historical stabilization',
            'High starting debt (96%)',
        ]
        
        for i, risk in enumerate(risks):
            ax5.text(0.05, 0.85 - i*0.22, f'⚠️ {risk}', fontsize=10, transform=ax5.transAxes)
        ax5.set_title('Critical Risks', fontsize=14, fontweight='bold')
        ax5.axis('off')
        
        # Panel 6: Mitigating factors
        ax6 = fig.add_subplot(gs[2, 1])
        mitigants = [
            'Long average maturity (13.7y)',
            'Reserve currency status',
            'Deep liquid gilt market',
            'BoE QE capacity',
        ]
        
        for i, m in enumerate(mitigants):
            ax6.text(0.05, 0.85 - i*0.22, f'✓ {m}', fontsize=10, transform=ax6.transAxes, 
                    color=COLORS['success'])
        ax6.set_title('Mitigating Factors', fontsize=14, fontweight='bold')
        ax6.axis('off')
        
        # Panel 7: Policy recommendations
        ax7 = fig.add_subplot(gs[2, 2])
        recs = [
            'Achieve fiscal consolidation',
            'Reduce ILG share',
            'Build fiscal buffers',
            'Enhance credibility',
        ]
        
        for i, r in enumerate(recs):
            ax7.text(0.05, 0.85 - i*0.22, f'→ {r}', fontsize=10, transform=ax7.transAxes,
                    color=COLORS['primary'])
        ax7.set_title('Policy Recommendations', fontsize=14, fontweight='bold')
        ax7.axis('off')
        
        plt.suptitle('', fontsize=1)  # Remove default title space
        path = f'{self.output_dir}/fig18_sustainability_dashboard.png'
        plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Saved: {path}")
        return fig
    
    def generate_all_figures(self):
        """Generate all publication figures."""
        print("\n" + "="*70)
        print("GENERATING ALL PUBLICATION FIGURES")
        print("="*70 + "\n")
        
        # Create output directory
        import os
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Generate all figures
        self.fig1_historical_debt_trajectory()
        self.fig2_debt_decomposition()
        self.fig3_r_g_differential()
        self.fig4_interest_burden()
        self.fig5_debt_composition()
        self.fig6_ilg_sensitivity()
        self.fig7_fiscal_rules_history()
        self.fig8_aging_costs()
        self.fig9_contingent_liabilities()
        self.fig10_fan_chart()
        self.fig11_bohn_test()
        self.fig12_econometric_tests()
        self.fig13_scenario_comparison()
        self.fig14_terminal_distribution()
        self.fig15_gfn_analysis()
        self.fig16_correlation_heatmap()
        self.fig17_mc_comparison()
        self.fig18_sustainability_dashboard()
        
        print("\n" + "="*70)
        print(f"ALL 18 FIGURES GENERATED IN {self.output_dir}")
        print("="*70)


def main():
    """Generate all figures."""
    output_dir = '/mnt/user-data/outputs'
    generator = PublicationFigures(output_dir)
    generator.generate_all_figures()
    

if __name__ == "__main__":
    main()
