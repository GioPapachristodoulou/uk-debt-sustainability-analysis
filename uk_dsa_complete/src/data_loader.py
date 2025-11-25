"""
UK Debt Sustainability Analysis - Comprehensive Data Loader
============================================================
Imperial College London UROP Project

This module loads and processes:
- 162 historical CSV files from ONS/DMO/BoE
- 14 OBR March 2025 forecast XLSX files
- Builds unified time series from 1997-2035

Data Sources:
- ONS: GDP, inflation, public finances
- DMO: Gilts, auctions, yields
- BoE: Interest rates, exchange rates
- OBR: March 2025 Economic and Fiscal Outlook
"""

import pandas as pd
import numpy as np
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


@dataclass
class DataPaths:
    """Configuration for data file locations."""
    csv_dir: str = '/mnt/project'
    xlsx_dir: str = '/mnt/user-data/uploads'
    output_dir: str = '/home/claude/uk_dsa/data/processed'


class DateParser:
    """
    Utility class for parsing various date formats in UK fiscal data.
    """
    
    MONTH_MAP = {
        'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6,
        'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12
    }
    
    @staticmethod
    def parse_ons_date(date_str: str) -> Optional[pd.Timestamp]:
        """
        Parse ONS-style dates: '2024 MAR', '2024 Q2', '2024', '31 Oct 25'
        """
        if pd.isna(date_str):
            return None
            
        date_str = str(date_str).strip()
        
        # Annual: '2024'
        if re.match(r'^\d{4}$', date_str):
            return pd.Timestamp(year=int(date_str), month=12, day=31)
        
        # Monthly: '2024 MAR' or 'MAR 2024'
        match = re.match(r'(\d{4})\s+([A-Z]{3})', date_str.upper())
        if match:
            year, month = int(match.group(1)), DateParser.MONTH_MAP.get(match.group(2))
            if month:
                return pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)
        
        # Quarterly: '2024 Q2'
        match = re.match(r'(\d{4})\s+Q(\d)', date_str.upper())
        if match:
            year, quarter = int(match.group(1)), int(match.group(2))
            month = quarter * 3
            return pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)
        
        # DMO style: '31 Oct 25'
        match = re.match(r'(\d{1,2})\s+([A-Za-z]{3})\s+(\d{2})', date_str)
        if match:
            day, month_str, year_short = match.groups()
            month = DateParser.MONTH_MAP.get(month_str.upper())
            year = 2000 + int(year_short) if int(year_short) < 50 else 1900 + int(year_short)
            if month:
                return pd.Timestamp(year=year, month=month, day=int(day))
        
        # Try pandas default
        try:
            return pd.to_datetime(date_str)
        except:
            return None
    
    @staticmethod
    def to_fiscal_year(date: pd.Timestamp) -> str:
        """
        Convert date to fiscal year string (e.g., '2024-25').
        UK fiscal year runs April to March.
        """
        if date.month >= 4:
            return f"{date.year}-{str(date.year + 1)[2:]}"
        else:
            return f"{date.year - 1}-{str(date.year)[2:]}"
    
    @staticmethod
    def fiscal_year_to_date(fy: str) -> pd.Timestamp:
        """
        Convert fiscal year string to end-of-fiscal-year date.
        '2024-25' -> 2025-03-31
        """
        start_year = int(fy.split('-')[0])
        return pd.Timestamp(year=start_year + 1, month=3, day=31)


class CSVLoader:
    """
    Loader for ONS/DMO/BoE CSV files.
    """
    
    def __init__(self, csv_dir: str = '/mnt/project'):
        self.csv_dir = Path(csv_dir)
        self.date_parser = DateParser()
    
    def list_files(self) -> List[str]:
        """List all CSV files in directory."""
        return sorted([f.name for f in self.csv_dir.glob('*.csv')])
    
    def load_single_csv(self, filename: str) -> pd.DataFrame:
        """
        Load a single CSV file with appropriate parsing.
        Handles ONS metadata headers.
        """
        filepath = self.csv_dir / filename
        
        # First, read raw to detect header rows
        raw = pd.read_csv(filepath, header=None, nrows=20, encoding='utf-8', 
                          on_bad_lines='skip')
        
        # Find where data starts (usually after metadata rows)
        data_start = 0
        for i, row in raw.iterrows():
            first_cell = str(row[0]).strip() if pd.notna(row[0]) else ''
            # Data rows typically start with a year or date
            if re.match(r'^\d{4}', first_cell) or first_cell in ['Date', 'Title']:
                if first_cell in ['Date', 'Title']:
                    data_start = i
                else:
                    data_start = i
                break
        
        # Load full file
        df = pd.read_csv(filepath, encoding='utf-8', on_bad_lines='skip')
        
        # Clean column names
        df.columns = [str(c).strip() for c in df.columns]
        
        # Identify date column
        date_col = df.columns[0]
        
        # Parse dates
        df['parsed_date'] = df[date_col].apply(self.date_parser.parse_ons_date)
        
        # Filter to valid dates only
        df = df[df['parsed_date'].notna()].copy()
        
        if len(df) == 0:
            return pd.DataFrame()
        
        # Get value column (second column typically)
        value_cols = [c for c in df.columns if c not in [date_col, 'parsed_date']]
        if value_cols:
            value_col = value_cols[0]
            # Clean and convert to numeric
            df['value'] = pd.to_numeric(df[value_col].astype(str).str.replace(',', ''), errors='coerce')
        
        # Sort by date
        df = df.sort_values('parsed_date').reset_index(drop=True)
        
        return df[['parsed_date', 'value']].rename(columns={'parsed_date': 'date'})
    
    def load_public_sector_net_debt(self) -> pd.DataFrame:
        """Load PSND series (key debt series)."""
        df = self.load_single_csv('Public_Sector_Net_Debt.csv')
        df['value'] = df['value']  # Already in £bn
        return df.rename(columns={'value': 'psnd_bn'})
    
    def load_gdp(self) -> pd.DataFrame:
        """Load GDP series (quarterly NSA)."""
        df = self.load_single_csv('Gross_Domestic_Product_at_market_prices_NSA.csv')
        df['value'] = df['value'] / 1000  # Convert £m to £bn
        return df.rename(columns={'value': 'gdp_bn'})
    
    def load_gilt_yields_10y(self) -> pd.DataFrame:
        """Load 10-year nominal zero-coupon gilt yields."""
        df = self.load_single_csv('Zerocoupon_yields_nominal__10year.csv')
        return df.rename(columns={'value': 'yield_10y'})
    
    def load_gilt_yields_5y(self) -> pd.DataFrame:
        """Load 5-year nominal zero-coupon gilt yields."""
        df = self.load_single_csv('Zerocoupon_yields_nominal__5year.csv')
        return df.rename(columns={'value': 'yield_5y'})
    
    def load_gilt_yields_20y(self) -> pd.DataFrame:
        """Load 20-year nominal zero-coupon gilt yields."""
        df = self.load_single_csv('Zerocoupon_yields_nominal__20year.csv')
        return df.rename(columns={'value': 'yield_20y'})
    
    def load_real_yields_10y(self) -> pd.DataFrame:
        """Load 10-year real zero-coupon gilt yields."""
        df = self.load_single_csv('Zerocoupon_yields_real__10year.csv')
        return df.rename(columns={'value': 'real_yield_10y'})
    
    def load_rpi(self) -> pd.DataFrame:
        """Load RPI index."""
        df = self.load_single_csv('Retail_Prices_Index_RPI_level.csv')
        return df.rename(columns={'value': 'rpi_index'})
    
    def load_cpi(self) -> pd.DataFrame:
        """Load CPI index."""
        df = self.load_single_csv('Consumer_Prices_Index.csv')
        return df.rename(columns={'value': 'cpi_index'})
    
    def load_bank_rate(self) -> pd.DataFrame:
        """Load Bank of England official rate."""
        df = self.load_single_csv('Official_Bank_Rate__EndMonth.csv')
        return df.rename(columns={'value': 'bank_rate'})
    
    def load_psnb(self) -> pd.DataFrame:
        """Load Public Sector Net Borrowing."""
        df = self.load_single_csv('Public_Sector_Net_Borrowing_NSA.csv')
        df['value'] = df['value'] / 1000  # Convert £m to £bn
        return df.rename(columns={'value': 'psnb_bn'})
    
    def load_debt_interest_cg(self) -> pd.DataFrame:
        """Load Central Government interest payments."""
        df = self.load_single_csv('CG_interestdividends_paid_to_private_sector__RoW.csv')
        df['value'] = df['value'] / 1000  # Convert £m to £bn
        return df.rename(columns={'value': 'debt_interest_cg_bn'})
    
    def load_receipts(self) -> pd.DataFrame:
        """Load Public Sector Current Receipts."""
        df = self.load_single_csv('Public_Sector_Current_Receipts.csv')
        df['value'] = df['value'] / 1000  # Convert £m to £bn
        return df.rename(columns={'value': 'receipts_bn'})
    
    def load_tme(self) -> pd.DataFrame:
        """Load Total Managed Expenditure."""
        df = self.load_single_csv('Total_Managed_Expenditure.csv')
        df['value'] = df['value'] / 1000  # Convert £m to £bn
        return df.rename(columns={'value': 'tme_bn'})
    
    def load_gilts_in_issue(self) -> pd.DataFrame:
        """
        Load current gilts in issue (DMO data).
        Returns breakdown of gilt stock.
        """
        filepath = self.csv_dir / 'Gilts_in_Issue_D1A.csv'
        
        # This file has special format - parse carefully
        lines = []
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        gilts = []
        current_category = 'Ultra-Short'
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Detect category
            if line.startswith('Ultra-Short'):
                current_category = 'Ultra-Short'
            elif line.startswith('Short,'):
                current_category = 'Short'
            elif line.startswith('Medium,'):
                current_category = 'Medium'
            elif line.startswith('Long,'):
                current_category = 'Long'
            
            # Parse gilt lines
            parts = line.split(',')
            if len(parts) >= 7 and '%' in parts[0]:
                name = parts[0].strip('"')
                amount_str = parts[6].strip().strip('"').replace(',', '')
                try:
                    amount = float(amount_str)
                    gilts.append({
                        'name': name,
                        'category': current_category,
                        'amount_mn': amount,
                        'type': 'conventional'
                    })
                except:
                    pass
        
        return pd.DataFrame(gilts)
    
    def load_ilg_in_issue(self) -> pd.DataFrame:
        """
        Load index-linked gilts in issue (DMO data).
        """
        filepath = self.csv_dir / 'Indexlinked_Gilts_in_Issue_D1D.csv'
        
        lines = []
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        gilts = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split(',')
            # ILG lines have gilt name with % in first column
            if len(parts) >= 7 and '%' in parts[0] and 'Index' in parts[0]:
                name = parts[0].strip('"')
                # Nominal in issue is column 5, with uplift is column 6
                nominal_str = parts[5].strip().strip('"').replace(',', '')
                uplift_str = parts[6].strip().strip('"').replace(',', '')
                try:
                    nominal = float(nominal_str)
                    uplift = float(uplift_str) if uplift_str else nominal
                    gilts.append({
                        'name': name,
                        'nominal_mn': nominal,
                        'with_uplift_mn': uplift,
                        'type': 'index-linked'
                    })
                except:
                    pass
        
        return pd.DataFrame(gilts)


class OBRLoader:
    """
    Loader for OBR March 2025 XLSX forecast files.
    """
    
    def __init__(self, xlsx_dir: str = '/mnt/user-data/uploads'):
        self.xlsx_dir = Path(xlsx_dir)
        self.forecast_years = ['2024-25', '2025-26', '2026-27', '2027-28', '2028-29', '2029-30']
    
    def load_economy_forecasts(self) -> Dict[str, pd.DataFrame]:
        """Load economic forecasts from Economy file."""
        filepath = self.xlsx_dir / 'Economy_Detailed_forecast_tables_March_2025.xlsx'
        
        forecasts = {}
        
        # GDP growth (real)
        try:
            df = pd.read_excel(filepath, sheet_name='1.1', header=None, skiprows=3)
            # Find GDP row
            for i, row in df.iterrows():
                if 'Gross domestic product' in str(row[1]) or 'GDP' in str(row[1]):
                    values = row[3:9].values
                    forecasts['real_gdp_growth'] = {
                        fy: float(v) if pd.notna(v) else None 
                        for fy, v in zip(self.forecast_years, values)
                    }
                    break
        except Exception as e:
            print(f"Warning: Could not load GDP growth: {e}")
        
        return forecasts
    
    def load_aggregates_forecasts(self) -> Dict[str, pd.DataFrame]:
        """Load fiscal aggregates from Aggregates file."""
        filepath = self.xlsx_dir / 'Aggregates_Detailed_forecast_tables_March_2025.xlsx'
        
        forecasts = {}
        
        # Load Table 6.1 - Expenditure breakdown
        try:
            df = pd.read_excel(filepath, sheet_name='6.1', header=None)
            
            # Parse the expenditure data
            for i, row in df.iterrows():
                label = str(row[2]) if pd.notna(row[2]) else str(row[1]) if pd.notna(row[1]) else ''
                
                if 'Interest and dividends paid' in label:
                    values = row[3:9].values
                    cg_interest = {
                        fy: float(v) if pd.notna(v) else None 
                        for fy, v in zip(self.forecast_years, values)
                    }
                    if i < 20:  # CG section
                        forecasts['cg_interest'] = cg_interest
                    
        except Exception as e:
            print(f"Warning: Could not load aggregates: {e}")
        
        return forecasts
    
    def load_debt_interest_forecasts(self) -> Dict[str, any]:
        """
        Load debt interest forecasts and ready reckoners.
        This is the crucial file for sensitivity analysis.
        """
        filepath = self.xlsx_dir / 'Debt_interest_Detailed_forecast_tables_March_2025.xlsx'
        
        forecasts = {}
        
        # Load all sheets
        xl = pd.ExcelFile(filepath)
        print(f"Debt interest sheets: {xl.sheet_names}")
        
        for sheet in xl.sheet_names:
            try:
                df = pd.read_excel(filepath, sheet_name=sheet, header=None)
                forecasts[sheet] = df
            except:
                pass
        
        return forecasts
    
    def load_chapter5_debt_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load Chapter 5 (fiscal sustainability) charts and tables.
        Contains debt forecasts and fiscal rule assessments.
        """
        filepath = self.xlsx_dir / 'Chapter_5_charts_and_tables_March_2025.xlsx'
        
        data = {}
        
        xl = pd.ExcelFile(filepath)
        
        for sheet in xl.sheet_names:
            try:
                df = pd.read_excel(filepath, sheet_name=sheet, header=None)
                data[sheet] = df
            except:
                pass
        
        return data


class HistoricalDataBuilder:
    """
    Builds consistent historical time series from multiple sources.
    """
    
    def __init__(self, csv_loader: CSVLoader):
        self.csv = csv_loader
    
    def build_annual_series(self, start_year: int = 1997, end_year: int = 2024) -> pd.DataFrame:
        """
        Build annual fiscal year time series.
        """
        # Load all series
        psnd = self.csv.load_public_sector_net_debt()
        gdp = self.csv.load_gdp()
        yields_10y = self.csv.load_gilt_yields_10y()
        rpi = self.csv.load_rpi()
        cpi = self.csv.load_cpi()
        bank_rate = self.csv.load_bank_rate()
        psnb = self.csv.load_psnb()
        receipts = self.csv.load_receipts()
        tme = self.csv.load_tme()
        
        # Create fiscal year index
        fiscal_years = [f"{y}-{str(y+1)[2:]}" for y in range(start_year, end_year + 1)]
        
        result = pd.DataFrame({'fiscal_year': fiscal_years})
        
        # For each fiscal year, get end-of-year values (March)
        for i, fy in enumerate(fiscal_years):
            year_end = int(fy.split('-')[0]) + 1
            march_date = pd.Timestamp(year=year_end, month=3, day=31)
            
            # PSND - get March value
            psnd_val = psnd[
                (psnd['date'].dt.year == year_end) & 
                (psnd['date'].dt.month == 3)
            ]['psnd_bn'].values
            result.loc[i, 'psnd_bn'] = psnd_val[0] if len(psnd_val) > 0 else np.nan
            
            # GDP - sum of 4 quarters (Apr-Mar)
            gdp_fy = gdp[
                ((gdp['date'].dt.year == year_end - 1) & (gdp['date'].dt.month >= 4)) |
                ((gdp['date'].dt.year == year_end) & (gdp['date'].dt.month <= 3))
            ]['gdp_bn'].sum()
            result.loc[i, 'gdp_bn'] = gdp_fy if gdp_fy > 0 else np.nan
            
            # Yields - average over fiscal year
            yields_fy = yields_10y[
                ((yields_10y['date'].dt.year == year_end - 1) & (yields_10y['date'].dt.month >= 4)) |
                ((yields_10y['date'].dt.year == year_end) & (yields_10y['date'].dt.month <= 3))
            ]['yield_10y'].mean()
            result.loc[i, 'gilt_yield_10y'] = yields_fy
            
            # Bank rate - average
            rate_fy = bank_rate[
                ((bank_rate['date'].dt.year == year_end - 1) & (bank_rate['date'].dt.month >= 4)) |
                ((bank_rate['date'].dt.year == year_end) & (bank_rate['date'].dt.month <= 3))
            ]['bank_rate'].mean()
            result.loc[i, 'bank_rate'] = rate_fy
            
            # RPI - calculate annual inflation
            rpi_start = rpi[
                (rpi['date'].dt.year == year_end - 1) & 
                (rpi['date'].dt.month == 4)
            ]['rpi_index'].values
            rpi_end = rpi[
                (rpi['date'].dt.year == year_end) & 
                (rpi['date'].dt.month == 3)
            ]['rpi_index'].values
            if len(rpi_start) > 0 and len(rpi_end) > 0:
                result.loc[i, 'rpi_inflation'] = (rpi_end[0] / rpi_start[0] - 1) * 100
            
            # PSNB - sum over fiscal year (monthly data)
            psnb_fy = psnb[
                ((psnb['date'].dt.year == year_end - 1) & (psnb['date'].dt.month >= 4)) |
                ((psnb['date'].dt.year == year_end) & (psnb['date'].dt.month <= 3))
            ]['psnb_bn'].sum()
            result.loc[i, 'psnb_bn'] = psnb_fy if psnb_fy != 0 else np.nan
            
            # Receipts - sum
            receipts_fy = receipts[
                ((receipts['date'].dt.year == year_end - 1) & (receipts['date'].dt.month >= 4)) |
                ((receipts['date'].dt.year == year_end) & (receipts['date'].dt.month <= 3))
            ]['receipts_bn'].sum()
            result.loc[i, 'receipts_bn'] = receipts_fy if receipts_fy > 0 else np.nan
            
            # TME - sum
            tme_fy = tme[
                ((tme['date'].dt.year == year_end - 1) & (tme['date'].dt.month >= 4)) |
                ((tme['date'].dt.year == year_end) & (tme['date'].dt.month <= 3))
            ]['tme_bn'].sum()
            result.loc[i, 'tme_bn'] = tme_fy if tme_fy > 0 else np.nan
        
        # Calculate derived metrics
        result['debt_to_gdp'] = result['psnd_bn'] / result['gdp_bn'] * 100
        result['deficit_to_gdp'] = result['psnb_bn'] / result['gdp_bn'] * 100
        result['receipts_to_gdp'] = result['receipts_bn'] / result['gdp_bn'] * 100
        result['tme_to_gdp'] = result['tme_bn'] / result['gdp_bn'] * 100
        
        # Nominal GDP growth
        result['nominal_gdp_growth'] = result['gdp_bn'].pct_change() * 100
        
        return result


class DebtCompositionAnalyzer:
    """
    Analyzes the composition of UK government debt.
    Critical for understanding inflation sensitivity.
    """
    
    def __init__(self, csv_loader: CSVLoader):
        self.csv = csv_loader
    
    def get_current_composition(self) -> Dict[str, float]:
        """
        Get current debt composition from DMO data.
        """
        conventional = self.csv.load_gilts_in_issue()
        ilg = self.csv.load_ilg_in_issue()
        
        conv_total = conventional['amount_mn'].sum() / 1000  # £bn
        ilg_nominal = ilg['nominal_mn'].sum() / 1000  # £bn
        ilg_with_uplift = ilg['with_uplift_mn'].sum() / 1000  # £bn
        
        total_gilts = conv_total + ilg_with_uplift
        
        return {
            'conventional_gilts_bn': conv_total,
            'ilg_nominal_bn': ilg_nominal,
            'ilg_with_uplift_bn': ilg_with_uplift,
            'total_gilts_bn': total_gilts,
            'ilg_share': ilg_with_uplift / total_gilts if total_gilts > 0 else 0,
            'conventional_share': conv_total / total_gilts if total_gilts > 0 else 0
        }
    
    def get_maturity_profile(self) -> pd.DataFrame:
        """
        Get maturity profile of conventional gilts.
        """
        conventional = self.csv.load_gilts_in_issue()
        
        # Aggregate by category
        profile = conventional.groupby('category')['amount_mn'].sum().reset_index()
        profile['amount_bn'] = profile['amount_mn'] / 1000
        
        return profile


class DataLoader:
    """
    Main data loader class - coordinates all data loading.
    """
    
    def __init__(self, paths: DataPaths = None):
        self.paths = paths or DataPaths()
        self.csv = CSVLoader(self.paths.csv_dir)
        self.obr = OBRLoader(self.paths.xlsx_dir)
        self.history_builder = HistoricalDataBuilder(self.csv)
        self.composition = DebtCompositionAnalyzer(self.csv)
    
    def load_all_historical(self, start_year: int = 1997, end_year: int = 2024) -> pd.DataFrame:
        """Load all historical data as annual fiscal year series."""
        return self.history_builder.build_annual_series(start_year, end_year)
    
    def get_debt_composition(self) -> Dict[str, float]:
        """Get current debt composition."""
        return self.composition.get_current_composition()
    
    def load_obr_forecasts(self) -> Dict[str, any]:
        """Load all OBR forecast data."""
        return {
            'economy': self.obr.load_economy_forecasts(),
            'aggregates': self.obr.load_aggregates_forecasts(),
            'debt_interest': self.obr.load_debt_interest_forecasts(),
            'chapter5': self.obr.load_chapter5_debt_data()
        }
    
    def build_complete_dataset(self) -> pd.DataFrame:
        """
        Build complete dataset from 1997 to 2035.
        Merges historical data with OBR forecasts.
        """
        # Get historical
        historical = self.load_all_historical(1997, 2024)
        
        # OBR forecast years (hardcoded from config for reliability)
        obr_data = {
            'fiscal_year': ['2024-25', '2025-26', '2026-27', '2027-28', '2028-29', '2029-30'],
            'gdp_bn': [2864, 3002, 3113, 3223, 3336, 3457],
            'psnd_bn': [2746.0, 2855.7, 2981.0, 3098.2, 3212.0, 3322.0],
            'debt_to_gdp': [95.9, 95.1, 95.8, 96.1, 96.3, 96.1],
            'psnb_bn': [137.3, 117.7, 97.2, 80.2, 77.4, 74.0],
            'debt_interest_bn': [105.2, 111.2, 111.4, 117.9, 124.2, 131.6],
            'receipts_bn': [1194.6, 1272.7, 1324.5, 1377.5, 1431.7, 1489.9],
            'tme_bn': [1331.9, 1390.4, 1421.7, 1457.7, 1509.1, 1563.9],
            'rpi_inflation': [3.6, 4.0, 3.1, 2.9, 2.8, 2.8],
            'cpi_inflation': [2.6, 3.2, 2.4, 2.1, 2.0, 2.0],
            'gilt_yield_10y': [4.4, 4.5, 4.4, 4.3, 4.2, 4.1],
            'bank_rate': [4.5, 4.0, 3.5, 3.25, 3.0, 3.0],
            'nominal_gdp_growth': [4.8, 4.8, 3.7, 3.5, 3.5, 3.6],
        }
        
        forecast = pd.DataFrame(obr_data)
        forecast['is_forecast'] = True
        historical['is_forecast'] = False
        
        # Extend to 2035 using simple extrapolation
        extension_years = ['2030-31', '2031-32', '2032-33', '2033-34', '2034-35']
        extension_data = []
        
        last_row = forecast.iloc[-1].copy()
        for i, fy in enumerate(extension_years):
            new_row = last_row.copy()
            new_row['fiscal_year'] = fy
            # Simple extrapolation - debt grows at nominal GDP growth minus primary surplus
            growth_rate = 1 + new_row['nominal_gdp_growth'] / 100
            new_row['gdp_bn'] = new_row['gdp_bn'] * growth_rate
            new_row['psnd_bn'] = new_row['psnd_bn'] + new_row['psnb_bn'] * 0.95  # Slight improvement
            new_row['psnb_bn'] = new_row['psnb_bn'] * 0.97  # 3% annual improvement
            new_row['debt_interest_bn'] = new_row['debt_interest_bn'] * 1.03  # 3% growth
            new_row['debt_to_gdp'] = new_row['psnd_bn'] / new_row['gdp_bn'] * 100
            new_row['is_forecast'] = True
            extension_data.append(new_row)
            last_row = new_row
        
        extension_df = pd.DataFrame(extension_data)
        
        # Combine all
        combined = pd.concat([
            historical[historical['fiscal_year'] < '2024-25'],
            forecast,
            extension_df
        ], ignore_index=True)
        
        return combined


def main():
    """Test the data loader."""
    loader = DataLoader()
    
    print("=" * 60)
    print("UK DSA DATA LOADER TEST")
    print("=" * 60)
    
    # Test CSV loading
    print("\n[1] Loading historical data...")
    historical = loader.load_all_historical(2010, 2024)
    print(f"    Loaded {len(historical)} years of historical data")
    print(f"    Columns: {list(historical.columns)}")
    print(f"\n    Sample data (2020-2024):")
    print(historical[historical['fiscal_year'] >= '2020-21'].to_string(index=False))
    
    # Test debt composition
    print("\n[2] Loading debt composition...")
    composition = loader.get_debt_composition()
    print(f"    Conventional gilts: £{composition['conventional_gilts_bn']:.1f}bn")
    print(f"    Index-linked (with uplift): £{composition['ilg_with_uplift_bn']:.1f}bn")
    print(f"    ILG share: {composition['ilg_share']*100:.1f}%")
    
    # Test complete dataset
    print("\n[3] Building complete dataset (1997-2035)...")
    complete = loader.build_complete_dataset()
    print(f"    Total rows: {len(complete)}")
    print(f"    Historical: {(~complete['is_forecast']).sum()}")
    print(f"    Forecast: {complete['is_forecast'].sum()}")
    
    print("\n[4] Debt trajectory summary:")
    key_years = ['1997-98', '2007-08', '2019-20', '2024-25', '2029-30', '2034-35']
    for fy in key_years:
        row = complete[complete['fiscal_year'] == fy]
        if len(row) > 0:
            print(f"    {fy}: Debt/GDP = {row['debt_to_gdp'].values[0]:.1f}%")
    
    return complete


if __name__ == '__main__':
    data = main()
