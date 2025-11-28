"""
UK Debt Sustainability Analysis Package
=======================================
Imperial College London UROP Project

This package provides comprehensive tools for analyzing UK public debt
sustainability using OBR forecasts and Monte Carlo simulation.

Modules:
    - config: Configuration parameters and OBR forecasts
    - debt_dynamics: Core debt dynamics model
    - monte_carlo: Stochastic simulation engine
    - visualization: Publication-quality charts
    - main: Main analysis script
"""

from .config import (
    OBR_GDP, OBR_PSND, OBR_DEBT_TO_GDP, OBR_PSNB, 
    OBR_DEBT_INTEREST, SCENARIOS, READY_RECKONERS
)
from .debt_dynamics import DebtDynamicsModel, FiscalReactionFunction
from .monte_carlo import MonteCarloEngine, MonteCarloConfig
from .visualization import DSAVisualizer

__version__ = '1.0.0'
__author__ = 'Imperial College London UROP'
__all__ = [
    'DebtDynamicsModel',
    'FiscalReactionFunction', 
    'MonteCarloEngine',
    'MonteCarloConfig',
    'DSAVisualizer'
]
