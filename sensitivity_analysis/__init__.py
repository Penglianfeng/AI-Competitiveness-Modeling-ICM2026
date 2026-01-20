# -*- coding: utf-8 -*-
"""
Sensitivity Analysis Module for Global AI Competition Assessment
================================================================

This module provides comprehensive sensitivity analysis for the AI competition
assessment model, covering:
- SA-1: Data robustness analysis (extrapolation uncertainty)
- SA-2: Weight tornado chart analysis (indicator importance)
- SA-3: GLV parameter sensitivity analysis (model stability)
- SA-4: Budget/constraint sensitivity analysis (optimization robustness)

Usage:
    # Method 1: Import individual functions
    from sensitivity_analysis import run_data_robustness
    results = run_data_robustness()
    
    # Method 2: Command line
    python -m sensitivity_analysis.sa_main --all
    python -m sensitivity_analysis.sa_main --task SA1 SA2

Author: AI Modeling Assistant
Date: January 2026
"""

# Lazy imports to avoid circular dependencies
def run_data_robustness(*args, **kwargs):
    """SA-1: Data robustness analysis"""
    from .sa_problem12_data_robustness import run_data_robustness as _run
    return _run(*args, **kwargs)


def run_weight_tornado(*args, **kwargs):
    """SA-2: Weight tornado chart analysis"""
    from .sa_problem12_weight_tornado import run_weight_tornado as _run
    return _run(*args, **kwargs)


def run_glv_parameter_sensitivity(*args, **kwargs):
    """SA-3: GLV parameter sensitivity analysis"""
    from .sa_problem3_glv_parameters import run_glv_parameter_sensitivity as _run
    return _run(*args, **kwargs)


def run_budget_sensitivity(*args, **kwargs):
    """SA-4: Budget/constraint sensitivity analysis"""
    from .sa_problem4_budget_scenarios import run_budget_sensitivity as _run
    return _run(*args, **kwargs)


def run_all():
    """Run all sensitivity analyses"""
    from .sa_main import run_all_tasks
    return run_all_tasks()


__all__ = [
    'run_data_robustness',
    'run_weight_tornado',
    'run_glv_parameter_sensitivity',
    'run_budget_sensitivity',
    'run_all',
]

__version__ = '1.0.0'
