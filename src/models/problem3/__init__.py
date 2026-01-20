# -*- coding: utf-8 -*-
"""
问题三模块：基于广义 Lotka-Volterra 的分层生态演化模型
===========================================================

预测 2026-2035 年全球 AI 竞争力发展趋势
"""

from .glv_forecast import main as run_glv_forecast
from .glv_forecast import GLVParameters

__all__ = ['run_glv_forecast', 'GLVParameters']
