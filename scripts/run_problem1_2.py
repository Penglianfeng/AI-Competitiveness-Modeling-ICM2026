#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
运行问题一&二解决方案
====================
入口脚本，用于执行问题一和问题二的完整求解流程

Usage:
    python scripts/run_problem1_2.py
"""

import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

# 导入并运行主函数
from src.models.problem1_2.solution import main

if __name__ == '__main__':
    print(f"项目根目录: {PROJECT_ROOT}")
    results = main()
