#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
运行所有数据爬虫脚本
====================
批量执行数据采集任务

Usage:
    python scripts/run_scrapers.py
"""

import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.scrapers import ai_talent_scraper
from src.data.scrapers import uis_rd_scraper
from src.data.scrapers import world_bank_scraper


def main():
    print("=" * 60)
    print("数据爬虫脚本运行器")
    print("=" * 60)
    
    # TODO: 根据需要运行特定的爬虫
    print("\n可用的爬虫脚本:")
    print("  1. ai_talent_scraper - AI人才数据")
    print("  2. uis_rd_scraper - UNESCO R&D数据")
    print("  3. world_bank_scraper - 世界银行数据")
    
    print("\n请直接运行相应的爬虫脚本，例如:")
    print("  python -m src.data.scrapers.ai_talent_scraper")


if __name__ == '__main__':
    main()
