# -*- coding: utf-8 -*-
"""
问题三运行脚本
==============

运行 GLV-SD 模型预测 2026-2035 年 AI 竞争力
"""

import sys
import os

# 添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.models.problem3.glv_forecast import main

if __name__ == '__main__':
    print("=" * 60)
    print("运行问题三：GLV-SD 分层生态演化模型")
    print("=" * 60)
    
    results = main()
    
    print("\n运行完成！")
    print(f"结果已保存到: outputs/problem3/")
