# -*- coding: utf-8 -*-
"""
敏感性分析主运行脚本
Sensitivity Analysis Main Runner
================================

支持命令行参数：
- --all: 运行所有敏感性分析
- --task SA1|SA2|SA3|SA4: 运行指定的分析任务

Usage:
    python -m sensitivity_analysis.sa_main --all
    python -m sensitivity_analysis.sa_main --task SA1
    python -m sensitivity_analysis.sa_main --task SA1 SA3

Author: AI Modeling Assistant
Date: January 2026
"""

import argparse
import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def get_base_path() -> Path:
    """动态获取项目根目录"""
    current_file = Path(__file__).resolve()
    for parent in current_file.parents:
        if (parent / 'configs').exists() or (parent / 'outputs').exists():
            return parent
    return Path.cwd()


BASE_PATH = get_base_path()
OUTPUT_PATH = BASE_PATH / 'outputs' / 'sensitivity_analysis'


# =============================================================================
# 任务定义
# =============================================================================

TASKS = {
    'SA1': {
        'name': '数据口径敏感性分析',
        'name_en': 'Data Robustness Analysis (Problem 1&2)',
        'description': '比较2024 vs 2025数据口径，Monte Carlo噪声扰动分析',
        'function': 'run_data_robustness',
        'module': 'sa_problem12_data_robustness'
    },
    'SA2': {
        'name': '权重龙卷风图分析',
        'name_en': 'Weight Tornado Chart Analysis (Problem 1&2)',
        'description': '指标权重敏感性分析，生成龙卷风图',
        'function': 'run_weight_tornado',
        'module': 'sa_problem12_weight_tornado'
    },
    'SA3': {
        'name': 'GLV参数敏感性分析',
        'name_en': 'GLV Parameter Sensitivity Analysis (Problem 3)',
        'description': 'Sobol敏感性指数，预测轨迹不确定性分析',
        'function': 'run_glv_parameter_sensitivity',
        'module': 'sa_problem3_glv_parameters'
    },
    'SA4': {
        'name': '预算/约束敏感性分析',
        'name_en': 'Budget Sensitivity Analysis (Problem 4)',
        'description': '多场景预算分析，Pareto前沿计算',
        'function': 'run_budget_sensitivity',
        'module': 'sa_problem4_budget_scenarios'
    }
}


# =============================================================================
# 任务执行器
# =============================================================================

def run_task(task_id: str, output_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    运行指定的敏感性分析任务
    
    Args:
        task_id: 任务ID (SA1, SA2, SA3, SA4)
        output_dir: 输出目录
    
    Returns:
        Dict: 任务执行结果
    """
    if task_id not in TASKS:
        raise ValueError(f"未知任务ID: {task_id}. 有效值: {list(TASKS.keys())}")
    
    task_info = TASKS[task_id]
    logger.info(f"\n{'='*60}")
    logger.info(f"执行任务: {task_id} - {task_info['name']}")
    logger.info(f"描述: {task_info['description']}")
    logger.info(f"{'='*60}")
    
    # 动态导入模块
    module_name = f".{task_info['module']}"
    try:
        import importlib
        module = importlib.import_module(module_name, package='sensitivity_analysis')
        func = getattr(module, task_info['function'])
    except ImportError as e:
        logger.error(f"无法导入模块 {module_name}: {e}")
        return {'status': 'error', 'message': str(e)}
    except AttributeError as e:
        logger.error(f"模块中找不到函数 {task_info['function']}: {e}")
        return {'status': 'error', 'message': str(e)}
    
    # 执行任务
    start_time = datetime.now()
    try:
        results = func(output_dir=output_dir)
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        logger.info(f"✅ 任务 {task_id} 完成！耗时: {duration:.1f}秒")
        
        return {
            'status': 'success',
            'task_id': task_id,
            'duration_seconds': duration,
            'results': results
        }
    
    except Exception as e:
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        logger.error(f"❌ 任务 {task_id} 失败: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            'status': 'error',
            'task_id': task_id,
            'duration_seconds': duration,
            'message': str(e)
        }


def run_all_tasks(output_dir: Optional[Path] = None) -> Dict[str, Dict]:
    """
    运行所有敏感性分析任务
    
    Args:
        output_dir: 输出目录
    
    Returns:
        Dict: 所有任务执行结果
    """
    logger.info("\n" + "="*70)
    logger.info("🚀 开始运行全部敏感性分析任务")
    logger.info("="*70)
    
    all_results = {}
    total_start = datetime.now()
    
    for task_id in TASKS.keys():
        result = run_task(task_id, output_dir)
        all_results[task_id] = result
    
    total_end = datetime.now()
    total_duration = (total_end - total_start).total_seconds()
    
    # 汇总
    logger.info("\n" + "="*70)
    logger.info("📊 敏感性分析执行汇总")
    logger.info("="*70)
    
    n_success = sum(1 for r in all_results.values() if r['status'] == 'success')
    n_failed = sum(1 for r in all_results.values() if r['status'] == 'error')
    
    for task_id, result in all_results.items():
        status_icon = "✅" if result['status'] == 'success' else "❌"
        duration = result.get('duration_seconds', 0)
        logger.info(f"  {status_icon} {task_id}: {TASKS[task_id]['name']} ({duration:.1f}s)")
    
    logger.info(f"\n总耗时: {total_duration:.1f}秒")
    logger.info(f"成功: {n_success}/{len(TASKS)}, 失败: {n_failed}/{len(TASKS)}")
    
    return all_results


# =============================================================================
# 报告生成
# =============================================================================

def generate_summary_report(
    results: Dict[str, Dict],
    output_dir: Path
) -> str:
    """
    生成敏感性分析汇总报告（Markdown格式）
    
    Args:
        results: 所有任务执行结果
        output_dir: 输出目录
    
    Returns:
        str: 报告文件路径
    """
    report_path = output_dir / 'sensitivity_analysis_report.md'
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# 敏感性分析报告 / Sensitivity Analysis Report\n\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## 概述 / Overview\n\n")
        f.write("本报告汇总了全球AI竞争力评估模型的敏感性分析结果，")
        f.write("涵盖问题1-4的关键参数稳健性检验。\n\n")
        
        f.write("## 任务执行状态 / Task Execution Status\n\n")
        f.write("| 任务ID | 名称 | 状态 | 耗时(秒) |\n")
        f.write("|--------|------|------|----------|\n")
        
        for task_id, result in results.items():
            task_info = TASKS[task_id]
            status = "✅ 成功" if result['status'] == 'success' else "❌ 失败"
            duration = result.get('duration_seconds', 0)
            f.write(f"| {task_id} | {task_info['name']} | {status} | {duration:.1f} |\n")
        
        f.write("\n## 生成的图表 / Generated Figures\n\n")
        f.write("1. **fig_sa1_data_robustness.png**: 数据口径敏感性分析（2024 vs 2025）\n")
        f.write("2. **fig_sa2_weight_tornado.png**: 权重龙卷风图（中美对比）\n")
        f.write("3. **fig_sa3_glv_sobol_heatmap.png**: GLV Sobol敏感性指数热力图\n")
        f.write("4. **fig_sa4_trajectory_bands.png**: 预测轨迹不确定性带\n")
        f.write("5. **fig_sa5_budget_scenarios.png**: 多场景预算分配堆叠面积图\n")
        f.write("6. **fig_sa6_pareto_frontier.png**: 预算-绩效Pareto前沿\n")
        
        f.write("\n## 生成的表格 / Generated Tables\n\n")
        f.write("- `sa1_ranking_stability.csv`: 排名稳定性分析结果\n")
        f.write("- `sa2_weight_sensitivity_CHN.csv`: 中国权重敏感性\n")
        f.write("- `sa2_weight_sensitivity_USA.csv`: 美国权重敏感性\n")
        f.write("- `sa3_sobol_indices.csv`: Sobol敏感性指数\n")
        f.write("- `sa3_trajectory_statistics.csv`: 轨迹统计量\n")
        f.write("- `sa4_optimal_allocation_*.csv`: 各场景最优分配\n")
        f.write("- `sa4_budget_elasticity.csv`: 预算弹性分析\n")
        f.write("- `sa4_pareto_frontier.csv`: Pareto前沿数据\n")
        
        # 如果有具体结果，添加关键发现
        f.write("\n## 关键发现 / Key Findings\n\n")
        
        for task_id, result in results.items():
            if result['status'] == 'success' and 'results' in result:
                task_info = TASKS[task_id]
                f.write(f"### {task_id}: {task_info['name']}\n\n")
                
                task_results = result['results']
                
                if task_id == 'SA1' and task_results:
                    if 'ranking_stability' in task_results:
                        f.write("- 排名稳定性分析完成\n")
                
                elif task_id == 'SA2' and task_results:
                    if 'key_indicators_chn' in task_results:
                        f.write(f"- 中国最敏感指标: {', '.join(task_results['key_indicators_chn'][:3])}\n")
                    if 'key_indicators_usa' in task_results:
                        f.write(f"- 美国最敏感指标: {', '.join(task_results['key_indicators_usa'][:3])}\n")
                
                elif task_id == 'SA3' and task_results:
                    if 'most_sensitive_param' in task_results:
                        f.write(f"- 最敏感参数: {task_results['most_sensitive_param']}\n")
                    if 'highest_uncertainty_country' in task_results:
                        f.write(f"- 预测不确定性最高国家: {task_results['highest_uncertainty_country']}\n")
                
                f.write("\n")
        
        f.write("\n---\n")
        f.write("*本报告由敏感性分析模块自动生成*\n")
    
    logger.info(f"📝 报告已生成: {report_path}")
    return str(report_path)


# =============================================================================
# 命令行入口
# =============================================================================

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='敏感性分析主运行脚本 / Sensitivity Analysis Runner',
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument(
        '--all',
        action='store_true',
        help='运行所有敏感性分析任务'
    )
    
    parser.add_argument(
        '--task',
        nargs='+',
        choices=['SA1', 'SA2', 'SA3', 'SA4'],
        help='运行指定的分析任务（可多选）\n'
             '  SA1: 数据口径敏感性分析\n'
             '  SA2: 权重龙卷风图分析\n'
             '  SA3: GLV参数敏感性分析\n'
             '  SA4: 预算/约束敏感性分析'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='输出目录（默认: outputs/sensitivity_analysis）'
    )
    
    parser.add_argument(
        '--list',
        action='store_true',
        help='列出所有可用任务'
    )
    
    parser.add_argument(
        '--no-report',
        action='store_true',
        help='不生成汇总报告'
    )
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    # 设置输出目录
    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_PATH
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'figures').mkdir(parents=True, exist_ok=True)
    (output_dir / 'tables').mkdir(parents=True, exist_ok=True)
    
    # 列出任务
    if args.list:
        print("\n可用敏感性分析任务:")
        print("-" * 60)
        for task_id, info in TASKS.items():
            print(f"  {task_id}: {info['name']}")
            print(f"       {info['description']}")
            print()
        return
    
    # 运行任务
    results = {}
    
    if args.all:
        results = run_all_tasks(output_dir)
    elif args.task:
        for task_id in args.task:
            results[task_id] = run_task(task_id, output_dir)
    else:
        # 默认运行所有任务
        print("提示: 未指定任务，使用 --all 运行全部或 --task 指定任务")
        print("使用 --list 查看可用任务，--help 查看帮助")
        return
    
    # 生成报告
    if not args.no_report and results:
        generate_summary_report(results, output_dir)
    
    # 返回状态码
    n_failed = sum(1 for r in results.values() if r['status'] == 'error')
    sys.exit(1 if n_failed > 0 else 0)


if __name__ == '__main__':
    main()
