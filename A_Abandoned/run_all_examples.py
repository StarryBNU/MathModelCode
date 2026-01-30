"""
运行所有电池模型示例
生成完整的结果报告
"""

import sys
import time
from datetime import datetime

def print_header(title):
    """打印美化的标题"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")

def main():
    print_header("智能手机电池建模 - 完整示例演示")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n本程序将运行所有建模示例，生成完整的分析结果。")
    print("预计用时: 2-3分钟")
    print("\n" + "-" * 70)
    
    results = []
    
    # ========== 1. 基础模型 ==========
    print_header("1/4 - 运行基础模型 (一阶RC)")
    start = time.time()
    try:
        import battery_model_basic
        elapsed = time.time() - start
        results.append(("基础模型", "成功", elapsed))
        print(f"✓ 完成 (耗时: {elapsed:.2f}秒)")
    except Exception as e:
        elapsed = time.time() - start
        results.append(("基础模型", f"失败: {str(e)}", elapsed))
        print(f"✗ 失败: {str(e)}")
    
    # ========== 2. 高级模型 ==========
    print_header("2/4 - 运行高级模型 (二阶RC + 参数估计)")
    start = time.time()
    try:
        import battery_model_advanced
        elapsed = time.time() - start
        results.append(("高级模型", "成功", elapsed))
        print(f"✓ 完成 (耗时: {elapsed:.2f}秒)")
    except Exception as e:
        elapsed = time.time() - start
        results.append(("高级模型", f"失败: {str(e)}", elapsed))
        print(f"✗ 失败: {str(e)}")
    
    # ========== 3. 智能手机模型 ==========
    print_header("3/4 - 运行智能手机完整模型")
    start = time.time()
    try:
        import smartphone_model
        elapsed = time.time() - start
        results.append(("智能手机模型", "成功", elapsed))
        print(f"✓ 完成 (耗时: {elapsed:.2f}秒)")
    except Exception as e:
        elapsed = time.time() - start
        results.append(("智能手机模型", f"失败: {str(e)}", elapsed))
        print(f"✗ 失败: {str(e)}")
    
    # ========== 4. 温度和老化模型 ==========
    print_header("4/4 - 运行温度和老化模型")
    start = time.time()
    try:
        import temperature_aging_model
        elapsed = time.time() - start
        results.append(("温度老化模型", "成功", elapsed))
        print(f"✓ 完成 (耗时: {elapsed:.2f}秒)")
    except Exception as e:
        elapsed = time.time() - start
        results.append(("温度老化模型", f"失败: {str(e)}", elapsed))
        print(f"✗ 失败: {str(e)}")
    
    # ========== 生成总结报告 ==========
    print_header("执行总结")
    
    total_time = sum(r[2] for r in results)
    success_count = sum(1 for r in results if r[1] == "成功")
    
    print("\n模块执行情况:")
    print("-" * 70)
    for name, status, elapsed in results:
        status_symbol = "✓" if status == "成功" else "✗"
        print(f"  {status_symbol} {name:20s}: {status:20s} ({elapsed:.2f}秒)")
    
    print("\n" + "-" * 70)
    print(f"\n总计: {success_count}/{len(results)} 个模块成功")
    print(f"总耗时: {total_time:.2f} 秒")
    
    if success_count == len(results):
        print("\n🎉 所有模块运行成功！")
        print("\n生成的图表文件:")
        print("  1. battery_constant_discharge.png - 恒定电流放电")
        print("  2. battery_smartphone_usage.png - 手机使用场景")
        print("  3. battery_sensitivity.png - 敏感性分析")
        print("  4. battery_model_comparison.png - 模型对比")
        print("  5. smartphone_daily_usage.png - 24小时使用模拟")
        print("  6. smartphone_scenario_comparison.png - 场景对比")
        print("  7. smartphone_optimization.png - 优化建议")
        print("  8. battery_temperature_effect.png - 温度影响")
        print("  9. battery_thermal_dynamics.png - 热动态")
        print(" 10. battery_aging.png - 老化模拟")
        
        print("\n📊 接下来可以:")
        print("  1. 查看生成的图表文件")
        print("  2. 阅读 README.md 了解详细说明")
        print("  3. 修改参数重新运行单个模块")
        print("  4. 基于这些代码开发自己的模型")
    else:
        print("\n⚠️  部分模块运行失败，请检查错误信息")
        print("常见问题:")
        print("  - 确保已安装: numpy, scipy, matplotlib")
        print("  - 检查 Python 版本 (建议 3.8+)")
    
    print("\n" + "=" * 70)
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  程序被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n✗ 发生未预期的错误: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
