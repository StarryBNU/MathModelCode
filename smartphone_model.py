"""
智能手机多组件功耗模型
考虑屏幕、CPU、网络、GPS等组件的电流消耗

功耗分解：
I_total = I_screen + I_CPU + I_network + I_GPS + I_background + I_idle
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from battery_model_advanced import BatteryModel2RC

class SmartphoneComponents:
    """智能手机组件功耗模型"""
    
    def __init__(self):
        # 屏幕参数
        self.screen_type = 'OLED'  # 'OLED' or 'LCD'
        self.screen_size = 6.5  # 英寸
        self.resolution = (1080, 2400)  # 像素
        self.max_brightness = 500  # nits
        
        # 功耗参数（基于实验测量，单位：mA）
        self.I_screen_base = 50  # 屏幕基础功耗（黑屏但开启）
        self.I_screen_per_nit = 0.5  # 每nit亮度的额外功耗
        self.I_refresh_120hz = 80  # 120Hz额外功耗
        self.I_refresh_60hz = 0   # 60Hz基准
        
        # CPU功耗
        self.I_cpu_idle = 30  # 空闲功耗
        self.I_cpu_max = 1200  # 满载功耗
        
        # 网络功耗
        self.I_wifi_idle = 10
        self.I_wifi_active = 150
        self.I_4g_idle = 20
        self.I_4g_active = 300
        self.I_5g_idle = 30
        self.I_5g_active = 450
        
        # GPS功耗
        self.I_gps = 100
        
        # 其他组件
        self.I_idle = 20  # 基础待机（操作系统等）
        self.I_background_per_app = 5  # 每个后台应用
    
    def screen_current(self, brightness_percent, refresh_rate=60, content_brightness=0.5):
        """
        屏幕电流消耗
        
        参数:
            brightness_percent: 亮度百分比 (0-100)
            refresh_rate: 刷新率 (60 or 120 Hz)
            content_brightness: 内容平均亮度 (0-1，OLED专用)
        
        返回:
            电流 (mA)
        """
        # 基础功耗
        I = self.I_screen_base
        
        # 亮度相关
        brightness_nits = self.max_brightness * (brightness_percent / 100)
        I += brightness_nits * self.I_screen_per_nit
        
        # OLED特性：内容亮度影响（暗像素省电）
        if self.screen_type == 'OLED':
            I *= content_brightness
        
        # 刷新率
        if refresh_rate == 120:
            I += self.I_refresh_120hz
        
        return I
    
    def cpu_current(self, utilization, temperature=25):
        """
        CPU电流消耗
        
        参数:
            utilization: CPU利用率 (0-1)
            temperature: 温度 (°C)
        
        返回:
            电流 (mA)
        """
        # 基础：线性模型
        I = self.I_cpu_idle + (self.I_cpu_max - self.I_cpu_idle) * utilization
        
        # 温度throttling（超过40°C降频）
        if temperature > 40:
            throttle_factor = 1 - 0.01 * (temperature - 40)
            throttle_factor = max(0.5, throttle_factor)  # 最多降至50%
            I *= throttle_factor
        
        return I
    
    def network_current(self, network_type, data_rate_mbps=0):
        """
        网络电流消耗
        
        参数:
            network_type: 'wifi', '4g', '5g', 'airplane'
            data_rate_mbps: 数据传输速率 (Mbps)
        
        返回:
            电流 (mA)
        """
        if network_type == 'airplane':
            return 0
        elif network_type == 'wifi':
            if data_rate_mbps > 0:
                return self.I_wifi_active
            else:
                return self.I_wifi_idle
        elif network_type == '4g':
            if data_rate_mbps > 0:
                return self.I_4g_active
            else:
                return self.I_4g_idle
        elif network_type == '5g':
            if data_rate_mbps > 0:
                return self.I_5g_active
            else:
                return self.I_5g_idle
        else:
            return 0
    
    def total_current(self, scenario):
        """
        计算总电流
        
        参数:
            scenario: 字典，包含各组件状态
                {
                    'screen_on': bool,
                    'brightness': 0-100,
                    'refresh_rate': 60 or 120,
                    'content_brightness': 0-1,
                    'cpu_utilization': 0-1,
                    'temperature': float,
                    'network_type': str,
                    'data_rate': float,
                    'gps_on': bool,
                    'background_apps': int
                }
        
        返回:
            总电流 (A)
        """
        I_total = self.I_idle
        
        # 屏幕
        if scenario.get('screen_on', False):
            I_total += self.screen_current(
                scenario.get('brightness', 50),
                scenario.get('refresh_rate', 60),
                scenario.get('content_brightness', 0.5)
            )
        
        # CPU
        I_total += self.cpu_current(
            scenario.get('cpu_utilization', 0.1),
            scenario.get('temperature', 25)
        )
        
        # 网络
        I_total += self.network_current(
            scenario.get('network_type', 'wifi'),
            scenario.get('data_rate', 0)
        )
        
        # GPS
        if scenario.get('gps_on', False):
            I_total += self.I_gps
        
        # 后台应用
        I_total += scenario.get('background_apps', 5) * self.I_background_per_app
        
        return I_total / 1000  # 转换为A


class SmartphoneBatteryModel:
    """完整的智能手机电池模型（组件+电池）"""
    
    def __init__(self):
        self.battery = BatteryModel2RC(Q_max=3000, R0=0.05, R1=0.03, 
                                       C1=2000, R2=0.02, C2=5000)
        self.components = SmartphoneComponents()
    
    def simulate_usage(self, usage_timeline, duration):
        """
        模拟手机使用
        
        参数:
            usage_timeline: 列表，每个元素是 (time, scenario)
            duration: 总时长 (秒)
        
        返回:
            t, SOC, V_terminal, I, scenarios
        """
        # 创建分段电流函数
        def current_profile(t):
            # 找到当前时间对应的场景
            current_scenario = usage_timeline[0][1]
            for time_point, scenario in usage_timeline:
                if t >= time_point:
                    current_scenario = scenario
                else:
                    break
            return self.components.total_current(current_scenario)
        
        # 模拟电池
        t, SOC, V1, V2, V_terminal, I = self.battery.simulate(
            [0, duration], current_profile, SOC0=1.0
        )
        
        return t, SOC, V_terminal, I, usage_timeline
    
    def predict_time_to_empty(self, scenario):
        """
        预测特定使用场景下的续航时间
        
        参数:
            scenario: 使用场景字典
        
        返回:
            续航时间 (小时)
        """
        def constant_usage(t):
            return self.components.total_current(scenario)
        
        max_time = 48 * 3600  # 最多48小时
        
        def event_cutoff(t, state):
            soc, v1, v2 = state
            I = constant_usage(t)
            V = self.battery.terminal_voltage(soc, v1, v2, I)
            return V - 3.0  # 截止电压
        
        event_cutoff.terminal = True
        event_cutoff.direction = -1
        
        sol = solve_ivp(
            fun=lambda t, y: self.battery.state_equations(t, y, constant_usage),
            t_span=[0, max_time],
            y0=[1.0, 0.0, 0.0],
            events=event_cutoff,
            method='RK45'
        )
        
        if sol.t_events[0].size > 0:
            return sol.t_events[0][0] / 3600  # 转换为小时
        else:
            return max_time / 3600


# ============ 示例使用 ============

def example_daily_usage():
    """示例1：模拟一天的手机使用"""
    print("=" * 60)
    print("示例1：模拟典型一天的手机使用")
    print("=" * 60)
    
    model = SmartphoneBatteryModel()
    
    # 定义一天的使用场景（时间单位：秒）
    usage_timeline = [
        # 早上7点：起床，查看通知
        (0, {
            'screen_on': True, 'brightness': 30, 'refresh_rate': 60,
            'content_brightness': 0.3, 'cpu_utilization': 0.2,
            'network_type': 'wifi', 'data_rate': 1, 'gps_on': False,
            'background_apps': 10, 'temperature': 25
        }),
        
        # 7:30 - 8:00: 通勤（听音乐，屏幕关闭）
        (1800, {
            'screen_on': False, 'cpu_utilization': 0.1,
            'network_type': '4g', 'data_rate': 0.5, 'gps_on': False,
            'background_apps': 10, 'temperature': 25
        }),
        
        # 8:00 - 12:00: 工作（偶尔查看）
        (3600, {
            'screen_on': False, 'cpu_utilization': 0.05,
            'network_type': 'wifi', 'data_rate': 0, 'gps_on': False,
            'background_apps': 15, 'temperature': 25
        }),
        
        # 12:00 - 13:00: 午餐刷视频
        (18000, {
            'screen_on': True, 'brightness': 80, 'refresh_rate': 60,
            'content_brightness': 0.7, 'cpu_utilization': 0.4,
            'network_type': 'wifi', 'data_rate': 5, 'gps_on': False,
            'background_apps': 15, 'temperature': 30
        }),
        
        # 13:00 - 18:00: 继续工作
        (21600, {
            'screen_on': False, 'cpu_utilization': 0.05,
            'network_type': 'wifi', 'data_rate': 0, 'gps_on': False,
            'background_apps': 15, 'temperature': 25
        }),
        
        # 18:00 - 19:00: 通勤（导航+音乐）
        (39600, {
            'screen_on': True, 'brightness': 100, 'refresh_rate': 60,
            'content_brightness': 0.8, 'cpu_utilization': 0.3,
            'network_type': '4g', 'data_rate': 2, 'gps_on': True,
            'background_apps': 15, 'temperature': 28
        }),
        
        # 19:00 - 21:00: 晚餐和休闲
        (43200, {
            'screen_on': True, 'brightness': 50, 'refresh_rate': 60,
            'content_brightness': 0.5, 'cpu_utilization': 0.2,
            'network_type': 'wifi', 'data_rate': 1, 'gps_on': False,
            'background_apps': 15, 'temperature': 25
        }),
        
        # 21:00 - 23:00: 玩游戏（高负载）
        (50400, {
            'screen_on': True, 'brightness': 70, 'refresh_rate': 120,
            'content_brightness': 0.8, 'cpu_utilization': 0.8,
            'network_type': 'wifi', 'data_rate': 3, 'gps_on': False,
            'background_apps': 15, 'temperature': 38
        }),
        
        # 23:00 - 7:00: 睡觉（待机）
        (57600, {
            'screen_on': False, 'cpu_utilization': 0.02,
            'network_type': 'wifi', 'data_rate': 0, 'gps_on': False,
            'background_apps': 10, 'temperature': 20
        }),
    ]
    
    # 模拟24小时
    t, SOC, V, I, timeline = model.simulate_usage(usage_timeline, 24*3600)
    
    # 绘图
    fig, axes = plt.subplots(4, 1, figsize=(14, 12))
    
    # 电流消耗
    axes[0].plot(t/3600, I*1000, 'b-', linewidth=1)
    axes[0].set_ylabel('Current (mA)')
    axes[0].set_title('Current Consumption Throughout the Day')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(0, 24)
    
    # 添加场景标注
    scenario_names = ['起床', '通勤', '工作', '午餐', '工作', '通勤', '晚餐', '游戏', '睡觉']
    for i, (time, scenario) in enumerate(timeline):
        if i < len(scenario_names):
            axes[0].axvline(x=time/3600, color='gray', linestyle='--', alpha=0.5)
            axes[0].text(time/3600, axes[0].get_ylim()[1]*0.9, 
                        scenario_names[i], fontsize=8, rotation=45)
    
    # SOC
    axes[1].plot(t/3600, SOC*100, 'g-', linewidth=2)
    axes[1].set_ylabel('SOC (%)')
    axes[1].set_title('Battery State of Charge')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(0, 24)
    axes[1].axhline(y=20, color='r', linestyle='--', alpha=0.7, label='Low Battery')
    axes[1].legend()
    
    # 电压
    axes[2].plot(t/3600, V, 'r-', linewidth=1.5)
    axes[2].set_ylabel('Voltage (V)')
    axes[2].set_title('Terminal Voltage')
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xlim(0, 24)
    axes[2].axhline(y=3.0, color='k', linestyle='--', alpha=0.7, label='Cutoff')
    axes[2].legend()
    
    # 功率
    Power = V * I
    axes[3].plot(t/3600, Power, 'purple', linewidth=1.5)
    axes[3].set_ylabel('Power (W)')
    axes[3].set_xlabel('Time (hours)')
    axes[3].set_title('Power Consumption')
    axes[3].grid(True, alpha=0.3)
    axes[3].set_xlim(0, 24)
    
    plt.tight_layout()
    plt.savefig('smartphone_daily_usage.png', dpi=300, bbox_inches='tight')
    
    print(f"最终SOC: {SOC[-1]*100:.1f}%")
    print(f"24小时后剩余电量: {SOC[-1]*100:.1f}%")
    print("图表已保存: smartphone_daily_usage.png\n")


def example_scenario_comparison():
    """示例2：不同使用场景的续航时间对比"""
    print("\n" + "=" * 60)
    print("示例2：不同使用场景的续航时间预测")
    print("=" * 60)
    
    model = SmartphoneBatteryModel()
    
    scenarios = {
        '待机（WiFi）': {
            'screen_on': False, 'cpu_utilization': 0.02,
            'network_type': 'wifi', 'data_rate': 0, 'gps_on': False,
            'background_apps': 5, 'temperature': 25
        },
        '待机（4G）': {
            'screen_on': False, 'cpu_utilization': 0.02,
            'network_type': '4g', 'data_rate': 0, 'gps_on': False,
            'background_apps': 5, 'temperature': 25
        },
        '浏览网页': {
            'screen_on': True, 'brightness': 50, 'refresh_rate': 60,
            'content_brightness': 0.5, 'cpu_utilization': 0.2,
            'network_type': 'wifi', 'data_rate': 1, 'gps_on': False,
            'background_apps': 10, 'temperature': 25
        },
        '看视频（WiFi）': {
            'screen_on': True, 'brightness': 70, 'refresh_rate': 60,
            'content_brightness': 0.7, 'cpu_utilization': 0.4,
            'network_type': 'wifi', 'data_rate': 5, 'gps_on': False,
            'background_apps': 10, 'temperature': 30
        },
        '看视频（5G）': {
            'screen_on': True, 'brightness': 70, 'refresh_rate': 60,
            'content_brightness': 0.7, 'cpu_utilization': 0.4,
            'network_type': '5g', 'data_rate': 5, 'gps_on': False,
            'background_apps': 10, 'temperature': 32
        },
        '玩游戏（60Hz）': {
            'screen_on': True, 'brightness': 80, 'refresh_rate': 60,
            'content_brightness': 0.9, 'cpu_utilization': 0.8,
            'network_type': 'wifi', 'data_rate': 2, 'gps_on': False,
            'background_apps': 10, 'temperature': 38
        },
        '玩游戏（120Hz）': {
            'screen_on': True, 'brightness': 80, 'refresh_rate': 120,
            'content_brightness': 0.9, 'cpu_utilization': 0.8,
            'network_type': 'wifi', 'data_rate': 2, 'gps_on': False,
            'background_apps': 10, 'temperature': 40
        },
        '导航（GPS）': {
            'screen_on': True, 'brightness': 100, 'refresh_rate': 60,
            'content_brightness': 0.8, 'cpu_utilization': 0.3,
            'network_type': '4g', 'data_rate': 1, 'gps_on': True,
            'background_apps': 10, 'temperature': 28
        },
    }
    
    results = []
    for name, scenario in scenarios.items():
        tte = model.predict_time_to_empty(scenario)
        current = model.components.total_current(scenario) * 1000  # mA
        results.append((name, tte, current))
        print(f"{name:20s}: {tte:6.2f} 小时 (平均电流: {current:6.1f} mA)")
    
    # 绘制对比图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    names = [r[0] for r in results]
    times = [r[1] for r in results]
    currents = [r[2] for r in results]
    
    # 续航时间
    colors = plt.cm.RdYlGn([t/max(times) for t in times])
    ax1.barh(names, times, color=colors)
    ax1.set_xlabel('Time-to-Empty (hours)')
    ax1.set_title('Battery Life Comparison')
    ax1.grid(True, alpha=0.3, axis='x')
    
    # 平均电流
    colors2 = plt.cm.YlOrRd([c/max(currents) for c in currents])
    ax2.barh(names, currents, color=colors2)
    ax2.set_xlabel('Average Current (mA)')
    ax2.set_title('Power Consumption Comparison')
    ax2.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig('smartphone_scenario_comparison.png', dpi=300, bbox_inches='tight')
    print("\n图表已保存: smartphone_scenario_comparison.png")


def example_optimization_recommendations():
    """示例3：节电优化建议"""
    print("\n" + "=" * 60)
    print("示例3：节电优化建议（敏感性分析）")
    print("=" * 60)
    
    model = SmartphoneBatteryModel()
    
    # 基准场景：看视频
    baseline = {
        'screen_on': True, 'brightness': 80, 'refresh_rate': 120,
        'content_brightness': 0.7, 'cpu_utilization': 0.4,
        'network_type': '5g', 'data_rate': 5, 'gps_on': False,
        'background_apps': 15, 'temperature': 30
    }
    
    baseline_tte = model.predict_time_to_empty(baseline)
    baseline_current = model.components.total_current(baseline) * 1000
    
    print(f"基准场景（看视频 - 最差设置）:")
    print(f"  续航时间: {baseline_tte:.2f} 小时")
    print(f"  平均电流: {baseline_current:.1f} mA\n")
    
    print("优化建议（单项改变）:")
    
    optimizations = [
        ('降低亮度 (80% → 50%)', {'brightness': 50}),
        ('切换到60Hz刷新率', {'refresh_rate': 60}),
        ('切换到WiFi', {'network_type': 'wifi'}),
        ('关闭后台应用 (15 → 5)', {'background_apps': 5}),
        ('降低亮度+60Hz', {'brightness': 50, 'refresh_rate': 60}),
        ('全部优化', {'brightness': 50, 'refresh_rate': 60, 
                    'network_type': 'wifi', 'background_apps': 5}),
    ]
    
    improvements = []
    
    for name, changes in optimizations:
        optimized = baseline.copy()
        optimized.update(changes)
        
        opt_tte = model.predict_time_to_empty(optimized)
        opt_current = model.components.total_current(optimized) * 1000
        
        improvement = (opt_tte - baseline_tte) / baseline_tte * 100
        current_reduction = (baseline_current - opt_current) / baseline_current * 100
        
        improvements.append((name, improvement, current_reduction))
        
        print(f"  {name}:")
        print(f"    续航提升: +{improvement:.1f}% ({opt_tte:.2f}小时)")
        print(f"    电流降低: -{current_reduction:.1f}% ({opt_current:.1f}mA)\n")
    
    # 绘制优化效果
    names = [i[0] for i in improvements]
    gains = [i[1] for i in improvements]
    
    plt.figure(figsize=(12, 6))
    colors = plt.cm.Greens([0.3 + 0.7*g/max(gains) for g in gains])
    bars = plt.barh(names, gains, color=colors)
    plt.xlabel('Battery Life Improvement (%)')
    plt.title('Impact of Different Optimizations on Battery Life')
    plt.grid(True, alpha=0.3, axis='x')
    
    # 添加数值标签
    for bar, gain in zip(bars, gains):
        plt.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, 
                f'+{gain:.1f}%', va='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('smartphone_optimization.png', dpi=300, bbox_inches='tight')
    print("图表已保存: smartphone_optimization.png")


if __name__ == "__main__":
    example_daily_usage()
    example_scenario_comparison()
    example_optimization_recommendations()
    
    print("\n" + "=" * 60)
    print("智能手机模型示例运行完成！")
    print("=" * 60)
