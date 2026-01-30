"""
基础电池模型 - 一阶等效电路（Thevenin模型）
适合新手入门和快速原型

物理模型：
- OCV(SOC): 开路电压
- R0: 欧姆电阻
- R1, C1: 极化电阻和电容

状态方程：
dSOC/dt = -I(t) / Q_max
dV1/dt = -V1/(R1*C1) + I(t)/C1
V_terminal = OCV(SOC) - I*R0 - V1
"""

import numpy as np
from scipy.integrate import odeint, solve_ivp
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

class BatteryModel1RC:
    """一阶等效电路电池模型"""
    
    def __init__(self, Q_max=3000, R0=0.05, R1=0.03, C1=2000):
        """
        参数:
            Q_max: 最大容量 (mAh)
            R0: 欧姆电阻 (Ohm)
            R1: 极化电阻 (Ohm)
            C1: 极化电容 (F)
        """
        self.Q_max = Q_max  # mAh
        self.R0 = R0
        self.R1 = R1
        self.C1 = C1
        self.tau1 = R1 * C1  # 时间常数
        
        # OCV-SOC查找表（基于实验数据）
        # 这是锂离子电池的典型OCV曲线
        self.soc_points = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        self.ocv_points = np.array([3.0, 3.3, 3.5, 3.6, 3.65, 3.7, 3.75, 3.85, 4.0, 4.1, 4.2])
        self.ocv_func = interp1d(self.soc_points, self.ocv_points, kind='cubic', 
                                  fill_value='extrapolate')
    
    def OCV(self, soc):
        """开路电压作为SOC的函数"""
        return self.ocv_func(np.clip(soc, 0, 1))
    
    def dOCV_dSOC(self, soc):
        """OCV对SOC的导数（用于卡尔曼滤波）"""
        delta = 0.001
        return (self.OCV(soc + delta) - self.OCV(soc - delta)) / (2 * delta)
    
    def state_equations(self, t, state, current_func):
        """
        状态方程
        
        参数:
            t: 时间 (s)
            state: [SOC, V1]
            current_func: 电流函数 I(t)，正值为放电
        
        返回:
            d_state/dt
        """
        SOC, V1 = state
        I = current_func(t)
        
        # 库仑计数
        dSOC_dt = -I / (self.Q_max * 3.6)  # 3.6转换mAh到As
        
        # RC极化电压
        dV1_dt = -V1 / self.tau1 + I / self.C1
        
        return [dSOC_dt, dV1_dt]
    
    def terminal_voltage(self, soc, V1, I):
        """计算端电压"""
        return self.OCV(soc) - I * self.R0 - V1
    
    def simulate(self, t_span, current_profile, SOC0=1.0, V10=0.0):
        """
        模拟电池放电过程
        
        参数:
            t_span: 时间范围 [t_start, t_end] (秒)
            current_profile: 电流曲线函数 I(t)
            SOC0: 初始SOC
            V10: 初始极化电压
        
        返回:
            t, SOC, V1, V_terminal, I
        """
        # 求解ODE
        t_eval = np.linspace(t_span[0], t_span[1], 1000)
        
        sol = solve_ivp(
            fun=lambda t, y: self.state_equations(t, y, current_profile),
            t_span=t_span,
            y0=[SOC0, V10],
            t_eval=t_eval,
            method='RK45'
        )
        
        t = sol.t
        SOC = sol.y[0]
        V1 = sol.y[1]
        
        # 计算电流和端电压
        I = np.array([current_profile(ti) for ti in t])
        V_terminal = np.array([self.terminal_voltage(soc, v1, i) 
                               for soc, v1, i in zip(SOC, V1, I)])
        
        return t, SOC, V1, V_terminal, I
    
    def time_to_empty(self, SOC0, current_profile, V_cutoff=3.0):
        """
        计算剩余时间（Time-to-Empty）
        
        参数:
            SOC0: 初始SOC
            current_profile: 电流曲线
            V_cutoff: 截止电压
        
        返回:
            剩余时间（秒）
        """
        # 最大模拟时间（10小时）
        max_time = 10 * 3600
        
        def event_cutoff(t, state):
            """当电压降至截止电压时停止"""
            soc, v1 = state
            I = current_profile(t)
            V = self.terminal_voltage(soc, v1, I)
            return V - V_cutoff
        
        event_cutoff.terminal = True
        event_cutoff.direction = -1
        
        sol = solve_ivp(
            fun=lambda t, y: self.state_equations(t, y, current_profile),
            t_span=[0, max_time],
            y0=[SOC0, 0.0],
            events=event_cutoff,
            method='RK45'
        )
        
        if sol.t_events[0].size > 0:
            return sol.t_events[0][0]
        else:
            return max_time  # 未达到截止电压


# ============ 示例使用 ============

def example_constant_discharge():
    """示例1：恒定电流放电"""
    print("=" * 50)
    print("示例1：恒定电流放电（500mA）")
    print("=" * 50)
    
    # 创建模型
    battery = BatteryModel1RC(Q_max=3000, R0=0.05, R1=0.03, C1=2000)
    
    # 定义恒定电流（500mA放电）
    def constant_current(t):
        return 0.5  # A
    
    # 模拟2小时
    t_span = [0, 2*3600]
    t, SOC, V1, V_terminal, I = battery.simulate(t_span, constant_current, SOC0=1.0)
    
    # 绘图
    fig, axes = plt.subplots(3, 1, figsize=(10, 8))
    
    axes[0].plot(t/3600, SOC*100)
    axes[0].set_ylabel('SOC (%)')
    axes[0].set_title('State of Charge')
    axes[0].grid(True)
    
    axes[1].plot(t/3600, V_terminal)
    axes[1].set_ylabel('Voltage (V)')
    axes[1].set_title('Terminal Voltage')
    axes[1].axhline(y=3.0, color='r', linestyle='--', label='Cutoff Voltage')
    axes[1].legend()
    axes[1].grid(True)
    
    axes[2].plot(t/3600, I*1000)
    axes[2].set_ylabel('Current (mA)')
    axes[2].set_xlabel('Time (hours)')
    axes[2].set_title('Discharge Current')
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig('battery_constant_discharge.png', dpi=300, bbox_inches='tight')
    print("图表已保存: battery_constant_discharge.png\n")
    
    # 计算剩余时间
    tte = battery.time_to_empty(SOC0=1.0, current_profile=constant_current)
    print(f"Time-to-Empty: {tte/3600:.2f} hours")


def example_variable_load():
    """示例2：变化负载（模拟手机使用）"""
    print("\n" + "=" * 50)
    print("示例2：变化负载（模拟手机使用场景）")
    print("=" * 50)
    
    battery = BatteryModel1RC(Q_max=3000, R0=0.05, R1=0.03, C1=2000)
    
    # 模拟手机使用：待机→视频→游戏→待机
    def smartphone_usage(t):
        """
        时间段划分：
        0-1800s (0-30min): 待机 (100mA)
        1800-3600s (30-60min): 看视频 (800mA)
        3600-5400s (60-90min): 玩游戏 (1500mA)
        5400-7200s (90-120min): 待机 (100mA)
        """
        if t < 1800:
            return 0.1
        elif t < 3600:
            return 0.8
        elif t < 5400:
            return 1.5
        else:
            return 0.1
    
    t_span = [0, 2*3600]
    t, SOC, V1, V_terminal, I = battery.simulate(t_span, smartphone_usage, SOC0=1.0)
    
    # 绘图
    fig, axes = plt.subplots(4, 1, figsize=(10, 10))
    
    axes[0].plot(t/60, I*1000)
    axes[0].set_ylabel('Current (mA)')
    axes[0].set_title('Smartphone Usage Pattern')
    axes[0].grid(True)
    
    axes[1].plot(t/60, SOC*100)
    axes[1].set_ylabel('SOC (%)')
    axes[1].set_title('State of Charge')
    axes[1].grid(True)
    
    axes[2].plot(t/60, V_terminal)
    axes[2].set_ylabel('Voltage (V)')
    axes[2].set_title('Terminal Voltage')
    axes[2].grid(True)
    
    # 功率
    Power = V_terminal * I
    axes[3].plot(t/60, Power)
    axes[3].set_ylabel('Power (W)')
    axes[3].set_xlabel('Time (minutes)')
    axes[3].set_title('Power Consumption')
    axes[3].grid(True)
    
    plt.tight_layout()
    plt.savefig('battery_smartphone_usage.png', dpi=300, bbox_inches='tight')
    print("图表已保存: battery_smartphone_usage.png\n")
    
    # 计算不同场景的剩余时间
    print("不同使用场景的Time-to-Empty预测:")
    scenarios = [
        ("待机", lambda t: 0.1),
        ("看视频", lambda t: 0.8),
        ("玩游戏", lambda t: 1.5),
    ]
    
    for name, current_func in scenarios:
        tte = battery.time_to_empty(SOC0=1.0, current_profile=current_func)
        print(f"  {name}: {tte/3600:.2f} hours")


def example_sensitivity_analysis():
    """示例3：敏感性分析"""
    print("\n" + "=" * 50)
    print("示例3：敏感性分析 - 电流对电池寿命的影响")
    print("=" * 50)
    
    battery = BatteryModel1RC(Q_max=3000, R0=0.05, R1=0.03, C1=2000)
    
    # 不同电流下的放电曲线
    currents = [0.3, 0.5, 1.0, 1.5, 2.0]  # A
    
    plt.figure(figsize=(10, 6))
    
    for I_discharge in currents:
        def current_func(t):
            return I_discharge
        
        t_span = [0, 4*3600]
        t, SOC, V1, V_terminal, I = battery.simulate(t_span, current_func, SOC0=1.0)
        
        plt.plot(t/3600, SOC*100, label=f'{I_discharge*1000:.0f}mA')
    
    plt.xlabel('Time (hours)')
    plt.ylabel('SOC (%)')
    plt.title('Discharge Curves at Different Current Rates')
    plt.legend()
    plt.grid(True)
    plt.savefig('battery_sensitivity.png', dpi=300, bbox_inches='tight')
    print("图表已保存: battery_sensitivity.png\n")


if __name__ == "__main__":
    # 运行所有示例
    example_constant_discharge()
    example_variable_load()
    example_sensitivity_analysis()
    
    print("\n" + "=" * 50)
    print("所有示例运行完成！")
    print("=" * 50)
