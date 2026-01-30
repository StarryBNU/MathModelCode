"""
温度和电池老化效应模型
包含：
1. 温度对容量和内阻的影响
2. 电池老化（循环寿命和日历寿命）
3. 考虑温度的动态热模型
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from battery_model_advanced import BatteryModel2RC

class TemperatureDependentBattery(BatteryModel2RC):
    """考虑温度影响的电池模型"""
    
    def __init__(self, Q_max_25C=3000, R0_25C=0.05, R1_25C=0.03, 
                 C1_25C=2000, R2_25C=0.02, C2_25C=5000,
                 T_ref=25, T_ambient=25):
        """
        参数:
            *_25C: 25°C时的参数
            T_ref: 参考温度 (°C)
            T_ambient: 环境温度 (°C)
        """
        super().__init__(Q_max_25C, R0_25C, R1_25C, C1_25C, R2_25C, C2_25C)
        
        self.Q_max_ref = Q_max_25C
        self.R0_ref = R0_25C
        self.R1_ref = R1_25C
        self.R2_ref = R2_25C
        
        self.T_ref = T_ref + 273.15  # 转换为K
        self.T_ambient = T_ambient + 273.15
        self.T_battery = self.T_ambient  # 初始电池温度
        
        # Arrhenius方程参数
        self.Ea_capacity = 20000  # 容量激活能 (J/mol)
        self.Ea_resistance = 15000  # 电阻激活能 (J/mol)
        self.R_gas = 8.314  # 气体常数
        
        # 热参数
        self.C_thermal = 500  # 热容 (J/K)
        self.h_conv = 10  # 对流换热系数 (W/(m^2·K))
        self.A_surface = 0.01  # 表面积 (m^2)
    
    def capacity_temperature_factor(self, T_kelvin):
        """温度对容量的影响"""
        # 基于Arrhenius方程
        factor = np.exp(-self.Ea_capacity / self.R_gas * 
                       (1/T_kelvin - 1/self.T_ref))
        
        # 低温时容量显著下降
        if T_kelvin < 273.15:  # 0°C
            extra_reduction = 0.02 * (273.15 - T_kelvin)  # 每降低1°C减少2%
            factor *= (1 - extra_reduction)
        
        return np.clip(factor, 0.5, 1.2)  # 限制在50%-120%
    
    def resistance_temperature_factor(self, T_kelvin):
        """温度对内阻的影响"""
        factor = np.exp(self.Ea_resistance / self.R_gas * 
                       (1/T_kelvin - 1/self.T_ref))
        
        # 低温时内阻显著增加
        if T_kelvin < 273.15:
            extra_increase = 0.05 * (273.15 - T_kelvin)
            factor *= (1 + extra_increase)
        
        return np.clip(factor, 0.5, 5.0)  # 限制在50%-500%
    
    def update_temperature_parameters(self, T_celsius):
        """更新温度相关参数"""
        T_kelvin = T_celsius + 273.15
        self.T_battery = T_kelvin
        
        # 更新容量
        self.Q_max = self.Q_max_ref * self.capacity_temperature_factor(T_kelvin)
        
        # 更新电阻
        R_factor = self.resistance_temperature_factor(T_kelvin)
        self.R0 = self.R0_ref * R_factor
        self.R1 = self.R1_ref * R_factor
        self.R2 = self.R2_ref * R_factor
        
        # 更新时间常数
        self.tau1 = self.R1 * self.C1
        self.tau2 = self.R2 * self.C2
    
    def thermal_dynamics(self, t, T_battery, I):
        """电池温度动态"""
        # 发热：I²R损耗
        P_loss = I**2 * (self.R0 + self.R1 + self.R2)
        
        # 散热：对流换热
        Q_conv = self.h_conv * self.A_surface * (T_battery - self.T_ambient)
        
        # 温度变化率
        dT_dt = (P_loss - Q_conv) / self.C_thermal
        
        return dT_dt
    
    def simulate_with_temperature(self, t_span, current_profile, 
                                  SOC0=1.0, T0_celsius=25):
        """考虑温度动态的模拟"""
        t_eval = np.linspace(t_span[0], t_span[1], 2000)
        
        # 状态: [SOC, V1, V2, T_battery]
        def augmented_dynamics(t, state):
            SOC, V1, V2, T_battery = state
            I = current_profile(t)
            
            # 更新温度参数
            self.update_temperature_parameters(T_battery - 273.15)
            
            # 电气动态
            dSOC_dt = -I / (self.Q_max * 3.6)
            dV1_dt = -V1 / self.tau1 + I / self.C1
            dV2_dt = -V2 / self.tau2 + I / self.C2
            
            # 热动态
            dT_dt = self.thermal_dynamics(t, T_battery, I)
            
            return [dSOC_dt, dV1_dt, dV2_dt, dT_dt]
        
        sol = solve_ivp(
            fun=augmented_dynamics,
            t_span=t_span,
            y0=[SOC0, 0.0, 0.0, T0_celsius + 273.15],
            t_eval=t_eval,
            method='RK45'
        )
        
        t = sol.t
        SOC = sol.y[0]
        V1 = sol.y[1]
        V2 = sol.y[2]
        T_battery = sol.y[3] - 273.15  # 转换为°C
        
        I = np.array([current_profile(ti) for ti in t])
        V_terminal = np.array([self.terminal_voltage(soc, v1, v2, i) 
                               for soc, v1, v2, i in zip(SOC, V1, V2, I)])
        
        return t, SOC, V1, V2, T_battery, V_terminal, I


class AgingBattery(BatteryModel2RC):
    """考虑老化的电池模型"""
    
    def __init__(self, Q_max_new=3000, **kwargs):
        super().__init__(Q_max=Q_max_new, **kwargs)
        
        self.Q_max_new = Q_max_new  # 新电池容量
        self.cycles_completed = 0  # 已完成循环次数
        self.calendar_days = 0  # 日历天数
        self.total_Ah_throughput = 0  # 累计安时吞吐量
        
        # 老化参数
        self.cycle_fade_rate = 0.0002  # 每循环容量衰减率
        self.calendar_fade_rate = 0.0001  # 每天容量衰减率
        self.resistance_growth_rate = 0.001  # 每循环电阻增长率
    
    def update_aging(self, cycles=0, days=0, Ah_throughput=0):
        """更新老化状态"""
        self.cycles_completed += cycles
        self.calendar_days += days
        self.total_Ah_throughput += Ah_throughput
        
        # 循环老化（主要影响容量）
        cycle_fade = 1 - self.cycle_fade_rate * self.cycles_completed
        
        # 日历老化（即使不使用也会衰减）
        calendar_fade = 1 - self.calendar_fade_rate * self.calendar_days
        
        # 容量衰减（两种老化效应叠加）
        self.Q_max = self.Q_max_new * cycle_fade * calendar_fade
        
        # 内阻增长
        resistance_growth = 1 + self.resistance_growth_rate * self.cycles_completed
        self.R0 *= resistance_growth
        
    def estimate_remaining_life(self, DOD_typical=0.8):
        """
        估计剩余寿命
        
        参数:
            DOD_typical: 典型放电深度
        
        返回:
            剩余循环次数, 剩余年数
        """
        # 当容量降至80%时认为寿命结束
        SOH = self.Q_max / self.Q_max_new
        
        if SOH <= 0.8:
            return 0, 0
        
        # 剩余容量衰减空间
        remaining_fade = SOH - 0.8
        
        # 估计剩余循环
        remaining_cycles = remaining_fade / self.cycle_fade_rate
        
        # 假设每天1次循环
        remaining_years = remaining_cycles / 365
        
        return int(remaining_cycles), remaining_years


# ============ 示例使用 ============

def example_temperature_effect():
    """示例1：温度对电池性能的影响"""
    print("=" * 60)
    print("示例1：温度对电池性能的影响")
    print("=" * 60)
    
    # 测试不同温度
    temperatures = [-10, 0, 10, 25, 40]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    for T in temperatures:
        battery = TemperatureDependentBattery(T_ambient=T)
        battery.update_temperature_parameters(T)
        
        # 恒流放电
        def constant_current(t):
            return 1.0  # 1A
        
        t_span = [0, 3600]
        t, SOC, V1, V2, V_terminal, I = battery.simulate(
            t_span, constant_current, SOC0=1.0
        )
        
        # SOC
        axes[0, 0].plot(t/60, SOC*100, label=f'{T}°C')
        
        # 电压
        axes[0, 1].plot(t/60, V_terminal, label=f'{T}°C')
        
        # 容量因子
        capacity_factor = battery.capacity_temperature_factor(T + 273.15)
        resistance_factor = battery.resistance_temperature_factor(T + 273.15)
        
        print(f"温度 {T}°C:")
        print(f"  容量因子: {capacity_factor:.3f} ({capacity_factor*100:.1f}%)")
        print(f"  内阻因子: {resistance_factor:.3f} ({resistance_factor*100:.1f}%)")
    
    axes[0, 0].set_xlabel('Time (minutes)')
    axes[0, 0].set_ylabel('SOC (%)')
    axes[0, 0].set_title('SOC vs Temperature')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].set_xlabel('Time (minutes)')
    axes[0, 1].set_ylabel('Voltage (V)')
    axes[0, 1].set_title('Terminal Voltage vs Temperature')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 温度因子曲线
    T_range = np.linspace(-20, 50, 100)
    capacity_factors = [TemperatureDependentBattery().capacity_temperature_factor(T+273.15) 
                       for T in T_range]
    resistance_factors = [TemperatureDependentBattery().resistance_temperature_factor(T+273.15) 
                         for T in T_range]
    
    axes[1, 0].plot(T_range, capacity_factors, 'b-', linewidth=2)
    axes[1, 0].set_xlabel('Temperature (°C)')
    axes[1, 0].set_ylabel('Capacity Factor')
    axes[1, 0].set_title('Temperature Effect on Capacity')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axvline(x=0, color='r', linestyle='--', alpha=0.5, label='Freezing')
    axes[1, 0].axvline(x=25, color='g', linestyle='--', alpha=0.5, label='Optimal')
    axes[1, 0].legend()
    
    axes[1, 1].plot(T_range, resistance_factors, 'r-', linewidth=2)
    axes[1, 1].set_xlabel('Temperature (°C)')
    axes[1, 1].set_ylabel('Resistance Factor')
    axes[1, 1].set_title('Temperature Effect on Resistance')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axvline(x=0, color='r', linestyle='--', alpha=0.5, label='Freezing')
    axes[1, 1].axvline(x=25, color='g', linestyle='--', alpha=0.5, label='Optimal')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('battery_temperature_effect.png', dpi=300, bbox_inches='tight')
    print("\n图表已保存: battery_temperature_effect.png\n")


def example_thermal_dynamics():
    """示例2：电池温度动态"""
    print("\n" + "=" * 60)
    print("示例2：高负载下的电池温度动态")
    print("=" * 60)
    
    battery = TemperatureDependentBattery(T_ambient=25)
    
    # 高电流放电（模拟游戏）
    def high_current(t):
        return 2.0  # 2A
    
    t_span = [0, 3600]
    t, SOC, V1, V2, T_battery, V_terminal, I = \
        battery.simulate_with_temperature(t_span, high_current, 
                                        SOC0=1.0, T0_celsius=25)
    
    # 绘图
    fig, axes = plt.subplots(4, 1, figsize=(12, 12))
    
    axes[0].plot(t/60, I, 'b-', linewidth=2)
    axes[0].set_ylabel('Current (A)')
    axes[0].set_title('Discharge Current')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(t/60, T_battery, 'r-', linewidth=2)
    axes[1].set_ylabel('Temperature (°C)')
    axes[1].set_title('Battery Temperature')
    axes[1].axhline(y=25, color='g', linestyle='--', label='Ambient')
    axes[1].axhline(y=40, color='orange', linestyle='--', label='Warning')
    axes[1].axhline(y=50, color='r', linestyle='--', label='Critical')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(t/60, SOC*100, 'g-', linewidth=2)
    axes[2].set_ylabel('SOC (%)')
    axes[2].set_title('State of Charge')
    axes[2].grid(True, alpha=0.3)
    
    axes[3].plot(t/60, V_terminal, 'purple', linewidth=2)
    axes[3].set_ylabel('Voltage (V)')
    axes[3].set_xlabel('Time (minutes)')
    axes[3].set_title('Terminal Voltage')
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('battery_thermal_dynamics.png', dpi=300, bbox_inches='tight')
    
    print(f"初始温度: {25}°C")
    print(f"最终温度: {T_battery[-1]:.1f}°C")
    print(f"温升: {T_battery[-1] - 25:.1f}°C")
    print("图表已保存: battery_thermal_dynamics.png\n")


def example_aging_simulation():
    """示例3：电池老化模拟"""
    print("\n" + "=" * 60)
    print("示例3：电池老化模拟（2年使用）")
    print("=" * 60)
    
    battery = AgingBattery(Q_max_new=3000)
    
    # 模拟2年，每天1次循环（80% DOD）
    days = 365 * 2
    cycles_per_day = 1
    DOD = 0.8
    
    SOH_history = []
    cycle_history = []
    
    for day in range(0, days, 30):  # 每30天记录一次
        # 更新老化
        battery.update_aging(cycles=30*cycles_per_day, days=30, 
                           Ah_throughput=30*DOD*battery.Q_max_new/1000)
        
        SOH = battery.Q_max / battery.Q_max_new
        SOH_history.append(SOH * 100)
        cycle_history.append(day * cycles_per_day)
        
        if day % 180 == 0:
            print(f"第 {day} 天 (循环 {day*cycles_per_day}次):")
            print(f"  SOH: {SOH*100:.1f}%")
            print(f"  容量: {battery.Q_max:.0f} mAh")
            print(f"  内阻增长: {(battery.R0/0.05 - 1)*100:.1f}%")
    
    # 估计剩余寿命
    remaining_cycles, remaining_years = battery.estimate_remaining_life(DOD)
    print(f"\n当前SOH: {battery.Q_max/battery.Q_max_new*100:.1f}%")
    print(f"估计剩余寿命: {remaining_cycles} 循环 ({remaining_years:.1f} 年)")
    
    # 绘图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # SOH vs 循环次数
    ax1.plot(cycle_history, SOH_history, 'b-', linewidth=2, marker='o')
    ax1.axhline(y=80, color='r', linestyle='--', label='End of Life (80%)')
    ax1.set_xlabel('Cycle Number')
    ax1.set_ylabel('State of Health (%)')
    ax1.set_title('Battery Aging: Capacity Fade')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 剩余寿命预测
    total_life_cycles = 0.2 / battery.cycle_fade_rate  # 衰减20%需要的循环数
    ax2.barh(['Initial', 'After 2 Years', 'Remaining'], 
            [total_life_cycles, cycle_history[-1], remaining_cycles],
            color=['green', 'orange', 'blue'])
    ax2.set_xlabel('Cycles')
    ax2.set_title('Battery Life Expectancy')
    ax2.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig('battery_aging.png', dpi=300, bbox_inches='tight')
    print("\n图表已保存: battery_aging.png")


if __name__ == "__main__":
    example_temperature_effect()
    example_thermal_dynamics()
    example_aging_simulation()
    
    print("\n" + "=" * 60)
    print("温度和老化模型示例运行完成！")
    print("=" * 60)
