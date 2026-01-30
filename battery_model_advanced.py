"""
高级电池模型 - 二阶等效电路（双RC模型）
更精确的模型，包含两个时间常数

物理模型：
- OCV(SOC): 开路电压
- R0: 欧姆电阻
- R1, C1: 快速极化（激活极化）
- R2, C2: 慢速极化（浓度极化）

状态方程：
dSOC/dt = -I(t) / Q_max
dV1/dt = -V1/(R1*C1) + I(t)/C1
dV2/dt = -V2/(R2*C2) + I(t)/C2
V_terminal = OCV(SOC) - I*R0 - V1 - V2
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.optimize import minimize
import matplotlib.pyplot as plt

class BatteryModel2RC:
    """二阶等效电路电池模型（更精确）"""
    
    def __init__(self, Q_max=3000, R0=0.05, R1=0.03, C1=2000, R2=0.02, C2=5000):
        """
        参数:
            Q_max: 最大容量 (mAh)
            R0: 欧姆电阻 (Ohm)
            R1: 快速极化电阻 (Ohm)
            C1: 快速极化电容 (F)
            R2: 慢速极化电阻 (Ohm)
            C2: 慢速极化电容 (F)
        """
        self.Q_max = Q_max
        self.R0 = R0
        self.R1 = R1
        self.C1 = C1
        self.R2 = R2
        self.C2 = C2
        self.tau1 = R1 * C1  # 快速时间常数（秒级）
        self.tau2 = R2 * C2  # 慢速时间常数（分钟级）
        
        # OCV-SOC查找表
        self.soc_points = np.linspace(0, 1, 21)
        # 更详细的OCV曲线（基于真实锂离子电池）
        self.ocv_points = 3.0 + 1.2 * (
            0.5 * self.soc_points + 
            0.3 * np.tanh(10 * (self.soc_points - 0.1)) +
            0.2 * np.tanh(10 * (self.soc_points - 0.9))
        )
        self.ocv_func = interp1d(self.soc_points, self.ocv_points, kind='cubic',
                                  fill_value='extrapolate')
    
    def OCV(self, soc):
        """开路电压"""
        return self.ocv_func(np.clip(soc, 0, 1))
    
    def state_equations(self, t, state, current_func):
        """状态方程"""
        SOC, V1, V2 = state
        I = current_func(t)
        
        dSOC_dt = -I / (self.Q_max * 3.6)
        dV1_dt = -V1 / self.tau1 + I / self.C1
        dV2_dt = -V2 / self.tau2 + I / self.C2
        
        return [dSOC_dt, dV1_dt, dV2_dt]
    
    def terminal_voltage(self, soc, V1, V2, I):
        """端电压"""
        return self.OCV(soc) - I * self.R0 - V1 - V2
    
    def simulate(self, t_span, current_profile, SOC0=1.0, V10=0.0, V20=0.0):
        """模拟"""
        t_eval = np.linspace(t_span[0], t_span[1], 2000)
        
        sol = solve_ivp(
            fun=lambda t, y: self.state_equations(t, y, current_profile),
            t_span=t_span,
            y0=[SOC0, V10, V20],
            t_eval=t_eval,
            method='RK45',
            rtol=1e-6
        )
        
        t = sol.t
        SOC = sol.y[0]
        V1 = sol.y[1]
        V2 = sol.y[2]
        
        I = np.array([current_profile(ti) for ti in t])
        V_terminal = np.array([self.terminal_voltage(soc, v1, v2, i) 
                               for soc, v1, v2, i in zip(SOC, V1, V2, I)])
        
        return t, SOC, V1, V2, V_terminal, I
    
    def get_parameters(self):
        """获取参数向量"""
        return np.array([self.Q_max, self.R0, self.R1, self.C1, self.R2, self.C2])
    
    def set_parameters(self, params):
        """设置参数"""
        self.Q_max, self.R0, self.R1, self.C1, self.R2, self.C2 = params
        self.tau1 = self.R1 * self.C1
        self.tau2 = self.R2 * self.C2


class ParameterEstimator:
    """参数估计器（使用实验数据拟合模型参数）"""
    
    def __init__(self, model):
        self.model = model
    
    def estimate(self, t_data, V_data, I_data, SOC_data=None):
        """
        参数估计
        
        参数:
            t_data: 时间数据
            V_data: 电压测量值
            I_data: 电流测量值
            SOC_data: SOC真实值（如果有）
        
        返回:
            优化后的参数
        """
        # 定义目标函数（最小化电压误差）
        def objective(params):
            # 约束参数为正
            if np.any(params <= 0):
                return 1e10
            
            # 设置模型参数
            self.model.set_parameters(params)
            
            # 构造电流插值函数
            current_func = interp1d(t_data, I_data, kind='linear', 
                                   fill_value='extrapolate')
            
            # 模拟
            try:
                t_span = [t_data[0], t_data[-1]]
                t_sim, SOC_sim, V1_sim, V2_sim, V_sim, I_sim = \
                    self.model.simulate(t_span, current_func, SOC0=1.0)
                
                # 插值到数据点
                V_interp = np.interp(t_data, t_sim, V_sim)
                
                # 计算误差（均方根误差）
                rmse = np.sqrt(np.mean((V_data - V_interp)**2))
                
                return rmse
            except:
                return 1e10
        
        # 初始参数
        params0 = self.model.get_parameters()
        
        # 参数边界
        bounds = [
            (1000, 5000),   # Q_max
            (0.01, 0.2),    # R0
            (0.01, 0.1),    # R1
            (500, 5000),    # C1
            (0.01, 0.1),    # R2
            (1000, 10000),  # C2
        ]
        
        # 优化
        print("开始参数估计...")
        result = minimize(objective, params0, method='Nelder-Mead', 
                         bounds=bounds, options={'maxiter': 1000})
        
        optimal_params = result.x
        print(f"优化完成！最终RMSE: {result.fun:.4f} V")
        
        return optimal_params


# ============ 示例使用 ============

def example_compare_models():
    """比较一阶和二阶模型"""
    print("=" * 50)
    print("示例：比较一阶和二阶模型的精度")
    print("=" * 50)
    
    # 创建真实系统（二阶模型）
    true_battery = BatteryModel2RC(
        Q_max=3000, R0=0.05, R1=0.03, C1=2000, R2=0.02, C2=5000
    )
    
    # 脉冲电流（能显示动态特性）
    def pulse_current(t):
        period = 600  # 10分钟周期
        duty = 0.5
        if (t % period) < (period * duty):
            return 1.0  # 1A
        else:
            return 0.2  # 0.2A
    
    # 生成"真实"数据
    t_span = [0, 2*3600]
    t_true, SOC_true, V1_true, V2_true, V_true, I_true = \
        true_battery.simulate(t_span, pulse_current, SOC0=1.0)
    
    # 一阶模型拟合
    from battery_model_basic import BatteryModel1RC
    model_1rc = BatteryModel1RC(Q_max=3000, R0=0.05, R1=0.03, C1=2000)
    t_1rc, SOC_1rc, V1_1rc, V_1rc, I_1rc = \
        model_1rc.simulate(t_span, pulse_current, SOC0=1.0)
    
    # 二阶模型拟合
    model_2rc = BatteryModel2RC(Q_max=3000, R0=0.05, R1=0.03, C1=2000, 
                                R2=0.02, C2=5000)
    t_2rc, SOC_2rc, V1_2rc, V2_2rc, V_2rc, I_2rc = \
        model_2rc.simulate(t_span, pulse_current, SOC0=1.0)
    
    # 计算误差
    V_1rc_interp = np.interp(t_true, t_1rc, V_1rc)
    V_2rc_interp = np.interp(t_true, t_2rc, V_2rc)
    
    rmse_1rc = np.sqrt(np.mean((V_true - V_1rc_interp)**2))
    rmse_2rc = np.sqrt(np.mean((V_true - V_2rc_interp)**2))
    
    print(f"一阶模型 RMSE: {rmse_1rc:.4f} V")
    print(f"二阶模型 RMSE: {rmse_2rc:.4f} V")
    print(f"精度提升: {(1 - rmse_2rc/rmse_1rc)*100:.1f}%\n")
    
    # 绘图
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # 电流
    axes[0].plot(t_true/60, I_true*1000, 'k-', linewidth=2)
    axes[0].set_ylabel('Current (mA)')
    axes[0].set_title('Pulse Current Profile')
    axes[0].grid(True)
    
    # SOC
    axes[1].plot(t_true/60, SOC_true*100, 'k-', label='True (2RC)', linewidth=2)
    axes[1].plot(t_1rc/60, SOC_1rc*100, 'b--', label='1RC Model', linewidth=1.5)
    axes[1].plot(t_2rc/60, SOC_2rc*100, 'r:', label='2RC Model', linewidth=1.5)
    axes[1].set_ylabel('SOC (%)')
    axes[1].set_title('State of Charge')
    axes[1].legend()
    axes[1].grid(True)
    
    # 电压（放大显示差异）
    axes[2].plot(t_true/60, V_true, 'k-', label='True (2RC)', linewidth=2)
    axes[2].plot(t_1rc/60, V_1rc, 'b--', label='1RC Model', linewidth=1.5, alpha=0.7)
    axes[2].plot(t_2rc/60, V_2rc, 'r:', label='2RC Model', linewidth=1.5, alpha=0.7)
    axes[2].set_ylabel('Voltage (V)')
    axes[2].set_xlabel('Time (minutes)')
    axes[2].set_title(f'Terminal Voltage (1RC RMSE={rmse_1rc:.4f}V, 2RC RMSE={rmse_2rc:.4f}V)')
    axes[2].legend()
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig('battery_model_comparison.png', dpi=300, bbox_inches='tight')
    print("图表已保存: battery_model_comparison.png")


def example_parameter_estimation():
    """参数估计示例"""
    print("\n" + "=" * 50)
    print("示例：从实验数据估计参数")
    print("=" * 50)
    
    # 生成模拟实验数据
    true_params = np.array([3000, 0.05, 0.03, 2000, 0.02, 5000])
    true_battery = BatteryModel2RC(*true_params)
    
    # 恒流放电
    def constant_current(t):
        return 1.0  # 1A
    
    t_span = [0, 1800]  # 30分钟
    t_exp, SOC_exp, V1_exp, V2_exp, V_exp, I_exp = \
        true_battery.simulate(t_span, constant_current, SOC0=1.0)
    
    # 添加测量噪声
    np.random.seed(42)
    noise_std = 0.01  # 10mV标准差
    V_measured = V_exp + np.random.normal(0, noise_std, len(V_exp))
    
    # 初始猜测（故意偏离真值）
    initial_guess = np.array([2500, 0.08, 0.05, 1500, 0.03, 3000])
    model = BatteryModel2RC(*initial_guess)
    
    print("真实参数:", true_params)
    print("初始猜测:", initial_guess)
    print()
    
    # 参数估计
    estimator = ParameterEstimator(model)
    optimal_params = estimator.estimate(t_exp, V_measured, I_exp)
    
    print("\n优化后参数:", optimal_params)
    print("参数误差:")
    for i, name in enumerate(['Q_max', 'R0', 'R1', 'C1', 'R2', 'C2']):
        error = abs(optimal_params[i] - true_params[i]) / true_params[i] * 100
        print(f"  {name}: {error:.2f}%")


if __name__ == "__main__":
    example_compare_models()
    example_parameter_estimation()
    
    print("\n" + "=" * 50)
    print("高级示例运行完成！")
    print("=" * 50)
