# 智能手机电池建模 - Python代码框架
## MCM 2026 Problem A

### 📁 文件结构

```
battery_modeling/
├── battery_model_basic.py          # 基础模型（一阶RC）- 适合入门
├── battery_model_advanced.py       # 高级模型（二阶RC）- 更精确
├── smartphone_model.py             # 智能手机组件模型 - 完整应用
├── temperature_aging_model.py      # 温度和老化效应 - 高级特性
├── run_all_examples.py            # 运行所有示例
└── README.md                       # 本文件
```

---

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install numpy scipy matplotlib
```

### 2. 运行示例

```bash
# 运行单个模块
python battery_model_basic.py
python battery_model_advanced.py
python smartphone_model.py
python temperature_aging_model.py

# 或运行所有示例
python run_all_examples.py
```

---

## 📚 模型说明

### 🔹 基础模型 (`battery_model_basic.py`)

**适用场景**：快速原型开发，理解基本原理

**核心方程**：

```
dSOC/dt = -I(t) / Q_max
dV1/dt = -V1/(R1*C1) + I(t)/C1
V_terminal = OCV(SOC) - I*R0 - V1
```

**主要功能**：
- ✅ 恒定电流放电
- ✅ 变化负载模拟
- ✅ Time-to-Empty预测
- ✅ 敏感性分析

**示例输出**：
- 恒定放电曲线
- 手机使用场景模拟
- 不同电流下的放电对比

---

### 🔹 高级模型 (`battery_model_advanced.py`)

**适用场景**：高精度预测，参数估计

**核心方程**：
```
dSOC/dt = -I(t) / Q_max
dV1/dt = -V1/(R1*C1) + I(t)/C1  # 快速极化
dV2/dt = -V2/(R2*C2) + I(t)/C2  # 慢速极化
V_terminal = OCV(SOC) - I*R0 - V1 - V2
```

**主要功能**：
- ✅ 双RC模型（双时间常数）
- ✅ 参数估计（Nelder-Mead优化）
- ✅ 模型对比（1RC vs 2RC）
- ✅ 实验数据拟合

**适合物理系同学**：
- 深入理解极化机制（激活极化 vs 浓度极化）
- 推导等效电路的物理意义

**适合数学系同学**：
- 参数优化算法实现
- 数值求解器选择（RK45）

---

### 🔹 智能手机模型 (`smartphone_model.py`)

**适用场景**：完整的应用级建模

**组件功耗模型**：
```
I_total = I_screen + I_CPU + I_network + I_GPS + I_background + I_idle
```

**各组件建模**：

1. **屏幕**：
   - 亮度影响（线性）
   - 刷新率（120Hz vs 60Hz，+50-100%）
   - OLED vs LCD（暗像素省电）
   - 分辨率影响

2. **CPU**：
   - 利用率线性模型
   - 温度throttling（>40°C降频）

3. **网络**：
   - WiFi (10-150 mA)
   - 4G (20-300 mA)
   - 5G (30-450 mA)

4. **GPS**：100 mA

5. **后台应用**：5 mA/app

**主要功能**：
- ✅ 完整一天使用模拟
- ✅ 不同场景续航对比
- ✅ 节电优化建议
- ✅ 敏感性分析

**示例输出**：
- 24小时使用曲线
- 8种场景续航对比
- 优化建议（量化改善效果）

---

### 🔹 温度和老化模型 (`temperature_aging_model.py`)

**适用场景**：长期预测，极端条件

**温度效应**（基于Arrhenius方程）：
```
Q_eff(T) = Q_max * exp(-Ea/(R*T))
R_eff(T) = R_0 * exp(Ea/(R*T))
```

**热动力学**：
```
C_thermal * dT/dt = I²R - h*A*(T - T_ambient)
```

**老化模型**：
```
SOH = (1 - α*cycles) * (1 - β*days)
```

**主要功能**：

- ✅ -20°C到50°C温度测试
- ✅ 电池温升模拟
- ✅ 2年老化预测
- ✅ 剩余寿命估计

---

## 🎯 团队分工建议

### 物理系同学负责：

1. **模型推导**
   - [ ] 从电化学原理推导等效电路
   - [ ] 解释各参数的物理意义
   - [ ] 推导温度依赖关系（Arrhenius方程）
   - [ ] 分析各组件功耗机制

2. **参数确定**
   - [ ] 从规格书/文献获取电池参数
   - [ ] 估算各组件功耗（屏幕、CPU等）
   - [ ] 验证参数的合理性

3. **物理分析**
   - [ ] 解释模型预测结果
   - [ ] 分析极端情况（高温、低温）
   - [ ] 讨论模型假设的物理合理性

### 数学系同学负责：

1. **数值实现**
   - [ ] 选择合适的ODE求解器（RK45 vs LSODA）
   - [ ] 实现参数估计算法
   - [ ] 编写数值稳定的代码

2. **优化与估计**
   - [ ] 参数优化（Nelder-Mead, L-BFGS-B）
   - [ ] 卡尔曼滤波器实现（EKF/UKF）
   - [ ] 不确定性量化

3. **敏感性分析**
   - [ ] 单因素敏感性
   - [ ] 全局敏感性（Sobol指数）
   - [ ] 置信区间计算

---

## 📊 建议的论文结构

### 1. Introduction (2页)
- 问题背景
- 建模目标
- 文献综述

### 2. Model Development (8-10页)

**2.1 物理基础**（物理系主导）
- 锂离子电池电化学原理
- 等效电路推导
- 参数物理意义

**2.2 数学模型**（数学系主导）
- 状态空间表示
- ODE系统
- 求解方法

**2.3 组件模型**（联合）
- 屏幕功耗
- CPU功耗
- 网络功耗
- 总功耗模型

**2.4 扩展模型**（联合）
- 温度效应
- 老化效应

### 3. Parameter Estimation (3-4页)
- 数据来源
- 估计方法
- 验证结果

### 4. Results (6-8页)
- Time-to-Empty预测
- 不同场景对比
- 敏感性分析
- 温度影响
- 老化分析

### 5. Recommendations (2页)
- 用户节电建议
- 操作系统优化策略
- 模型应用扩展

### 6. Strengths & Limitations (1页)

### 7. Conclusion (1页)

---

## 🔧 使用技巧

### 修改参数

```python
# 在任何模型中修改电池参数
battery = BatteryModel2RC(
    Q_max=3500,    # 更大容量
    R0=0.03,       # 更小内阻（更好的电池）
    R1=0.02,
    C1=2500,
    R2=0.015,
    C2=6000
)
```

### 自定义使用场景

```python
# 添加新的使用场景
def my_scenario(t):
    if t < 3600:
        return 0.8  # 第一小时：800mA
    else:
        return 0.3  # 之后：300mA

t, SOC, V1, V2, V_terminal, I = battery.simulate(
    [0, 7200], my_scenario, SOC0=1.0
)
```

### 批量测试

```python
# 测试多个初始SOC
for SOC0 in [1.0, 0.8, 0.6, 0.4, 0.2]:
    tte = battery.time_to_empty(SOC0, current_profile)
    print(f"SOC {SOC0*100}%: {tte/3600:.2f} hours")
```

---

## 📈 预期结果

运行所有示例后，会生成以下图表：

1. `battery_constant_discharge.png` - 恒定电流放电
2. `battery_smartphone_usage.png` - 手机使用模拟
3. `battery_sensitivity.png` - 敏感性分析
4. `battery_model_comparison.png` - 1RC vs 2RC对比
5. `smartphone_daily_usage.png` - 24小时使用
6. `smartphone_scenario_comparison.png` - 场景对比
7. `smartphone_optimization.png` - 优化建议
8. `battery_temperature_effect.png` - 温度影响
9. `battery_thermal_dynamics.png` - 热动态
10. `battery_aging.png` - 老化模拟

---

## 💡 高级扩展建议

### 对于追求高分的团队：

1. **实现卡尔曼滤波器**
   ```python
   # 扩展卡尔曼滤波器（EKF）
   # 用于实时SOC估计和参数自适应
   ```

2. **神经ODE（Neural ODE）**
   ```python
   # 使用PyTorch实现混合物理-数据驱动模型
   # 学习难以建模的非线性效应
   ```

3. **不确定性量化**
   ```python
   # 蒙特卡洛模拟
   # 计算预测的置信区间
   ```

4. **多目标优化**
   ```python
   # 续航时间 vs 性能权衡
   # Pareto前沿分析
   ```

---

## 📖 参考资料

### 论文
1. NASA TM-20205008059 - Battery Modeling
2. Scientific Reports 2025 - UKBF for SOC Estimation
3. Energy Informatics 2021 - Neural ODEs for Batteries

### 数据集
1. CALCE Battery Data
2. NASA Prognostics Data Repository
3. BatteryLife Dataset (2025)

---

## 🎓 学习路径

### 第1天：理解基础
- [ ] 运行 `battery_model_basic.py`
- [ ] 理解ODE方程
- [ ] 修改参数看效果

### 第2天：深入模型
- [ ] 运行 `battery_model_advanced.py`
- [ ] 理解双RC模型
- [ ] 尝试参数估计

### 第3天：完整应用
- [ ] 运行 `smartphone_model.py`
- [ ] 添加自己的使用场景
- [ ] 分析优化建议

### 第4天：高级特性
- [ ] 运行 `temperature_aging_model.py`
- [ ] 理解温度和老化效应
- [ ] 整合所有模型

---

## ⚠️ 常见问题

### Q: 模型预测不准确？
A: 检查参数是否合理，尝试从实验数据拟合参数

### Q: 求解器报错？
A: 检查电流函数是否连续，尝试降低tolerances

### Q: 图表不显示？
A: 确保安装了matplotlib，使用 `plt.savefig()` 保存图片

### Q: 想要更快的求解速度？
A: 使用 'LSODA' 求解器，或减少 `t_eval` 的点数

---

## 🏆 比赛提示

1. **物理推导要详细**：从Maxwell方程到等效电路
2. **数学要严谨**：证明解的存在性、稳定性
3. **结果要可视化**：高质量的图表很重要
4. **验证要充分**：与文献数据对比
5. **讨论要深入**：分析模型假设的影响

---

## 📞 联系方式

如有问题，可以：
1. 查看代码中的注释
2. 阅读引用的论文
3. 在代码中添加 `print()` 调试

---

**祝你们建模成功！Good luck!** 🎉
