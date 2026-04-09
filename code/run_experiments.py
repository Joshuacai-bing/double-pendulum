import sys
import os
import matplotlib.pyplot as plt
import numpy as np

# 确保能引入同目录的模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from double_pendulum_simulation import simulate_and_analyze

# 三组实验参数: (L1, L2, M1, M2, name)
configs = [
    (1.0, 1.0, 1.0, 1.0, "Experiment_A_Baseline"),
    (1.0, 1.0, 1.0, 5.0, "Experiment_B_HeavyBottom"),
    (1.0, 2.0, 1.0, 1.0, "Experiment_C_LongBottom")
]

# 通用模拟参数
theta0 = 0.1
t_max = 20.0
dt = 0.01
use_nonlinear = False
g = 9.8
integrator_name = 'GaussLegendreRK4'

# 使用 mixed_phase 初始条件 (theta1=0.1, theta2=0)，能同时激发系统内部的所有振动模态
mode = 'mixed_phase'

for L1, L2, M1, M2, name in configs:
    print(f"\n{'='*40}")
    print(f"Running {name}: L1={L1}, L2={L2}, M1={M1}, M2={M2}")
    t_vals, states, energies, fig = simulate_and_analyze(
        L1, L2, M1, M2, theta0, mode, t_max, dt, use_nonlinear, g, integrator_name
    )
    
    # 调整标题以免重叠
    fig.suptitle(f"{name} (L1={L1}, L2={L2}, M1={M1}, M2={M2})", fontsize=14)
    
    # 保存图片
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"{name}.png")
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot to {output_path}")
