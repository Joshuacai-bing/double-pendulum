import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import scipy.signal
from integrators import INTEGRATORS

class DoublePendulum:
    """
    双摆系统模型封装，包含参数、导数函数、RK4 积分和能量计算。
    """
    def __init__(self, L1, L2, M1, M2, g=9.8):
        self.L1 = L1
        self.L2 = L2
        self.M1 = M1
        self.M2 = M2
        self.g = g

    def derivatives_linear(self, t, state):
        """
        线性化双摆系统（小角度近似）的运动方程导数。
        state: [theta1, omega1, theta2, omega2]
        """
        theta1, omega1, theta2, omega2 = state
        d_theta1 = omega1
        d_theta2 = omega2
        
        # 根据小角度拉格朗日方程推导
        # (M1+M2)*L1^2 * theta1'' + M2*L1*L2 * theta2'' + (M1+M2)*g*L1 * theta1 = 0 
        # M2*L2^2 * theta2'' + M2*L1*L2 * theta1'' + M2*g*L2 * theta2 = 0 
        
        # Solving the linear system for theta1'' and theta2''
        # A * x = b
        # [ (M1+M2)*L1^2    M2*L1*L2 ] [ theta1'' ] = [ -(M1+M2)*g*L1*theta1 ]
        # [ M2*L1*L2        M2*L2^2  ] [ theta2'' ] = [ -M2*g*L2*theta2      ]
        
        a11 = (self.M1 + self.M2) * self.L1**2
        a12 = self.M2 * self.L1 * self.L2
        a21 = self.M2 * self.L1 * self.L2
        a22 = self.M2 * self.L2**2
        
        b1 = -(self.M1 + self.M2) * self.g * self.L1 * theta1
        b2 = -self.M2 * self.g * self.L2 * theta2
        
        det = a11 * a22 - a12 * a21
        
        d_omega1 = (b1 * a22 - b2 * a12) / det
        d_omega2 = (a11 * b2 - a21 * b1) / det
        
        return np.array([d_theta1, d_omega1, d_theta2, d_omega2])

    def derivatives_nonlinear(self, t, state):
        """
        非线性双摆系统（完整形式）的运动方程导数。
        来源: 经典力学中双摆的标准拉格朗日方程解。
        """
        theta1, omega1, theta2, omega2 = state
        d_theta1 = omega1
        d_theta2 = omega2
        
        delta = theta1 - theta2
        den = 2 * self.M1 + self.M2 - self.M2 * np.cos(2 * delta)
        
        num1 = -self.g * (2 * self.M1 + self.M2) * np.sin(theta1) \
               - self.M2 * self.g * np.sin(theta1 - 2 * theta2) \
               - 2 * np.sin(delta) * self.M2 * (omega2**2 * self.L2 + omega1**2 * self.L1 * np.cos(delta))
        d_omega1 = num1 / (self.L1 * den)
        
        num2 = 2 * np.sin(delta) * (omega1**2 * self.L1 * (self.M1 + self.M2) \
               + self.g * (self.M1 + self.M2) * np.cos(theta1) \
               + omega2**2 * self.L2 * self.M2 * np.cos(delta))
        d_omega2 = num2 / (self.L2 * den)
        
        return np.array([d_theta1, d_omega1, d_theta2, d_omega2])

    def calculate_energy(self, state, use_nonlinear=False):
        """
        计算系统的总机械能（动能 + 势能）。
        """
        theta1, omega1, theta2, omega2 = state
        if use_nonlinear:
            T = 0.5 * self.M1 * (self.L1 * omega1)**2 + \
                0.5 * self.M2 * ((self.L1 * omega1)**2 + (self.L2 * omega2)**2 + \
                                 2 * self.L1 * self.L2 * omega1 * omega2 * np.cos(theta1 - theta2))
            V = - (self.M1 + self.M2) * self.g * self.L1 * np.cos(theta1) - self.M2 * self.g * self.L2 * np.cos(theta2)
        else:
            T = 0.5 * (self.M1 + self.M2) * self.L1**2 * omega1**2 + \
                0.5 * self.M2 * self.L2**2 * omega2**2 + \
                self.M2 * self.L1 * self.L2 * omega1 * omega2
            
            # 线性模型的势能基准设定在平衡位置，这样能量守恒更明显
            V = 0.5 * (self.M1 + self.M2) * self.g * self.L1 * theta1**2 + \
                0.5 * self.M2 * self.g * self.L2 * theta2**2
        return T + V

def calculate_theoretical_frequencies(L1, L2, M1, M2, g):
    """
    计算线性化双摆系统的理论简正频率。
    """
    K = g / M1
    trace = (M1 + M2) * K * (1 / L1 + 1 / L2)
    det = g**2 * (M1 + M2) / (M1 * L1 * L2)
    
    discriminant = trace**2 - 4 * det
    w2_plus = 0.5 * (trace + np.sqrt(discriminant))
    w2_minus = 0.5 * (trace - np.sqrt(discriminant))
    
    f_plus = np.sqrt(w2_plus) / (2 * np.pi)
    f_minus = np.sqrt(w2_minus) / (2 * np.pi)
    
    return sorted([f_plus, f_minus])

def simulate_and_analyze(L1, L2, M1, M2, theta0, phase_type, t_max, dt, use_nonlinear=False, g=9.8, integrator_name='RK4'):
    """
    执行一次完整的模拟和分析。
    """
    pendulum = DoublePendulum(L1, L2, M1, M2, g)
    
    # 初始条件：三种相位模式
    if phase_type == 'in_phase':
        state = np.array([theta0, 0.0, theta0, 0.0])
    elif phase_type == 'anti_phase':
        state = np.array([theta0, 0.0, -theta0, 0.0])
    elif phase_type == 'mixed_phase':
        state = np.array([theta0, 0.0, 0.0, 0.0])
    else:
        raise ValueError(f"Unknown phase_type: {phase_type}")
        
    t_vals = np.arange(0, t_max, dt)
    n_steps = len(t_vals)
    
    states = np.zeros((n_steps, 4))
    energies = np.zeros(n_steps)
    
    states[0] = state
    energies[0] = pendulum.calculate_energy(state, use_nonlinear)
    
    warned = False
    
    integrator_class = INTEGRATORS.get(integrator_name)
    if integrator_class is None:
        raise ValueError(f"Unknown integrator: {integrator_name}")
    integrator = integrator_class()
    
    derivs = pendulum.derivatives_nonlinear if use_nonlinear else pendulum.derivatives_linear
    
    # 主循环
    for i in range(1, n_steps):
        state = integrator.step(derivs, t_vals[i-1], state, dt)
        states[i] = state
        energies[i] = pendulum.calculate_energy(state, use_nonlinear)
        
        # 小角度假设检查
        if not warned and not use_nonlinear and (abs(state[0]) > 0.3 or abs(state[2]) > 0.3):
            print(f"[{phase_type}] 警告: 摆角超过 0.3 rad，小角度近似可能不再准确。")
            warned = True
            
    theta1_vals = states[:, 0]
    theta2_vals = states[:, 2]
    
    # 傅里叶分析
    N = len(theta1_vals)
    fft_vals = np.fft.fft(theta1_vals)
    fft_freq = np.fft.fftfreq(N, dt)
    
    # 提取正频率部分
    pos_mask = fft_freq > 0
    freqs = fft_freq[pos_mask]
    amps = np.abs(fft_vals)[pos_mask] * 2.0 / N
    
    # 寻找峰值
    peaks, _ = scipy.signal.find_peaks(amps, height=np.max(amps)*0.1, distance=10)
    peak_freqs = freqs[peaks]
    peak_amps = amps[peaks]
    
    # 获取最大的两个峰值
    sorted_indices = np.argsort(peak_amps)[::-1]
    top_peaks = sorted_indices[:2]
    dom_freqs = sorted(peak_freqs[top_peaks])
    
    print(f"\n--- 模式: {phase_type} ---")
    print(f"观测到的主导频率: {[round(f, 3) for f in dom_freqs]} Hz")
    
    theo_freqs = calculate_theoretical_frequencies(L1, L2, M1, M2, g)
    print(f"理论简正频率: {theo_freqs[0]:.3f} Hz, {theo_freqs[1]:.3f} Hz")
    
    # 绘制可视化
    fig, axs = plt.subplots(3, 1, figsize=(10, 12))
    fig.canvas.manager.set_window_title(f"Double Pendulum - {phase_type} (Nonlinear: {use_nonlinear})")
    
    # 角度-时间曲线
    axs[0].plot(t_vals, theta1_vals, label=r'$\theta_1$ (Upper Pendulum)')
    axs[0].plot(t_vals, theta2_vals, label=r'$\theta_2$ (Lower Pendulum)')
    axs[0].set_xlabel('Time (s)')
    axs[0].set_ylabel('Angle (rad)')
    axs[0].set_title(f'Angles vs Time ({phase_type})')
    axs[0].legend()
    axs[0].grid(True)
    
    # 频谱图
    axs[1].plot(freqs, amps)
    if len(dom_freqs) > 0:
        axs[1].scatter(dom_freqs, peak_amps[top_peaks], color='red', zorder=5, label='Peaks')
        for f in dom_freqs:
            axs[1].annotate(f'{f:.3f} Hz', xy=(f, np.interp(f, freqs, amps)), 
                            xytext=(5, 5), textcoords='offset points')
    axs[1].set_xlabel('Frequency (Hz)')
    axs[1].set_ylabel('Amplitude')
    axs[1].set_title(f'Fourier Spectrum of $\\theta_1$ ({phase_type})')
    axs[1].set_xlim(0, max(theo_freqs)*2.5 if theo_freqs else 5)
    axs[1].legend()
    axs[1].grid(True)
    
    # 能量曲线
    # 计算能量漂移
    energy_drift = np.max(energies) - np.min(energies)
    print(f"最大能量漂移: {energy_drift:.2e} J")
    
    axs[2].plot(t_vals, energies, color='purple', label='Total Energy')
    axs[2].set_xlabel('Time (s)')
    axs[2].set_ylabel('Energy (J)')
    axs[2].set_title(f'Mechanical Energy vs Time ({phase_type})')
    # 如果能量漂移非常小，适当放大纵轴范围，以免全是数值噪声
    if energy_drift < 1e-10:
        mean_e = np.mean(energies)
        axs[2].set_ylim(mean_e - 1e-9, mean_e + 1e-9)
    axs[2].legend()
    axs[2].grid(True)
    
    plt.tight_layout()
    
    return t_vals, states, energies, fig

def animate_pendulums(results_dict, L1, L2, phase_to_animate='all'):
    """
    生成并展示双摆运动的实时动画。
    由于耗时计算已在模拟阶段完成，动画循环仅刷新绘图。
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    max_len = (L1 + L2) * 1.2
    ax.set_xlim(-max_len, max_len)
    ax.set_ylim(-max_len, max_len)
    ax.set_aspect('equal')
    ax.grid(True)
    title_str = "All Phases" if phase_to_animate == 'all' else phase_to_animate
    ax.set_title(f"Double Pendulum Animation ({title_str})")
    
    lines = {}
    traces = {}
    colors = {'in_phase': 'blue', 'anti_phase': 'red', 'mixed_phase': 'green'}
    
    coords = {}
    phases = list(results_dict.keys()) if phase_to_animate == 'all' else [phase_to_animate]
    
    t_vals = None
    for phase_type in phases:
        t, states = results_dict[phase_type]
        if t_vals is None:
            t_vals = t
        theta1 = states[:, 0]
        theta2 = states[:, 2]
        
        # 预先计算好轨迹，避免动画循环中重复计算
        x1 = L1 * np.sin(theta1)
        y1 = -L1 * np.cos(theta1)
        x2 = x1 + L2 * np.sin(theta2)
        y2 = y1 - L2 * np.cos(theta2)
        
        coords[phase_type] = (x1, y1, x2, y2)
        
        line, = ax.plot([], [], 'o-', lw=2, markersize=8, color=colors[phase_type], label=phase_type)
        trace, = ax.plot([], [], '-', lw=1, color=colors[phase_type], alpha=0.5)
        lines[phase_type] = line
        traces[phase_type] = trace
        
    ax.legend(loc="upper right")
    
    dt = t_vals[1] - t_vals[0]
    fps = 30
    step = max(1, int((1/fps) / dt))
    
    def init():
        for p in phases:
            lines[p].set_data([], [])
            traces[p].set_data([], [])
        return list(lines.values()) + list(traces.values())
        
    def update(frame):
        idx = frame * step
        for p in phases:
            x1, y1, x2, y2 = coords[p]
            if idx < len(x1):
                # 更新摆杆和质点
                lines[p].set_data([0, x1[idx], x2[idx]], [0, y1[idx], y2[idx]])
                
                # 绘制末端轨迹的“拖尾”效果（保留过去50帧）
                start_idx = max(0, idx - 50*step)
                traces[p].set_data(x2[start_idx:idx], y2[start_idx:idx])
                
        return list(lines.values()) + list(traces.values())
        
    frames = len(t_vals) // step
    ani = FuncAnimation(fig, update, frames=frames, init_func=init, blit=True, interval=1000/fps)
    
    # 可选保存为 MP4 或 GIF: 
    # ani.save("double_pendulum.mp4", writer="ffmpeg", fps=fps)
    # ani.save("double_pendulum.gif", writer="imagemagick", fps=fps)
    
    plt.show()
    return ani

if __name__ == "__main__":
    # --- 1. 参数设置 ---
    L1 = 1.0       # 上摆长 (m)
    L2 = 1.1   # 下摆长 (m)
    M1 = 1.0       # 上摆质量 (kg)
    M2 = 1.0       # 下摆质量 (kg)
    g = 9.8        # 重力加速度 (m/s^2)
    theta0 = 0.1   # 初始小角度 (rad)
    t_max = 20.0   # 模拟时长 (s)
    dt = 0.01      # 时间步长 (s)
    
    # 切换此开关以验证完整非线性模型（False 为线性小角度模型）
    USE_NONLINEAR = False 
    
    # 可选的积分器: 'RK4', 'SymplecticEuler', 'VelocityVerlet', 'ImplicitMidpoint', 'GaussLegendreRK4', 'Yoshida4'
    INTEGRATOR_NAME = 'GaussLegendreRK4'

    modes = ['in_phase', 'anti_phase', 'mixed_phase']
    results = {}
    
    # --- 2. 模拟与分析循环 ---
    for mode in modes:
        t_vals, states, energies, fig = simulate_and_analyze(
            L1, L2, M1, M2, theta0, mode, t_max, dt, use_nonlinear=USE_NONLINEAR, g=g, integrator_name=INTEGRATOR_NAME
        )
        results[mode] = (t_vals, states)
        
        # 可选保存为 CSV (这里仅为注释示例，用户可自行取消注释以启用):
        # np.savetxt(f"{mode}_data.csv", np.column_stack((t_vals, states, energies)), 
        #            delimiter=",", header="t,theta1,omega1,theta2,omega2,energy")

    # 显示所有图表窗口
    plt.show()
    
    # --- 3. 动画演示 ---
    # 可以将 'all' 替换为 'in_phase', 'anti_phase', 或 'mixed_phase' 以只看一种模式
    print("\n准备播放动画，关闭图表窗口以退出程序。")
    ani = animate_pendulums(results, L1, L2, phase_to_animate='all')
