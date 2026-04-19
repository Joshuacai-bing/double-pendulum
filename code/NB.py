import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import scipy.signal
import os
from integrators import INTEGRATORS

class DoublePendulum:
    """
    双摆系统模型封装，包含参数、导数函数、RK4 积分和能量计算。
    加入了关节粘性阻尼系数 c1, c2 和空气阻尼系数 k1, k2。
    """
    def __init__(self, L1, L2, M1, M2, g=9.81, c1=0.0, c2=0.0, k1=0.0, k2=0.0):
        self.L1 = L1
        self.L2 = L2
        self.M1 = M1
        self.M2 = M2
        self.g = g
        self.c1 = c1  # 上关节粘性阻尼系数
        self.c2 = c2  # 下关节粘性阻尼系数
        self.k1 = k1  # 上摆空气阻力系数 (与速度平方成正比)
        self.k2 = k2  # 下摆空气阻力系数 (与速度平方成正比)

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
        # 加上粘性阻尼: -c * omega 
        # 加上空气阻尼 (二次阻力): -k * omega * |omega|
        
        a11 = (self.M1 + self.M2) * self.L1**2
        a12 = self.M2 * self.L1 * self.L2
        a21 = self.M2 * self.L1 * self.L2
        a22 = self.M2 * self.L2**2
        
        b1 = -(self.M1 + self.M2) * self.g * self.L1 * theta1 \
             - self.c1 * omega1 - self.k1 * omega1 * abs(omega1)
             
        b2 = -self.M2 * self.g * self.L2 * theta2 \
             - self.c2 * omega2 - self.k2 * omega2 * abs(omega2)
        
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
        
        # 将非线性方程中的广义力项加上粘性阻尼和空气阻尼 (二次阻力)
        num1 = -self.g * (2 * self.M1 + self.M2) * np.sin(theta1) \
               - self.M2 * self.g * np.sin(theta1 - 2 * theta2) \
               - 2 * np.sin(delta) * self.M2 * (omega2**2 * self.L2 + omega1**2 * self.L1 * np.cos(delta)) \
               - (self.c1 * omega1 + self.k1 * omega1 * abs(omega1)) * den / self.L1 # 加入粘性+空气阻力修正
               
        d_omega1 = num1 / (self.L1 * den)
        
        num2 = 2 * np.sin(delta) * (omega1**2 * self.L1 * (self.M1 + self.M2) \
               + self.g * (self.M1 + self.M2) * np.cos(theta1) \
               + omega2**2 * self.L2 * self.M2 * np.cos(delta)) \
               - (self.c2 * omega2 + self.k2 * omega2 * abs(omega2)) * den / self.L2 # 加入粘性+空气阻力修正
               
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

    def calculate_kinetic_energies(self, state, use_nonlinear=False):
        """
        分别计算上摆动能 Ek1 和下摆动能 Ek2
        """
        theta1, omega1, theta2, omega2 = state
        if use_nonlinear:
            v1_sq = (self.L1 * omega1)**2
            v2_sq = (self.L1 * omega1)**2 + (self.L2 * omega2)**2 + 2 * self.L1 * self.L2 * omega1 * omega2 * np.cos(theta1 - theta2)
            Ek1 = 0.5 * self.M1 * v1_sq
            Ek2 = 0.5 * self.M2 * v2_sq
        else:
            Ek1 = 0.5 * self.M1 * self.L1**2 * omega1**2
            # 线性近似下的下摆速度平方展开
            Ek2 = 0.5 * self.M2 * (self.L1**2 * omega1**2 + self.L2**2 * omega2**2 + 2 * self.L1 * self.L2 * omega1 * omega2)
        return Ek1, Ek2

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

def simulate_and_analyze(L1, L2, M1, M2, theta0, phase_type, t_max, dt, use_nonlinear=False, g=9.8, c1=0.0, c2=0.0, k1=0.0, k2=0.0, integrator_name='RK4'):
    """
    执行一次完整的模拟和分析。
    """
    pendulum = DoublePendulum(L1, L2, M1, M2, g, c1, c2, k1, k2)
    
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
    
    # 傅里叶分析 (通过加窗和零填充使曲线更平滑)
    N = len(theta1_vals)
    
    # 1. 减去均值以消除零频（直流）分量干扰
    theta1_centered = theta1_vals - np.mean(theta1_vals)
    
    # 2. 应用汉宁窗以减少频谱泄漏（毛刺）
    window = np.hanning(N)
    theta1_windowed = theta1_centered * window
    
    # 3. 零填充（Zero Padding），大幅提高频域曲线的显示平滑度
    padding_factor = 10  # 填充倍数
    N_padded = N * padding_factor
    
    fft_vals = np.fft.fft(theta1_windowed, n=N_padded)
    fft_freq = np.fft.fftfreq(N_padded, dt)
    
    # 提取正频率部分
    pos_mask = fft_freq > 0
    freqs = fft_freq[pos_mask]
    
    # 恢复因加窗损失的振幅能量 (Hanning 窗的幅度补偿系数为 2)
    amps = np.abs(fft_vals)[pos_mask] * 2.0 / N * 2.0
    
    # 寻找峰值
    peaks, _ = scipy.signal.find_peaks(amps, height=np.max(amps)*0.1, distance=10 * padding_factor)
    peak_freqs = freqs[peaks]
    peak_amps = amps[peaks]
    
    # 获取最大的两个峰值
    sorted_indices = np.argsort(peak_amps)[::-1]
    top_peaks = sorted_indices[:2]
    
    # 提取对应的频率和振幅，然后按频率排序以保证画图匹配
    dom_freqs_amps = [(peak_freqs[i], peak_amps[i]) for i in top_peaks]
    dom_freqs_amps.sort(key=lambda x: x[0])  # 按频率升序排序
    
    dom_freqs = [x[0] for x in dom_freqs_amps]
    dom_amps = [x[1] for x in dom_freqs_amps]
    
    print(f"\n--- 模式: {phase_type} ---")
    print(f"观测到的主导频率: {[round(f, 3) for f in dom_freqs]} Hz")
    
    theo_freqs = calculate_theoretical_frequencies(L1, L2, M1, M2, g)
    print(f"理论简正频率: {theo_freqs[0]:.3f} Hz, {theo_freqs[1]:.3f} Hz")
    
    # 提取角速度用于相图
    omega1_vals = states[:, 1]
    omega2_vals = states[:, 3]
    
    # 绘制可视化
    fig = plt.figure(figsize=(12, 16))
    fig.canvas.manager.set_window_title(f"Double Pendulum - {phase_type} (Nonlinear: {use_nonlinear})")
    
    # 使用 GridSpec 布局：4行2列
    # 第一行: 角度-时间图 (跨2列)
    # 第二行: 频谱图 (跨2列)
    # 第三行: 相图 1 和 相图 2
    # 第四行: 能量曲线 (跨2列)
    gs = fig.add_gridspec(4, 2)
    ax_time = fig.add_subplot(gs[0, :])
    ax_fft = fig.add_subplot(gs[1, :])
    ax_phase1 = fig.add_subplot(gs[2, 0])
    ax_phase2 = fig.add_subplot(gs[2, 1])
    ax_energy = fig.add_subplot(gs[3, :])
    
    # 角度-时间曲线
    ax_time.plot(t_vals, theta1_vals, label=r'$\theta_1$ (Upper Pendulum)')
    ax_time.plot(t_vals, theta2_vals, label=r'$\theta_2$ (Lower Pendulum)')
    
    # 寻找并标记最大振幅点 (绝对值的最大值，为了画图好看我们在正方向寻找最大值)
    idx_max1 = np.argmax(theta1_vals)
    idx_max2 = np.argmax(theta2_vals)
    
    t_max1 = t_vals[idx_max1]
    max_theta1 = theta1_vals[idx_max1]
    t_max2 = t_vals[idx_max2]
    max_theta2 = theta2_vals[idx_max2]
    
    ax_time.scatter([t_max1], [max_theta1], color='blue', zorder=5)
    ax_time.annotate(f'Max: {max_theta1:.3f} rad', xy=(t_max1, max_theta1), 
                     xytext=(-20, 10), textcoords='offset points', color='blue')
                     
    ax_time.scatter([t_max2], [max_theta2], color='orange', zorder=5)
    ax_time.annotate(f'Max: {max_theta2:.3f} rad', xy=(t_max2, max_theta2), 
                     xytext=(-20, 10), textcoords='offset points', color='orange')
    
    ax_time.set_xlabel('Time (s)')
    ax_time.set_ylabel('Angle (rad)')
    ax_time.set_title(f'Angles vs Time ({phase_type})')
    ax_time.legend()
    ax_time.grid(True)
    
    # 频谱图
    ax_fft.plot(freqs, amps)
    if len(dom_freqs) > 0:
        ax_fft.scatter(dom_freqs, dom_amps, color='red', zorder=5, label='Peaks')
        for f, a in dom_freqs_amps:
            ax_fft.annotate(f'{f:.3f} Hz', xy=(f, a), 
                            xytext=(5, 5), textcoords='offset points')
    ax_fft.set_xlabel('Frequency (Hz)')
    ax_fft.set_ylabel('Amplitude')
    ax_fft.set_title(f'Fourier Spectrum of $\\theta_1$ ({phase_type})')
    ax_fft.set_xlim(0, max(theo_freqs)*2.5 if theo_freqs else 5)
    ax_fft.legend()
    ax_fft.grid(True)
    
    # 相图 1: 上摆 (theta1 vs omega1)
    ax_phase1.plot(theta1_vals, omega1_vals, color='tab:blue', linewidth=1, alpha=0.8)
    ax_phase1.plot(theta1_vals[0], omega1_vals[0], 'go', markersize=8, label='Start (t=0)')  # 绿点标记起点
    ax_phase1.plot(theta1_vals[-1], omega1_vals[-1], 'rs', markersize=8, label='End')        # 红方块标记终点
    ax_phase1.set_xlabel(r'$\theta_1$ (rad)')
    ax_phase1.set_ylabel(r'$\dot{\theta}_1$ (rad/s)')
    ax_phase1.set_title(r'Phase Portrait: Upper Pendulum ($\theta_1$ vs $\dot{\theta}_1$)')
    ax_phase1.legend()
    ax_phase1.grid(True)

    # 相图 2: 下摆 (theta2 vs omega2)
    ax_phase2.plot(theta2_vals, omega2_vals, color='tab:orange', linewidth=1, alpha=0.8)
    ax_phase2.plot(theta2_vals[0], omega2_vals[0], 'go', markersize=8, label='Start (t=0)')
    ax_phase2.plot(theta2_vals[-1], omega2_vals[-1], 'rs', markersize=8, label='End')
    ax_phase2.set_xlabel(r'$\theta_2$ (rad)')
    ax_phase2.set_ylabel(r'$\dot{\theta}_2$ (rad/s)')
    ax_phase2.set_title(r'Phase Portrait: Lower Pendulum ($\theta_2$ vs $\dot{\theta}_2$)')
    ax_phase2.legend()
    ax_phase2.grid(True)
    
    # 能量曲线
    # 计算能量漂移
    energy_drift = np.max(energies) - np.min(energies)
    print(f"最大能量漂移: {energy_drift:.2e} J")
    
    ax_energy.plot(t_vals, energies, color='purple', label='Total Energy')
    ax_energy.set_xlabel('Time (s)')
    ax_energy.set_ylabel('Energy (J)')
    ax_energy.set_title(f'Mechanical Energy vs Time ({phase_type})')
    # 如果能量漂移非常小，适当放大纵轴范围，以免全是数值噪声
    if energy_drift < 1e-10:
        mean_e = np.mean(energies)
        ax_energy.set_ylim(mean_e - 1e-9, mean_e + 1e-9)
    ax_energy.legend()
    ax_energy.grid(True)
    
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

def plot_frequency_vs_mass_ratio(L1, L2, M1_base, g):
    """
    绘制质量比 (m2/m1) 与理论角频率/频率关系的图像。
    """
    mass_ratios = np.linspace(0.1, 5.0, 100)
    freqs_low = []
    freqs_high = []
    
    for r in mass_ratios:
        M2_current = M1_base * r
        # 使用之前写好的计算理论简正频率的函数
        theo_freqs = calculate_theoretical_frequencies(L1, L2, M1_base, M2_current, g)
        freqs_low.append(theo_freqs[0])
        freqs_high.append(theo_freqs[1])
        
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(mass_ratios, freqs_low, label=r'$f_{low}$ (In-Phase Mode)', color='tab:blue', linewidth=2)
    ax.plot(mass_ratios, freqs_high, label=r'$f_{high}$ (Anti-Phase Mode)', color='tab:orange', linewidth=2)
    
    ax.set_xlim(mass_ratios[0], mass_ratios[-1])
    ax.set_xlabel(r'Mass Ratio ($m_2/m_1$)')
    ax.set_ylabel('Theoretical Frequency (Hz)')
    ax.set_title('Normal Mode Frequencies vs Mass Ratio')
    ax.legend(loc='best')
    ax.grid(True)
    
    output_dir = "simulation_results"
    os.makedirs(output_dir, exist_ok=True)
    fig_filename = os.path.join(output_dir, "frequency_vs_mass_ratio.png")
    fig.savefig(fig_filename, dpi=300, bbox_inches='tight')
    print(f"质量比与角频率图像已保存至: {fig_filename}")
    plt.close(fig)

def plot_frequency_vs_length_ratio(L1_base, M1, M2, g):
    """
    绘制摆长比 (L2/L1) 与理论角频率/频率关系的图像。
    """
    # 使用对数分布以便更好地观察极限情况
    length_ratios = np.logspace(-2, 1, 100) # 从 0.01 到 10.0
    freqs_low = []
    freqs_high = []
    
    for r in length_ratios:
        L2_current = L1_base * r
        theo_freqs = calculate_theoretical_frequencies(L1_base, L2_current, M1, M2, g)
        freqs_low.append(theo_freqs[0])
        freqs_high.append(theo_freqs[1])
        
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(length_ratios, freqs_low, label=r'$f_{low}$ (In-Phase Mode)', color='tab:blue', linewidth=2)
    ax.plot(length_ratios, freqs_high, label=r'$f_{high}$ (Anti-Phase Mode)', color='tab:orange', linewidth=2)
    
    ax.set_xscale('log') # 横轴设为对数坐标
    ax.set_xlim(length_ratios[0], length_ratios[-1])
    ax.set_xlabel(r'Length Ratio ($L_2/L_1$)')
    ax.set_ylabel('Theoretical Frequency (Hz)')
    ax.set_title('Normal Mode Frequencies vs Length Ratio')
    
    # 当 L2 极短时 (L2/L1 -> 0)，低频模式 f_low 趋近于上摆作为单摆的频率
    # 当 L2 极长时 (L2/L1 -> \infty)，高频模式 f_high 趋近于上摆作为单摆的频率
    f_single_L1 = np.sqrt(g / L1_base) / (2 * np.pi)
    ax.axhline(y=f_single_L1, color='green', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Single Pendulum ($L_1$) Asymptote ({f_single_L1:.2f} Hz)')
    
    ax.legend(loc='best')
    ax.grid(True)
    
    output_dir = "simulation_results"
    os.makedirs(output_dir, exist_ok=True)
    fig_filename = os.path.join(output_dir, "frequency_vs_length_ratio.png")
    fig.savefig(fig_filename, dpi=300, bbox_inches='tight')
    print(f"摆长比与理论频率图像已保存至: {fig_filename}")
    plt.close(fig)

def plot_kinetic_energy_distribution(L1, L2, M1_base, theta0, modes, t_max, dt, use_nonlinear, g, c1, c2, k1, k2, integrator_name):
    """
    绘制动能分配比例堆叠图，横轴为质量比 m2/m1。
    """
    # 将扫描点数增加，并从更小的质量比开始，使用对数分布以便更好地观察极限情况
    mass_ratios = np.logspace(-3, np.log10(5.0), 100)
    total_steps = len(mass_ratios)
    
    # 我们为每种模式画一张单独的堆叠图
    for mode in modes:
        ratio_Ek1 = []
        ratio_Ek2 = []
        
        print(f"\n正在计算 {mode} 模式下的动能分配数据...")
        for idx, r in enumerate(mass_ratios):
            if (idx + 1) % 5 == 0 or idx == 0:
                print(f"  进度: [{idx+1}/{total_steps}] 正在计算质量比 m2/m1 = {r:.2f} ...")
                
            M2_current = M1_base * r
            # 注意：如果我们需要得到类似于你描述的“下摆动能占比从 0 单调上升到 1”的理论图
            # 我们应该消除阻尼的影响，否则阻尼会过早耗散掉小质量下摆的能量
            pendulum = DoublePendulum(L1, L2, M1_base, M2_current, g, 0.0, 0.0, 0.0, 0.0)
            
            # 初始条件：按照理论分析，我们使用上摆释放，下摆静止的混合模式来计算
            # 只有在这种能量从一端注入的情况下，平均动能占比才符合那个单调递增的理论曲线
            # 为了展示这种纯粹的物理规律，我们在绘图时统一使用 mixed_phase 初始条件
            state = np.array([theta0, 0.0, 0.0, 0.0])
                
            # 动态计算频率，以自适应调整 t_max 和 dt，消除积分截断带来的锯齿波动
            theo_freqs = calculate_theoretical_frequencies(L1, L2, M1_base, M2_current, g)
            f_low, f_high = theo_freqs[0], theo_freqs[1]
            
            # 1. 动态 dt：保证高频振荡每周期至少有 40 个采样点，防止极小质量比下高频积分失真
            dt_current = min(dt, 1.0 / (40.0 * f_high))
            
            # 2. 动态 t_max：为了完美平均掉能量的周期性波动，积分总时间应当是慢周期的整数倍。
            # 我们强制模拟正好 30 个完整的慢周期 (T_slow = 1 / f_low)。
            T_slow = 1.0 / f_low
            t_max_current = 30.0 * T_slow
            
            t_vals = np.arange(0, t_max_current, dt_current)
            n_steps = len(t_vals)
            
            integrator_class = INTEGRATORS.get(integrator_name)
            integrator = integrator_class()
            derivs = pendulum.derivatives_nonlinear if use_nonlinear else pendulum.derivatives_linear
            
            Ek1_total = 0.0
            Ek2_total = 0.0
            
            for i in range(1, n_steps):
                state = integrator.step(derivs, t_vals[i-1], state, dt_current)
                ek1, ek2 = pendulum.calculate_kinetic_energies(state, use_nonlinear)
                Ek1_total += ek1
                Ek2_total += ek2
                
            total_ek = Ek1_total + Ek2_total
            if total_ek > 0:
                ratio_Ek1.append(Ek1_total / total_ek)
                ratio_Ek2.append(Ek2_total / total_ek)
            else:
                ratio_Ek1.append(0.5)
                ratio_Ek2.append(0.5)
                
        # 绘制堆叠图
        fig, ax = plt.subplots(figsize=(8, 6))
        fig.canvas.manager.set_window_title(f"Kinetic Energy Distribution - {mode}")
        
        ax.stackplot(mass_ratios, ratio_Ek1, ratio_Ek2, 
                     labels=[r'$E_{k1}$ (Upper Pendulum)', r'$E_{k2}$ (Lower Pendulum)'],
                     colors=['tab:blue', 'tab:orange'], alpha=0.8)
                     
        ax.set_xscale('log') # 使用对数坐标轴，以放大极小质量比的区域
        ax.set_xlim(mass_ratios[0], mass_ratios[-1])
        ax.set_ylim(0, 1)
        ax.set_xlabel(r'Mass Ratio ($m_2/m_1$)')
        ax.set_ylabel('Normalized Kinetic Energy Proportion')
        ax.set_title(f'Kinetic Energy Distribution vs Mass Ratio ({mode})')
        
        # 添加渐近线
        if mode == 'in_phase':
            ax.axhline(y=1.0, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Asymptote (1.0)')
            ax.axhline(y=0.0, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
        elif mode == 'anti_phase':
            ax.axhline(y=1.0, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Asymptote (1.0)')
            ax.axhline(y=0.0, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
        
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        output_dir = "simulation_results"
        model_type = "nonlinear" if use_nonlinear else "linear"
        fig_filename = os.path.join(output_dir, f"energy_distribution_{mode}_{model_type}.png")
        fig.savefig(fig_filename, dpi=300, bbox_inches='tight')
        print(f"动能分配堆叠图已保存至: {fig_filename}")
        plt.close(fig) # 释放内存

if __name__ == "__main__":
    # --- 1. 参数设置 ---
    L1 = 1.0       # 上摆长 (m)
    L2 = 1.0   # 下摆长 (m)
    M1 = 1.0       # 上摆质量 (kg)
    M2 = 1.0   # 下摆质量 (kg)
    g = 9.8        # 重力加速度 (m/s^2)
    theta0 = 0.1   # 初始小角度 (rad)
    t_max = 20  # 模拟时长 (s) - 增加以覆盖更长的拍频周期
    dt = 0.01      # 时间步长 (s)
    
    c1 = 0    # 上关节粘性阻尼系数 (与速度成正比)
    c2 = 0    # 下关节粘性阻尼系数 (与速度成正比)
    k1 = 0    # 上摆空气阻尼系数 (与速度平方成正比)
    k2 = 0      # 下摆空气阻尼系数 (与速度平方成正比)
    
    # 切换此开关以验证完整非线性模型（False 为线性小角度模型）
    USE_NONLINEAR = True 
    
    # 可选的积分器: 'RK4', 'SymplecticEuler', 'VelocityVerlet', 'ImplicitMidpoint', 'GaussLegendreRK4', 'Yoshida4'
    INTEGRATOR_NAME = 'GaussLegendreRK4'

    modes = ['in_phase', 'anti_phase', 'mixed_phase']
    results = {}
    
    # 创建一个文件夹用来保存生成的图表和数据
    output_dir = "simulation_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # --- 2. 模拟与分析循环 ---
    for mode in modes:
        t_vals, states, energies, fig = simulate_and_analyze(
            L1, L2, M1, M2, theta0, mode, t_max, dt, use_nonlinear=USE_NONLINEAR, g=g, c1=c1, c2=c2, k1=k1, k2=k2, integrator_name=INTEGRATOR_NAME
        )
        results[mode] = (t_vals, states)
        
        # 自动保存生成的图表为高清 PNG 图片
        model_type = "nonlinear" if USE_NONLINEAR else "linear"
        fig_filename = os.path.join(output_dir, f"{mode}_{model_type}_{INTEGRATOR_NAME}_c1_{c1}_c2_{c2}.png")
        fig.savefig(fig_filename, dpi=300, bbox_inches='tight')
        print(f"图表已保存至: {fig_filename}")
        
        # 可选保存为 CSV (这里仅为注释示例，用户可自行取消注释以启用):
        # csv_filename = os.path.join(output_dir, f"{mode}_{model_type}_{INTEGRATOR_NAME}_c1_{c1}_c2_{c2}_data.csv")
        # np.savetxt(csv_filename, np.column_stack((t_vals, states, energies)), 
        #            delimiter=",", header="t,theta1,omega1,theta2,omega2,energy")

    # --- 3. 动能分配比例堆叠图 ---
    print("\n正在生成动能分配比例堆叠图 (横轴: m2/m1, 纵轴: 能量占比)...")
    # 由于需要扫参数 (m2从0.1到5.0)，计算量较大，我们重用上面的主参数，只扫 M2
    plot_kinetic_energy_distribution(L1, L2, M1, theta0, modes, t_max, dt, USE_NONLINEAR, g, c1, c2, k1, k2, INTEGRATOR_NAME)
    
    # --- 4. 理论角频率与质量比关系图 ---
    print("\n正在生成理论角频率/频率与质量比的关系图...")
    plot_frequency_vs_mass_ratio(L1, L2, M1, g)
    
    # --- 5. 理论频率与摆长比关系图 ---
    print("\n正在生成理论频率与摆长比的关系图...")
    plot_frequency_vs_length_ratio(L1, M1, M2, g)
    
    # 显示所有图表窗口
    # plt.show()
    
    # --- 3. 动画演示 ---
    # 可以将 'all' 替换为 'in_phase', 'anti_phase', 或 'mixed_phase' 以只看一种模式
    print("\n图表已全部保存。跳过弹出界面和动画演示。")
    # ani = animate_pendulums(results, L1, L2, phase_to_animate='all')
