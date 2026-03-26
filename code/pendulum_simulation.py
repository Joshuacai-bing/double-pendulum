"""
单摆物理模拟 - 生成动画视频/GIF
使用matplotlib生成高质量的单摆动画
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation, PillowWriter
import os

# 配置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 物理参数
L = 1.0  # 摆长 (m)
g = 9.81  # 重力加速度 (m/s^2)
m = 1.0  # 摆球质量 (kg)
theta0 = np.pi / 3  # 初始角度 (60度)
omega0 = 0.0  # 初始角速度 (rad/s)

# Simulation parameters
t_max = 10.0  # Total simulation time (s)
dt = 0.04  # Time step (s) - increased to reduce frame count
t = np.arange(0, t_max, dt)

# 使用欧拉-克罗默方法求解运动方程
def simulate_pendulum(L, g, theta0, omega0, t, dt):
    """使用欧拉-克罗默方法求解单摆运动方程"""
    theta = np.zeros(len(t))
    omega = np.zeros(len(t))
    
    theta[0] = theta0
    omega[0] = omega0
    
    for i in range(1, len(t)):
        # 角加速度
        alpha = -(g / L) * np.sin(theta[i-1])
        
        # 更新角速度和角度（欧拉-克罗默方法）
        omega[i] = omega[i-1] + alpha * dt
        theta[i] = theta[i-1] + omega[i] * dt
    
    return theta, omega

# 运行模拟
theta, omega = simulate_pendulum(L, g, theta0, omega0, t, dt)

# 计算位置
x = L * np.sin(theta)
y = -L * np.cos(theta)

# 计算能量
kinetic_energy = 0.5 * m * (L * omega) ** 2
potential_energy = m * g * (L + y)
total_energy = kinetic_energy + potential_energy

print(f"Simulation completed! Total frames: {len(t)}")
print(f"Pendulum period (small angle approximation): {2 * np.pi * np.sqrt(L/g):.3f} s")

# Create figure
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Pendulum Physics Simulation', fontsize=16, fontweight='bold')

# Subplot 1: Pendulum animation
ax1 = axes[0, 0]
ax1.set_xlim(-1.5, 1.5)
ax1.set_ylim(-1.5, 0.5)
ax1.set_aspect('equal')
ax1.grid(True, alpha=0.3)
ax1.set_title('Pendulum Motion')
ax1.set_xlabel('x (m)')
ax1.set_ylabel('y (m)')

# 绘制支点
pivot = Circle((0, 0), 0.03, color='black', zorder=5)
ax1.add_patch(pivot)

# 绘制摆线
line, = ax1.plot([], [], 'b-', linewidth=2, zorder=2)

# 绘制摆球
ball = Circle((0, 0), 0.08, color='red', zorder=4)
ax1.add_patch(ball)

# 绘制轨迹
trail, = ax1.plot([], [], 'r--', alpha=0.5, linewidth=1)
trail_x = []
trail_y = []

# Subplot 2: Angle vs time
ax2 = axes[0, 1]
ax2.set_xlim(0, t_max)
ax2.set_ylim(-np.pi, np.pi)
ax2.grid(True, alpha=0.3)
ax2.set_title('Angle vs Time')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Angle (rad)')
theta_line, = ax2.plot([], [], 'b-', linewidth=1.5)
theta_point, = ax2.plot([], [], 'ro', markersize=6)

# Subplot 3: Phase space
ax3 = axes[1, 0]
ax3.set_xlim(-np.pi, np.pi)
ax3.set_ylim(-8, 8)
ax3.grid(True, alpha=0.3)
ax3.set_title('Phase Space (Angle - Angular Velocity)')
ax3.set_xlabel('Angle (rad)')
ax3.set_ylabel('Angular Velocity (rad/s)')
phase_line, = ax3.plot([], [], 'g-', linewidth=1, alpha=0.7)
phase_point, = ax3.plot([], [], 'ro', markersize=6)

# Subplot 4: Energy vs time
ax4 = axes[1, 1]
ax4.set_xlim(0, t_max)
energy_margin = 0.1 * np.max(total_energy)
ax4.set_ylim(0, np.max(total_energy) + energy_margin)
ax4.grid(True, alpha=0.3)
ax4.set_title('Energy vs Time')
ax4.set_xlabel('Time (s)')
ax4.set_ylabel('Energy (J)')
ke_line, = ax4.plot([], [], 'r-', linewidth=1.5, label='Kinetic Energy')
pe_line, = ax4.plot([], [], 'b-', linewidth=1.5, label='Potential Energy')
te_line, = ax4.plot([], [], 'g--', linewidth=2, label='Total Energy')
ax4.legend(loc='upper right')

# 添加时间显示
time_text = ax1.text(0.02, 0.98, '', transform=ax1.transAxes, 
                     fontsize=12, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# 初始化函数
def init():
    line.set_data([], [])
    ball.center = (0, -L)
    trail.set_data([], [])
    theta_line.set_data([], [])
    theta_point.set_data([], [])
    phase_line.set_data([], [])
    phase_point.set_data([], [])
    ke_line.set_data([], [])
    pe_line.set_data([], [])
    te_line.set_data([], [])
    time_text.set_text('')
    return line, ball, trail, theta_line, theta_point, phase_line, phase_point, ke_line, pe_line, te_line, time_text

# 动画更新函数
def update(frame):
    # 更新摆线
    line.set_data([0, x[frame]], [0, y[frame]])
    
    # 更新摆球位置
    ball.center = (x[frame], y[frame])
    
    # 更新轨迹
    trail_x.append(x[frame])
    trail_y.append(y[frame])
    if len(trail_x) > 100:  # 只保留最近100个点
        trail_x.pop(0)
        trail_y.pop(0)
    trail.set_data(trail_x, trail_y)
    
    # 更新角度图
    theta_line.set_data(t[:frame+1], theta[:frame+1])
    theta_point.set_data([t[frame]], [theta[frame]])
    
    # 更新相空间图
    phase_line.set_data(theta[:frame+1], omega[:frame+1])
    phase_point.set_data([theta[frame]], [omega[frame]])
    
    # Update energy plot
    ke_line.set_data(t[:frame+1], kinetic_energy[:frame+1])
    pe_line.set_data(t[:frame+1], potential_energy[:frame+1])
    te_line.set_data(t[:frame+1], total_energy[:frame+1])
    
    # Update time display
    time_text.set_text(f'Time: {t[frame]:.2f} s\n'
                       f'Angle: {theta[frame]:.3f} rad\n'
                       f'Angular Velocity: {omega[frame]:.3f} rad/s')
    
    return line, ball, trail, theta_line, theta_point, phase_line, phase_point, ke_line, pe_line, te_line, time_text

# Create animation
print("Creating animation...")
anim = FuncAnimation(fig, update, init_func=init, frames=len(t), 
                     interval=dt*1000, blit=True)

# Ensure output directory exists
output_dir = '../output'
os.makedirs(output_dir, exist_ok=True)

# Save as GIF animation
gif_path = os.path.join(output_dir, 'pendulum_simulation.gif')
print(f"Saving GIF animation to: {gif_path}")

# Use PillowWriter to save GIF
writer = PillowWriter(fps=25)  # Reduced fps for faster generation
anim.save(gif_path, writer=writer)

print(f"GIF animation saved to: {gif_path}")
plt.close()

# Create a static summary plot
fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))
fig2.suptitle('Pendulum Simulation Summary', fontsize=16, fontweight='bold')

# Subplot 1: Pendulum trajectory
ax1 = axes2[0, 0]
ax1.set_xlim(-1.5, 1.5)
ax1.set_ylim(-1.5, 0.5)
ax1.set_aspect('equal')
ax1.grid(True, alpha=0.3)
ax1.set_title('Pendulum Trajectory')
ax1.set_xlabel('x (m)')
ax1.set_ylabel('y (m)')
ax1.plot(x, y, 'b-', alpha=0.7, linewidth=1)
ax1.plot(0, 0, 'ko', markersize=8)
ax1.plot(x[-1], y[-1], 'ro', markersize=10)

# Subplot 2: Angle vs time
ax2 = axes2[0, 1]
ax2.grid(True, alpha=0.3)
ax2.set_title('Angle vs Time')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Angle (rad)')
ax2.plot(t, theta, 'b-', linewidth=1.5)

# Subplot 3: Phase space
ax3 = axes2[1, 0]
ax3.grid(True, alpha=0.3)
ax3.set_title('Phase Space')
ax3.set_xlabel('Angle (rad)')
ax3.set_ylabel('Angular Velocity (rad/s)')
ax3.plot(theta, omega, 'g-', linewidth=1, alpha=0.7)

# Subplot 4: Energy vs time
ax4 = axes2[1, 1]
ax4.grid(True, alpha=0.3)
ax4.set_title('Energy vs Time')
ax4.set_xlabel('Time (s)')
ax4.set_ylabel('Energy (J)')
ax4.plot(t, kinetic_energy, 'r-', linewidth=1.5, label='Kinetic Energy')
ax4.plot(t, potential_energy, 'b-', linewidth=1.5, label='Potential Energy')
ax4.plot(t, total_energy, 'g--', linewidth=2, label='Total Energy')
ax4.legend()

plt.tight_layout()
summary_path = os.path.join(output_dir, 'pendulum_summary.png')
plt.savefig(summary_path, dpi=300, bbox_inches='tight')
print(f"Summary plot saved to: {summary_path}")
plt.close()

print("\nSimulation completed!")
print(f"GIF animation file: {gif_path}")
print(f"Summary plot: {summary_path}")
