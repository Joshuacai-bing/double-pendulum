import numpy as np 
import matplotlib.pyplot as plt 
from scipy.integrate import solve_ivp 
from matplotlib.animation import FuncAnimation 
 
# 参数设置 
g = 9.8        # 重力加速度 (m/s²) 
L = 1.0        # 摆长 (m) 
t_max = 10.0   # 模拟时长 (秒) 
fps = 60       # 动画帧率 
 
# 初始条件 (小角度，单位弧度) 
theta1_0 = 0.1   # 上摆初始角 (约5.7°) 
theta2_0 = -0.05 # 下摆初始角 
omega1_0 = 0.0 
omega2_0 = 0.0 
 
y0 = [theta1_0, omega1_0, theta2_0, omega2_0] 
 
# 定义微分方程 
def small_angle_pendulum(t, y): 
    theta1, omega1, theta2, omega2 = y 
    dtheta1 = omega1 
    domega1 = -(g/L) * (2*theta1 - theta2) 
    dtheta2 = omega2 
    domega2 = -(g/L) * (2*theta2 - theta1) 
    return [dtheta1, domega1, dtheta2, domega2] 
 
# 求解 
t_eval = np.linspace(0, t_max, int(t_max * fps)) 
sol = solve_ivp(small_angle_pendulum, [0, t_max], y0, t_eval=t_eval, method='RK45') 
 
theta1 = sol.y[0] 
theta2 = sol.y[2] 
 
# 将角度转换为笛卡尔坐标用于绘图 
x1 = L * np.sin(theta1) 
y1 = -L * np.cos(theta1) 
x2 = x1 + L * np.sin(theta2) 
y2 = y1 - L * np.cos(theta2) 
 
# 绘制角度随时间变化 
plt.rcParams['font.sans-serif'] = ['SimHei'] # 指定默认字体为黑体
plt.rcParams['axes.unicode_minus'] = False   # 解决保存图像是负号'-'显示为方块的问题

plt.figure(figsize=(12, 4)) 
plt.subplot(1, 2, 1) 
plt.plot(sol.t, theta1 * 180/np.pi, label='θ1') 
plt.plot(sol.t, theta2 * 180/np.pi, label='θ2') 
plt.xlabel('时间 (s)') 
plt.ylabel('角度 (度)') 
plt.legend() 
plt.grid(True) 
 
# 动画 
fig, ax = plt.subplots(figsize=(5, 5)) 
ax.set_xlim(-2.2, 2.2) 
ax.set_ylim(-2.2, 2.2) 
ax.set_aspect('equal') 
line, = ax.plot([], [], 'o-', lw=2, markersize=8) 
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes) 
 
def init(): 
    line.set_data([], []) 
    time_text.set_text('') 
    return line, time_text 
 
def animate(i): 
    xx = [0, x1[i], x2[i]] 
    yy = [0, y1[i], y2[i]] 
    line.set_data(xx, yy) 
    time_text.set_text(f't = {sol.t[i]:.2f} s') 
    return line, time_text 
 
ani = FuncAnimation(fig, animate, frames=len(sol.t), init_func=init, interval=1000/fps, blit=True) 
 
plt.tight_layout() 
plt.show()