import numpy as np
from abc import ABC, abstractmethod

class Integrator(ABC):
    """
    数值积分算法的抽象基类。
    """
    @abstractmethod
    def step(self, f, t, state, dt):
        """
        执行单步积分。
        
        :param f: 导数函数，形式为 f(t, state) 返回状态导数
        :param t: 当前时间
        :param state: 当前状态向量
        :param dt: 时间步长
        :return: 下一步的状态向量
        """
        pass

class RK4(Integrator):
    """
    经典四阶 Runge-Kutta 法 (显式，非辛)。
    稳定、精度高，常用于普通 ODE 求解。
    """
    def step(self, f, t, state, dt):
        k1 = f(t, state)
        k2 = f(t + dt/2, state + k1 * dt/2)
        k3 = f(t + dt/2, state + k2 * dt/2)
        k4 = f(t + dt, state + k3 * dt)
        return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

class SymplecticIntegrator(Integrator):
    """
    辛积分器基类。
    由于辛积分需要对坐标 q 和动量/速度 v 进行分步交替更新，
    这里提取出 q 和 v 的索引。
    默认针对双摆的 [theta1, omega1, theta2, omega2] 状态结构，
    其中 q_indices 为 [0, 2]，v_indices 为 [1, 3]。
    """
    def __init__(self, q_indices=[0, 2], v_indices=[1, 3]):
        self.q_idx = q_indices
        self.v_idx = v_indices

    def get_a(self, f, t, state):
        # 计算当前的加速度 (即 dv/dt)
        return f(t, state)[self.v_idx]

    def update_state(self, state, q, v):
        # 构建并返回更新后的状态向量
        new_state = np.copy(state)
        new_state[self.q_idx] = q
        new_state[self.v_idx] = v
        return new_state

class SymplecticEuler(SymplecticIntegrator):
    """
    辛欧拉法 (1阶，半隐式)。
    最简单的辛结构原型，常用于教学、构造高阶方法。
    """
    def step(self, f, t, state, dt):
        q = state[self.q_idx]
        v = state[self.v_idx]
        
        # v_{n+1} = v_n + a(q_n, v_n) * dt
        a = self.get_a(f, t, state)
        v_new = v + a * dt
        
        # q_{n+1} = q_n + v_{n+1} * dt
        q_new = q + v_new * dt
        
        return self.update_state(state, q_new, v_new)

class VelocityVerlet(SymplecticIntegrator):
    """
    速度韦尔莱法 (2阶，显式)。
    稳定、简单，分子动力学和行星轨道的标准辛算法。
    注：此处使用了一种预测-校正的形式来近似处理速度相关的加速度。
    """
    def step(self, f, t, state, dt):
        q = state[self.q_idx]
        v = state[self.v_idx]
        
        # 1. 预测速度半步: v_{n+1/2} = v_n + a_n * dt / 2
        a1 = self.get_a(f, t, state)
        v_half = v + a1 * dt / 2.0
        
        # 2. 更新位置全步: q_{n+1} = q_n + v_{n+1/2} * dt
        q_new = q + v_half * dt
        
        # 3. 计算新位置处的加速度 (用 v_{n+1/2} 作为速度近似)
        mid_state = self.update_state(state, q_new, v_half)
        a2 = self.get_a(f, t + dt, mid_state)
        
        # 4. 更新速度后半步: v_{n+1} = v_{n+1/2} + a_{n+1} * dt / 2
        v_new = v_half + a2 * dt / 2.0
        
        return self.update_state(state, q_new, v_new)

class ImplicitMidpoint(Integrator):
    """
    隐式中点法 (2阶，隐式辛积分器)。
    对称性好，基础模块，常用于刚性问题、构造辛龙格-库塔方法。
    采用不动点迭代求解隐式方程。
    """
    def __init__(self, tol=1e-9, max_iter=100):
        self.tol = tol
        self.max_iter = max_iter

    def step(self, f, t, state, dt):
        y_guess = state + dt * f(t, state)
        for _ in range(self.max_iter):
            y_mid = (state + y_guess) / 2.0
            y_new = state + dt * f(t + dt/2.0, y_mid)
            if np.linalg.norm(y_new - y_guess) < self.tol:
                return y_new
            y_guess = y_new
        return y_guess

class GaussLegendreRK4(Integrator):
    """
    高斯-勒让德 RK法 (4阶，隐式辛积分器)。
    极高精度，常用于高精度轨道计算和控制理论。
    采用不动点迭代求解隐式方程。
    """
    def __init__(self, tol=1e-9, max_iter=100):
        self.tol = tol
        self.max_iter = max_iter
        sq3 = np.sqrt(3) / 6.0
        self.c = np.array([0.5 - sq3, 0.5 + sq3])
        self.a = np.array([[0.25, 0.25 - sq3],
                           [0.25 + sq3, 0.25]])
        self.b = np.array([0.5, 0.5])

    def step(self, f, t, state, dt):
        k1 = f(t, state)
        k2 = f(t, state)
        for _ in range(self.max_iter):
            y1 = state + dt * (self.a[0,0]*k1 + self.a[0,1]*k2)
            y2 = state + dt * (self.a[1,0]*k1 + self.a[1,1]*k2)
            
            k1_new = f(t + self.c[0]*dt, y1)
            k2_new = f(t + self.c[1]*dt, y2)
            
            if np.linalg.norm(k1_new - k1) + np.linalg.norm(k2_new - k2) < self.tol:
                k1, k2 = k1_new, k2_new
                break
            k1, k2 = k1_new, k2_new
            
        return state + dt * (self.b[0]*k1 + self.b[1]*k2)

class Yoshida4(SymplecticIntegrator):
    """
    Yoshida 分步法 (4阶，显式辛积分器)。
    通过多个 Verlet 步的组合构造出高阶方法，非常适合长期天文模拟。
    """
    def __init__(self, q_indices=[0, 2], v_indices=[1, 3]):
        super().__init__(q_indices, v_indices)
        w1 = 1.0 / (2.0 - 2.0**(1.0/3.0))
        w0 = 1.0 - 2.0 * w1
        self.w = [w1, w0, w1]
        self.verlet = VelocityVerlet(q_indices, v_indices)

    def step(self, f, t, state, dt):
        s = state
        curr_t = t
        for wi in self.w:
            s = self.verlet.step(f, curr_t, s, wi * dt)
            curr_t += wi * dt
        return s

# 导出所有可用的积分器
INTEGRATORS = {
    'RK4': RK4,
    'SymplecticEuler': SymplecticEuler,
    'VelocityVerlet': VelocityVerlet,
    'ImplicitMidpoint': ImplicitMidpoint,
    'GaussLegendreRK4': GaussLegendreRK4,
    'Yoshida4': Yoshida4
}
