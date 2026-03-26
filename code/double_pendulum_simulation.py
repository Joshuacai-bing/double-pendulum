"""
Double Pendulum Physics Simulation
Chaotic dynamics visualization and analysis
Using Runge-Kutta 4th order method for numerical integration
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation, PillowWriter
import os

# Physical parameters
g = 9.81  # gravitational acceleration (m/s^2)
l1 = 1.0  # length of first pendulum (m)
l2 = 1.0  # length of second pendulum (m)
m1 = 1.0  # mass of first pendulum (kg)
m2 = 1.0  # mass of second pendulum (kg)

# Initial conditions
theta1_0 = np.pi / 2  # initial angle of first pendulum (rad)
theta2_0 = np.pi / 2  # initial angle of second pendulum (rad)
omega1_0 = 0.0  # initial angular velocity of first pendulum (rad/s)
omega2_0 = 0.0  # initial angular velocity of second pendulum (rad/s)

# Simulation parameters
t_max = 20.0  # total simulation time (s)
dt = 0.02  # time step (s)
t = np.arange(0, t_max, dt)

def derivatives(state, t, l1, l2, m1, m2, g):
    """
    Calculate derivatives for double pendulum system
    state = [theta1, omega1, theta2, omega2]
    """
    theta1, omega1, theta2, omega2 = state
    
    delta_theta = theta2 - theta1
    
    sin_delta = np.sin(delta_theta)
    cos_delta = np.cos(delta_theta)
    
    # Common terms
    sin1 = np.sin(theta1)
    sin2 = np.sin(theta2)
    
    # Denominator for equations
    denom = 2 * m1 + m2 - m2 * np.cos(2 * delta_theta)
    
    # Angular acceleration for first pendulum
    alpha1 = (-g * (2 * m1 + m2) * sin1 
              - m2 * g * np.sin(theta1 - 2 * theta2)
              - 2 * sin_delta * m2 * (omega2**2 * l2 + omega1**2 * l1 * cos_delta)) / (l1 * denom)
    
    # Angular acceleration for second pendulum
    alpha2 = (2 * sin_delta * (omega1**2 * l1 * (m1 + m2)
              + g * (m1 + m2) * np.cos(theta1)
              + omega2**2 * l2 * m2 * cos_delta)) / (l2 * denom)
    
    return np.array([omega1, alpha1, omega2, alpha2])

def rk4_step(state, t, dt, l1, l2, m1, m2, g):
    """
    4th order Runge-Kutta method for numerical integration
    """
    k1 = derivatives(state, t, l1, l2, m1, m2, g)
    k2 = derivatives(state + 0.5 * dt * k1, t + 0.5 * dt, l1, l2, m1, m2, g)
    k3 = derivatives(state + 0.5 * dt * k2, t + 0.5 * dt, l1, l2, m1, m2, g)
    k4 = derivatives(state + dt * k3, t + dt, l1, l2, m1, m2, g)
    
    return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

def simulate_double_pendulum(l1, l2, m1, m2, g, theta1_0, theta2_0, omega1_0, omega2_0, t):
    """
    Simulate double pendulum using RK4 method
    """
    n_steps = len(t)
    state = np.zeros((n_steps, 4))
    
    state[0] = [theta1_0, omega1_0, theta2_0, omega2_0]
    
    for i in range(1, n_steps):
        state[i] = rk4_step(state[i-1], t[i-1], dt, l1, l2, m1, m2, g)
    
    return state

def compute_positions(theta1, theta2, l1, l2):
    """
    Compute x, y coordinates of both pendulum masses
    """
    x1 = l1 * np.sin(theta1)
    y1 = -l1 * np.cos(theta1)
    
    x2 = x1 + l2 * np.sin(theta2)
    y2 = y1 - l2 * np.cos(theta2)
    
    return x1, y1, x2, y2

def compute_energy(theta1, theta2, omega1, omega2, l1, l2, m1, m2, g):
    """
    Compute kinetic and potential energy of the system
    """
    y1 = -l1 * np.cos(theta1)
    y2 = -l1 * np.cos(theta1) - l2 * np.cos(theta2)
    
    # Kinetic energy
    T = (0.5 * (m1 + m2) * l1**2 * omega1**2
         + 0.5 * m2 * l2**2 * omega2**2
         + m2 * l1 * l2 * omega1 * omega2 * np.cos(theta1 - theta2))
    
    # Potential energy
    V = -(m1 + m2) * g * l1 * np.cos(theta1) - m2 * g * l2 * np.cos(theta2)
    
    return T, V, T + V

def compute_lyapunov_sensitivity(theta1_init, theta2_init, l1, l2, m1, m2, g, t, epsilon=1e-8):
    """
    Compute Lyapunov exponent to characterize chaos
    """
    state1 = simulate_double_pendulum(l1, l2, m1, m2, g, 
                                       theta1_init, theta2_init, omega1_0, omega2_0, t)
    
    state2 = simulate_double_pendulum(l1, l2, m1, m2, g, 
                                       theta1_init + epsilon, theta2_init + epsilon, omega1_0, omega2_0, t)
    
    delta_theta1 = np.abs(state1[:, 0] - state2[:, 0])
    delta_theta2 = np.abs(state1[:, 2] - state2[:, 2])
    
    return np.sqrt(delta_theta1**2 + delta_theta2**2)

print("Starting double pendulum simulation...")
print(f"Parameters: l1={l1}m, l2={l2}m, m1={m1}kg, m2={m2}kg, g={g}m/s^2")
print(f"Initial conditions: theta1={theta1_0:.3f}rad, theta2={theta2_0:.3f}rad")

# Run simulation
state = simulate_double_pendulum(l1, l2, m1, m2, g, 
                                  theta1_0, theta2_0, omega1_0, omega2_0, t)

theta1 = state[:, 0]
omega1 = state[:, 1]
theta2 = state[:, 2]
omega2 = state[:, 3]

# Compute positions
x1, y1, x2, y2 = compute_positions(theta1, theta2, l1, l2)

# Compute energy
T, V, E = compute_energy(theta1, theta2, omega1, omega2, l1, l2, m1, m2, g)

# Compute Lyapunov sensitivity
print("Computing Lyapunov sensitivity...")
delta = compute_lyapunov_sensitivity(theta1_0, theta2_0, l1, l2, m1, m2, g, t)

print(f"Simulation completed! Total frames: {len(t)}")

# Create visualization
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Double Pendulum Chaotic Dynamics Simulation', fontsize=16, fontweight='bold')

# Subplot 1: Pendulum animation (top-left)
ax1 = axes[0, 0]
ax1.set_xlim(-3, 3)
ax1.set_ylim(-3, 1)
ax1.set_aspect('equal')
ax1.grid(True, alpha=0.3)
ax1.set_title('Double Pendulum Motion')
ax1.set_xlabel('x (m)')
ax1.set_ylabel('y (m)')

# Pivot point
pivot = Circle((0, 0), 0.05, color='black', zorder=5)
ax1.add_patch(pivot)

# Pendulum lines and masses
line1, = ax1.plot([], [], 'b-', linewidth=2, zorder=2)
line2, = ax1.plot([], [], 'g-', linewidth=2, zorder=2)
mass1 = Circle((0, 0), 0.1, color='red', zorder=4)
mass2 = Circle((0, 0), 0.1, color='orange', zorder=4)
ax1.add_patch(mass1)
ax1.add_patch(mass2)

# Trajectory of mass2
trajectory, = ax1.plot([], [], 'm-', alpha=0.5, linewidth=0.8)
traj_x = []
traj_y = []

time_text = ax1.text(0.02, 0.98, '', transform=ax1.transAxes, 
                     fontsize=10, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Subplot 2: Angle vs time (top-middle)
ax2 = axes[0, 1]
ax2.set_xlim(0, t_max)
ax2.set_ylim(-4, 4)
ax2.grid(True, alpha=0.3)
ax2.set_title('Angles vs Time')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Angle (rad)')
theta1_line, = ax2.plot([], [], 'r-', linewidth=1.5, label=r'$\theta_1$')
theta2_line, = ax2.plot([], [], 'b-', linewidth=1.5, label=r'$\theta_2$')
ax2.legend(loc='upper right')

# Subplot 3: Phase space (top-right)
ax3 = axes[0, 2]
ax3.set_xlim(-4, 4)
ax3.set_ylim(-10, 10)
ax3.grid(True, alpha=0.3)
ax3.set_title('Phase Space ($\\theta_1$ vs $\\omega_1$)')
ax3.set_xlabel('Angle $\\theta_1$ (rad)')
ax3.set_ylabel('Angular Velocity $\\omega_1$ (rad/s)')
phase1_line, = ax3.plot([], [], 'purple', linewidth=0.8, alpha=0.7)

# Subplot 4: Energy analysis (bottom-left)
ax4 = axes[1, 0]
ax4.set_xlim(0, t_max)
E_min = np.min(E)
E_max = np.max(E)
E_margin = 0.1 * (E_max - E_min) if E_max != E_min else 1
ax4.set_ylim(E_min - E_margin, E_max + E_margin)
ax4.grid(True, alpha=0.3)
ax4.set_title('Energy vs Time')
ax4.set_xlabel('Time (s)')
ax4.set_ylabel('Energy (J)')
ke_line, = ax4.plot([], [], 'r-', linewidth=1.5, label='Kinetic Energy')
pe_line, = ax4.plot([], [], 'b-', linewidth=1.5, label='Potential Energy')
te_line, = ax4.plot([], [], 'g--', linewidth=2, label='Total Energy')
ax4.legend(loc='upper right')

# Energy conservation error
energy_error = np.abs(E - E[0]) / np.abs(E[0]) * 100 if E[0] != 0 else np.abs(E - E[0])
print(f"Energy conservation error: max={np.max(energy_error):.6f}%, mean={np.mean(energy_error):.6f}%")

# Subplot 5: Lyapunov sensitivity (bottom-middle)
ax5 = axes[1, 1]
ax5.set_xlim(0, t_max)
ax5.set_ylim(1e-10, 1e3)
ax5.set_yscale('log')
ax5.grid(True, alpha=0.3)
ax5.set_title('Lyapunov Sensitivity (Trajectory Divergence)')
ax5.set_xlabel('Time (s)')
ax5.set_ylabel('$\\Delta\\theta$ (rad)')
lyap_line, = ax5.plot([], [], 'm-', linewidth=1.5)

# Subplot 6: Poincaré-like plot (bottom-right)
ax6 = axes[1, 2]
ax6.set_xlim(-4, 4)
ax6.set_ylim(-4, 4)
ax6.grid(True, alpha=0.3)
ax6.set_title('Phase Space ($\\theta_1$ vs $\\theta_2$)')
ax6.set_xlabel('$\\theta_1$ (rad)')
ax6.set_ylabel('$\\theta_2$ (rad)')
poincare, = ax6.plot([], [], 'c.', markersize=0.5, alpha=0.6)

def init():
    line1.set_data([], [])
    line2.set_data([], [])
    mass1.center = (0, -l1)
    mass2.center = (0, -l1 - l2)
    trajectory.set_data([], [])
    theta1_line.set_data([], [])
    theta2_line.set_data([], [])
    phase1_line.set_data([], [])
    ke_line.set_data([], [])
    pe_line.set_data([], [])
    te_line.set_data([], [])
    lyap_line.set_data([], [])
    time_text.set_text('')
    return line1, line2, mass1, mass2, trajectory, theta1_line, theta2_line, phase1_line, ke_line, pe_line, te_line, lyap_line, time_text

def update(frame):
    # Update pendulum
    line1.set_data([0, x1[frame]], [0, y1[frame]])
    line2.set_data([x1[frame], x2[frame]], [y1[frame], y2[frame]])
    mass1.center = (x1[frame], y1[frame])
    mass2.center = (x2[frame], y2[frame])
    
    # Update trajectory
    traj_x.append(x2[frame])
    traj_y.append(y2[frame])
    if len(traj_x) > 500:
        traj_x = traj_x[-500:]
        traj_y = traj_y[-500:]
    trajectory.set_data(traj_x, traj_y)
    
    # Update angle plot
    theta1_line.set_data(t[:frame+1], theta1[:frame+1])
    theta2_line.set_data(t[:frame+1], theta2[:frame+1])
    
    # Update phase space
    phase1_line.set_data(theta1[:frame+1], omega1[:frame+1])
    
    # Update energy
    ke_line.set_data(t[:frame+1], T[:frame+1])
    pe_line.set_data(t[:frame+1], V[:frame+1])
    te_line.set_data(t[:frame+1], E[:frame+1])
    
    # Update Lyapunov sensitivity
    lyap_line.set_data(t[:frame+1], delta[:frame+1])
    
    # Update time display
    time_text.set_text(f'Time: {t[frame]:.2f} s\n'
                       f'$\\theta_1$: {theta1[frame]:.3f} rad\n'
                       f'$\\theta_2$: {theta2[frame]:.3f} rad\n'
                       f'Total E: {E[frame]:.3f} J')
    
    return line1, line2, mass1, mass2, trajectory, theta1_line, theta2_line, phase1_line, ke_line, pe_line, te_line, lyap_line, time_text

print("Creating animation...")
anim = FuncAnimation(fig, update, init_func=init, frames=len(t), 
                    interval=dt*1000, blit=True)

output_dir = '../output'
os.makedirs(output_dir, exist_ok=True)

gif_path = os.path.join(output_dir, 'double_pendulum_simulation.gif')
print(f"Saving GIF animation to: {gif_path}")

writer = PillowWriter(fps=30)
anim.save(gif_path, writer=writer)
print(f"GIF animation saved to: {gif_path}")
plt.close()

# Create static summary plots
fig2, axes2 = plt.subplots(2, 3, figsize=(18, 12))
fig2.suptitle('Double Pendulum Simulation Summary', fontsize=16, fontweight='bold')

# Trajectory
ax1 = axes2[0, 0]
ax1.set_xlim(-3, 3)
ax1.set_ylim(-3, 1)
ax1.set_aspect('equal')
ax1.grid(True, alpha=0.3)
ax1.set_title('Trajectory of Second Mass')
ax1.set_xlabel('x (m)')
ax1.set_ylabel('y (m)')
ax1.plot(x2, y2, 'm-', alpha=0.6, linewidth=0.8)
ax1.plot(0, 0, 'ko', markersize=10)

# Angles
ax2 = axes2[0, 1]
ax2.grid(True, alpha=0.3)
ax2.set_title('Angles vs Time')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Angle (rad)')
ax2.plot(t, theta1, 'r-', linewidth=1.5, label=r'$\theta_1$')
ax2.plot(t, theta2, 'b-', linewidth=1.5, label=r'$\theta_2$')
ax2.legend()

# Phase space
ax3 = axes2[0, 2]
ax3.grid(True, alpha=0.3)
ax3.set_title('Phase Space')
ax3.set_xlabel('$\\theta_1$ (rad)')
ax3.set_ylabel('$\\omega_1$ (rad/s)')
ax3.plot(theta1, omega1, 'purple', linewidth=0.8, alpha=0.7)

# Energy
ax4 = axes2[1, 0]
ax4.grid(True, alpha=0.3)
ax4.set_title('Energy vs Time')
ax4.set_xlabel('Time (s)')
ax4.set_ylabel('Energy (J)')
ax4.plot(t, T, 'r-', linewidth=1.5, label='Kinetic')
ax4.plot(t, V, 'b-', linewidth=1.5, label='Potential')
ax4.plot(t, E, 'g--', linewidth=2, label='Total')
ax4.legend()

# Lyapunov
ax5 = axes2[1, 1]
ax5.set_xlim(0, t_max)
ax5.set_yscale('log')
ax5.grid(True, alpha=0.3)
ax5.set_title('Lyapunov Sensitivity')
ax5.set_xlabel('Time (s)')
ax5.set_ylabel('$\\Delta\\theta$ (rad)')
ax5.plot(t, delta, 'm-', linewidth=1.5)

# Phase space 2D
ax6 = axes2[1, 2]
ax6.grid(True, alpha=0.3)
ax6.set_title('Phase Space ($\\theta_1$ vs $\\theta_2$)')
ax6.set_xlabel('$\\theta_1$ (rad)')
ax6.set_ylabel('$\\theta_2$ (rad)')
ax6.plot(theta1, theta2, 'c-', markersize=0.5, alpha=0.6)

plt.tight_layout()
summary_path = os.path.join(output_dir, 'double_pendulum_summary.png')
plt.savefig(summary_path, dpi=150, bbox_inches='tight')
print(f"Summary plot saved to: {summary_path}")
plt.close()

# Angular velocities plot
fig3, axes3 = plt.subplots(1, 2, figsize=(14, 5))
fig3.suptitle('Angular Velocities Analysis', fontsize=14, fontweight='bold')

axes3[0].grid(True, alpha=0.3)
axes3[0].set_title('Angular Velocities vs Time')
axes3[0].set_xlabel('Time (s)')
axes3[0].set_ylabel('Angular Velocity (rad/s)')
axes3[0].plot(t, omega1, 'r-', linewidth=1.5, label=r'$\omega_1$')
axes3[0].plot(t, omega2, 'b-', linewidth=1.5, label=r'$\omega_2$')
axes3[0].legend()

axes3[1].grid(True, alpha=0.3)
axes3[1].set_title('Phase Space ($\\theta_2$ vs $\\omega_2$)')
axes3[1].set_xlabel('$\\theta_2$ (rad)')
axes3[1].set_ylabel('$\\omega_2$ (rad/s)')
axes3[1].plot(theta2, omega2, 'orange', linewidth=0.8, alpha=0.7)

plt.tight_layout()
velocity_path = os.path.join(output_dir, 'double_pendulum_velocities.png')
plt.savefig(velocity_path, dpi=150, bbox_inches='tight')
print(f"Velocity plot saved to: {velocity_path}")
plt.close()

print("\n" + "="*60)
print("DATA VALIDATION AND VERIFICATION")
print("="*60)

# Validation 1: Small Angle Approximation - Compare with Linear Theory
print("\n[1] Small Angle Approximation Validation")
print("-" * 40)

theta1_small = np.pi / 18  # 10 degrees
theta2_small = np.pi / 18  # 10 degrees

state_small = simulate_double_pendulum(l1, l2, m1, m2, g, 
                                        theta1_small, theta2_small, omega1_0, omega2_0, t)
theta1_sim = state_small[:, 0]
theta2_sim = state_small[:, 2]

# Linearized equations analytical solution (for comparison with simulation)
omega_linear_minus = np.sqrt(g / l1)

theta1_theory = theta1_small * np.cos(omega_linear_minus * t)
theta2_theory = theta2_small * np.cos(omega_linear_minus * t)

# Calculate RMS error
rms_error_theta1 = np.sqrt(np.mean((theta1_sim - theta1_theory)**2))
rms_error_theta2 = np.sqrt(np.mean((theta2_sim - theta2_theory)**2))
print(f"RMS Error (theta1): {rms_error_theta1:.6f} rad")
print(f"RMS Error (theta2): {rms_error_theta2:.6f} rad")
print(f"Maximum deviation (theta1): {np.max(np.abs(theta1_sim - theta1_theory)):.6f} rad")
print(f"Maximum deviation (theta2): {np.max(np.abs(theta2_sim - theta2_theory)):.6f} rad")

# Validation 2: Energy Conservation
print("\n[2] Energy Conservation Verification")
print("-" * 40)

T_full, V_full, E_full = compute_energy(theta1, theta2, omega1, omega2, l1, l2, m1, m2, g)

energy_drift = np.abs(E_full - E_full[0])
max_energy_drift = np.max(energy_drift)
relative_energy_error = max_energy_drift / np.abs(E_full[0]) * 100

print(f"Initial total energy: {E_full[0]:.6f} J")
print(f"Final total energy: {E_full[-1]:.6f} J")
print(f"Maximum absolute energy drift: {max_energy_drift:.8f} J")
print(f"Relative energy error: {relative_energy_error:.6f}%")
print(f"Energy conservation: {'PASSED' if relative_energy_error < 0.1 else 'FAILED'} (threshold: 0.1%)")

# Validation 3: RK4 vs Euler Method Comparison
print("\n[3] Numerical Method Comparison (RK4 vs Euler)")
print("-" * 40)

def euler_step(state, t, dt, l1, l2, m1, m2, g):
    """Euler method for comparison"""
    derivs = derivatives(state, t, l1, l2, m1, m2, g)
    return state + dt * derivs

def simulate_euler(l1, l2, m1, m2, g, theta1_0, theta2_0, omega1_0, omega2_0, t):
    """Simulate using Euler method"""
    n_steps = len(t)
    state = np.zeros((n_steps, 4))
    state[0] = [theta1_0, omega1_0, theta2_0, omega2_0]
    for i in range(1, n_steps):
        state[i] = euler_step(state[i-1], t[i-1], dt, l1, l2, m1, m2, g)
    return state

state_euler = simulate_euler(l1, l2, m1, m2, g, theta1_0, theta2_0, omega1_0, omega2_0, t)
theta1_euler = state_euler[:, 0]
theta2_euler = state_euler[:, 2]
omega1_euler = state_euler[:, 1]
omega2_euler = state_euler[:, 3]

T_euler, V_euler, E_euler = compute_energy(theta1_euler, theta2_euler, omega1_euler, omega2_euler, l1, l2, m1, m2, g)

# Calculate energy drift for Euler
E_drift_euler = np.abs(E_euler - E_euler[0])
E_drift_rk4 = np.abs(E_full - E_full[0])

print(f"Euler method - Energy drift: {np.max(E_drift_euler):.6f} J")
print(f"RK4 method - Energy drift: {np.max(E_drift_rk4):.6f} J")
print(f"Energy drift improvement: {np.max(E_drift_euler)/np.max(E_drift_rk4):.1f}x better with RK4")

# Angle comparison
angle_diff_euler = np.sqrt(np.mean((theta1_euler - theta1)**2 + (theta2_euler - theta2)**2))
print(f"Euler vs RK4 angle RMS difference: {angle_diff_euler:.6f} rad")

print("\n" + "="*60)
print("Creating validation plots...")
print("="*60)

# Create validation plots
fig_val, axes_val = plt.subplots(2, 3, figsize=(18, 12))
fig_val.suptitle('Double Pendulum Simulation - Data Validation', fontsize=16, fontweight='bold')

# Plot 1: Small angle - Simulation vs Theory
ax1 = axes_val[0, 0]
ax1.set_xlim(0, 5)
ax1.set_ylim(-0.6, 0.6)
ax1.grid(True, alpha=0.3)
ax1.set_title('Small Angle Validation: Simulation vs Theory')
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Angle (rad)')
ax1.plot(t, theta1_sim, 'r-', linewidth=1.5, label='Sim $\\theta_1$', alpha=0.8)
ax1.plot(t, theta1_theory, 'r--', linewidth=2, label='Theory $\\theta_1$', alpha=0.6)
ax1.plot(t, theta2_sim, 'b-', linewidth=1.5, label='Sim $\\theta_2$', alpha=0.8)
ax1.plot(t, theta2_theory, 'b--', linewidth=2, label='Theory $\\theta_2$', alpha=0.6)
ax1.legend(loc='upper right', fontsize=9)

# Plot 2: Energy Conservation
ax2 = axes_val[0, 1]
ax2.set_xlim(0, t_max)
E_range = np.max(np.abs(E_full)) * 1.1
ax2.set_ylim(-E_range, E_range)
ax2.grid(True, alpha=0.3)
ax2.set_title('Energy Conservation (RK4)')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Energy (J)')
ax2.plot(t, T_full, 'r-', linewidth=1.5, label='Kinetic Energy', alpha=0.8)
ax2.plot(t, V_full, 'b-', linewidth=1.5, label='Potential Energy', alpha=0.8)
ax2.plot(t, E_full, 'g-', linewidth=2, label='Total Energy', alpha=0.9)
ax2.axhline(y=E_full[0], color='k', linestyle='--', linewidth=1, alpha=0.5, label=f'Initial E={E_full[0]:.3f}J')
ax2.legend(loc='upper right', fontsize=9)

# Plot 3: Energy Drift Comparison
ax3 = axes_val[0, 2]
ax3.set_xlim(0, t_max)
ax3.set_ylim(1e-8, 1)
ax3.set_yscale('log')
ax3.grid(True, alpha=0.3)
ax3.set_title('Energy Drift Comparison')
ax3.set_xlabel('Time (s)')
ax3.set_ylabel('Energy Drift |E - E₀| (J)')
ax3.plot(t, E_drift_rk4, 'g-', linewidth=2, label='RK4', alpha=0.9)
ax3.plot(t, E_drift_euler, 'r-', linewidth=2, label='Euler', alpha=0.7)
ax3.legend(loc='upper left', fontsize=9)

# Plot 4: RK4 vs Euler Angle Comparison
ax4 = axes_val[1, 0]
ax4.set_xlim(0, t_max)
ax4.set_ylim(-7, 7)
ax4.grid(True, alpha=0.3)
ax4.set_title('RK4 vs Euler Method Comparison')
ax4.set_xlabel('Time (s)')
ax4.set_ylabel('Angle (rad)')
ax4.plot(t, theta1, 'b-', linewidth=1.5, label='RK4 $\\theta_1$', alpha=0.8)
ax4.plot(t, theta1_euler, 'b--', linewidth=1.5, label='Euler $\\theta_1$', alpha=0.6)
ax4.plot(t, theta2, 'r-', linewidth=1.5, label='RK4 $\\theta_2$', alpha=0.8)
ax4.plot(t, theta2_euler, 'r--', linewidth=1.5, label='Euler $\\theta_2$', alpha=0.6)
ax4.legend(loc='upper right', fontsize=9)

# Plot 5: Phase Space Comparison
ax5 = axes_val[1, 1]
ax5.set_xlim(-4, 4)
ax5.set_ylim(-10, 10)
ax5.grid(True, alpha=0.3)
ax5.set_title('Phase Space: RK4 vs Euler')
ax5.set_xlabel('$\\theta_1$ (rad)')
ax5.set_ylabel('$\\omega_1$ (rad/s)')
ax5.plot(theta1, omega1, 'b-', linewidth=0.8, label='RK4', alpha=0.7)
ax5.plot(theta1_euler, omega1_euler, 'r-', linewidth=0.8, label='Euler', alpha=0.5)
ax5.legend(loc='upper right', fontsize=9)

# Plot 6: Error Analysis Summary
ax6 = axes_val[1, 2]
ax6.axis('off')
error_text = f"""
╔════════════════════════════════════════════╗
║         VALIDATION SUMMARY                 ║
╠════════════════════════════════════════════╣
║                                            ║
║  [1] Small Angle Approximation             ║
║      RMS Error (θ₁): {rms_error_theta1:.2e} rad        ║
║      RMS Error (θ₂): {rms_error_theta2:.2e} rad        ║
║      Max Deviation:  {np.max(np.abs(theta1_sim - theta1_theory)):.2e} rad        ║
║                                            ║
║  [2] Energy Conservation (RK4)              ║
║      Initial Energy:  {E_full[0]:.4f} J           ║
║      Final Energy:    {E_full[-1]:.4f} J           ║
║      Relative Error:  {relative_energy_error:.4f}%            ║
║      Status: {'✓ PASSED' if relative_energy_error < 0.1 else '✗ FAILED'}                      ║
║                                            ║
║  [3] Numerical Method Comparison            ║
║      Euler Drift:     {np.max(E_drift_euler):.4f} J           ║
║      RK4 Drift:       {np.max(E_drift_rk4):.6f} J          ║
║      Improvement:     {np.max(E_drift_euler)/np.max(E_drift_rk4):.1f}x                 ║
║                                            ║
╚════════════════════════════════════════════╝
"""
ax6.text(0.1, 0.5, error_text, transform=ax6.transAxes, fontsize=11,
         verticalalignment='center', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

plt.tight_layout()
validation_path = os.path.join(output_dir, 'double_pendulum_validation.png')
plt.savefig(validation_path, dpi=150, bbox_inches='tight')
print(f"Validation plot saved to: {validation_path}")
plt.close()

# Additional Plot: Lyapunov Exponent Estimation
print("\n[4] Lyapunov Exponent Estimation")
print("-" * 40)

# Estimate Lyapunov exponent from divergence rate
log_delta = np.log(delta[100:])
time_for_fit = t[100:] - t[100]
coeffs = np.polyfit(time_for_fit, log_delta, 1)
lambda_estimated = coeffs[0]

print(f"Estimated Largest Lyapunov Exponent: {lambda_estimated:.4f} 1/s")
print(f"Positive λ indicates chaotic dynamics: {'YES' if lambda_estimated > 0 else 'NO'}")

# Create Lyapunov plot
fig_lyap, ax_lyap = plt.subplots(1, 2, figsize=(14, 5))
fig_lyap.suptitle('Lyapunov Exponent Analysis', fontsize=14, fontweight='bold')

ax_lyap[0].set_xlim(0, t_max)
ax_lyap[0].set_ylim(1e-10, 1e3)
ax_lyap[0].set_yscale('log')
ax_lyap[0].grid(True, alpha=0.3)
ax_lyap[0].set_title('Trajectory Divergence (Initial Condition Sensitivity)')
ax_lyap[0].set_xlabel('Time (s)')
ax_lyap[0].set_ylabel('$\\Delta\\theta$ (rad)')
ax_lyap[0].plot(t, delta, 'm-', linewidth=1.5, label='Measured divergence')
ax_lyap[0].plot(t[100:], np.exp(coeffs[0] * t[100:] + coeffs[1]), 'c--', linewidth=2, 
                label=f'Fit: λ={lambda_estimated:.3f}')
ax_lyap[0].legend(loc='upper left', fontsize=10)

ax_lyap[1].set_xlim(0, t_max)
ax_lyap[1].set_ylim(-2, 2)
ax_lyap[1].grid(True, alpha=0.3)
ax_lyap[1].set_title('Log Divergence for λ Estimation')
ax_lyap[1].set_xlabel('Time (s)')
ax_lyap[1].set_ylabel('ln($\\Delta\\theta$)')
mask = delta > 1e-10
ax_lyap[1].plot(t[mask], np.log(delta[mask]), 'm-', linewidth=1.5, alpha=0.7)
ax_lyap[1].plot(t[100:], coeffs[0] * t[100:] + coeffs[1], 'c--', linewidth=2,
                label=f'Linear fit: λ={lambda_estimated:.3f}')
ax_lyap[1].legend(loc='upper right', fontsize=10)

plt.tight_layout()
lyapunov_path = os.path.join(output_dir, 'double_pendulum_lyapunov.png')
plt.savefig(lyapunov_path, dpi=150, bbox_inches='tight')
print(f"Lyapunov plot saved to: {lyapunov_path}")
plt.close()

# Create comprehensive comparison plot
fig_comp, axes_comp = plt.subplots(2, 2, figsize=(14, 12))
fig_comp.suptitle('Double Pendulum - Comprehensive Analysis', fontsize=16, fontweight='bold')

# Full trajectory
ax1 = axes_comp[0, 0]
ax1.set_xlim(-3, 3)
ax1.set_ylim(-3, 1)
ax1.set_aspect('equal')
ax1.grid(True, alpha=0.3)
ax1.set_title('Phase Space Trajectory (Second Mass)')
ax1.set_xlabel('x (m)')
ax1.set_ylabel('y (m)')
scatter = ax1.scatter(x2[::10], y2[::10], c=t[::10], cmap='viridis', s=1, alpha=0.6)
plt.colorbar(scatter, ax=ax1, label='Time (s)')
ax1.plot(0, 0, 'ko', markersize=10)

# Angular velocities
ax2 = axes_comp[0, 1]
ax2.grid(True, alpha=0.3)
ax2.set_title('Angular Velocities vs Time')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Angular Velocity (rad/s)')
ax2.plot(t, omega1, 'r-', linewidth=1.2, label=r'$\omega_1$', alpha=0.8)
ax2.plot(t, omega2, 'b-', linewidth=1.2, label=r'$\omega_2$', alpha=0.8)
ax2.legend(loc='upper right', fontsize=10)

# Energy distribution
ax3 = axes_comp[1, 0]
ax3.grid(True, alpha=0.3)
ax3.set_title('Energy Components vs Time')
ax3.set_xlabel('Time (s)')
ax3.set_ylabel('Energy (J)')
ax3.fill_between(t, 0, T_full, alpha=0.3, color='red', label='Kinetic')
ax3.fill_between(t, np.minimum(V_full, 0), np.maximum(V_full, 0), alpha=0.3, color='blue', label='Potential')
ax3.plot(t, T_full, 'r-', linewidth=1.5)
ax3.plot(t, V_full, 'b-', linewidth=1.5)
ax3.plot(t, E_full, 'g-', linewidth=2, label='Total')
ax3.legend(loc='upper right', fontsize=10)

# Poincaré section (when theta1 crosses zero with positive velocity)
ax4 = axes_comp[1, 1]
zero_crossings = np.where(np.diff(np.sign(theta1)) > 0)[0]
poincare_theta1 = theta1[zero_crossings]
poincare_omega1 = omega1[zero_crossings]
ax4.set_xlim(-4, 4)
ax4.set_ylim(-10, 10)
ax4.grid(True, alpha=0.3)
ax4.set_title('Poincaré Section ($\\dot{\\theta}_1 > 0$)')
ax4.set_xlabel('$\\theta_1$ (rad)')
ax4.set_ylabel('$\\omega_1$ (rad/s)')
ax4.plot(poincare_theta1, poincare_omega1, 'c.', markersize=4, alpha=0.6)

plt.tight_layout()
comprehensive_path = os.path.join(output_dir, 'double_pendulum_comprehensive.png')
plt.savefig(comprehensive_path, dpi=150, bbox_inches='tight')
print(f"Comprehensive plot saved to: {comprehensive_path}")
plt.close()

print("\n" + "="*60)
print("ALL OUTPUT FILES:")
print("="*60)
print(f"1. Animation:     {gif_path}")
print(f"2. Summary:       {summary_path}")
print(f"3. Velocities:    {velocity_path}")
print(f"4. Validation:    {validation_path}")
print(f"5. Lyapunov:      {lyapunov_path}")
print(f"6. Comprehensive: {comprehensive_path}")
print("="*60)
print(f"\nTotal energy at t=0: {E[0]:.4f} J")
print(f"Total energy at t={t_max}: {E[-1]:.4f} J")
print(f"Energy drift: {((E[-1]-E[0])/E[0]*100):.6f}%")