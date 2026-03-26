import csv
import os

import matplotlib.pyplot as plt
import numpy as np


def derivatives(state, l1, l2, m1, m2, g):
    theta1, omega1, theta2, omega2 = state
    delta = theta2 - theta1
    sin_delta = np.sin(delta)
    cos_delta = np.cos(delta)
    denom = 2 * m1 + m2 - m2 * np.cos(2 * delta)
    alpha1 = (
        -g * (2 * m1 + m2) * np.sin(theta1)
        - m2 * g * np.sin(theta1 - 2 * theta2)
        - 2 * sin_delta * m2 * (omega2**2 * l2 + omega1**2 * l1 * cos_delta)
    ) / (l1 * denom)
    alpha2 = (
        2
        * sin_delta
        * (
            omega1**2 * l1 * (m1 + m2)
            + g * (m1 + m2) * np.cos(theta1)
            + omega2**2 * l2 * m2 * cos_delta
        )
    ) / (l2 * denom)
    return np.array([omega1, alpha1, omega2, alpha2], dtype=float)


def rk4_step(state, dt, l1, l2, m1, m2, g):
    k1 = derivatives(state, l1, l2, m1, m2, g)
    k2 = derivatives(state + 0.5 * dt * k1, l1, l2, m1, m2, g)
    k3 = derivatives(state + 0.5 * dt * k2, l1, l2, m1, m2, g)
    k4 = derivatives(state + dt * k3, l1, l2, m1, m2, g)
    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def simulate(initial_state, t, l1, l2, m1, m2, g):
    state = np.zeros((len(t), 4), dtype=float)
    state[0] = initial_state
    dt = t[1] - t[0]
    for i in range(1, len(t)):
        state[i] = rk4_step(state[i - 1], dt, l1, l2, m1, m2, g)
    return state


def estimate_omega_zero_crossing(signal, t):
    idx = np.where((signal[:-1] <= 0) & (signal[1:] > 0))[0]
    if idx.size < 3:
        return np.nan
    t_cross = []
    for i in idx:
        y0 = signal[i]
        y1 = signal[i + 1]
        x0 = t[i]
        x1 = t[i + 1]
        if y1 == y0:
            continue
        x_cross = x0 - y0 * (x1 - x0) / (y1 - y0)
        t_cross.append(x_cross)
    if len(t_cross) < 3:
        return np.nan
    periods = np.diff(np.array(t_cross))
    period = np.mean(periods[1:]) if periods.size > 1 else np.mean(periods)
    return 2 * np.pi / period


def estimate_omega_fft(signal, t):
    dt = t[1] - t[0]
    centered = signal - np.mean(signal)
    spectrum = np.abs(np.fft.rfft(centered))
    freq = np.fft.rfftfreq(centered.size, dt)
    mask = freq > 0.05
    if not np.any(mask):
        return np.nan
    peak = np.argmax(spectrum[mask])
    f = freq[mask][peak]
    return 2 * np.pi * f


def top_two_omega_fft(signal, t):
    dt = t[1] - t[0]
    centered = signal - np.mean(signal)
    spectrum = np.abs(np.fft.rfft(centered))
    freq = np.fft.rfftfreq(centered.size, dt)
    mask = freq > 0.05
    sf = spectrum[mask]
    ff = freq[mask]
    if sf.size < 2:
        return np.array([np.nan, np.nan]), ff, sf
    top_idx = np.argpartition(sf, -2)[-2:]
    top_freq = np.sort(ff[top_idx])
    return 2 * np.pi * top_freq, ff, sf


def to_cartesian(theta1, theta2, l1, l2):
    x1 = l1 * np.sin(theta1)
    y1 = -l1 * np.cos(theta1)
    x2 = x1 + l2 * np.sin(theta2)
    y2 = y1 - l2 * np.cos(theta2)
    return x1, y1, x2, y2


def main():
    g = 9.81
    l1 = 1.0
    l2 = 1.0
    m1 = 1.0
    m2 = 1.0
    theta0 = 0.03
    t = np.arange(0.0, 40.0, 0.002)

    omega_minus_theory = np.sqrt(g / l1 * (2 - np.sqrt(2)))
    omega_plus_theory = np.sqrt(g / l1 * (2 + np.sqrt(2)))

    init_mode_minus = np.array([theta0, 0.0, np.sqrt(2) * theta0, 0.0], dtype=float)
    init_mode_plus = np.array([theta0, 0.0, -np.sqrt(2) * theta0, 0.0], dtype=float)
    init_super = np.array([2 * theta0, 0.0, 0.0, 0.0], dtype=float)

    state_minus = simulate(init_mode_minus, t, l1, l2, m1, m2, g)
    state_plus = simulate(init_mode_plus, t, l1, l2, m1, m2, g)
    state_super = simulate(init_super, t, l1, l2, m1, m2, g)

    theta1_minus = state_minus[:, 0]
    theta1_plus = state_plus[:, 0]
    theta2_super = state_super[:, 2]
    theta1_super = state_super[:, 0]
    _, _, x2_super, y2_super = to_cartesian(theta1_super, theta2_super, l1, l2)

    omega_minus_sim_zc = estimate_omega_zero_crossing(theta1_minus, t)
    omega_plus_sim_zc = estimate_omega_zero_crossing(theta1_plus, t)
    omega_minus_sim_fft = estimate_omega_fft(theta1_minus, t)
    omega_plus_sim_fft = estimate_omega_fft(theta1_plus, t)
    omega_super_peaks, super_freq, super_amp = top_two_omega_fft(theta1_super, t)

    err_minus = abs(omega_minus_sim_zc - omega_minus_theory) / omega_minus_theory * 100
    err_plus = abs(omega_plus_sim_zc - omega_plus_theory) / omega_plus_theory * 100

    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "output"))
    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, "double_pendulum_frequency_validation.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["case", "omega_theory_rad_s", "omega_sim_zero_crossing_rad_s", "omega_sim_fft_rad_s", "relative_error_percent"])
        writer.writerow(["in_phase_mode_minus", omega_minus_theory, omega_minus_sim_zc, omega_minus_sim_fft, err_minus])
        writer.writerow(["out_of_phase_mode_plus", omega_plus_theory, omega_plus_sim_zc, omega_plus_sim_fft, err_plus])
        writer.writerow(["superposition_peak_1", omega_minus_theory, omega_super_peaks[0], omega_super_peaks[0], abs(omega_super_peaks[0] - omega_minus_theory) / omega_minus_theory * 100])
        writer.writerow(["superposition_peak_2", omega_plus_theory, omega_super_peaks[1], omega_super_peaks[1], abs(omega_super_peaks[1] - omega_plus_theory) / omega_plus_theory * 100])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Double Pendulum Frequency Validation (Small-Angle Regime)", fontsize=14)

    axes[0, 0].plot(t, theta1_minus, label=r"Simulation $\theta_1$")
    axes[0, 0].plot(t, theta0 * np.cos(omega_minus_theory * t), "--", label=r"Theory $\omega_-$")
    axes[0, 0].set_title(r"In-Phase Mode ($\omega_-$)")
    axes[0, 0].set_xlabel("Time (s)")
    axes[0, 0].set_ylabel(r"$\theta_1$ (rad)")
    axes[0, 0].grid(alpha=0.3)
    axes[0, 0].legend()

    axes[0, 1].plot(t, theta1_plus, label=r"Simulation $\theta_1$")
    axes[0, 1].plot(t, theta0 * np.cos(omega_plus_theory * t), "--", label=r"Theory $\omega_+$")
    axes[0, 1].set_title(r"Out-of-Phase Mode ($\omega_+$)")
    axes[0, 1].set_xlabel("Time (s)")
    axes[0, 1].set_ylabel(r"$\theta_1$ (rad)")
    axes[0, 1].grid(alpha=0.3)
    axes[0, 1].legend()

    axes[1, 0].plot(t, theta1_super, color="purple")
    axes[1, 0].set_title("Superposition Initial Condition")
    axes[1, 0].set_xlabel("Time (s)")
    axes[1, 0].set_ylabel(r"$\theta_1$ (rad)")
    axes[1, 0].grid(alpha=0.3)

    axes[1, 1].plot(2 * np.pi * super_freq, super_amp, color="teal")
    axes[1, 1].axvline(omega_minus_theory, color="r", linestyle="--", label=r"Theory $\omega_-$")
    axes[1, 1].axvline(omega_plus_theory, color="b", linestyle="--", label=r"Theory $\omega_+$")
    axes[1, 1].set_title("Superposition Spectrum")
    axes[1, 1].set_xlabel(r"Angular Frequency $\omega$ (rad/s)")
    axes[1, 1].set_ylabel("Amplitude")
    axes[1, 1].grid(alpha=0.3)
    axes[1, 1].legend()

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "double_pendulum_frequency_validation.png")
    plt.savefig(plot_path, dpi=180, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(7, 7))
    plt.plot(x2_super, y2_super, color="navy", linewidth=1.2)
    plt.scatter([x2_super[0]], [y2_super[0]], color="green", s=40, label="Start")
    plt.scatter([x2_super[-1]], [y2_super[-1]], color="red", s=40, label="End")
    plt.title("Second Mass Trajectory (Small-Angle Superposition)")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.axis("equal")
    plt.grid(alpha=0.3)
    plt.legend()
    trajectory_path = os.path.join(output_dir, "double_pendulum_trajectory_small_angle.png")
    plt.savefig(trajectory_path, dpi=180, bbox_inches="tight")
    plt.close()

    print("双摆频率验证完成")
    print(f"理论角频率 omega_- = {omega_minus_theory:.6f} rad/s, omega_+ = {omega_plus_theory:.6f} rad/s")
    print(f"模拟角频率(零交叉) omega_- = {omega_minus_sim_zc:.6f} rad/s, omega_+ = {omega_plus_sim_zc:.6f} rad/s")
    print(f"相对误差 omega_- = {err_minus:.4f}%, omega_+ = {err_plus:.4f}%")
    print(f"叠加态主峰角频率 = {omega_super_peaks[0]:.6f}, {omega_super_peaks[1]:.6f} rad/s")
    print(f"结果数据: {csv_path}")
    print(f"验证图像: {plot_path}")
    print(f"轨迹图像: {trajectory_path}")


if __name__ == "__main__":
    main()
