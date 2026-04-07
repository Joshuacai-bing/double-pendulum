import numpy as np
import matplotlib.pyplot as plt
from double_pendulum_simulation import DoublePendulum
from integrators import INTEGRATORS

def run_comparison():
    L1, L2, M1, M2, g = 1.0, 1.0, 1.0, 1.0, 9.8
    theta0 = 0.5
    t_max = 10.0
    dt = 0.01
    use_nonlinear = True
    
    pendulum = DoublePendulum(L1, L2, M1, M2, g)
    initial_state = np.array([theta0, 0.0, 0.0, 0.0]) # mixed phase or similar
    t_vals = np.arange(0, t_max, dt)
    n_steps = len(t_vals)
    
    results = {}
    
    for name, integrator_class in INTEGRATORS.items():
        print(f"Running simulation with {name}...")
        integrator = integrator_class()
        
        state = initial_state.copy()
        energies = np.zeros(n_steps)
        energies[0] = pendulum.calculate_energy(state, use_nonlinear)
        
        derivs = pendulum.derivatives_nonlinear if use_nonlinear else pendulum.derivatives_linear
        
        for i in range(1, n_steps):
            state = integrator.step(derivs, t_vals[i-1], state, dt)
            energies[i] = pendulum.calculate_energy(state, use_nonlinear)
            
        # Calculate energy error relative to initial energy
        energy_drift = np.abs(energies - energies[0])
        results[name] = energy_drift
        print(f"{name} max energy drift: {np.max(energy_drift):.2e} J")
        
    # Plot results
    plt.figure(figsize=(10, 6))
    for name, drift in results.items():
        # Plot in log scale for better visualization of differences
        plt.plot(t_vals, drift + 1e-16, label=name)
        
    plt.yscale('log')
    plt.xlabel('Time (s)')
    plt.ylabel('Energy Error |E(t) - E(0)| (J)')
    plt.title(f'Energy Conservation Comparison (dt={dt}s, Nonlinear Model)')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig('energy_comparison.png')
    print("Comparison plot saved as 'energy_comparison.png'")

if __name__ == "__main__":
    run_comparison()