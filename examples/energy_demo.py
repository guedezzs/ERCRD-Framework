"""
Energy System Demo - ERCRD Framework
Reproducible research example

Author: Fabiano Mello Guedes
Contact: professor@fabianoguedes.com.br
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add the src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from ercrd.core import EnergyERCRD

def run_demo():
    print("ERCRD Framework - Energy System Demo")
    print("Author: Fabiano Mello Guedes")
    print("=" * 50)
    
    # System configuration
    num_nodes = 3
    B = np.array([
        [0, 0.2, 0.1],
        [0.2, 0, 0.15], 
        [0.1, 0.15, 0]
    ])
    
    generator_costs = [
        [0.01, 10, 100],  # [quadratic, linear, max_capacity]
        [0.02, 15, 120],
        [0.03, 20, 90]
    ]
    
    # Create optimizer
    optimizer = EnergyERCRD(
        num_nodes=num_nodes,
        line_parameters=B,
        generator_costs=generator_costs,
        time_horizon=24,
        discretization_steps=24
    )
    
    # Initial state: [P1, P2, P3, theta1, theta2, theta3]
    x0 = np.array([50, 60, 40, 0, 0, 0])
    
    print("Running optimization...")
    result = optimizer.optimize(x0)
    
    if result['success']:
        print(f"✅ Optimization successful!")
        print(f"📊 Final cost: ${result['final_cost']:.2f}")
        print(f"🔄 Iterations: {result['nit']}")
        
        # Plot results
        trajectory = result['optimal_trajectory']
        time = np.arange(24)
        
        plt.figure(figsize=(12, 8))
        
        # Plot generation
        plt.subplot(2, 2, 1)
        for i in range(num_nodes):
            plt.plot(time, trajectory[:, i], label=f'Generator {i+1}', linewidth=2)
        plt.xlabel('Time (hours)')
        plt.ylabel('Power (MW)')
        plt.title('Optimal Generation Dispatch')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot voltage angles
        plt.subplot(2, 2, 2)
        for i in range(num_nodes):
            plt.plot(time, trajectory[:, num_nodes + i], label=f'Node {i+1}')
        plt.xlabel('Time (hours)')
        plt.ylabel('Voltage Angle (rad)')
        plt.title('Voltage Angles')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot costs over time
        plt.subplot(2, 2, 3)
        costs = [optimizer.efficiency_term(trajectory[i], i) for i in range(24)]
        plt.plot(time, costs, 'r-', linewidth=2)
        plt.xlabel('Time (hours)')
        plt.ylabel('Cost ($)')
        plt.title('Generation Costs Over Time')
        plt.grid(True, alpha=0.3)
        
        plt.suptitle('ERCRD Framework - Energy System Optimization\nFabiano Mello Guedes', 
                    fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        print("\n🎉 Demo completed successfully!")
        print("📈 Check the generated plots for optimization results.")
        
    else:
        print(f"❌ Optimization failed: {result.get('message', 'Unknown error')}")

if __name__ == "__main__":
    run_demo()
