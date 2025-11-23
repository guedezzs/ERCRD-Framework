# ERCRD Framework: Mathematical Foundation for Dynamic Resource Optimization

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Research Software - Implementing the ERCRD Mathematical Framework**

This repository contains the reference implementation of the ERCRD (Resource Optimization Equation with Dynamic Constraints) framework, a novel mathematical approach to dynamic optimization across multiple domains.

## 🎯 Mathematical Foundation

The core ERCRD equation:

```math
min┬x(t)∫₀ᵀ[α·‖∇F(x(t))‖² + β·Φ(x(t),t) + γ·Ψ(x(t))]dt
dx(t)/dt = G(x(t), u(t), ξ(t))
H(x(t)) ≤ R(t)
x(0) = x₀
# Clone the repository
git clone https://github.com/fabianomelloguedes/ERCRD-Framework.git
cd ERCRD-Framework

# Install the framework
pip install -e .

# Run the example
python examples/energy_demo.py
from ercrd import EnergyERCRD
import numpy as np

# Configure a 3-node power system
optimizer = EnergyERCRD(
    num_nodes=3,
    line_parameters=np.array([[0, 0.2, 0.1], [0.2, 0, 0.15], [0.1, 0.15, 0]]),
    generator_costs=[[0.01, 10, 100], [0.02, 15, 120], [0.03, 20, 90]]
)

# Solve optimal dispatch
result = optimizer.optimize(np.array([50, 60, 40, 0, 0, 0]))
print(f"Optimal cost: ${result['final_cost']:.2f}")
