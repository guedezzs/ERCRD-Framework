"""
ERCRD Framework Core Implementation
Mathematical Foundation for Dynamic Resource Optimization

Author: Fabiano Mello Guedes
Contact: professor@fabianoguedes.com.br
"""

import numpy as np
from scipy.optimize import minimize

class ERCRDOptimizer:
    """
    Implements the ERCRD mathematical framework:
    
    min┬x(t)∫₀ᵀ[α·‖∇F(x(t))‖² + β·Φ(x(t),t) + γ·Ψ(x(t))]dt
    
    Subject to:
    dx(t)/dt = G(x(t), u(t), ξ(t))
    H(x(t)) ≤ R(t)
    x(0) = x₀
    """
    
    def __init__(self, alpha=0.4, beta=0.3, gamma=0.3, 
                 time_horizon=1.0, discretization_steps=100):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.T = time_horizon
        self.n_steps = discretization_steps
        self.dt = time_horizon / discretization_steps
        
    def efficiency_term(self, x, t):
        """Term 1: Operational efficiency ∇F(x(t))"""
        raise NotImplementedError("Subclasses must implement efficiency_term")
    
    def adaptability_term(self, x, t):
        """Term 2: Adaptive response Φ(x(t), t)"""
        raise NotImplementedError("Subclasses must implement adaptability_term")
    
    def collective_efficiency_term(self, x, t):
        """Term 3: Collective performance Ψ(x(t))"""
        raise NotImplementedError("Subclasses must implement collective_efficiency_term")
    
    def system_dynamics(self, x, u, uncertainty, t):
        """System dynamics G(x(t), u(t), ξ(t))"""
        raise NotImplementedError("Subclasses must implement system_dynamics")
    
    def constraints(self, x, t):
        """Constraints H(x(t)) ≤ R(t)"""
        raise NotImplementedError("Subclasses must implement constraints")
    
    def objective_function(self, x_flat):
        """Compute the complete ERCRD objective"""
        n_states = x_flat.size // self.n_steps
        x_trajectory = x_flat.reshape(self.n_steps, n_states)
        
        total_cost = 0.0
        for k in range(self.n_steps):
            t = k * self.dt
            x_k = x_trajectory[k]
            
            efficiency = self.alpha * self.efficiency_term(x_k, t)
            adaptability = self.beta * self.adaptability_term(x_k, t)
            collective = self.gamma * self.collective_efficiency_term(x_k, t)
            
            if k == 0 or k == self.n_steps - 1:
                weight = 0.5
            else:
                weight = 1.0
                
            total_cost += weight * (efficiency + adaptability + collective) * self.dt
        
        return total_cost
    
    def optimize(self, x0, method='SLSQP', max_iter=1000):
        """Solve the ERCRD optimization problem"""
        n_states = x0.size
        x0_trajectory = np.tile(x0, self.n_steps)
        
        def path_constraints(x_flat):
            x_trajectory = x_flat.reshape(self.n_steps, n_states)
            constraint_values = []
            for k in range(self.n_steps):
                t = k * self.dt
                constraint_vals = self.constraints(x_trajectory[k], t)
                constraint_values.extend(constraint_vals)
            return np.array(constraint_values)
        
        constraints = [{'type': 'ineq', 'fun': lambda x: -path_constraints(x)}]
        bounds = [(None, None)] * (n_states * self.n_steps)
        
        result = minimize(
            self.objective_function,
            x0_trajectory,
            method=method,
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': max_iter, 'ftol': 1e-6}
        )
        
        optimal_trajectory = result.x.reshape(self.n_steps, n_states)
        return {
            'success': result.success,
            'optimal_trajectory': optimal_trajectory,
            'final_cost': result.fun,
            'nit': result.nit
        }

class EnergyERCRD(ERCRDOptimizer):
    """ERCRD implementation for energy systems"""
    
    def __init__(self, num_nodes, line_parameters, generator_costs, **kwargs):
        super().__init__(**kwargs)
        self.num_nodes = num_nodes
        self.line_params = line_parameters
        self.generator_costs = generator_costs
        
    def efficiency_term(self, x, t):
        P = x[:self.num_nodes]
        cost = 0
        for i in range(self.num_nodes):
            a, b, c = self.generator_costs[i]
            cost += a * P[i]**2 + b * P[i] + c
        return cost
    
    def adaptability_term(self, x, t):
        P = x[:self.num_nodes]
        if hasattr(self, 'prev_P'):
            change = np.sum((P - self.prev_P)**2)
        else:
            change = 0
        self.prev_P = P.copy()
        return change
    
    def collective_efficiency_term(self, x, t):
        theta = x[self.num_nodes:]
        angle_diff = 0
        for i in range(self.num_nodes):
            for j in range(i+1, self.num_nodes):
                if self.line_params[i,j] != 0:
                    angle_diff += (theta[i] - theta[j])**2
        return angle_diff
    
    def constraints(self, x, t):
        P = x[:self.num_nodes]
        constraint_values = []
        total_power = np.sum(P) - np.sum(self.generator_costs, axis=0)[2] * 0.8
        constraint_values.append(total_power)
        for i in range(self.num_nodes):
            max_gen = self.generator_costs[i][2]
            constraint_values.append(P[i] - max_gen)
            constraint_values.append(-P[i])
        return np.array(constraint_values)
