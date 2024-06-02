import copy

import numpy as np
from scipy.optimize import minimize

EPSILON = 1e-10

def get_objective(k):
    if k ==3:
        print("Hello")
    def objective(q):
        # Objective function to maximize q_k
        return -q[k]
    return objective

def get_kl_constraint(p, beta):
    def constraint_kl(q):
        # Constraint for KL divergence <= beta
        kl_divergence = np.sum(q * np.log(q / p))
        return beta - kl_divergence
    return constraint_kl


# Example distribution p and parameters
# p = np.array([0.2, 0.3, 0.1, 0.4])  # Replace with your actual distribution
# BETA = 0.1  # Replace with your desired KL divergence threshold
# k = 2  # Index of the outcome you want to maximize

def get_max_q(p, beta, k):
    # Initial guess for q
    initial_guess = copy.deepcopy(p)
    constraint_kl = get_kl_constraint(p, beta)
    objective = get_objective(k)

    # Constraints
    constraints = [
        {'type': 'eq', 'fun': lambda q: np.sum(q) - 1},  # Sum of q_i equals 1
        {'type': 'ineq', 'fun': constraint_kl}  # KL divergence constraint
    ]

    # Bounds for q_i (between 0 and 1)
    bounds = [(EPSILON, 1-EPSILON) for _ in range(len(p))]

    # Solve the optimization problem
    # result = minimize(objective, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)
    result = minimize(objective, initial_guess, bounds=bounds, constraints=constraints)

    # Extract the optimized distribution q
    optimal_q = result.x
    print(f"Old prob = {p}\tnew_q = {optimal_q}\t KL = {np.sum(optimal_q * np.log(optimal_q/ p))}")
    return optimal_q

