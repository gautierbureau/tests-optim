import pyoptinterface as poi
from pyoptinterface import xpress
import numpy as np


class BNFMinMaxSolver:
    def __init__(self, x_bounds, y_bounds, f_func, eps=1e-6, max_iter=50):
        """
        Solves: min_x max_y f(x, y)
        x_bounds: List of (low, high) for x
        y_bounds: List of (low, high) for y
        f_func: A function f(x, y) that returns a pyoptinterface expression
        """
        self.x_bounds = x_bounds
        self.y_bounds = y_bounds
        self.f_func = f_func
        self.eps = eps
        self.max_iter = max_iter

        # Internal state
        self.discretized_y = []

    def solve(self, initial_y):
        self.discretized_y = [initial_y]

        # 1. Initialize Master Problem Model
        master_model = xpress.Model()
        master_model.set_model_attribute(poi.ModelAttribute.Silent, True)
        x = [master_model.add_variable(lb=b[0], ub=b[1], name=f"x_{i}") for i, b in enumerate(self.x_bounds)]
        eta = master_model.add_variable(lb=-1e9, name="eta")  # The max value helper
        master_model.set_objective(eta, poi.ObjectiveSense.Minimize)

        print(f"{'Iter':<5} | {'Upper Bound':<12} | {'Lower Bound':<12} | {'Gap':<10}")
        print("-" * 50)

        for i in range(self.max_iter):
            # --- STEP A: Solve Master Problem ---
            # Add constraints: f(x, y_i) <= eta for all y_i in discretized set
            # In a real implementation, we only add the NEWEST y_i
            latest_y = self.discretized_y[-1]
            master_model.add_quadratic_constraint(self.f_func(x, latest_y) <= eta)

            master_model.optimize()

            status = master_model.get_model_attribute(poi.ModelAttribute.TerminationStatus)
            if status != poi.TerminationStatusCode.OPTIMAL:
                print("Master problem not optimal")
                break

            current_x_vals = [master_model.get_value(var) for var in x]
            eta_val = master_model.get_value(eta)

            # --- STEP B: Solve Subproblem (The "Max" part) ---
            # Maximize f(x_fixed, y) over y
            sub_model = xpress.Model()
            sub_model.set_model_attribute(poi.ModelAttribute.Silent, True)
            y_vars = [sub_model.add_variable(lb=b[0], ub=b[1], name=f"y_{j}") for j, b in enumerate(self.y_bounds)]

            # Objective: Maximize f(current_x, y)
            sub_obj = self.f_func(current_x_vals, y_vars)
            sub_model.set_objective(sub_obj, poi.ObjectiveSense.Maximize)

            sub_model.optimize()

            f_max_val = sub_model.get_model_attribute(poi.ModelAttribute.ObjectiveValue)
            current_y_vals = [sub_model.get_value(v) for v in y_vars]

            # --- STEP C: Convergence Check ---
            gap = f_max_val - eta_val
            print(f"{i:<5} | {f_max_val:<12.6f} | {eta_val:<12.6f} | {gap:<10.2e}")

            if gap < self.eps:
                print("\nConvergence reached.")
                return current_x_vals, eta_val

            # Add the new worst-case y to the discretization set
            self.discretized_y.append(current_y_vals)

        print("\nReached max iterations.")
        return current_x_vals, eta_val


# --- Example Usage ---
# Problem: min_x max_y (x - y)^2  where x in [0, 2], y in [0, 1]
# Note: This is a trivial convex-concave example.

def my_objective(x, y):
    # f(x, y) = x^2 - 2xy + y^2
    # In pyoptinterface, we use standard operators
    return x[0] ** 2 - 2 * x[0] * y[0] + y[0] ** 2

if __name__ == "__main__":
    solver = BNFMinMaxSolver(
        x_bounds=[(0, 2)],
        y_bounds=[(0, 1)],
        f_func=my_objective
    )

    # Start with an initial guess for y
    best_x, min_max_val = solver.solve(initial_y=[0.5])

    print(f"Optimal x: {best_x}")
    print(f"Min-Max Value: {min_max_val}")