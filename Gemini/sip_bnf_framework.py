import pyoptinterface as poi
from pyoptinterface import xpress

class SIP_Framework:
    def __init__(self, lbp_solver, llp_solver):
        self.lbp = lbp_solver
        self.llp = llp_solver
        self.Y_grid = []

    def solve(self, max_iter=20, tol=0.01):
        print(f"{'Iter':<5} | {'LBP x':<10} | {'LLP y*':<10} | {'Violation':<10}")
        print("-" * 45)
        for i in range(max_iter):
            # 1. Résoudre le niveau "Haut" (lbp)
            self.lbp.model.optimize()
            current_x = self.lbp.get_decision_value()

            # 2. Passer l'info au niveau "Bas" (llp-problem / Discretization)
            # On met à jour le sous-problème avec la solution actuelle du maître
            self.llp.update_with_lbp_solution(current_x)
            self.llp.model.optimize()

            worst_y = self.llp.get_violation_point()
            violation_score = self.llp.get_violation_score(current_x)

            print(f"{i:<5} | {current_x:<10.4f} | {worst_y:<10.4f} | {max(0, violation_score):<10.4f}")

            # 3. Test de convergence
            if violation_score <= tol:
                return current_x

            # 4. Interaction : Générer une nouvelle "coupe" (Discretization)
            # On définit comment le point y devient une contrainte pour x
            self.lbp.add_discretization(worst_y)
            self.Y_grid.append(worst_y)

        return None

class LBP:
    def __init__(self):
        self.model = xpress.Model()
        self.model.set_model_attribute(poi.ModelAttribute.Silent, True)
        self.x = self.model.add_variable(lb=-10, ub=10)
        self.model.set_objective(self.x, poi.ObjectiveSense.Minimize)
        self.model.add_linear_constraint(self.x >= -1.0)  # Contrainte de base

    def get_decision_value(self):
        return self.model.get_value(self.x)

    def add_discretization(self, y_value):
        # La règle d'interaction : x >= y
        self.model.add_linear_constraint(self.x >= y_value)

class LLP:
    def __init__(self):
        self.model = xpress.Model()
        self.model.set_model_attribute(poi.ModelAttribute.Silent, True)
        self.y = self.model.add_variable(lb=0.0, ub=0.5)
        self.current_x = 0

    def update_with_lbp_solution(self, x_val):
        self.current_x = x_val
        # Ici on maximise la violation : y - x
        self.model.set_objective(-(self.y - self.current_x), poi.ObjectiveSense.Minimize)

    def get_violation_point(self):
        return self.model.get_value(self.y)

    def get_violation_score(self, x_val):
        return self.get_violation_point() - x_val

# --- Exécution ---
if __name__ == "__main__":
    framework = SIP_Framework(LBP(), LLP())
    result = framework.solve()
    print(f"Résultat final : {result}")