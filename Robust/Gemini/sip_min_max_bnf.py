from sip_bnf_framework import SIP_Framework
import pyoptinterface as poi
from pyoptinterface import xpress

class LBP:
    def __init__(self):
        self.model = xpress.Model()
        self.model.set_model_attribute(poi.ModelAttribute.Silent, True)
        self.x1 = self.model.add_variable(lb=-10, ub=10)
        self.mu = self.model.add_variable(lb=-1000, ub=1000)
        self.x = [self.x1, self.mu]
        self.model.set_objective(self.mu, poi.ObjectiveSense.Minimize)

    def get_decision_value(self):
        return [self.model.get_value(xi) for xi in self.x]

    def add_discretization(self, y_value):
        self.model.add_linear_constraint(self.x1 * y_value - self.mu <= 0)

class LLP:
    def __init__(self):
        self.model = xpress.Model()
        self.model.set_model_attribute(poi.ModelAttribute.Silent, True)
        self.y = self.model.add_variable(lb=-10, ub=10)

        self.current_x = []

    def update_with_lbp_solution(self, x_val):
        self.current_x = x_val
        x = self.current_x[0]
        mu = self.current_x[1]
        # Ici on maximise la violation : y - x
        self.model.set_objective(-(x * self.y), poi.ObjectiveSense.Minimize)

    def get_violation_point(self):
        return self.model.get_value(self.y)

    def get_violation_score(self, x_val):
        x = x_val[0]
        y = self.get_violation_point()
        return y * x

if __name__ == "__main__":
    framework = SIP_Framework(LBP(), LLP())
    result = framework.solve()
    print(f"Résultat final : {result}")