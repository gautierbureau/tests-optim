import pyoptinterface as poi
from pyoptinterface import xpress

def test():
    model = xpress.Model()

    variables = {}
    for i in range(1, 3):
        variables[f"x{i}"] = model.add_variable(lb=0, name=f"x{i}")

    variables['z'] = model.add_variable(domain=poi.VariableDomain.Binary, name="z")

    model.add_linear_constraint(variables["x1"] + variables["x2"], poi.Eq, 1)
    #model.add_linear_constraint(variables["x1"], poi.Geq, 0)
    #model.add_linear_constraint(variables["x2"], poi.Geq, 0)

    M = 100
    rhs = 10

    # Constraint: x <= 10 + M * (1 - z)
    # Rearranged for poi: x + M*z <= 10 + M
    expr = variables['x1'] + M * variables['z']

    model.add_linear_constraint(expr <= rhs + M)

    objective = variables["x1"] * variables["x1"] + 2 * variables["x2"] * variables["x2"]
    model.set_objective(objective, poi.ObjectiveSense.Minimize)

    model.optimize()

    print(
        f"x1 = {model.get_value(variables['x1'])}, x2 = {model.get_value(variables['x2'])}"
    )

if __name__ == "__main__":
    test()