import pyoptinterface as poi
# from pyoptinterface import highs
from pyoptinterface import xpress

def solve_sip_full_dynamic():
    # --- 1. CONFIGURATION DU PROBLÈME MAÎTRE ---
    # Min x st x >= 1 ET x >= y pour tout y dans [0, 0.5]
    master = xpress.Model()
    master.set_model_attribute(poi.ModelAttribute.Silent, True)
    x = master.add_variable(lb=-10.0, ub=10.0, name="x")
    master.set_objective(x, poi.ObjectiveSense.Minimize)

    # Contrainte initiale simple
    master.add_linear_constraint(x >= -1.0)

    tol = 0.01
    max_iter = 10

    print(f"{'Iter':<5} | {'Master x':<10} | {'Sub y*':<10} | {'Violation':<10}")
    print("-" * 45)

    Y_grid = []

    for i in range(max_iter):
        # ÉTAPE A : Résoudre le Master Problem
        master.optimize()
        current_x = master.get_value(x)

        # ÉTAPE B : Résoudre le Sub-Problem (L'Oracle)
        # On cherche le y qui viole le plus la contrainte : Maximize (y - x)
        # s.t. y \in [0, 0.5]
        sub = xpress.Model()
        sub.set_model_attribute(poi.ModelAttribute.Silent, True)
        y = sub.add_variable(lb=0.0, ub=0.5, name="y")


        # Objectif : maximiser la violation f(y) - x
        # Comme x est constant pour le sous-problème, on maximise simplement y
        # sub.set_objective(y - current_x, poi.ObjectiveSense.Maximize)
        sub.set_objective(-(y - current_x), poi.ObjectiveSense.Minimize)
        sub.optimize()

        worst_y = sub.get_value(y)
        violation = worst_y - current_x

        print(f"{i:<5} | {current_x:<10.4f} | {worst_y:<10.4f} | {max(0, violation):<10.4f}")

        # ÉTAPE C : Test de convergence
        if violation <= tol:
            print("-" * 45)
            print(f"Convergence atteinte ! Solution optimale : x = {current_x}")
            break

        # ÉTAPE D : Ajouter la coupe (Cut) au Master
        # On ajoute la contrainte x >= worst_y pour la prochaine itération
        master.add_linear_constraint(x >= worst_y)
        Y_grid.append(worst_y)
    else:
        print("Nombre max d'itérations atteint.")

if __name__ == "__main__":
    solve_sip_full_dynamic()