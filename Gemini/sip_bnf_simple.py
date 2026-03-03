import pyoptinterface as poi
# from pyoptinterface import highs
from pyoptinterface import xpress


def solve_sip_blankenship_falk():
    # 1. Initialisation du solveur Maître (Master Problem)
    # model = highs.Model()
    model = xpress.Model()
    model.set_model_attribute(poi.ModelAttribute.Silent, True)
    x = model.add_variable(lb=-10.0, ub=10.0)  # x dans [-10, 10]
    model.set_objective(x, poi.ObjectiveSense.Minimize)

    # Contrainte déterministe simple
    model.add_linear_constraint(x >= -1.0)

    # Ensemble de points y "critiques" (au début vide)
    Y_grid = []

    iteration = 0
    tol = 1e-6

    print(f"{'Iter':<10} | {'x value':<10} | {'Worst y':<10} | {'Violation':<10}")
    print("-" * 50)

    while True:
        # A. Résoudre le problème Maître avec les contraintes actuelles
        model.optimize()
        current_x = model.get_value(x)
        #print("TerminationStatus " + str(model.get_model_attribute(poi.ModelAttribute.TerminationStatus)))

        # B. Problème Esclave (Subproblem) : Trouver le pire y
        # On veut maximiser la violation : Violation = y - x
        # Sur l'intervalle [0, 0.5], le y qui maximise (y - current_x) est 0.5
        worst_y = 0.5
        violation = worst_y - current_x

        print(f"{iteration:<10} | {current_x:<10.4f} | {worst_y:<10.4f} | {max(0, violation):<10.4f}")

        # C. Test de convergence
        if violation <= tol:
            break

        # D. Ajouter la nouvelle contrainte et itérer
        # On ajoute : x >= worst_y
        model.add_linear_constraint(x >= worst_y)
        Y_grid.append(worst_y)

        iteration += 1
        if iteration > 20: break  # Sécurité

    print("-" * 50)
    print(f"Solution finale : x = {model.get_value(x)}")


if __name__ == "__main__":
    solve_sip_blankenship_falk()