import pyoptinterface as poi
from pyoptinterface import xpress


class RRHS_Solver:
    def __init__(self, x_bounds, y_bounds):
        self.x_bounds = x_bounds
        self.y_bounds = y_bounds

        self.Y_lbp_disc = []  # Discrétisation initiale
        self.Y_ubp_disc = []  # Discrétisation initiale
        self.epsilon = 0.1  # Paramètre de restriction initial
        self.beta = 0.5  # Facteur de réduction de epsilon

        self.LB = -float('inf')
        self.UB = float('inf')
        self.best_x = None

    def _create_master_model(self, restriction=0.0):
        """Crée le modèle LBP (si restriction=0) ou UBP (si restriction > 0)"""
        model = xpress.Model()
        model.set_model_attribute(poi.ModelAttribute.Silent, True)
        x = model.add_variable(lb=self.x_bounds[0], ub=self.x_bounds[1], name="x")
        model.set_objective(x, poi.ObjectiveSense.Minimize)  # Simplifié: min x

        model.add_linear_constraint(x >= -1.0)

        # Ajout des coupes accumulées dans Y_disc
        for y_val in self.Y_lbp_disc:
            # g(x, y) + restriction <= 0
            # Pour l'exemple simple x >= y, on adapte la contrainte
            model.add_linear_constraint(x >= y_val)

        if restriction > 0:
            for y_val in self.Y_ubp_disc:
                # g(x, y) + restriction <= 0
                # Pour l'exemple simple x >= y, on adapte la contrainte
                model.add_linear_constraint(x - y_val >= -restriction)

        return model, x

    def solve_llp(self, current_x):
        """Lower Level Problem: Maximize g(current_x, y) over y"""
        model = xpress.Model()
        model.set_model_attribute(poi.ModelAttribute.Silent, True)
        # On force Xpress en mode global pour le non-linéaire si nécessaire
        y = model.add_variable(lb=self.y_bounds[0], ub=self.y_bounds[1], name="y")

        # Objectif: maximiser la violation (y - current_x)
        # Dans un vrai cas complexe, g(x, y) serait une expression non-linéaire
        # model.set_objective(y, poi.ObjectiveSense.Maximize)
        model.set_objective(-(y - current_x), poi.ObjectiveSense.Minimize)

        model.optimize()
        worst_y = model.get_value(y)
        return model.get_value(y), worst_y - current_x

    def run(self, max_iter=20, tol=1e-4):
        print(f"{'Iter':<5} | {'LB':<8} | {'UB':<8} | {'eps':<8} | {'Status'}")
        print("-" * 55)

        for i in range(max_iter):
            # --- 1. LBP (Lower Bounding Problem) ---
            lbp_model, lbp_x = self._create_master_model(restriction=0.0)
            lbp_model.set_model_attribute(poi.ModelAttribute.Silent, True)
            lbp_model.optimize()

            if lbp_model.get_model_attribute(poi.ModelAttribute.TerminationStatus) == poi.TerminationStatusCode.INFEASIBLE:
                print("Le problème SIP est globalement infaisable.")
                return None

            current_x_lbp = lbp_model.get_value(lbp_x)
            self.LB = current_x_lbp

            # --- 2. LLP pour enrichir la discrétisation (Coupure de Benders/SIP) ---
            y_star_lbp, viol_lbp = self.solve_llp(current_x_lbp)
            if y_star_lbp not in self.Y_lbp_disc:
                self.Y_lbp_disc.append(y_star_lbp)

            # --- 3. UBP (Upper Bounding Problem - Restriction) ---
            ubp_model, ubp_x = self._create_master_model(restriction=self.epsilon)
            ubp_model.optimize()

            status_ubp = "Infeasible"
            if ubp_model.get_model_attribute(poi.ModelAttribute.TerminationStatus) == poi.TerminationStatusCode.OPTIMAL:
                current_x_ubp = ubp_model.get_value(ubp_x)

                # Vérification de la faisabilité GLOBALE de la solution restreinte
                y_star_ubp, viol_global = self.solve_llp(current_x_ubp)
                if y_star_ubp not in self.Y_ubp_disc:
                    self.Y_ubp_disc.append(y_star_ubp)

                if viol_global <= 1e-7:  # Si vraiment faisable partout
                    self.UB = min(self.UB, current_x_ubp)
                    self.best_x = current_x_ubp
                    status_ubp = "Feasible"
                else:
                    status_ubp = "Violated"

            print(f"{i:<5} | {self.LB:<8.4f} | {self.UB:<8.4f} | {self.epsilon:<8.4f} | {status_ubp}")

            # --- 4. Critère d'arrêt et mise à jour de epsilon ---
            if self.UB - self.LB <= tol:
                print("-" * 55)
                print(f"Convergence! x* = {self.best_x}")
                break

            # Logique libDIPS : On réduit epsilon si l'UBP ne donne rien de faisable
            if status_ubp != "Feasible":
                self.epsilon *= self.beta

        return self.best_x


if __name__ == "__main__":
    # --- Configuration du problème ---
    # Min x st x >= y pour tout y dans [0, 0.5]
    solver = RRHS_Solver(
        x_bounds=(-10, 10),
        y_bounds=(0.0, 0.5)
    )

    result = solver.run()