import pyoptinterface as poi
from pyoptinterface import highs

class SIP_Restriction_Framework:
    def __init__(self, master, sub):
        self.master = master
        self.sub = sub
        self.epsilon = 0.1  # Facteur de restriction initial
        self.UB = float('inf')

    def solve(self, max_iter=15):
        for i in range(max_iter):
            # 1. Résoudre le Master avec restriction
            # On cherche x tel que g(x, y) + epsilon <= 0
            self.master.solve_restricted(self.epsilon)
            current_x = self.master.get_x()

            # 2. Oracle : Calculer le maximum global de g(x, y) pour le x actuel
            # C'est ici que l'optimisation globale intervient
            max_g, worst_y = self.sub.compute_global_max(current_x)

            # 3. Mise à jour de la Borne Supérieure (UB)
            # Si le max_g est <= 0, alors current_x est réalisable pour le SIP original
            if max_g <= 0:
                self.UB = min(self.UB, self.master.get_objective_value())
                print(f"Iter {i}: Faisable! UB actualisée: {self.UB}")
            else:
                print(f"Iter {i}: Infaisable (Violation: {max_g})")

            # 4. Ajustement de epsilon (Logique de Mitsos/Barton)
            # On réduit epsilon ou on ajoute des coupes selon l'écart
            if max_g > 0:
                self.epsilon *= 1.1 # On durcit la restriction
            else:
                self.epsilon *= 0.5 # On peut relâcher un peu

        return self.UB


class Global_SIP_Solver:
    def __init__(self, f, g, Y_domain):
        self.Y_disc = {Y_domain.sample()}  # On commence avec un point au hasard
        self.epsilon = 0.1
        self.LB = -float('inf')
        self.UB = float('inf')
        self.best_x = None

    def step(self):
        # --- PHASE 1 : LOWER BOUND ---
        # Résoudre le problème relaxé sur Y_disc
        x_lb = self.solve_master(self.Y_disc, restriction=0)
        self.LB = f(x_lb)

        # --- PHASE 2 : UPPER BOUND (Restriction) ---
        # Résoudre le problème restreint sur Y_disc
        x_ub = self.solve_master(self.Y_disc, restriction=self.epsilon)

        if x_ub is not supported:  # Si la restriction est trop forte, le problème est infaisable
            self.epsilon *= 0.5  # On réduit la sévérité
        else:
            # Vérifier la faisabilité globale de x_ub
            # On cherche y* qui maximise g(x_ub, y)
            max_violation, y_star = self.global_subproblem(x_ub)

            if max_violation <= 0:
                # x_ub est réellement faisable pour le SIP !
                if f(x_ub) < self.UB:
                    self.UB = f(x_ub)
                    self.best_x = x_ub

        # --- PHASE 3 : ENRICHISSEMENT ---
        # On ajoute le pire point du problème LB pour affiner la relaxation
        _, y_lb_star = self.global_subproblem(x_lb)
        self.Y_disc.add(y_lb_star)

    def run(self, tol=1e-3):
        while (self.UB - self.LB) > tol:
            self.step()
        return self.best_x