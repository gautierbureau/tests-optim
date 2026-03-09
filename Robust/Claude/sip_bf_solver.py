"""
Algorithme Blankenship & Falk (B&F) pour problèmes SIP (Semi-Infinite Programming)
avec sous-problèmes MILP résolus via pyoptinterface + Xpress.

Structure du problème SIP :
    min   f(x)
    s.t.  g_i(x, y) <= 0    ∀y ∈ Y,  i = 1..m     (contraintes semi-infinies)
          h_j(x) <= 0        j = 1..p               (contraintes finies)
          x ∈ X ⊆ ℝⁿ (mixte-entier)

Deux sous-problèmes à chaque itération k :
    LBP : MILP sur discrétisation courante Y_k ⊂ Y  → borne inférieure + x*
    LLP : MILP / NLP pour chaque contrainte i       → pire y*, violation max
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pyoptinterface as poi
from pyoptinterface import xpress

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Structures de données
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BFConfig:
    """Configuration de l'algorithme B&F."""
    tol_violation: float = 1e-6       # Tolérance sur la violation (critère d'arrêt LLP)
    tol_optimality: float = 1e-6      # Tolérance sur le gap LB/UB
    max_iter: int = 200               # Nombre max d'itérations
    max_cuts_per_iter: int = 1        # Nb de points y* ajoutés par itération (multi-cut)
    time_limit: float = 3600.0        # Limite de temps globale (secondes)
    lbp_time_limit: float = 300.0     # Limite de temps pour chaque LBP
    llp_time_limit: float = 60.0      # Limite de temps pour chaque LLP
    verbose: bool = True              # Affichage des itérations
    xpress_log_level: int = 0         # 0=silencieux, 1=résumé, 5=détaillé


@dataclass
class BFResult:
    """Résultat de l'algorithme B&F."""
    status: str                        # "optimal", "infeasible", "timeout", "max_iter"
    x_opt: np.ndarray | None = None    # Solution optimale
    f_opt: float = float("inf")        # Valeur objectif optimale
    lower_bound: float = -float("inf") # Meilleure borne inférieure
    upper_bound: float = float("inf")  # Meilleure borne supérieure (= f_opt si faisable)
    iterations: int = 0               # Nombre d'itérations effectuées
    n_cuts_total: int = 0             # Nombre total de coupes ajoutées
    solve_time: float = 0.0           # Temps de résolution total
    history: list[dict] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Interface abstraite : à implémenter par l'utilisateur
# ─────────────────────────────────────────────────────────────────────────────

class SIPProblem(ABC):
    """
    Classe de base abstraite décrivant un problème SIP.

    L'utilisateur hérite de cette classe et implémente :
        - build_lbp_model  : construit le MILP LBP avec poi + Xpress
        - build_llp_model  : construit le MILP LLP pour trouver le pire y*
        - get_x_values     : extrait la solution x* du modèle LBP
        - get_y_values     : extrait la solution y* du modèle LLP
        - get_violation    : calcule la violation g(x*, y*) depuis le modèle LLP
        - initial_y_points : points y initiaux pour amorcer Y_0
    """

    # ── À implémenter ────────────────────────────────────────────────────────

    @abstractmethod
    def build_lbp_model(
        self,
        y_points: list[np.ndarray],
        config: BFConfig,
    ) -> xpress.Model:
        """
        Construit et retourne le modèle MILP du LBP avec pyoptinterface.

        Le LBP est le problème relaxé :
            min   f(x)
            s.t.  g_i(x, y) <= 0   ∀y ∈ y_points,  i = 1..m
                  h_j(x)    <= 0   j = 1..p
                  x ∈ X

        Args:
            y_points : liste des points y discrets accumulés (Y_k)
            config   : configuration de l'algorithme

        Returns:
            model : modèle pyoptinterface xpress prêt à être optimisé
                    (NE PAS appeler model.optimize() ici)
        """
        ...

    @abstractmethod
    def build_llp_model(
        self,
        x_values: np.ndarray,
        constraint_idx: int,
        config: BFConfig,
    ) -> xpress.Model:
        """
        Construit et retourne le modèle MILP du LLP pour la contrainte i.

        Le LLP cherche le pire cas :
            max   g_i(x*, y)
            s.t.  y ∈ Y

        Args:
            x_values       : solution x* issue du LBP
            constraint_idx : indice de la contrainte i (0-indexé)
            config         : configuration

        Returns:
            model : modèle pyoptinterface xpress prêt à être optimisé
        """
        ...

    @abstractmethod
    def get_x_values(self, lbp_model: xpress.Model) -> np.ndarray:
        """
        Extrait le vecteur x* de la solution du modèle LBP.

        Args:
            lbp_model : modèle LBP après résolution

        Returns:
            x* sous forme np.ndarray
        """
        ...

    @abstractmethod
    def get_y_values(self, llp_model: xpress.Model, constraint_idx: int):
        """
        Extrait le vecteur y* de la solution du modèle LLP.

        Args:
            llp_model      : modèle LLP après résolution
            constraint_idx : indice de la contrainte

        Returns:
            y* sous forme np.ndarray
        """
        ...

    @abstractmethod
    def get_violation(self, llp_model: xpress.Model, constraint_idx: int, x_k) -> float:
        """
        Retourne la valeur de la violation g_i(x*, y*) depuis le modèle LLP.
        Une valeur > 0 signifie que la contrainte est violée.

        Args:
            llp_model      : modèle LLP après résolution
            constraint_idx : indice de la contrainte

        Returns:
            valeur de g_i(x*, y*)
        """
        ...

    @abstractmethod
    def initial_y_points(self) -> list:
        """
        Retourne une liste de points y initiaux pour Y_0.
        Typiquement : quelques extrêmes/coins de Y + centre.

        Returns:
            liste de np.ndarray
        """
        ...

    # ── Propriétés optionnelles à surcharger ─────────────────────────────────

    @property
    def n_semi_infinite_constraints(self) -> int:
        """
        Nombre de familles de contraintes semi-infinies (défaut = 1).
        Surcharger si le problème a plusieurs familles g_1, g_2, ...
        """
        return 1

    def evaluate_objective(self, x: np.ndarray) -> float:
        """
        Évalue f(x) directement (utile pour le calcul de UB).
        Par défaut, on lit la valeur du LBP.
        Surcharger si nécessaire.
        """
        raise NotImplementedError(
            "Surcharger evaluate_objective() ou laisser le LBP calculer UB."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Solveur B&F principal
# ─────────────────────────────────────────────────────────────────────────────

class BFSolver:
    """
    Solveur Blankenship & Falk pour SIP avec sous-problèmes MILP (pyoptinterface + Xpress).

    Usage :
        solver = BFSolver(problem=MyProblem(), config=BFConfig())
        result = solver.solve()
    """

    def __init__(self, problem: SIPProblem, config: BFConfig | None = None):
        self.problem = problem
        self.config = config or BFConfig()
        self._setup_logging()

    def _setup_logging(self) -> None:
        if self.config.verbose:
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s [B&F] %(message)s",
                datefmt="%H:%M:%S",
            )

    # ── Boucle principale B&F ────────────────────────────────────────────────

    def solve(self) -> BFResult:
        """
        Lance l'algorithme Blankenship & Falk.

        Returns:
            BFResult avec solution optimale et informations de convergence
        """
        cfg = self.config
        result = BFResult(status="running")
        t_start = time.perf_counter()

        # ── Initialisation ───────────────────────────────────────────────────
        Y_k = self.problem.initial_y_points()
        # if not Y_k:
        #     raise ValueError("initial_y_points() doit retourner au moins un point.")

        logger.info("=" * 70)
        logger.info("Algorithme B&F - SIP MILP (pyoptinterface + Xpress)")
        logger.info(f"  Tolérance violation : {cfg.tol_violation:.2e}")
        logger.info(f"  Max itérations     : {cfg.max_iter}")
        logger.info(f"  Contraintes SI     : {self.problem.n_semi_infinite_constraints}")
        logger.info(f"  Points initiaux    : {len(Y_k)}")
        logger.info("=" * 70)
        logger.info(f"{'Iter':>5} {'LB':>14} {'UB':>14} {'Gap':>10} {'Viol.':>12} {'|Y_k|':>7} {'Time':>8}")
        logger.info("-" * 70)

        for k in range(cfg.max_iter):

            # ── Vérification limite de temps ──────────────────────────────
            elapsed = time.perf_counter() - t_start
            if elapsed > cfg.time_limit:
                result.status = "timeout"
                logger.warning(f"Limite de temps atteinte ({elapsed:.1f}s)")
                break

            # ── ÉTAPE 1 : Résoudre LBP ────────────────────────────────────
            lbp_status, lbp_model, lb = self._solve_lbp(Y_k)

            if lbp_status == "infeasible":
                result.status = "infeasible"
                logger.warning(f"Iter {k}: LBP infaisable → problème SIP infaisable.")
                break

            if lbp_status == "error":
                result.status = "error"
                logger.error(f"Iter {k}: Erreur lors de la résolution du LBP.")
                break

            result.lower_bound = lb
            x_k = self.problem.get_x_values(lbp_model)

            # ── ÉTAPE 2 : Résoudre LLP(s) ────────────────────────────────
            new_y_points, max_violation = self._solve_llps(x_k)

            # ── Mise à jour UB si x_k est faisable ───────────────────────
            if max_violation <= cfg.tol_violation:
                # x_k est (ε-)faisable → c'est une borne supérieure valide
                ub = lb  # lb == f(x_k) car LBP est relaxation de SIP
                if ub < result.upper_bound:
                    result.upper_bound = ub
                    result.x_opt = x_k.copy()
                    result.f_opt = ub

            gap = _compute_gap(result.lower_bound, result.upper_bound)
            elapsed = time.perf_counter() - t_start

            logger.info(
                f"{k:>5} {result.lower_bound:>14.6f} {result.upper_bound:>14.6f} "
                f"{gap:>10.2e} {max_violation:>12.2e} {len(Y_k):>7} {elapsed:>7.1f}s"
            )

            # ── Enregistrement historique ─────────────────────────────────
            result.history.append({
                "iter": k,
                "lb": result.lower_bound,
                "ub": result.upper_bound,
                "gap": gap,
                "max_violation": max_violation,
                "n_y_points": len(Y_k),
                "time": elapsed,
            })

            # ── ÉTAPE 3 : Critère d'arrêt ─────────────────────────────────
            if max_violation <= cfg.tol_violation:
                if gap <= cfg.tol_optimality:
                    result.status = "optimal"
                    logger.info(f"✓ Convergence à l'itération {k} | f* = {result.f_opt:.6f}")
                    break
                # Faisable mais pas encore optimal (cas rare avec MILP)
                # On continue quand même (la LB va remonter)

            # ── ÉTAPE 4 : Mise à jour Y_k (ajout des pires y*) ────────────
            n_before = len(Y_k)
            for y_new in new_y_points[: cfg.max_cuts_per_iter]:
                if not _is_duplicate(y_new, Y_k):
                    Y_k.append(y_new)

            result.n_cuts_total += len(Y_k) - n_before
            result.iterations = k + 1

        else:
            result.status = "max_iter"
            logger.warning(f"Nombre maximum d'itérations atteint ({cfg.max_iter})")

        result.solve_time = time.perf_counter() - t_start
        self._log_summary(result)
        return result

    # ── Résolution du LBP ────────────────────────────────────────────────────

    def _solve_lbp(
        self, y_points: list
    ) -> tuple[str, xpress.Model | None, float]:
        """
        Construit et résout le LBP.

        Returns:
            (status, model, lower_bound)
        """
        try:
            model = self.problem.build_lbp_model(y_points, self.config)
            _set_xpress_params(model, self.config.lbp_time_limit, self.config.xpress_log_level)
            model.optimize()
            termination = model.get_model_attribute(poi.ModelAttribute.TerminationStatus)
        except Exception as exc:
            logger.error(f"LBP exception: {exc}")
            return "error", None, -float("inf")

        if termination == poi.TerminationStatusCode.INFEASIBLE:
            return "infeasible", model, -float("inf")

        if termination not in (
            poi.TerminationStatusCode.OPTIMAL,
            poi.TerminationStatusCode.LOCALLY_SOLVED,
            poi.TerminationStatusCode.ALMOST_OPTIMAL,
        ):
            logger.warning(f"LBP statut inattendu : {termination}")
            return "error", model, -float("inf")

        lb = model.get_model_attribute(poi.ModelAttribute.ObjectiveValue)
        return "optimal", model, lb

    # ── Résolution des LLP(s) ────────────────────────────────────────────────

    def _solve_llps(
        self, x_k: np.ndarray
    ) -> tuple[list, float]:
        """
        Résout un LLP par famille de contraintes semi-infinies.

        Returns:
            (new_y_points triés par violation décroissante, violation max)
        """
        candidates: list[tuple[float, float]] = []  # (violation, y*)

        for i in range(self.problem.n_semi_infinite_constraints):
            try:
                model = self.problem.build_llp_model(x_k, i, self.config)
                _set_xpress_params(model, self.config.llp_time_limit, self.config.xpress_log_level)
                model.optimize()
                termination = model.get_model_attribute(poi.ModelAttribute.TerminationStatus)
            except Exception as exc:
                logger.error(f"LLP[{i}] exception: {exc}")
                continue

            if termination not in (
                poi.TerminationStatusCode.OPTIMAL,
                poi.TerminationStatusCode.LOCALLY_SOLVED,
                poi.TerminationStatusCode.ALMOST_OPTIMAL,
            ):
                logger.warning(f"LLP[{i}] statut inattendu : {termination}")
                continue

            violation = self.problem.get_violation(model, i, x_k)
            y_star = self.problem.get_y_values(model, i)
            candidates.append((violation, y_star))

        if not candidates:
            return [], 0.0

        # Trier par violation décroissante (pires violations en premier)
        candidates.sort(key=lambda t: t[0], reverse=True)
        max_violation = candidates[0][0]
        new_y_points = [y for _, y in candidates]

        return new_y_points, max_violation

    # ── Affichage résumé ──────────────────────────────────────────────────────

    def _log_summary(self, result: BFResult) -> None:
        logger.info("=" * 70)
        logger.info(f"Statut      : {result.status.upper()}")
        logger.info(f"Objectif    : {result.f_opt:.6f}")
        logger.info(f"Borne inf.  : {result.lower_bound:.6f}")
        logger.info(f"Borne sup.  : {result.upper_bound:.6f}")
        logger.info(f"Gap         : {_compute_gap(result.lower_bound, result.upper_bound):.2e}")
        logger.info(f"Itérations  : {result.iterations}")
        logger.info(f"Coupes tot. : {result.n_cuts_total}")
        logger.info(f"Temps (s)   : {result.solve_time:.2f}")
        logger.info("=" * 70)


# ─────────────────────────────────────────────────────────────────────────────
# Fonctions utilitaires
# ─────────────────────────────────────────────────────────────────────────────

def _set_xpress_params(model: xpress.Model, time_limit: float, log_level: int) -> None:
    """Configure les paramètres Xpress communs."""
    model.set_raw_control("MAXTIME", int(time_limit))
    model.set_raw_control("OUTPUTLOG", log_level)


def _compute_gap(lb: float, ub: float) -> float:
    """Gap relatif entre borne inférieure et supérieure."""
    if abs(ub) < 1e-10:
        return abs(ub - lb)
    return abs(ub - lb) / max(abs(ub), 1e-10)


def _is_duplicate(y_new: np.ndarray, Y_k: list[np.ndarray], tol: float = 1e-8) -> bool:
    """Vérifie si y_new est déjà (quasi-)présent dans Y_k."""
    for y in Y_k:
        if np.linalg.norm(y_new - y) < tol:
            return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Exemple d'implémentation : Problème SIP linéaire simple
# ─────────────────────────────────────────────────────────────────────────────

class ExampleLinearSIP(SIPProblem):
    """
    Exemple concret de problème SIP MILP :

        min   c' x
        s.t.  a(y)' x >= b(y)    ∀y ∈ [y_lb, y_ub]
              x >= 0, x ∈ ℤⁿ (optionnel)

    où a(y) et b(y) sont linéaires en y (donc LLP = LP/MILP).

    ─── Problème test ───────────────────────────────────────────────────────
        min   x1 + x2
        s.t.  x1*sin(y) + x2*cos(y) >= 1   ∀y ∈ [0, 2π]
        x1, x2 >= 0

    Reformulation de la contrainte en g(x,y) <= 0 :
        g(x, y) = 1 - x1*sin(y) - x2*cos(y) <= 0   ∀y ∈ [0, 2π]

    LBP : min x1 + x2
          s.t. x1*sin(y_j) + x2*cos(y_j) >= 1   ∀y_j ∈ Y_k
               x1, x2 >= 0

    LLP (pour x* fixé) : max  1 - x1*sin(y) - x2*cos(y)
                         s.t. y ∈ [0, 2π]
        → Pour un MILP, y est discrétisé ou on utilise une approximation.
        → Ici, cas LP : y continu, mais g non-linéaire en y.
        → On discrétise finement Y pour le LLP (LP approché).
        → Dans un vrai MILP, le LLP peut être un MILP exact.
    """

    def __init__(self):
        # Grille fine pour approximer le LLP continu par un LP
        self.y_grid = []
        self._x_vars: list[poi.VariableIndex] = []
        self._y_var: poi.VariableIndex | None = None
        self._llp_y_values: np.ndarray = np.array([])

    def initial_y_points(self) -> list:
        """Points initiaux : quelques valeurs de y dans [0, 2π]."""
        return []

    def build_lbp_model(
        self,
        y_points: list,
        config: BFConfig,
    ) -> xpress.Model:
        """LBP : LP avec contraintes pour chaque y_j ∈ Y_k."""
        model = xpress.Model()

        x = model.add_variable(lb=-10, ub=10)
        self._x_vars = [x]

        # Objectif : min x
        model.set_objective(x, poi.ObjectiveSense.Minimize)
        model.add_linear_constraint(x >= -1.0)  # Contrainte de base

        for y_pt in y_points:
            model.add_linear_constraint(x >= y_pt)

        return model

    def build_llp_model(
        self,
        x_values: np.ndarray,
        constraint_idx: int,
        config: BFConfig,
    ) -> xpress.Model:
        """
        LLP : trouver max g(x*, y) = max [1 - x1*sin(y) - x2*cos(y)] sur y ∈ [0, 2π].

        Ici on utilise une grille fine (approximation LP) car g est non-linéaire en y.
        Dans un vrai MILP SIP, le LLP serait un MILP exact.
        """
        x_val = float(x_values[0])

        # Évaluer g(x*, y) sur la grille
        # violations = 1.0 - x1_val * np.sin(self.y_grid) - x2_val * np.cos(self.y_grid)
        #
        # # Trouver le maximum
        # idx_max = int(np.argmax(violations))
        # self._llp_y_values = self.y_grid

        # Créer un modèle "fictif" qui stocke le résultat
        # (dans un vrai MILP LLP, on aurait de vraies variables y)
        model = xpress.Model()
        y_var = model.add_variable(lb=0.0, ub=0.5)

        self._y_var = y_var

        # Objectif = violation maximale (constante ici car y est fixé)
        model.set_objective(-(y_var - x_val), poi.ObjectiveSense.Minimize)

        return model

    def get_x_values(self, lbp_model: xpress.Model) -> np.ndarray:
        x_vals = [
            lbp_model.get_value(v) for v in self._x_vars
        ]
        return np.array(x_vals)

    def get_y_values(self, llp_model: xpress.Model, constraint_idx: int):
        y_val = llp_model.get_value(self._y_var)
        return y_val

    def get_violation(self, llp_model: xpress.Model, constraint_idx: int, x_k) -> float:
        return max(self.get_y_values(llp_model, constraint_idx) - x_k)


# ─────────────────────────────────────────────────────────────────────────────
# Point d'entrée
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # ── Exemple d'utilisation ─────────────────────────────────────────────
    problem = ExampleLinearSIP()

    config = BFConfig(
        tol_violation=1e-5,
        tol_optimality=1e-5,
        max_iter=50,
        max_cuts_per_iter=1,
        verbose=True,
        xpress_log_level=0,
    )

    solver = BFSolver(problem=problem, config=config)
    result = solver.solve()

    print(f"\nSolution : x* = {result.x_opt}")
    print(f"Objectif : f* = {result.f_opt:.6f}")
    print(f"Statut   : {result.status}")
