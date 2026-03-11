"""
DC Optimal Power Flow — Two-Bus System (PTDF Formulation)
==========================================================
Replaces the theta-angle flow model with a PTDF-based flow model.

Theta formulation (previous)        PTDF formulation (this file)
───────────────────────────         ──────────────────────────────
Variables  : Pg, θ, P12             Variables  : Pg, P12  (no angles!)
Flow def   : P12 = b·(θ1−θ2)       Flow def   : P12 = Σ PTDF·Pnet
Ref. angle : θ1 = 0                (absorbed into PTDF derivation)

PTDF derivation for a two-bus system
──────────────────────────────────────
Given susceptance matrix B:
    B = [[ b12, -b12],
         [-b12,  b12]]

Remove slack row/col (Bus 1) → B_red = [b12]  (1×1 scalar)

Line-bus incidence (Bus2 side, since B1 is slack):
    A_red = [-1]   (line L12 ends at Bus 2 → -1 for the "to" bus)

PTDF(L12, B2) = A_red · B_red⁻¹ = -1/b12 · b12 = -1.0

Interpretation: 1 MW injected at Bus 2, withdrawn at slack (Bus 1),
  sends −1 MW on line L12 (i.e., 1 MW flows Bus2 → Bus1).

Flow equation (non-slack buses only):
    P12 = PTDF(L12, B2) · Pnet_B2
        = −1.0 · (Pg2 − Pd2)

This replaces the two constraints  [line_flow_def + ref angle fix]
with a single equality that has no angle variables.
"""

import math
import pypowsybl as ppw
import pypowsybl.network as pn
import pypowsybl.sensitivity as psa
import pandas as pd
import pyoptinterface as poi
from pyoptinterface import xpress
from dc_opf_two_bus import extract_parameters
import pypowsybl.loadflow as lf

# ── 2. Compute PTDF via pypowsybl Sensitivity Analysis ───────────────────────
#
#  pypowsybl's DC sensitivity engine computes ∂P_line / ∂P_injection directly.
#  We query: "how does flow on L12 respond to a unit injection at each bus?"
#
#  Convention: Bus 1 is slack (chosen by pypowsybl based on the generator
#  with voltage_regulator_on=True that is connected to it).
#
#  Result column B1 ≈ 0  (slack absorbs perturbation, no net effect on flow)
#  Result column B2 ≈ -1 (inject at B2, withdraw at slack → flow reversed)

def compute_ptdf(network: pn.Network, change_slack: bool) -> dict[str, float]:
    """
    Use pypowsybl DC sensitivity analysis to compute the PTDF matrix.

    Returns a dict  {bus_id: PTDF_value}  for line L12.
    Slack bus (B1) will have PTDF ≈ 0 and is excluded from the OPF constraints.
    """
    sa = psa.create_dc_analysis()

    # Ask: sensitivity of L12 flow w.r.t. injection at each generator bus
    #   - branch_id       : the monitored line
    #   - variable_id     : the injection source (generator id is used as proxy for its bus)
    #   - matrix_id       : label for this sensitivity matrix
    sa.add_branch_flow_factor_matrix(
        branches_ids    = ["L12"],
        variables_ids   = ["G1",  "G2"],   # generators as injection points
        matrix_id       = "PTDF",
    )

    if change_slack:
        lf_params = lf.Parameters(
            distributed_slack=False,
        )
    else:
        lf_params = lf.Parameters(
        )
    result = sa.run(network, lf_params)
    df: pd.DataFrame = result.get_sensitivity_matrix("PTDF")
    # df rows = branches (L12), columns = variables (G1, G2)
    # Values are ∂P_L12 / ∂P_Gi  (in MW per MW injected)

    print(df)  # ← paste the output here so we can see the exact shape
    print(df.index)
    print(df.columns)

    print("\n── PTDF matrix (pypowsybl sensitivity) ────────────────────────")
    print(df.to_string())
    print()

    ptdf = {
        "G1": float(df.loc["G1", "L12"]),
        "G2": float(df.loc["G2", "L12"]),
    }
    return ptdf


def compute_ptdf_analytical(x_pu: float = 0.1, base_mva: float = 100.0) -> dict[str, float]:
    """
    Analytical PTDF for a two-bus, one-line network (for verification).

    With Bus 1 as slack and a single line of susceptance b = 1/x:
        PTDF(L12, B1) =  0   (slack bus)
        PTDF(L12, B2) = -1   (all injection at B2 returns via L12)

    The sign follows the "from-to" convention of L12 (B1 → B2).
    """
    # B_red = b12 (scalar), A_red = -1 (to-bus of L12 is B2)
    b = 1.0 / x_pu          # pu susceptance
    ptdf_B2 = (-1.0) * (1.0 / b) * b    # A_red · B_red⁻¹  = -1
    return {"G1": 0.0, "G2": ptdf_B2}

# ── 4. DC-OPF with PTDF Flow Constraints ─────────────────────────────────────

def solve_dc_opf_ptdf(params: dict) -> dict:
    """
    PTDF-based DC-OPF.

    Key difference vs theta formulation
    ────────────────────────────────────
    REMOVED  : theta_B1, theta_B2  (no angle variables)
    REMOVED  : line_flow_def constraint  P12 = b·(θ1-θ2)
    REMOVED  : reference bus constraint  θ1 = 0

    ADDED    : ptdf_flow constraint
                P12 = Σ_{k≠slack} PTDF(L12,k) · Pnet_k
                P12 = PTDF_G2 · (Pg2 - Pd_B2)

    Everything else (balance, bounds, objective) is identical.
    """
    Pg_min  = params["Pg_min"]
    Pg_max  = params["Pg_max"]
    Pd_bus  = params["Pd_bus"]
    cost    = params["cost"]
    P12_max = params["P12_max"]
    ptdf    = params["ptdf"]

    model = xpress.Model()
    model.set_model_attribute(poi.ModelAttribute.Silent, True)

    # ── decision variables (no theta!) ────────────────────────────────────────
    Pg = {
        g: model.add_variable(lb=Pg_min[g], ub=Pg_max[g], name=f"Pg_{g}")
        for g in ["G1", "G2"]
    }
    P12 = model.add_variable(lb=-P12_max, ub=P12_max, name="P12")

    # ── PTDF flow constraint ──────────────────────────────────────────────────
    # P12 = Σ PTDF(L12,k) · Pnet_k   for non-slack buses
    #
    # Only G2 (Bus 2) is non-slack.  G1 is at the slack bus → PTDF ≈ 0.
    #
    # Expand:  P12 = ptdf_G2 · (Pg2 - Pd_B2)
    #    →     P12 - ptdf_G2·Pg2 = -ptdf_G2·Pd_B2
    #
    # With ptdf_G2 = -1:
    #    P12 + Pg2 = Pd_B2 = 250  →  P12 = 250 - Pg2

    ptdf_G1 = ptdf["G1"]
    ptdf_G2 = ptdf["G2"]
    rhs_ptdf = ptdf_G1 * (-Pd_bus["B1"]) + ptdf_G2 * (-Pd_bus["B2"])

    con_ptdf_flow_L12 = model.add_linear_constraint(
        P12 - ptdf_G1 * Pg["G1"] - ptdf_G2 * Pg["G2"],
        poi.Eq, rhs_ptdf,
        name="ptdf_flow_L12"
    )

    # ── nodal power balance (identical to theta version) ──────────────────────
    con_balance_B1 = model.add_linear_constraint(
        Pg["G1"] - P12,
        poi.Eq, Pd_bus["B1"],
        name="balance_B1"
    )
    con_balance_B2 = model.add_linear_constraint(
        Pg["G2"] + P12,
        poi.Eq, Pd_bus["B2"],
        name="balance_B2"
    )

    # ── objective ─────────────────────────────────────────────────────────────
    obj = poi.ExprBuilder()
    for g in ["G1", "G2"]:
        obj += cost[g] * Pg[g]
    model.set_objective(obj, poi.ObjectiveSense.Minimize)

    # ── solve ─────────────────────────────────────────────────────────────────
    model.optimize()

    status = model.get_model_attribute(poi.ModelAttribute.TerminationStatus)
    if status != poi.TerminationStatusCode.OPTIMAL:
        raise RuntimeError(f"Solver did not reach optimality: {status}")

    def val(v):
        return model.get_value(v)

    return {
        "status"      : str(status),
        "total_cost"  : model.get_model_attribute(poi.ModelAttribute.ObjectiveValue),
        "Pg"          : {g: val(Pg[g]) for g in ["G1", "G2"]},
        "P12_MW"      : val(P12),
        "LMP"         : {
            "B1": model.get_constraint_attribute(con_balance_B1, poi.ConstraintAttribute.Dual),
            "B2": model.get_constraint_attribute(con_balance_B2, poi.ConstraintAttribute.Dual),
        },
        "shadow_ptdf" : model.get_constraint_attribute(con_ptdf_flow_L12, poi.ConstraintAttribute.Dual),
    }


# ── 5. Validate with pypowsybl DC Load-Flow ──────────────────────────────────

def apply_and_validate(network: pn.Network, results: dict) -> None:
    for g, pval in results["Pg"].items():
        network.update_generators(id=g, target_p=pval)

    lf = ppw.loadflow.run_dc(network)
    print("── pypowsybl DC load-flow validation ──────────────────────────")
    for comp in lf:
        print(f"  Component status : {comp.status}")
    lines = network.get_lines(all_attributes=True)
    print(f"  L12 flow (from B1) : {lines.loc['L12', 'p1']:.2f} MW")
    gens = network.get_generators(all_attributes=True)
    for g in gens.index:
        print(f"  {g} output : {gens.loc[g, 'p']:.2f} MW")


# ── 6. Side-by-Side Comparison ────────────────────────────────────────────────

def print_formulation_comparison():
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║          Theta formulation  vs  PTDF formulation                     ║
╠══════════════════════════════════════════╦═══════════════════════════╣
║  THETA                                   ║  PTDF                     ║
╠══════════════════════════════════════════╬═══════════════════════════╣
║  Variables: Pg, θ₁, θ₂, P₁₂             ║  Variables: Pg, P₁₂       ║
║  (4 variables for 2-bus system)          ║  (3 variables)            ║
╠══════════════════════════════════════════╬═══════════════════════════╣
║  Constraints:                            ║  Constraints:             ║
║    θ₁ = 0             (ref angle)        ║    P₁₂ = PTDF·Pnet       ║
║    P₁₂ = b·(θ₁−θ₂)   (flow def)         ║    balance B1             ║
║    balance B1                            ║    balance B2             ║
║    balance B2                            ║                           ║
╠══════════════════════════════════════════╬═══════════════════════════╣
║  Scales to N buses as:                   ║  Scales as:               ║
║    N angle vars + N-1 balance            ║    L flow vars +          ║
║    + L flow vars + L flow-def            ║    N-1 balance            ║
║    = O(N + L) vars, O(N + L) cons        ║    = O(L) cons, no angles ║
╠══════════════════════════════════════════╬═══════════════════════════╣
║  Dual of flow-def → line shadow price    ║  Dual of PTDF con →       ║
║                                          ║    congestion rent        ║
║  LMP = dual(balance)                     ║  LMP = dual(balance)      ║
╚══════════════════════════════════════════╩═══════════════════════════╝
""")


def print_results(params: dict, results: dict) -> None:
    sep = "─" * 56
    print(f"\n{'═'*56}")
    print(f"  DC OPF (PTDF) — Results")
    print(f"{'═'*56}")
    print(f"  Solver status   : {results['status']}")
    print(f"  Total cost      : ${results['total_cost']:,.2f}/h")
    print(sep)
    print(f"  {'Generator':<12}{'Pg (MW)':>10}  {'Min':>7}  {'Max':>7}  {'$/MWh':>7}")
    print(sep)
    for g in ["G1", "G2"]:
        pg = results["Pg"][g]
        mn, mx = params["Pg_min"][g], params["Pg_max"][g]
        flag = " ◄ LIMIT" if (abs(pg-mn)<0.01 or abs(pg-mx)<0.01) else ""
        print(f"  {g:<12}{pg:>10.2f}  {mn:>7.1f}  {mx:>7.1f}  {params['cost'][g]:>7.1f}{flag}")
    print(sep)
    print(f"  Line L12 flow   : {results['P12_MW']:>8.2f} MW  (limit ±{params['P12_max']} MW)")
    flag = " ◄ CONGESTED" if abs(abs(results['P12_MW']) - params['P12_max']) < 0.01 else ""
    print(f"  Congestion      :{flag if flag else ' none'}")
    print(sep)
    print(f"  PTDF used       : G1={params['ptdf']['G1']:.3f}, G2={params['ptdf']['G2']:.3f}")
    print(sep)
    print("  Locational Marginal Prices:")
    for b, lmp in results["LMP"].items():
        print(f"    LMP {b} : ${lmp:>8.4f}/MWh")
    print(f"  PTDF shadow price : {results['shadow_ptdf']:.4f}  (congestion rent if line binding)")
    print(f"{'═'*56}\n")

def sensitivity_analysis(params: dict) -> None:
    """
    Demonstrate constraint binding by sweeping the line thermal limit.
    Shows how tightening P12_max forces a more expensive dispatch.
    """
    print("── Constraint sensitivity: P12_max sweep ──────────────────────")
    print(f"  {'P12_max (MW)':>14}  {'Pg1 (MW)':>10}  {'Pg2 (MW)':>10}  {'Cost ($/h)':>12}  {'LMP_B1':>8}  {'LMP_B2':>8}  {'PTDF shadow':>8}")
    print("  " + "─" * 70)

    for limit in [200, 150, 120, 100, 80, 60]:
        p = {**params, "P12_max": limit}
        try:
            r = solve_dc_opf_ptdf(p)
            print(f"  {limit:>14}  {r['Pg']['G1']:>10.2f}  {r['Pg']['G2']:>10.2f}"
                  f"  {r['total_cost']:>12.2f}"
                  f"  {r['LMP']['B1']:>8.4f}  {r['LMP']['B2']:>8.4f}"
                  f"  {r['shadow_ptdf']:>8.4f}"
                  )
        except RuntimeError as e:
            print(f"  {limit:>14}  INFEASIBLE ({e})")
    print()

# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print_formulation_comparison()

    # 1. build network
    print("Building network …")
    network = pn.load("two_buses.xiidm")

    # 2. compute PTDF two ways and cross-check
    print("Computing PTDF via pypowsybl sensitivity analysis …")
    change_slack = True
    ptdf_pypow = compute_ptdf(network, change_slack)

    print("Computing PTDF analytically (verification) …")
    ptdf_ana = compute_ptdf_analytical(x_pu=0.1)

    if change_slack:
        print(f"  pypowsybl  →  G1: {ptdf_pypow['G1']:+.6f},  G2: {ptdf_pypow['G2']:+.6f}")
        print(f"  analytical →  G1: {ptdf_ana['G1']:+.6f},  G2: {ptdf_ana['G2']:+.6f}")
        assert abs(ptdf_pypow["G2"] - ptdf_ana["G2"]) < 1e-4, "PTDF mismatch!"
        print("  ✓ PTDFs match\n")

    # 3. extract parameters and solve
    params  = extract_parameters(network)
    params["ptdf"] = ptdf_pypow
    results = solve_dc_opf_ptdf(params)

    print_results(params, results)

    # 4. validate with pypowsybl DC load-flow
    apply_and_validate(network, results)

    sensitivity_analysis(params)
