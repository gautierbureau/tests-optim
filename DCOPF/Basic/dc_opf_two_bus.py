"""
DC Optimal Power Flow — Two-Bus Power System
============================================
Libraries : pypowsybl  (network modelling + DC load-flow validation)
            pyoptinterface (algebraic modelling + HiGHS solver)

Network topology
----------------
        Gen1 (slack)          Gen2
            |                   |
          Bus1 ---[Line12]--- Bus2
                                |
                              Load2

DC-OPF formulation
------------------
  min   c1·Pg1 + c2·Pg2          (linear cost)

  s.t.
  [Power balance — per bus]
    Bus1:  Pg1 - P12        = Pd1
    Bus2:  Pg2 + P12        = Pd2

  [DC line-flow model]
    P12 = b12 · (θ1 - θ2)        (susceptance × angle diff)

  [Reference bus]
    θ1 = 0

  [Generator limits]
    Pg_min_i ≤ Pg_i ≤ Pg_max_i   for i ∈ {1, 2}

  [Line thermal limit]
    -P12_max ≤ P12 ≤ P12_max
"""

# ── 0. Imports ────────────────────────────────────────────────────────────────
import math
import pypowsybl as ppw
import pypowsybl.network as pn
import pyoptinterface as poi
from pyoptinterface import highs          # HiGHS backend (free, open-source)
from pyoptinterface import xpress          # HiGHS backend (free, open-source)

def extract_parameters(network: pn.Network) -> dict:
    """
    Pull the numbers we need for the OPF from the pypowsybl network object.
    Returns a flat dict of scalars used in the optimisation.
    """
    base_mva = 100.0           # system base (MVA)

    # generators
    gens = network.get_generators()
    Pg_min = {g: gens.loc[g, "min_p"] for g in gens.index}
    Pg_max = {g: gens.loc[g, "max_p"] for g in gens.index}

    # loads
    loads = network.get_loads()
    Pd = {l: loads.loc[l, "p0"] for l in loads.index}

    # line susceptance & thermal limit
    lines = network.get_lines()
    x_pu   = lines.loc["L12", "x"]           # pu
    b_pu   = 1.0 / x_pu                      # pu susceptance
    b_mw   = b_pu * base_mva                 # MW/rad

    P12_max = 120.0   # MW  (thermal limit — set by hand here)

    # linear cost coefficients  $/MWh  (or $/MW in a single-period OPF)
    cost = {"G1": 30.0, "G2": 45.0}

    # loads per bus  (MW)
    Pd_bus = {"B1": 0.0, "B2": Pd["D2"]}

    return dict(
        Pg_min=Pg_min,  Pg_max=Pg_max,
        b_mw=b_mw,      P12_max=P12_max,
        cost=cost,      Pd_bus=Pd_bus,
        base_mva=base_mva,
    )


# ── 3. Build & Solve the DC-OPF with pyoptinterface ───────────────────────────

def solve_dc_opf(params: dict) -> dict:
    """
    Formulate and solve the DC-OPF using pyoptinterface + HiGHS.
    Returns a dict with optimal values and shadow prices.
    """
    Pg_min  = params["Pg_min"]
    Pg_max  = params["Pg_max"]
    b_mw    = params["b_mw"]
    P12_max = params["P12_max"]
    cost    = params["cost"]
    Pd_bus  = params["Pd_bus"]

    # ── create model ──────────────────────────────────────────────────────────
    # model = highs.Model()
    model = xpress.Model()
    model.set_model_attribute(poi.ModelAttribute.Silent, True)

    # ── decision variables ────────────────────────────────────────────────────
    # Generator outputs  (MW)
    Pg = {
        g: model.add_variable(lb=Pg_min[g], ub=Pg_max[g], name=f"Pg_{g}")
        for g in ["G1", "G2"]
    }

    # Voltage angles  (radians) — Bus 1 is reference (fixed to 0)
    theta = {
        "B1": model.add_variable(lb=0.0,   ub=0.0,  name="theta_B1"),   # ref
        "B2": model.add_variable(lb=-math.pi, ub=math.pi, name="theta_B2"),
    }

    # Line flow  P12 = b · (θ1 − θ2)  —  auxiliary variable for readability
    P12 = model.add_variable(lb=-P12_max, ub=P12_max, name="P12")

    # ── constraints ───────────────────────────────────────────────────────────

    # (C1) DC line-flow definition:  P12 - b·(θ1 - θ2) = 0
    con_line_flow_def = model.add_linear_constraint(
        P12 - b_mw * theta["B1"] + b_mw * theta["B2"],
        poi.Eq, 0.0,
        name="line_flow_def"
    )

    # (C2) Power balance — Bus 1:  Pg1 - P12 = Pd_B1
    con_balance_B1 = model.add_linear_constraint(
        Pg["G1"] - P12,
        poi.Eq, Pd_bus["B1"],
        name="balance_B1"
    )

    # (C3) Power balance — Bus 2:  Pg2 + P12 = Pd_B2
    con_balance_B2 = model.add_linear_constraint(
        Pg["G2"] + P12,
        poi.Eq, Pd_bus["B2"],
        name="balance_B2"
    )

    # ── objective — minimise total generation cost ─────────────────────────────
    obj = poi.ExprBuilder()
    for g in ["G1", "G2"]:
        obj += cost[g] * Pg[g]
    model.set_objective(obj, poi.ObjectiveSense.Minimize)

    # ── solve ─────────────────────────────────────────────────────────────────
    model.optimize()

    status = model.get_model_attribute(poi.ModelAttribute.TerminationStatus)
    if status != poi.TerminationStatusCode.OPTIMAL:
        raise RuntimeError(f"Solver did not find an optimal solution: {status}")

    # ── extract results ───────────────────────────────────────────────────────
    def val(v):
        return model.get_value(v)

    results = {
        "status"     : str(status),
        "total_cost" : model.get_model_attribute(poi.ModelAttribute.ObjectiveValue),
        "Pg"         : {g: val(Pg[g])    for g in ["G1", "G2"]},
        "theta_deg"  : {b: math.degrees(val(theta[b])) for b in ["B1", "B2"]},
        "P12_MW"     : val(P12),
        # Shadow prices (locational marginal prices) from dual variables
        "LMP"        : {
            "B1": model.get_constraint_attribute(con_balance_B1, poi.ConstraintAttribute.Dual),
            "B2": model.get_constraint_attribute(con_balance_B2, poi.ConstraintAttribute.Dual),
        },
        "shadow_line": model.get_constraint_attribute(con_line_flow_def, poi.ConstraintAttribute.Dual),
    }
    return results


# ── 4. Update pypowsybl Network with Optimal Dispatch ────────────────────────

def apply_dispatch(network: pn.Network, results: dict) -> None:
    """Write optimal Pg back into the pypowsybl network object."""
    for g, pval in results["Pg"].items():
        network.update_generators(id=g, target_p=pval)


# ── 5. Validate with pypowsybl DC Load-Flow ──────────────────────────────────

def validate_with_dc_lf(network: pn.Network) -> None:
    """Run pypowsybl's built-in DC load-flow and print line flows."""
    result = ppw.loadflow.run_dc(network)
    print("\n── pypowsybl DC load-flow validation ──────────────────────────")
    for comp in result:
        print(f"  Component status : {comp.status}")
    lines = network.get_lines(all_attributes=True)
    print(f"  L12 active flow (from bus1 side) : {lines.loc['L12', 'p1']:.2f} MW")
    gens = network.get_generators(all_attributes=True)
    for g in gens.index:
        print(f"  {g} output : {gens.loc[g, 'p']:.2f} MW")


# ── 6. Pretty-Print Results ───────────────────────────────────────────────────

def print_results(params: dict, results: dict) -> None:
    sep = "─" * 55
    print(f"\n{'═'*55}")
    print(f"  DC OPF — Two-Bus System Results")
    print(f"{'═'*55}")
    print(f"  Solver status   : {results['status']}")
    print(f"  Total cost      : ${results['total_cost']:,.2f}/h")
    print(sep)
    print(f"  {'Generator':<12} {'Pg (MW)':>10}  {'Min':>8}  {'Max':>8}  {'Cost $/MWh':>10}")
    print(sep)
    for g in ["G1", "G2"]:
        pg  = results["Pg"][g]
        mn  = params["Pg_min"][g]
        mx  = params["Pg_max"][g]
        c   = params["cost"][g]
        flag = " ◄ AT LIMIT" if (abs(pg - mn) < 0.01 or abs(pg - mx) < 0.01) else ""
        print(f"  {g:<12} {pg:>10.2f}  {mn:>8.1f}  {mx:>8.1f}  {c:>10.1f}{flag}")
    print(sep)
    print(f"  Line flow  L12  : {results['P12_MW']:>8.2f} MW"
          f"  (limit ±{params['P12_max']} MW)")
    print(sep)
    print(f"  Angle Bus 1     : {results['theta_deg']['B1']:>8.3f}°  (reference)")
    print(f"  Angle Bus 2     : {results['theta_deg']['B2']:>8.3f}°")
    print(sep)
    print(f"  Locational Marginal Prices (shadow prices on balance constraints):")
    for b, lmp in results["LMP"].items():
        print(f"    LMP {b} : ${lmp:>8.4f}/MWh")
    print(f"  Line shadow price : {results['shadow_line']:.4f}")
    print(f"{'═'*55}\n")


# ── 7. Sensitivity / Constraint Analysis ─────────────────────────────────────

def sensitivity_analysis(params: dict) -> None:
    """
    Demonstrate constraint binding by sweeping the line thermal limit.
    Shows how tightening P12_max forces a more expensive dispatch.
    """
    print("── Constraint sensitivity: P12_max sweep ──────────────────────")
    print(f"  {'P12_max (MW)':>14}  {'Pg1 (MW)':>10}  {'Pg2 (MW)':>10}  {'Cost ($/h)':>12}  {'LMP_B1':>8}  {'LMP_B2':>8} {'PTDF shadow'}")
    print("  " + "─" * 70)

    for limit in [200, 150, 120, 100, 80, 60]:
        p = {**params, "P12_max": limit}
        try:
            r = solve_dc_opf(p)
            print(f"  {limit:>14}  {r['Pg']['G1']:>10.2f}  {r['Pg']['G2']:>10.2f}"
                  f"  {r['total_cost']:>12.2f}"
                  f"  {r['LMP']['B1']:>8.4f}  {r['LMP']['B2']:>8.4f}"
                  f"  {r['shadow_line']:>8.4f}"
                  )
        except RuntimeError as e:
            print(f"  {limit:>14}  INFEASIBLE ({e})")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Building two-bus network with pypowsybl …")
    network = pn.load("two_buses.xiidm")

    print("Extracting network parameters …")
    params = extract_parameters(network)

    print(f"\nNetwork data summary")
    print(f"  Load at Bus 2    : {params['Pd_bus']['B2']} MW")
    print(f"  Line susceptance : {params['b_mw']:.1f} MW/rad")
    print(f"  Line limit       : ±{params['P12_max']} MW")

    print("\nSolving DC-OPF with pyoptinterface …")
    results = solve_dc_opf(params)

    print_results(params, results)

    # write optimal dispatch back to pypowsybl network
    apply_dispatch(network, results)
    validate_with_dc_lf(network)

    # sweep line limits to show constraint impact
    sensitivity_analysis(params)
