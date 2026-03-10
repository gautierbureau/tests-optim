"""
DC OPF with Automaton Line-Tripping — Big-M MILP
=================================================
Implements the mathematical formulation from automaton_bigM_formulation.md.

Key additions over the PTDF LP (dc_opf_ptdf.py):
  - P̃ₗ  : natural (unconstrained) flow variable
  - zₗ   : binary line status  (1=connected, 0=tripped)
  - βₗ   : binary overload direction (1=positive, 0=negative)
  - §5.3  P_l = P̃_l when connected   (Big-M link)
  - §5.4  P_l = 0   when tripped      (Big-M zero-flow)
  - A1±   forward implication: overload forces trip
  - A2±   backward implication: trip only if overload
  - A2    coupling: βₗ ≤ 1 − zₗ
"""

import math
import pypowsybl.network as pn
import pypowsybl.sensitivity as psa
import pypowsybl as ppw
import pandas as pd
import pyoptinterface as poi
from pyoptinterface import xpress
from dc_opf_ptdf import compute_ptdf
from dc_opf_two_bus import extract_parameters

# ─────────────────────────────────────────────────────────────────────────────
# 4. MILP with automaton Big-M constraints
# ─────────────────────────────────────────────────────────────────────────────

def solve_automaton_milp(params: dict, load_shedding: bool, curtailement: bool) -> dict:
    """
    Solve the DC-OPF + automaton trip logic as a MILP via HiGHS.

    Variables
    ---------
    Pg[g]      : generator output           (continuous, bounded)
    P12        : actual line flow            (continuous, free)
    P12_nat    : natural (PTDF) line flow    (continuous, free)
    z_L12      : line status binary          (1=on, 0=tripped)
    beta_L12   : overload direction binary   (1=positive OL, 0=negative OL)

    Automaton constraints (see formulation doc §5.3–5.5)
    ----------------------------------------------------
    §5.3  P12 - M(1-z) ≤ P12_nat ≤ P12 + M(1-z)     [flow=natural when on]
    §5.4  -Pmax·z ≤ P12 ≤ Pmax·z                     [flow=0 when tripped]
    A1+   P12_nat ≤  Pmax + M(1-z)                   [forward: +overload→trip]
    A1-   P12_nat ≥ -Pmax - M(1-z)                   [forward: -overload→trip]
    A2+   Pmax(1-z) - M·β ≤ P12_nat                  [backward: trip→+OL]
    A2-   P12_nat ≤ -Pmax(1-z) + M(1-β)              [backward: trip→-OL]
    A2c   β ≤ 1 - z                                   [coupling]
    """
    Pg_min  = params["Pg_min"]
    Pg_max  = params["Pg_max"]
    Pd_bus  = params["Pd_bus"]
    cost    = params["cost"]
    ptdf    = params["ptdf"]
    Pmax    = params["P12_max"]
    M       = params["M"]

    model = xpress.Model()
    #model.set_raw_control("PRESOLVE", 0)
    model.set_model_attribute(poi.ModelAttribute.Silent, True)

    obj = poi.ExprBuilder()

    # ── Continuous variables ──────────────────────────────────────────────────

    Pg = {
        g: model.add_variable(lb=Pg_min[g], ub=Pg_max[g], name=f"Pg_{g}")
        for g in ["G1", "G2"]
    }

    # Actual flow: bounded by ±Pmax only when connected (§5.4 handles this)
    P12 = model.add_variable(lb=-M, ub=M, name="P12")

    # Natural flow: no bounds here — the automaton constraints enforce limits
    P12_nat = model.add_variable(lb=-M, ub=M, name="P12_nat")

    # ── Binary variables ──────────────────────────────────────────────────────

    z    = model.add_variable(domain=poi.VariableDomain.Binary, name="z_L12")
    beta = model.add_variable(domain=poi.VariableDomain.Binary, name="beta_L12")

    # ── Natural flow definition (PTDF) ───────────────────────────────────────
    # P12_nat = PTDF_G2·(Pg_G2 - Pd_B2)   [PTDF_G1 = 0, slack bus]
    # Rearranged: P12_nat - PTDF_G2·Pg_G2 = -PTDF_G2·Pd_B2

    ptdf_G1 = ptdf["G1"]
    ptdf_G2 = ptdf["G2"]

    # ── Nodal power balance (uses ACTUAL flow P12) ───────────────────────────

    if load_shedding:
        shed = {
            b: model.add_variable(lb=0.0, ub=params["Pd_bus"][b], name=f"shed_{b}")
            for b in ["B1", "B2"]
        }

        if curtailement:
            curtail = {
                g: model.add_variable(lb=0.0, ub=Pg_max[g], name=f"curtail_{g}")
                for g in ["G1", "G2"]
            }
        else:
            curtail = {
                g: model.add_variable(lb=0.0, ub=0, name=f"curtail_{g}")
                for g in ["G1", "G2"]
            }

        rhs_nat = ptdf_G1 * (-Pd_bus["B1"]) + ptdf_G2 * (-Pd_bus["B2"])

        model.add_linear_constraint(
            P12_nat
            - ptdf_G1 * Pg["G1"] - ptdf_G2 * Pg["G2"],
            poi.Eq, rhs_nat,
            name="ptdf_natural_flow",
        )

        # Modified balance: generation - flow = load - shed
        model.add_linear_constraint(
            Pg["G1"] - curtail["G1"] - P12 + shed["B1"],
            poi.Eq, Pd_bus["B1"], name="balance_B1"
        )
        model.add_linear_constraint(
            Pg["G2"] - curtail["G2"] + P12 + shed["B2"],
            poi.Eq, Pd_bus["B2"], name="balance_B2"
        )

        SHED_COST = 10_000.0  # $/MWh — high penalty to only shed when unavoidable
        # Add shed cost to objective
        for b in ["B1", "B2"]:
            obj += SHED_COST * shed[b]

    else:
        rhs_nat = ptdf_G1 * (-Pd_bus["B1"]) + ptdf_G2 * (-Pd_bus["B2"])

        model.add_linear_constraint(
            P12_nat - ptdf_G1 * Pg["G1"] - ptdf_G2 * Pg["G2"],
            poi.Eq, rhs_nat,
            name="ptdf_natural_flow",
        )

        model.add_linear_constraint(
            Pg["G1"] - P12,
            poi.Eq, Pd_bus["B1"],
            name="balance_B1",
        )
        model.add_linear_constraint(
            Pg["G2"] + P12,
            poi.Eq, Pd_bus["B2"],
            name="balance_B2",
        )

    # ── §5.3  Actual flow = natural flow when connected ───────────────────────
    # P12_nat - M(1-z) ≤ P12 ≤ P12_nat + M(1-z)
    # Rearranged:
    #   P12 - P12_nat ≤  M(1-z)   →  P12 - P12_nat - M + M·z ≤ 0
    #   P12 - P12_nat ≥ -M(1-z)   →  P12 - P12_nat + M - M·z ≥ 0

    model.add_linear_constraint(
        P12 - P12_nat - M + M * z,
        poi.Leq, 0.0,
        name="s53_upper",
    )
    model.add_linear_constraint(
        P12 - P12_nat + M - M * z,
        poi.Geq, 0.0,
        name="s53_lower",
    )

    # ── §5.4  Zero flow when tripped ─────────────────────────────────────────
    # -Pmax·z ≤ P12 ≤ Pmax·z
    # Rearranged: P12 - Pmax·z ≤ 0  and  P12 + Pmax·z ≥ 0

    model.add_linear_constraint(
        P12 - Pmax * z,
        poi.Leq, 0.0,
        name="s54_upper",
    )
    model.add_linear_constraint(
        P12 + Pmax * z,
        poi.Geq, 0.0,
        name="s54_lower",
    )

    # ── A1+  Forward: positive overload forces trip ───────────────────────────
    # P12_nat ≤ Pmax + M(1-z)  →  P12_nat - M + M·z ≤ Pmax

    model.add_linear_constraint(
        P12_nat - M + M * z,
        poi.Leq, Pmax,
        name="A1_pos",
    )

    # ── A1-  Forward: negative overload forces trip ───────────────────────────
    # P12_nat ≥ -Pmax - M(1-z)  →  P12_nat + M - M·z ≥ -Pmax

    model.add_linear_constraint(
        P12_nat + M - M * z,
        poi.Geq, -Pmax,
        name="A1_neg",
    )

    # ── A2+  Backward: trip implies positive overload (if β=1) ───────────────
    # Pmax(1-z) - M·β ≤ P12_nat
    # Rearranged: Pmax - Pmax·z - M·β - P12_nat ≤ 0

    model.add_linear_constraint(
        Pmax - Pmax * z - M * beta - P12_nat,
        poi.Leq, 0.0,
        name="A2_pos",
    )

    # ── A2-  Backward: trip implies negative overload (if β=0) ───────────────
    # P12_nat ≤ -Pmax(1-z) + M(1-β)
    # Rearranged: P12_nat + Pmax - Pmax·z - M + M·β ≤ 0

    model.add_linear_constraint(
        P12_nat + Pmax - Pmax * z - M + M * beta,
        poi.Leq, 0.0,
        name="A2_neg",
    )

    # ── A2 coupling  β ≤ 1 - z  →  β + z ≤ 1 ───────────────────────────────

    model.add_linear_constraint(
        beta + z,
        poi.Leq, 1.0,
        name="A2_coupling",
    )

    # ── Objective ─────────────────────────────────────────────────────────────


    for g in ["G1", "G2"]:
        obj += cost[g] * Pg[g]

    model.set_objective(obj, poi.ObjectiveSense.Minimize)

    # ── Solve ─────────────────────────────────────────────────────────────────

    model.optimize()

    status = model.get_model_attribute(poi.ModelAttribute.TerminationStatus)
    if status != poi.TerminationStatusCode.OPTIMAL:
        return {
            "status": str(status),
            "total_cost": math.nan,
            "Pg": {g: 0 for g in ["G1", "G2"]},
            "P12": math.nan,
            "P12_nat": math.nan,
            "z_L12": math.nan,
            "beta_L12": math.nan,
            "M": M,
            "Pmax": Pmax,
        }

    def val(v):
        return model.get_value(v)

    res = {
        "status"     : str(status),
        "total_cost" : model.get_model_attribute(poi.ModelAttribute.ObjectiveValue),
        "Pg"         : {g: val(Pg[g]) for g in ["G1", "G2"]},
        "P12"        : val(P12),
        "P12_nat"    : val(P12_nat),
        "z_L12"      : int(round(val(z))),
        "beta_L12"   : int(round(val(beta))),
        "M"          : M,
        "Pmax"       : Pmax,
    }

    if load_shedding:
        res["shed_B1"] = val(shed["B1"])
        res["shed_B2"] = val(shed["B2"])

        if curtailement:
            res["curtail_G1"] = val(curtail["G1"])
            res["curtail_G2"] = val(curtail["G2"])

    return res


# ─────────────────────────────────────────────────────────────────────────────
# 5. Validate with pypowsybl DC load-flow
# ─────────────────────────────────────────────────────────────────────────────

def validate_with_dc_lf(network: pn.Network, results: dict) -> None:
    for g, pval in results["Pg"].items():
        network.update_generators(id=g, target_p=pval)

    # Disconnect line in pypowsybl if automaton tripped it
    if results["z_L12"] == 0:
        network.update_lines(id="L12", connected1=False, connected2=False)

    if "shed_B1" in results:
        if results["shed_B1"] > 0:
            network.create_loads(id='shed_B1', voltage_level_id='VL1', bus_id='B1', p0=-results["shed_B1"], q0=0)
    if "shed_B2" in results:
        if results["shed_B2"] > 0:
            network.create_loads(id='shed_B2', voltage_level_id='VL2', bus_id='B2', p0=-results["shed_B2"], q0=0)

    if "curtail_G1" in results:
        if results["curtail_G1"] > 0:
            network.create_loads(id='curtail_G1', voltage_level_id='VL1', bus_id='B1', p0=results["curtail_G1"], q0=0)
    if "curtail_G2" in results:
        if results["curtail_G2"] > 0:
            network.create_loads(id='curtail_G2', voltage_level_id='VL2', bus_id='B2', p0=results["curtail_G2"], q0=0)

    lf_params = ppw.loadflow.Parameters(
        component_mode=ppw.loadflow.ComponentMode.ALL_CONNECTED
    )
    lf = ppw.loadflow.run_dc(network, parameters=lf_params)
    print("\n── pypowsybl DC load-flow validation ──────────────────────────")
    for comp in lf:
        print(f"  Component {comp.connected_component_num}: {comp.status}")
    lines = network.get_lines(all_attributes=True)
    print(f"  L12 flow (p1) : {lines.loc['L12', 'p1']:.2f} MW")
    gens = network.get_generators(all_attributes=True)
    for g in gens.index:
        print(f"  {g} output : {gens.loc[g, 'p']:.2f} MW")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Results printer
# ─────────────────────────────────────────────────────────────────────────────

def print_results(params: dict, results: dict) -> None:
    Pmax     = results["Pmax"]
    P12_nat  = results["P12_nat"]
    P12      = results["P12"]
    z        = results["z_L12"]
    beta     = results["beta_L12"]

    line_status  = "CONNECTED" if z    == 1 else "TRIPPED ⚡"
    overload_dir = ("positive" if beta == 1 else "negative") if z == 0 else "n/a"
    overloaded = abs(P12_nat) > Pmax - 0.01

    sep = "─" * 58
    print(f"\n{'═'*58}")
    print(f"  DC OPF + Automaton (Big-M MILP) — Results")
    print(f"{'═'*58}")
    print(f"  Solver status  : {results['status']}")
    print(f"  Total cost     : ${results['total_cost']:,.2f} /h")
    print(sep)
    print(f"  {'Generator':<10} {'Pg (MW)':>10}  {'Min':>7}  {'Max':>7}  {'$/MWh':>7}")
    print(sep)
    for g in ["G1", "G2"]:
        pg = results["Pg"][g]
        mn, mx = params["Pg_min"][g], params["Pg_max"][g]
        flag = " ◄ LIMIT" if (abs(pg - mn) < 0.01 or abs(pg - mx) < 0.01) else ""
        print(f"  {g:<10} {pg:>10.2f}  {mn:>7.1f}  {mx:>7.1f}  "
              f"{params['cost'][g]:>7.1f}{flag}")
    if "shed_B1" in results:
        print(f"  {'B1':<10} {results['shed_B1']:>10.2f}  {'':>7}  {'':>7}  {'':>7}")
    if "shed_B2" in results:
        print(f"  {'B2':<10} {results['shed_B2']:>10.2f}  {'':>7}  {'':>7}  {'':>7}")
    if "curtail_G1" in results:
        print(f"  {'G1':<10} {results['curtail_G1']:>10.2f}  {'':>7}  {'':>7}  {'':>7}")
    if "curtail_G2" in results:
        print(f"  {'G2':<10} {results['curtail_G2']:>10.2f}  {'':>7}  {'':>7}  {'':>7}")
    print(sep)
    print(f"  Natural flow P̃₁₂  : {P12_nat:>8.2f} MW  (PTDF, no limits)")
    print(f"  Thermal limit Pmax : ±{Pmax:.1f} MW")
    print(f"  Overload?          : {'YES → ' + overload_dir if overloaded and z==0 else 'no'}")
    print(sep)
    print(f"  z_L12   (status)   : {z}  →  {line_status}")
    print(f"  β_L12   (dir)      : {beta}  →  overload direction = {overload_dir}")
    print(f"  Actual flow P₁₂    : {P12:>8.2f} MW")
    print(f"  Big-M used         : {results['M']:.1f} MW")
    print(f"{'═'*58}\n")

    # Logic verification against truth table (§9 of formulation doc)
    print("── Automaton logic verification ────────────────────────────────")
    checks = []
    if z == 1:
        ok = abs(P12_nat) <= Pmax + 0.01
        checks.append(("z=1 → |P̃| ≤ Pmax", ok))
        ok2 = abs(P12 - P12_nat) < 0.5
        checks.append(("z=1 → P12 = P̃12", ok2))
    else:
        ok  = abs(P12_nat) >= Pmax - 0.01
        checks.append(("z=0 → |P̃| ≥ Pmax", ok))
        ok2 = abs(P12) < 0.5
        checks.append(("z=0 → P12 = 0", ok2))
        if beta == 1:
            ok3 = P12_nat >= Pmax - 0.01
            checks.append(("β=1 → P̃ ≥ +Pmax", ok3))
        else:
            ok3 = P12_nat <= -Pmax + 0.01
            checks.append(("β=0 → P̃ ≤ −Pmax", ok3))
        ok4 = beta + z <= 1
        checks.append(("β + z ≤ 1", ok4))

    for label, ok in checks:
        print(f"  {'✓' if ok else '✗'}  {label}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# 7. Scenario sweep — vary P12_max to trigger the automaton
# ─────────────────────────────────────────────────────────────────────────────

def scenario_sweep(network: pn.Network, ptdf: dict) -> None:
    """
    Sweep thermal limit from 200 MW down to 60 MW.
    At each value the MILP decides whether to keep or trip the line.

    When the line is tripped (z=0), the load at Bus2 must be fully served
    by G2 (local generation), since the line carries nothing.
    Bus1 generation G1 is constrained to [Pmin=50, Pmax=200] MW — but with
    no line to export, it may hit its minimum.
    """
    print("── Scenario sweep: varying P12_max ─────────────────────────────")
    hdr = (f"  {'Pmax':>6}  {'z':>4}  {'β':>4}  {'P̃₁₂':>8}  "
           f"{'P₁₂':>8}  {'Pg1':>8}  {'Pg2':>8}  {'Cost':>10}  Status")
    print(hdr)
    print("  " + "─" * 74)

    for pmax in [200, 160, 130, 120, 100, 80, 60]:
        try:
            p   = extract_parameters(network)
            p["ptdf"] = ptdf
            p["P12_max"] = pmax
            p["M"] = sum(params1["Pg_max"].values())  # 200 + 150 = 350 M
            r   = solve_automaton_milp(p, False, False)
            sta = "CONNECTED" if r["z_L12"] == 1 else "TRIPPED ⚡"
            print(
                f"  {pmax:>6}  {r['z_L12']:>4}  {r['beta_L12']:>4}  "
                f"{r['P12_nat']:>8.2f}  {r['P12']:>8.2f}  "
                f"{r['Pg']['G1']:>8.2f}  {r['Pg']['G2']:>8.2f}  "
                f"{r['total_cost']:>10.2f}  {sta}"
            )
        except RuntimeError as e:
            print(f"  {pmax:>6}  {'':>4}  {'':>4}  {'':>8}  {'':>8}  "
                  f"{'':>8}  {'':>8}  {'':>10}  INFEASIBLE ({e})")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Building two-bus network …")
    network = pn.load("two_buses.xiidm")

    print("Computing PTDF …")
    change_slack = False
    ptdf = compute_ptdf(network, change_slack)
    print(f"  PTDF: G1={ptdf['G1']:+.4f}, G2={ptdf['G2']:+.4f}\n")

    # ── Case 1: Line within limit — automaton leaves it connected ─────────────
    print("═" * 58)
    print("  CASE 1 — Pmax = 130 MW  (natural flow ≈ 100 MW, no trip)")
    print("═" * 58)
    params1  = extract_parameters(network)
    params1["ptdf"] = ptdf
    params1["P12_max"] = 130
    params1["M"] = sum(params1["Pg_max"].values())  # 200 + 150 = 350 M
    results1 = solve_automaton_milp(params1, False, False)
    print_results(params1, results1)
    validate_with_dc_lf(pn.load("two_buses.xiidm"), results1)

    # ── Case 2: Line over limit — automaton trips it ──────────────────────────
    print("\n" + "═" * 58)
    print("  CASE 2 — Pmax = 80 MW  (natural flow ≈ 100 MW → TRIP)")
    print("═" * 58)
    params2  = extract_parameters(network)
    params2["ptdf"] = ptdf
    params2["P12_max"] = 80
    params2["M"] = sum(params2["Pg_max"].values())  # 200 + 150 = 350 M
    results2 = solve_automaton_milp(params2, False, False)
    print_results(params2, results2)
    validate_with_dc_lf(pn.load("two_buses.xiidm"), results2)

    # ── Sweep ─────────────────────────────────────────────────────────────────
    print("\n")
    scenario_sweep(network, ptdf)
