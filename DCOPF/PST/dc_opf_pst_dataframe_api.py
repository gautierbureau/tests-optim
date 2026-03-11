"""
PST creation with pypowsybl DataFrame API
==========================================

Demonstrates create_phase_tap_changers with explicit step definitions.

Each step carries:
  alpha  : phase shift angle in degrees (the key parameter for DC OPF)
  rho    : turns ratio (1.0 for ideal PST — no voltage magnitude change)
  r, x   : additional series resistance/reactance per step (usually 0)
  g, b   : additional shunt conductance/susceptance per step (usually 0)

Topology (bypass PST — from dc_opf_pst_focus.py)
--------------------------------------------------
         ┌──── L12a  (B1→B2b, bypass) ─────────────────┐
G1 ── B1─┤                                              ├── B2b ──L2b_3── B3 ── G2/Load
         └──── L1_2a (B1→B2a) ── B2a ──[PST_T φ]──────┘
                                  └── S2: VL2a→VL2b ───┘
"""

import math
import pandas as pd
import numpy as np
import pypowsybl.network as pn
import pypowsybl.loadflow as plf
import pypowsybl.sensitivity as psa
import pypowsybl as ppw
import pyoptinterface as poi
from pyoptinterface import xpress
from dc_opf_pst_focus import build_network
from common_pst import build_pst_dataframes

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

BASE_MVA    = 100.0
KV          = 400.0
Z_BASE      = KV**2 / BASE_MVA     # 1600 Ω

B_L12A      = BASE_MVA / 0.2       # 500  MW/rad
B_L1_2A     = BASE_MVA / 0.4       # 250  MW/rad
B_PST       = BASE_MVA / 0.1       # 1000 MW/rad
B_L2B_3     = BASE_MVA / 0.2       # 500  MW/rad

PMAX_L12A   = 130.0
PMAX_L1_2A  = 150.0
PMAX_PST    = 150.0
PMAX_L2B_3  = 200.0

# PST tap changer definition
# 21 steps: tap 0 = −30°, tap 10 = 0° (neutral), tap 20 = +30°
N_STEPS     = 21
NEUTRAL_TAP = 10
ALPHA_STEP  = 3.0   # degrees per tap step
PHI_MAX_DEG = ALPHA_STEP * NEUTRAL_TAP    # 30°
PHI_MAX_RAD = math.radians(PHI_MAX_DEG)

C_PST       = 5.0   # $/rad wear cost

BRANCHES    = ["L12a", "L1_2a", "PST_T", "L2b_3"]


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Build pypowsybl network with DataFrame PST API
# ─────────────────────────────────────────────────────────────────────────────

def inspect_pst(network: pn.Network) -> None:
    """Print the phase tap changer and its steps to verify the DataFrame API."""
    ptc   = network.get_phase_tap_changers()
    steps = network.get_phase_tap_changer_steps()

    print("── Phase tap changer (ptc) ─────────────────────────────────────")
    print(ptc.to_string())
    print()
    print("── Tap steps ───────────────────────────────────────────────────")
    print(steps.to_string())
    print()


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Compute PTDF and PSDF
# ─────────────────────────────────────────────────────────────────────────────

def compute_sensitivities(network: pn.Network) -> dict:
    """PTDF and PSDF via pypowsybl sensitivity analysis."""
    sa = psa.create_dc_analysis()
    sa.add_branch_flow_factor_matrix(
        branches_ids  = BRANCHES,
        variables_ids = ["G2"],
        matrix_id     = "PTDF",
    )
    sa.add_branch_flow_factor_matrix(
        branches_ids  = BRANCHES,
        variables_ids = ["PST_T"],
        matrix_id     = "PSDF",
    )

    lf_params = plf.Parameters(distributed_slack=False)
    result    = sa.run(network, parameters=lf_params)

    df_ptdf = result.get_sensitivity_matrix("PTDF")
    df_psdf = result.get_sensitivity_matrix("PSDF")

    ptdf = {br: float(df_ptdf.loc["G2",   br]) for br in BRANCHES}
    psdf = {br: float(df_psdf.loc["PST_T", br]) for br in BRANCHES}

    print("── Sensitivities ────────────────────────────────────────────────")
    print(f"  {'Branch':<10}  {'PTDF_G2':>10}  {'PSDF (MW/rad)':>14}")
    print("  " + "─" * 40)
    for br in BRANCHES:
        print(f"  {br:<10}  {ptdf[br]:>+10.6f}  {psdf[br]:>+14.4f}")
    print()

    return {"ptdf": ptdf, "psdf": psdf}


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Parameters
# ─────────────────────────────────────────────────────────────────────────────

def extract_parameters(network: pn.Network) -> dict:
    gens  = network.get_generators()
    loads = network.get_loads()
    return dict(
        Pg_min = {g: float(gens.loc[g,"min_p"]) for g in gens.index},
        Pg_max = {g: float(gens.loc[g,"max_p"]) for g in gens.index},
        Pd_B3  = float(loads.loc["D3","p0"]),
        cost   = {"G1": 30.0, "G2": 45.0},
    )


# ─────────────────────────────────────────────────────────────────────────────
# 4.  LP solver — PTDF formulation
# ─────────────────────────────────────────────────────────────────────────────

def solve_lp(params: dict,
             sens: dict,
             pmax_l12a: float = PMAX_L12A,
             phi_fixed: float = None) -> dict:
    """
    PTDF DC-OPF.

    PTDF flow constraint per branch l:
      P_l = PTDF(l,G2)·(Pg_G2 − Pd_B3) + PSDF(l)·φ
      → P_l − PTDF(l,G2)·Pg_G2 − PSDF(l)·φ = PTDF(l,G2)·(−Pd_B3)
    """
    Pg_min = params["Pg_min"]
    Pg_max = params["Pg_max"]
    Pd_B3  = params["Pd_B3"]
    cost   = params["cost"]
    ptdf   = sens["ptdf"]
    psdf   = sens["psdf"]

    pmax = {
        "L12a" : pmax_l12a,
        "L1_2a": PMAX_L1_2A,
        "PST_T": PMAX_PST,
        "L2b_3": PMAX_L2B_3,
    }

    model = xpress.Model()
    model.set_model_attribute(poi.ModelAttribute.Silent, True)
    obj = poi.ExprBuilder()

    Pg = {g: model.add_variable(lb=Pg_min[g], ub=Pg_max[g], name=f"Pg_{g}")
          for g in ["G1","G2"]}

    P = {br: model.add_variable(lb=-pmax[br], ub=pmax[br], name=f"P_{br}")
         for br in BRANCHES}

    if phi_fixed is not None:
        phi = model.add_variable(lb=phi_fixed, ub=phi_fixed, name="phi")
        psi = model.add_variable(lb=abs(phi_fixed), ub=abs(phi_fixed), name="psi")
    else:
        phi = model.add_variable(lb=-PHI_MAX_RAD, ub=PHI_MAX_RAD, name="phi")
        psi = model.add_variable(lb=0.0, ub=PHI_MAX_RAD, name="psi")

    # PTDF constraints
    cons_ptdf = {}
    for br in BRANCHES:
        rhs = ptdf[br] * (-Pd_B3)
        cons_ptdf[br] = model.add_linear_constraint(
            P[br] - ptdf[br] * Pg["G2"] - psdf[br] * phi,
            poi.Eq, rhs, name=f"ptdf_{br}",
        )

    # Nodal balances
    con_B1 = model.add_linear_constraint(
        Pg["G1"] - P["L12a"] - P["L1_2a"],
        poi.Eq, 0.0, name="bal_B1",
    )
    con_B3 = model.add_linear_constraint(
        Pg["G2"] + P["L2b_3"],
        poi.Eq, Pd_B3, name="bal_B3",
    )

    model.add_linear_constraint(psi - phi, poi.Geq, 0.0, name="psi_pos")
    model.add_linear_constraint(psi + phi, poi.Geq, 0.0, name="psi_neg")

    for g in ["G1","G2"]:
        obj += cost[g] * Pg[g]
    obj += C_PST * psi
    model.set_objective(obj, poi.ObjectiveSense.Minimize)

    model.optimize()
    status = model.get_model_attribute(poi.ModelAttribute.TerminationStatus)
    if status != poi.TerminationStatusCode.OPTIMAL:
        raise RuntimeError(f"LP not optimal: {status}")

    def val(v):  return model.get_value(v)
    def dual(c): return model.get_constraint_attribute(
        c, poi.ConstraintAttribute.Dual)

    phi_val = val(phi)
    return {
        "status"     : str(status),
        "total_cost" : model.get_model_attribute(poi.ModelAttribute.ObjectiveValue),
        "Pg"         : {g: val(Pg[g]) for g in ["G1","G2"]},
        "P"          : {br: val(P[br]) for br in BRANCHES},
        "phi_deg"    : math.degrees(phi_val),
        "phi_rad"    : phi_val,
        "psi_rad"    : val(psi),
        "LMP"        : {"B1": dual(con_B1), "B3": dual(con_B3)},
        "pmax"       : pmax,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Validate: map optimal φ to nearest tap, run DC LF
# ─────────────────────────────────────────────────────────────────────────────

def phi_to_tap(phi_deg: float) -> tuple[int, float]:
    """
    Map a continuous OPF angle to the nearest available tap step.
    Returns (tap_index, actual_alpha_at_tap).
    """
    # tap_idx = round((phi_deg - alpha_at_tap0) / ALPHA_STEP)
    #         = round((phi_deg - (0 - NEUTRAL_TAP)*ALPHA_STEP) / ALPHA_STEP)
    #         = round(phi_deg/ALPHA_STEP + NEUTRAL_TAP)
    tap = int(round(phi_deg / ALPHA_STEP + NEUTRAL_TAP))
    tap = max(0, min(N_STEPS - 1, tap))
    alpha_at_tap = (tap - NEUTRAL_TAP) * ALPHA_STEP
    return tap, alpha_at_tap


def validate(results: dict) -> None:
    net = build_network()

    for g, pval in results["Pg"].items():
        net.update_generators(id=g, target_p=pval)

    tap, alpha_at_tap = phi_to_tap(results["phi_deg"])
    net.update_phase_tap_changers(id="PST_T", tap=tap)

    lf_params = plf.Parameters(distributed_slack=False)
    lf = ppw.loadflow.run_dc(net, parameters=lf_params)

    lines  = net.get_lines(all_attributes=True)
    trafos = net.get_2_windings_transformers(all_attributes=True)
    gens   = net.get_generators(all_attributes=True)

    print("\n── pypowsybl DC LF validation ───────────────────────────────────")
    for c in lf:
        print(f"  Component {c.connected_component_num}: {c.status}")
    print(f"  OPF φ_opt={results['phi_deg']:+.2f}° → tap {tap} "
          f"→ α={alpha_at_tap:+.1f}°  "
          f"(discretisation error: {results['phi_deg']-alpha_at_tap:+.2f}°)")
    print()
    print(f"  {'Branch':<10}  {'OPF (MW)':>10}  {'LF (MW)':>10}  {'Δ':>7}")
    print("  " + "─" * 45)
    lf_flows = {
        "L12a" : lines.loc["L12a",  "p1"],
        "L1_2a": lines.loc["L1_2a", "p1"],
        "PST_T": trafos.loc["PST_T","p1"],
        "L2b_3": lines.loc["L2b_3", "p1"],
    }
    for br in BRANCHES:
        opf  = results["P"][br]
        lf_v = lf_flows[br]
        print(f"  {br:<10}  {opf:>+10.2f}  {lf_v:>+10.2f}  {opf-lf_v:>+7.3f}")

    # Also print tap changer state
    ptc = net.get_phase_tap_changers()
    print(f"\n  PTC state after update: tap={int(ptc.loc['PST_T','tap'])}  "
          f"regulating={ptc.loc['PST_T','regulating']}")


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Pretty printer
# ─────────────────────────────────────────────────────────────────────────────

def print_results(r: dict, title: str = "") -> None:
    sep = "─" * 64
    print(f"\n{'═'*64}")
    if title:
        print(f"  {title}")
        print(f"{'═'*64}")

    tap, alpha = phi_to_tap(r["phi_deg"])
    gen_cost   = sum(r["Pg"][g]*c for g,c in zip(["G1","G2"],[30,45]))
    pst_cost   = C_PST * r["psi_rad"]

    print(f"  Cost    : ${r['total_cost']:>10,.2f}/h  "
          f"(gen=${gen_cost:,.0f}  PST=${pst_cost:.2f})")
    print(sep)
    for g, mn, mx, c in [("G1",0,300,30),("G2",0,150,45)]:
        pg  = r["Pg"][g]
        lim = " ◄" if (abs(pg-mn)<0.5 or abs(pg-mx)<0.5) else ""
        print(f"  {g}: {pg:>7.2f} MW  [{mn}..{mx}]  ${c}/MWh{lim}")
    print(sep)
    print(f"  PST φ = {r['phi_deg']:>+6.2f}°  →  tap {tap:>2d}  "
          f"(α={alpha:>+5.1f}°, limit ±{PHI_MAX_DEG:.0f}°)")
    print(sep)
    print(f"  {'Branch':<10}  {'Flow':>9}  {'Pmax':>6}  {'Load%':>6}")
    print(sep)
    for br in BRANCHES:
        flow = r["P"][br]
        pm   = r["pmax"][br]
        pct  = 100*abs(flow)/pm
        flag = " ◄" if pct > 99.0 else ""
        print(f"  {br:<10}  {flow:>+9.2f}  {pm:>6.0f}  {pct:>5.1f}%{flag}")
    print(sep)
    print(f"  LMP B1=${r['LMP']['B1']:>8.4f}/MWh   B3=${r['LMP']['B3']:>8.4f}/MWh")
    print(f"{'═'*64}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # ── Print PST DataFrame structure ─────────────────────────────────────────
    ptc_df, steps_df = build_pst_dataframes()

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  PST DataFrame API                                           ║")
    print("╠══════════════════════════════════════════════════════════════╣")
    print(f"║  {N_STEPS} steps: α from {(0-NEUTRAL_TAP)*ALPHA_STEP:+.0f}° "
          f"to {(N_STEPS-1-NEUTRAL_TAP)*ALPHA_STEP:+.0f}°  "
          f"({ALPHA_STEP:.0f}° per step)          ║")
    print(f"║  Neutral tap: {NEUTRAL_TAP} (α=0°)                               ║")
    print("╚══════════════════════════════════════════════════════════════╝\n")

    print("── ptc_df (tap changer definition) ─────────────────────────────")
    print(ptc_df.to_string())
    print()

    print("── steps_df (first 5 and last 5 steps shown) ───────────────────")
    print(pd.concat([steps_df.head(5), steps_df.tail(5)]).to_string())
    print(f"  ... ({N_STEPS} steps total)")
    print()

    # ── Build network and inspect ─────────────────────────────────────────────
    net = build_network()
    inspect_pst(net)

    # ── Sensitivities and OPF ─────────────────────────────────────────────────
    sens   = compute_sensitivities(net)
    params = extract_parameters(net)

    r1 = solve_lp(params, sens, pmax_l12a=250.0)
    print_results(r1, "Case 1 — No congestion")
    validate(r1)

    r2 = solve_lp(params, sens, pmax_l12a=110.0)
    print_results(r2, "Case 2 — L12a tight (110 MW): PST activated")
    validate(r2)

    r3 = solve_lp(params, sens, pmax_l12a=65.0)
    print_results(r3, "Case 3 — L12a very tight (65 MW): PST near limit")
    validate(r3)
