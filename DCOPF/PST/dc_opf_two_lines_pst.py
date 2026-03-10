"""
DC OPF — Two-Bus, Two Parallel Lines + PST + Automata
======================================================

Network topology
----------------
                    ┌─── L12a (plain) ───────────────┐
  G1 (slack) ── B1 ─┤                                 ├── B2 ── G2 / Load
                    └─── L12b ──[PST φ]───────────────┘

Parameters
----------
  Base : 100 MVA, 400 kV
  Both lines : x = 0.4 pu  →  b = 250 MW/rad each
  L12a : Pmax = 95 MW   (tighter limit — no PST)
  L12b : Pmax = 150 MW  (roomier  limit — has PST, φ ∈ [−30°, +30°])
  G1   : 0–200 MW @ 30 $/MWh  (cheap, slack bus)
  G2   : 0–150 MW @ 45 $/MWh  (expensive)
  Load : 250 MW at Bus 2

Key additions over dc_opf_automaton.py
---------------------------------------
  P12_nat_a, P12_nat_b   : two natural flow variables
  P12_a, P12_b           : two actual flow variables
  z_a, β_a, z_b, β_b    : two sets of automaton binaries
  φ  (phi)               : PST angle variable (radians, continuous)
  ψ  (psi)               : |φ| via linearisation (for wear cost in objective)
  PSDF                   : Phase Shift Distribution Factor (analytical)
  PST–automaton coupling : φ = 0 when z_b = 0  (line tripped)
  Lost load / curtailment: ensures feasibility after trips

Sensitivities (analytical — two identical parallel lines, PST on L12b)
-----------------------------------------------------------------------
  Let b = 250 MW/rad (each line), b_total = 500 MW/rad.

  PTDF (distributed_slack=False, G1=slack):
    PTDF_L12a_G2 = −b/(2b) = −0.5
    PTDF_L12b_G2 = −b/(2b) = −0.5
    PTDF_L12a_G1 = PTDF_L12b_G1 = 0  (slack bus)

  Natural flows (from PTDF derivation with G1 slack):
    P̃_a = 0.5·(Pd − P_G2) − PSDF_a·φ
    P̃_b = 0.5·(Pd − P_G2) + PSDF_b·φ

  PSDF (PST on L12b):
    Δθ_B2 = −b_b·φ / (b_a + b_b) = −φ/2
    ΔP_L12a = b_a·(−Δθ_B2)       = b/2·φ  →  φ>0 INCREASES P̃_a
    ΔP_L12b = b_b·(−Δθ_B2 + φ)   = b/2·φ  →  same direction?

  Wait — re-derive with sign:
    θ_B2 = (P_G2 − Pd + b_b·φ) / (b_a + b_b)
    P̃_a  = b_a·(0 − θ_B2) = −b_a·θ_B2
          = [b_a/(b_a+b_b)]·(Pd − P_G2) − [b_a·b_b/(b_a+b_b)]·φ
    P̃_b  = b_b·(0 − θ_B2 + φ)
          = [b_b/(b_a+b_b)]·(Pd − P_G2) + [b_a·b_b/(b_a+b_b)]·φ

  With b_a = b_b = b = 250:
    PSDF_L12a = −b²/(2b) = −b/2 = −125 MW/rad  (φ>0 → P̃_a decreases)
    PSDF_L12b = +b²/(2b) = +b/2 = +125 MW/rad  (φ>0 → P̃_b increases)

  So φ > 0 shifts flow from L12a to L12b.
  The PST protects the tighter line L12a by pushing excess flow to L12b.
"""

import math
import pandas as pd
import pypowsybl.network as pn
import pypowsybl.loadflow as plf
import pypowsybl as ppw
import pyoptinterface as poi
from pyoptinterface import xpress    # swap for highs if needed

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

BASE_MVA    = 100.0
BASE_KV     = 400.0
Z_BASE      = BASE_KV**2 / BASE_MVA   # 1600 Ω

LINE_X_PU   = 0.4                      # per-unit reactance for each line
LINE_B_MW   = BASE_MVA / LINE_X_PU     # susceptance in MW/rad = 250

PHI_MAX_DEG = 30.0
PHI_MAX_RAD = math.radians(PHI_MAX_DEG)

VOLL        = 10_000.0    # value of lost load  $/MWh
C_CURTAIL   =     0.1     # curtailment penalty  $/MWh
C_PST       =     5.0     # PST wear cost        $/rad  (≈ 0.09 $/degree)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Build pypowsybl network
# ─────────────────────────────────────────────────────────────────────────────

def build_network() -> pn.Network:
    """
    Two-bus network with two parallel lines.
    L12a : plain line
    L12b : two-winding transformer — models the PST branch
    Both share the same per-unit reactance (x = 0.4 pu).
    """
    net = pn.create_empty("two_bus_two_lines_pst")

    net.create_substations(id=["S1", "S2"], country=["FR", "FR"])
    net.create_voltage_levels(
        id=["VL1", "VL2"], substation_id=["S1", "S2"],
        topology_kind=["BUS_BREAKER", "BUS_BREAKER"],
        nominal_v=[BASE_KV, BASE_KV],
    )
    net.create_buses(id=["B1", "B2"], voltage_level_id=["VL1", "VL2"])

    # ── L12a : plain line ─────────────────────────────────────────────────────
    net.create_lines(
        id=["L12a"],
        voltage_level1_id=["VL1"], bus1_id=["B1"],
        voltage_level2_id=["VL2"], bus2_id=["B2"],
        r=[0.0], x=[LINE_X_PU],
        g1=[0.0], b1=[0.0], g2=[0.0], b2=[0.0],
    )

    # ── L12b : transformer branch (PST represented by phase tap changer) ──────
    # net.create_2_windings_transformers(
    #     id=["L12b"],
    #     voltage_level1_id=["VL1"], bus1_id=["B1"],
    #     voltage_level2_id=["VL2"], bus2_id=["B2"],
    #     rated_u1=[BASE_KV], rated_u2=[BASE_KV],
    #     r=[0.0], x=[LINE_X_PU * Z_BASE],   # Ω
    #     g=[0.0], b=[0.0],
    # )

    net.create_lines(
        id=["L12b"],
        voltage_level1_id=["VL1"], bus1_id=["B1"],
        voltage_level2_id=["VL2"], bus2_id=["B2"],
        r=[0.0], x=[LINE_X_PU],  # pu directly, not in Ω
        g1=[0.0], b1=[0.0], g2=[0.0], b2=[0.0],
    )

    # ── Generators ────────────────────────────────────────────────────────────
    net.create_generators(
        id=["G1", "G2"],
        voltage_level_id=["VL1", "VL2"],
        bus_id=["B1", "B2"],
        energy_source=["OTHER", "OTHER"],
        min_p=[0.0, 0.0], max_p=[200.0, 150.0],
        target_p=[125.0, 125.0],
        target_v=[BASE_KV, BASE_KV],
        target_q=[0, 1],
        voltage_regulator_on=[True, False],
    )

    # ── Load ──────────────────────────────────────────────────────────────────
    net.create_loads(
        id=["D2"], voltage_level_id=["VL2"], bus_id=["B2"],
        p0=[250.0], q0=[0.0],
    )

    return net


# ─────────────────────────────────────────────────────────────────────────────
# 2. Analytical sensitivities
# ─────────────────────────────────────────────────────────────────────────────

def compute_sensitivities() -> dict:
    """
    Analytical PTDF and PSDF for two identical parallel lines (b_a = b_b = b).
    PST is on L12b.

    PTDF (G1 = slack, distributed_slack=False):
        P̃_a = PTDF_a_G2·(P_G2 − Pd)  →  PTDF_a_G2 = −0.5
        P̃_b = PTDF_b_G2·(P_G2 − Pd)  →  PTDF_b_G2 = −0.5

    PSDF (φ in radians):
        ∂P̃_a/∂φ = −b/2 = −125 MW/rad
        ∂P̃_b/∂φ = +b/2 = +125 MW/rad
    """
    b = LINE_B_MW   # 250 MW/rad

    return {
        "ptdf": {
            ("L12a", "G1"): 0.0,
            ("L12a", "G2"): -0.5,
            ("L12b", "G1"): 0.0,
            ("L12b", "G2"): -0.5,
        },
        "psdf": {
            "L12a": -b / 2,   # −125 MW/rad  (φ>0 → P̃_a ↓)
            "L12b": +b / 2,   # +125 MW/rad  (φ>0 → P̃_b ↑)
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. Parameters
# ─────────────────────────────────────────────────────────────────────────────

def extract_parameters(network: pn.Network,
                       sens: dict,
                       pmax_a: float,
                       pmax_b: float) -> dict:
    gens  = network.get_generators()
    loads = network.get_loads()

    Pg_min = {g: gens.loc[g, "min_p"] for g in gens.index}
    Pg_max = {g: gens.loc[g, "max_p"] for g in gens.index}
    Pd_bus = {"B1": 0.0, "B2": loads.loc["D2", "p0"]}
    cost   = {"G1": 30.0, "G2": 45.0}

    M = sum(Pg_max.values())   # 350 MW  — valid Big-M

    return dict(
        Pg_min=Pg_min, Pg_max=Pg_max, Pd_bus=Pd_bus,
        cost=cost, sens=sens,
        pmax_a=pmax_a, pmax_b=pmax_b, M=M,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 4. Helper — add one line's automaton constraints
# ─────────────────────────────────────────────────────────────────────────────

def add_automaton_constraints(model, P12, P12_nat, z, beta, Pmax, M, label):
    """
    Add all Big-M automaton constraints for a single line.
    Returns a dict of constraint handles (needed if duals are queried later).
    """
    cons = {}

    # §5.3  P12 = P12_nat when connected
    cons[f"s53_up_{label}"] = model.add_linear_constraint(
        P12 - P12_nat - M + M * z, poi.Leq, 0.0, name=f"s53_up_{label}")
    cons[f"s53_lo_{label}"] = model.add_linear_constraint(
        P12 - P12_nat + M - M * z, poi.Geq, 0.0, name=f"s53_lo_{label}")

    # §5.4  P12 = 0 when tripped
    cons[f"s54_up_{label}"] = model.add_linear_constraint(
        P12 - Pmax * z, poi.Leq, 0.0, name=f"s54_up_{label}")
    cons[f"s54_lo_{label}"] = model.add_linear_constraint(
        P12 + Pmax * z, poi.Geq, 0.0, name=f"s54_lo_{label}")

    # A1+  positive overload forces trip
    cons[f"A1p_{label}"] = model.add_linear_constraint(
        P12_nat - M + M * z, poi.Leq, Pmax, name=f"A1p_{label}")

    # A1-  negative overload forces trip
    cons[f"A1n_{label}"] = model.add_linear_constraint(
        P12_nat + M - M * z, poi.Geq, -Pmax, name=f"A1n_{label}")

    # A2+  trip → positive overload  (β=1 branch)
    cons[f"A2p_{label}"] = model.add_linear_constraint(
        Pmax - Pmax * z - M * beta - P12_nat,
        poi.Leq, 0.0, name=f"A2p_{label}")

    # A2-  trip → negative overload  (β=0 branch)
    cons[f"A2n_{label}"] = model.add_linear_constraint(
        P12_nat + Pmax - Pmax * z - M + M * beta,
        poi.Leq, 0.0, name=f"A2n_{label}")

    # A2 coupling  β ≤ 1 − z
    cons[f"A2c_{label}"] = model.add_linear_constraint(
        beta + z, poi.Leq, 1.0, name=f"A2c_{label}")

    return cons


# ─────────────────────────────────────────────────────────────────────────────
# 5. MILP solver
# ─────────────────────────────────────────────────────────────────────────────

def solve_milp(params: dict) -> dict:
    """
    DC-OPF + two automata + PST as a MILP.

    Decision variables
    ------------------
    Continuous : Pg_G1, Pg_G2
                 P12_a, P12_nat_a   (L12a actual / natural flow)
                 P12_b, P12_nat_b   (L12b actual / natural flow)
                 phi                (PST angle, radians)
                 psi                (|phi|, for objective)
                 lostload_B2        (unserved demand at Bus 2)
                 curtail_G1         (excess generation at Bus 1)
    Binary     : z_a, β_a           (L12a automaton)
                 z_b, β_b           (L12b automaton)

    Natural flow equations (with PSDF)
    ------------------------------------
      P̃_a = 0.5·(Pd − P_G2) − 125·φ
      P̃_b = 0.5·(Pd − P_G2) + 125·φ

    PST–automaton coupling
    ----------------------
      φ ≤  φ_max · z_b    (φ=0 when L12b tripped)
      φ ≥ −φ_max · z_b
    """
    Pg_min  = params["Pg_min"]
    Pg_max  = params["Pg_max"]
    Pd_bus  = params["Pd_bus"]
    cost    = params["cost"]
    sens    = params["sens"]
    pmax_a  = params["pmax_a"]
    pmax_b  = params["pmax_b"]
    M       = params["M"]
    Pd      = Pd_bus["B2"]

    ptdf    = sens["ptdf"]
    psdf    = sens["psdf"]

    model = xpress.Model()
    model.set_model_attribute(poi.ModelAttribute.Silent, True)

    obj = poi.ExprBuilder()

    # ── Continuous variables ──────────────────────────────────────────────────

    Pg = {g: model.add_variable(lb=Pg_min[g], ub=Pg_max[g], name=f"Pg_{g}")
          for g in ["G1", "G2"]}

    P12_a     = model.add_variable(lb=-M, ub=M, name="P12_a")
    P12_nat_a = model.add_variable(lb=-M, ub=M, name="P12_nat_a")

    P12_b     = model.add_variable(lb=-M, ub=M, name="P12_b")
    P12_nat_b = model.add_variable(lb=-M, ub=M, name="P12_nat_b")

    phi = model.add_variable(lb=-PHI_MAX_RAD, ub=PHI_MAX_RAD, name="phi")
    psi = model.add_variable(lb=0.0, ub=PHI_MAX_RAD, name="psi")   # |phi|

    lostload = model.add_variable(lb=0.0, ub=Pd,           name="lostload_B2")
    curtail  = model.add_variable(lb=0.0, ub=Pg_max["G1"], name="curtail_G1")

    # ── Binary variables ──────────────────────────────────────────────────────

    z_a    = model.add_variable(domain=poi.VariableDomain.Binary, name="z_a")
    beta_a = model.add_variable(domain=poi.VariableDomain.Binary, name="beta_a")
    z_b    = model.add_variable(domain=poi.VariableDomain.Binary, name="z_b")
    beta_b = model.add_variable(domain=poi.VariableDomain.Binary, name="beta_b")

    # ── Natural flow: L12a ────────────────────────────────────────────────────
    # P̃_a = PTDF_a_G2·(P_G2 − Pd) + PSDF_a·φ
    #      = −0.5·(P_G2 − 250) − 125·φ
    #      = 0.5·(250 − P_G2) − 125·φ
    # Constraint: P̃_a + 0.5·P_G2 + 125·φ = 125

    ptdf_a = ptdf[("L12a", "G2")]   # −0.5
    psdf_a = psdf["L12a"]           # −125 MW/rad
    rhs_a  = ptdf_a * (-Pd)         # −0.5·(−250) = 125

    model.add_linear_constraint(
        P12_nat_a - ptdf_a * Pg["G2"] - psdf_a * phi,
        poi.Eq, rhs_a, name="nat_flow_a",
    )

    # ── Natural flow: L12b ────────────────────────────────────────────────────
    # P̃_b = −0.5·(P_G2 − 250) + 125·φ = 0.5·(250 − P_G2) + 125·φ
    # Constraint: P̃_b + 0.5·P_G2 − 125·φ = 125

    ptdf_b = ptdf[("L12b", "G2")]   # −0.5
    psdf_b = psdf["L12b"]           # +125 MW/rad
    rhs_b  = ptdf_b * (-Pd)         # 125

    model.add_linear_constraint(
        P12_nat_b - ptdf_b * Pg["G2"] - psdf_b * phi,
        poi.Eq, rhs_b, name="nat_flow_b",
    )

    # ── Nodal power balance ───────────────────────────────────────────────────
    # Bus 1: G1 − P12_a − P12_b − curtail = 0
    # Bus 2: G2 + P12_a + P12_b + lostload = Pd

    con_bal_B1 = model.add_linear_constraint(
        Pg["G1"] - P12_a - P12_b - curtail,
        poi.Eq, Pd_bus["B1"], name="balance_B1",
    )
    con_bal_B2 = model.add_linear_constraint(
        Pg["G2"] + P12_a + P12_b + lostload,
        poi.Eq, Pd_bus["B2"], name="balance_B2",
    )

    # ── Automaton constraints — L12a ─────────────────────────────────────────
    add_automaton_constraints(model, P12_a, P12_nat_a, z_a, beta_a,
                              pmax_a, M, "a")

    # ── Automaton constraints — L12b ─────────────────────────────────────────
    add_automaton_constraints(model, P12_b, P12_nat_b, z_b, beta_b,
                              pmax_b, M, "b")

    # ── PST–automaton coupling: φ = 0 when L12b tripped ──────────────────────
    # φ ≤  φ_max · z_b  →  φ − φ_max·z_b ≤ 0
    # φ ≥ −φ_max · z_b  →  φ + φ_max·z_b ≥ 0

    model.add_linear_constraint(
        phi - PHI_MAX_RAD * z_b, poi.Leq, 0.0, name="pst_coup_up")
    model.add_linear_constraint(
        phi + PHI_MAX_RAD * z_b, poi.Geq, 0.0, name="pst_coup_lo")

    # ── |φ| linearisation: ψ ≥ φ  and  ψ ≥ −φ ───────────────────────────────
    model.add_linear_constraint(psi - phi,  poi.Geq, 0.0, name="psi_pos")
    model.add_linear_constraint(psi + phi,  poi.Geq, 0.0, name="psi_neg")

    # ── Objective ─────────────────────────────────────────────────────────────
    for g in ["G1", "G2"]:
        obj += cost[g] * Pg[g]
    obj += VOLL    * lostload
    obj += C_CURTAIL * curtail
    obj += C_PST   * psi

    model.set_objective(obj, poi.ObjectiveSense.Minimize)

    # ── Solve ─────────────────────────────────────────────────────────────────
    model.optimize()

    status = model.get_model_attribute(poi.ModelAttribute.TerminationStatus)
    if status != poi.TerminationStatusCode.OPTIMAL:
        raise RuntimeError(f"MILP not optimal: {status}")

    def val(v): return model.get_value(v)

    phi_val = val(phi)

    return {
        "status"      : str(status),
        "total_cost"  : model.get_model_attribute(poi.ModelAttribute.ObjectiveValue),
        "Pg"          : {g: val(Pg[g]) for g in ["G1", "G2"]},
        "P12_a"       : val(P12_a),
        "P12_nat_a"   : val(P12_nat_a),
        "P12_b"       : val(P12_b),
        "P12_nat_b"   : val(P12_nat_b),
        "z_a"         : int(round(val(z_a))),
        "beta_a"      : int(round(val(beta_a))),
        "z_b"         : int(round(val(z_b))),
        "beta_b"      : int(round(val(beta_b))),
        "phi_rad"     : phi_val,
        "phi_deg"     : math.degrees(phi_val),
        "psi_rad"     : val(psi),
        "lostload"    : val(lostload),
        "curtail"     : val(curtail),
        "pmax_a"      : pmax_a,
        "pmax_b"      : pmax_b,
        "M"           : M,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 6. Validate with pypowsybl DC load-flow
# ─────────────────────────────────────────────────────────────────────────────

def validate(results: dict) -> None:
    """Rebuild a fresh network, apply optimal dispatch, run DC LF."""
    net = build_network()

    for g, pval in results["Pg"].items():
        net.update_generators(id=g, target_p=pval)

    # Disconnect tripped lines
    if results["z_a"] == 0:
        net.update_lines(id="L12a", connected1=False, connected2=False)
    if results["z_b"] == 0:
        net.update_lines(
            id="L12b", connected1=False, connected2=False)

    lf = ppw.loadflow.run_dc(net)
    print("\n── pypowsybl DC LF validation ──────────────────────────────────")
    for c in lf:
        print(f"  Component {c.connected_component_num}: {c.status}")

    lines = net.get_lines(all_attributes=True)
    if results["z_a"] == 1:
        print(f"  L12a flow : {lines.loc['L12a', 'p1']:+.2f} MW")

    if results["z_b"] == 1:
        print(f"  L12b flow : {lines.loc['L12b', 'p1']:+.2f} MW")

    gens = net.get_generators(all_attributes=True)
    for g in gens.index:
        print(f"  {g} output : {gens.loc[g, 'p']:+.2f} MW")


# ─────────────────────────────────────────────────────────────────────────────
# 7. Pretty printer
# ─────────────────────────────────────────────────────────────────────────────

STATUS = {1: "CONNECTED", 0: "TRIPPED ⚡"}

def print_results(results: dict, title: str = "") -> None:
    r       = results
    sep     = "─" * 64
    phi_deg = r["phi_deg"]

    def line_info(nat, actual, z, beta, Pmax, label):
        sta  = STATUS[z]
        ol   = abs(nat) > Pmax - 0.01
        flag = f"  ← OVERLOAD ({nat:+.1f} > {Pmax})" if ol and z == 0 else ""
        return (f"  {label:<6}  z={z}  β={beta}  "
                f"P̃={nat:>8.2f} MW  P={actual:>8.2f} MW  "
                f"Pmax={Pmax:>5.0f}  {sta}{flag}")

    print(f"\n{'═'*64}")
    if title:
        print(f"  {title}")
        print(f"{'═'*64}")
    print(f"  Solver  : {r['status']}")
    print(f"  Cost    : ${r['total_cost']:>12,.2f} /h  "
          f"(gen={30*r['Pg']['G1']+45*r['Pg']['G2']:,.0f}  "
          f"VOLL={VOLL*r['lostload']:,.0f}  "
          f"PST={C_PST*r['psi_rad']:.1f})")
    print(sep)
    print(f"  {'Gen':<6}  {'Pg (MW)':>9}  {'Min':>7}  {'Max':>7}  {'$/MWh':>7}")
    print(sep)
    for g, mn, mx, c in [("G1", 0, 200, 30), ("G2", 0, 150, 45)]:
        pg = r["Pg"][g]
        fl = " ◄" if (abs(pg - mn) < 0.5 or abs(pg - mx) < 0.5) else ""
        print(f"  {g:<6}  {pg:>9.2f}  {mn:>7.0f}  {mx:>7.0f}  {c:>7.0f}{fl}")
    print(sep)
    print(f"  PST angle φ = {phi_deg:+.2f}°  (limit ±{PHI_MAX_DEG}°)  "
          f"|φ| cost = {C_PST * r['psi_rad']:.2f} $/h")
    print(sep)
    print(f"  {'Line':<6}  {'z':>2}  {'β':>2}  "
          f"{'Nat. flow':>10}  {'Act. flow':>10}  {'Pmax':>5}  Status")
    print(sep)
    print(line_info(r["P12_nat_a"], r["P12_a"], r["z_a"], r["beta_a"],
                    r["pmax_a"], "L12a"))
    print(line_info(r["P12_nat_b"], r["P12_b"], r["z_b"], r["beta_b"],
                    r["pmax_b"], "L12b"))
    if r["lostload"] > 0.01:
        print(f"  ⚠  Lost load B2 = {r['lostload']:.2f} MW")
    if r["curtail"] > 0.01:
        print(f"  ⚠  Curtailment  = {r['curtail']:.2f} MW")
    print(f"{'═'*64}")


# ─────────────────────────────────────────────────────────────────────────────
# 8. Scenario sweep
# ─────────────────────────────────────────────────────────────────────────────

def scenario_sweep(network: pn.Network, sens: dict) -> None:
    """
    Vary Pmax_a (L12a thermal limit) from 150 MW down to 60 MW.
    Shows how the PST adapts to protect L12a as its limit tightens.

    Key transitions
    ---------------
    Pmax_a ≥ 125 : no PST needed  (symmetric flow 125 MW each at G2_min)
    Pmax_a ≈ 100 : PST activated to push excess flow to L12b
    Pmax_a ≤ ~70 : even max PST cannot prevent L12a trip
    """
    print(f"\n{'═'*100}")
    print("  Scenario sweep — varying Pmax_a  (Pmax_b fixed at 150 MW)")
    print(f"{'═'*100}")
    hdr = (f"  {'Pmax_a':>7}  {'φ (°)':>7}  "
           f"{'z_a':>4}  {'z_b':>4}  "
           f"{'P̃_a':>8}  {'P̃_b':>8}  "
           f"{'Pg1':>7}  {'Pg2':>7}  "
           f"{'LL':>7}  {'Cost':>12}  Note")
    print(hdr)
    print("  " + "─" * 97)

    prev_z_a = 1
    for pmax_a in [150, 130, 110, 100, 95, 90, 80, 70, 60]:
        try:
            p = extract_parameters(network, sens, pmax_a=pmax_a, pmax_b=150.0)
            r = solve_milp(p)

            note = ""
            if r["z_a"] == 1 and prev_z_a == 1 and abs(r["phi_deg"]) > 0.5:
                note = "← PST activated"
            elif r["z_a"] == 0 and prev_z_a == 1:
                note = "← L12a TRIPS"
            elif r["lostload"] > 0.5:
                note = f"← lost load {r['lostload']:.0f} MW"

            print(
                f"  {pmax_a:>7}  {r['phi_deg']:>+7.2f}  "
                f"{r['z_a']:>4}  {r['z_b']:>4}  "
                f"{r['P12_nat_a']:>8.2f}  {r['P12_nat_b']:>8.2f}  "
                f"{r['Pg']['G1']:>7.2f}  {r['Pg']['G2']:>7.2f}  "
                f"{r['lostload']:>7.2f}  {r['total_cost']:>12.2f}  {note}"
            )
            prev_z_a = r["z_a"]

        except RuntimeError as e:
            print(f"  {pmax_a:>7}  {'INFEASIBLE':>90}  ({e})")

    print()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # ── Print sensitivity summary ─────────────────────────────────────────────
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  Network sensitivities (analytical)                          ║")
    print("╠══════════════════════════════════════════════════════════════╣")
    print(f"║  Line susceptance  b = {LINE_B_MW:.0f} MW/rad  (x = {LINE_X_PU} pu)          ║")
    print(f"║  PTDF_L12a_G2 = −0.5     PTDF_L12b_G2 = −0.5              ║")
    print(f"║  PSDF_L12a    = {-LINE_B_MW/2:+.0f} MW/rad  (φ>0 → P̃_a ↓)         ║")
    print(f"║  PSDF_L12b    = {+LINE_B_MW/2:+.0f} MW/rad  (φ>0 → P̃_b ↑)         ║")
    print(f"║  PST range    = ±{PHI_MAX_DEG:.0f}°  (±{PHI_MAX_RAD:.4f} rad)              ║")
    print(f"║  Max flow shift per line: {LINE_B_MW/2 * PHI_MAX_RAD:.1f} MW at φ_max          ║")
    print("╚══════════════════════════════════════════════════════════════╝\n")

    net  = build_network()
    sens = compute_sensitivities()

    # ── Case 1 : both lines within limits — minimal PST ──────────────────────
    print("CASE 1 — Pmax_a=130 MW, Pmax_b=150 MW")
    print("  Expected: both connected, small or zero φ, cheap dispatch")
    p1 = extract_parameters(net, sens, pmax_a=130.0, pmax_b=150.0)
    r1 = solve_milp(p1)
    print_results(r1, "Case 1 — No congestion")
    validate(r1)

    # ── Case 2 : L12a tight — PST prevents trip ───────────────────────────────
    print("\n\nCASE 2 — Pmax_a=95 MW, Pmax_b=150 MW")
    print("  Without PST: natural flow 100 MW > 95 MW → L12a would trip.")
    print("  PST shifts ~5 MW from L12a to L12b to stay connected.")
    p2 = extract_parameters(net, sens, pmax_a=95.0, pmax_b=150.0)
    r2 = solve_milp(p2)
    print_results(r2, "Case 2 — PST prevents L12a trip")
    validate(r2)

    # ── Case 3 : L12a very tight — PST cannot prevent trip ────────────────────
    print("\n\nCASE 3 — Pmax_a=60 MW, Pmax_b=150 MW")
    print("  Even at φ=+30°, max PST shift is +65.4 MW on L12b.")
    print("  Natural flow 100 MW can be reduced to 34.6 MW on L12a ≤ 60 ✓")
    print("  But optimal dispatch may change — check if G2 is needed more.")
    p3 = extract_parameters(net, sens, pmax_a=60.0, pmax_b=150.0)
    r3 = solve_milp(p3)
    print_results(r3, "Case 3 — Tight L12a, PST at high angle")
    validate(r3)

    # ── Scenario sweep ────────────────────────────────────────────────────────
    scenario_sweep(net, sens)
