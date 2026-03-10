"""
DC OPF — Valid PST Topology with pypowsybl
==========================================

Network topology
----------------
The PST substation (S2) has TWO internal busbars (B2a, B2b) at the same
voltage level. The PST transformer connects them inside S2.
External lines connect S1→S2a and S2b→S3 (different substations → lines).

                ┌────────── L12a (line, S1→S3) ──────────────────┐
                │                                                  │
G1(slack) ── B1(S1,400kV)                                     B3(S3,400kV) ── G2/Load
                │                                                  │
                └── L1_2a (line, S1→S2) ── B2a ──[PST]── B2b ── L2b_3 (line, S2→S3) ──┘
                                                  S2 (400kV internal)

pypowsybl rules respected
--------------------------
  Line    : connects two DIFFERENT substations  (L12a, L1_2a, L2b_3)
  Trafo   : connects two voltage levels of the SAME substation (PST inside S2)
  PST     : PhaseTapChanger on the PST transformer (B2a → B2b, both in S2)

OPF theta formulation
---------------------
  θ_B1 = 0  (slack reference)
  θ_B2a, θ_B2b, θ_B3  free angle variables

  P_L12a   = b_a   · (θ_B1 − θ_B3)     = −b_a · θ_B3
  P_L1_2a  = b_in  · (θ_B1 − θ_B2a)    = −b_in · θ_B2a
  P_PST    = b_pst · (θ_B2a − θ_B2b + φ)
  P_L2b_3  = b_out · (θ_B2b − θ_B3)

  HVDC removed from this file to keep focus on PST topology.

Automata
--------
  L12a  : automaton (z_a, β_a) — plain parallel path
  L1_2a : automaton (z_in, β_in) — entry leg of PST path
  PST path note: if L1_2a trips, PST path goes dark entirely
                 (B2a loses its only feed)
"""

import math
import pypowsybl.network as pn
import pypowsybl as ppw
import pyoptinterface as poi
from pyoptinterface import xpress
from dc_opf_pst_dataframe_api import build_pst_dataframes

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

BASE_MVA    = 100.0
KV          = 400.0
Z_BASE      = KV**2 / BASE_MVA     # 1600 Ω

# Line susceptances (MW/rad at 100 MVA base)
B_L12A      = BASE_MVA / 0.2       # 500 MW/rad   (direct path, x=0.2 pu)
B_L1_2A     = BASE_MVA / 0.4       # 250 MW/rad   (entry leg to PST, x=0.4 pu)
B_PST       = BASE_MVA / 0.1       # 1000 MW/rad  (PST transformer, x=0.1 pu)
B_L2B_3     = BASE_MVA / 0.4       # 250 MW/rad   (exit leg from PST, x=0.4 pu)

# Thermal limits
PMAX_L12A   = 130.0   # MW  direct path
PMAX_L1_2A  = 180.0   # MW  entry leg
PMAX_PST    = 180.0   # MW  PST transformer
PMAX_L2B_3  = 180.0   # MW  exit leg

# PST angle
PHI_MAX_DEG = 30.0
PHI_MAX_RAD = math.radians(PHI_MAX_DEG)
C_PST       = 5.0     # $/rad wear cost
N_STEPS     = 21
NEUTRAL_TAP = 10
ALPHA_STEP  = 3.0   # degrees per tap step

# Angle bounds
THETA_MAX   = math.pi / 3   # ±60°

# Costs
VOLL        = 10_000.0
C_CURTAIL   = 0.1


# ─────────────────────────────────────────────────────────────────────────────
# 1. Build pypowsybl network
# ─────────────────────────────────────────────────────────────────────────────
def build_network() -> pn.Network:
    """
    Valid pypowsybl network with a PST.

    Substation layout
    -----------------
    S1 : one 400 kV voltage level (VL1)
         one bus B1
    S2 : ONE substation with TWO 400 kV voltage levels (VL2a, VL2b)
         bus B2a in VL2a  (receives from S1 via L1_2a)
         bus B2b in VL2b  (sends to S3 via L2b_3)
         PST transformer between VL2a and VL2b — valid because SAME substation
    S3 : one 400 kV voltage level (VL3)
         one bus B3

    Lines  (different substations):
      L12a  : S1 (VL1/B1) → S3 (VL3/B3)   direct path
      L1_2a : S1 (VL1/B1) → S2 (VL2a/B2a) entry leg of PST path
      L2b_3 : S2 (VL2b/B2b) → S3 (VL3/B3) exit leg of PST path

    Transformer (same substation S2):
      PST   : S2 VL2a/B2a → S2 VL2b/B2b   with phase tap changer
    """
    net = pn.create_empty("pst_valid_topology")

    # ── Substations ───────────────────────────────────────────────────────────
    # S2 hosts both VL2a and VL2b — this is the PST substation
    net.create_substations(
        id      = ["S1",  "S2",  "S3"],
        country = ["FR",  "FR",  "FR"],
    )

    # ── Voltage levels ────────────────────────────────────────────────────────
    # S2 has TWO voltage levels at the same nominal voltage — valid in pypowsybl
    net.create_voltage_levels(
        id            = ["VL1",         "VL2a",        "VL2b",        "VL3"],
        substation_id = ["S1",          "S2",          "S2",          "S3"],
        topology_kind = ["BUS_BREAKER", "BUS_BREAKER", "BUS_BREAKER", "BUS_BREAKER"],
        nominal_v     = [KV,             KV,            KV,            KV],
    )

    # ── Buses ─────────────────────────────────────────────────────────────────
    net.create_buses(
        id               = ["B1",   "B2a",  "B2b",  "B3"],
        voltage_level_id = ["VL1",  "VL2a", "VL2b", "VL3"],
    )

    # ── Lines (different substations) ─────────────────────────────────────────

    # L12a : direct path S1 → S3  (bypasses PST entirely)
    net.create_lines(
        id=["L12a"],
        voltage_level1_id=["VL1"], bus1_id=["B1"],
        voltage_level2_id=["VL3"], bus2_id=["B3"],
        r=[0.0], x=[BASE_MVA / B_L12A],    # x = 1/b in pu
        g1=[0.0], b1=[0.0], g2=[0.0], b2=[0.0],
    )

    # L1_2a : entry leg  S1 → S2 (to B2a, before PST)
    net.create_lines(
        id=["L1_2a"],
        voltage_level1_id=["VL1"],  bus1_id=["B1"],
        voltage_level2_id=["VL2a"], bus2_id=["B2a"],
        r=[0.0], x=[BASE_MVA / B_L1_2A],
        g1=[0.0], b1=[0.0], g2=[0.0], b2=[0.0],
    )

    # L2b_3 : exit leg  S2 (from B2b, after PST) → S3
    net.create_lines(
        id=["L2b_3"],
        voltage_level1_id=["VL2b"], bus1_id=["B2b"],
        voltage_level2_id=["VL3"],  bus2_id=["B3"],
        r=[0.0], x=[BASE_MVA / B_L2B_3],
        g1=[0.0], b1=[0.0], g2=[0.0], b2=[0.0],
    )

    # ── PST transformer (SAME substation S2 → valid!) ─────────────────────────
    # Connects VL2a (B2a) to VL2b (B2b) — both inside S2
    # The phase tap changer is attached after creation
    net.create_2_windings_transformers(
        id                = ["PST_T"],
        voltage_level1_id = ["VL2a"], bus1_id = ["B2a"],
        voltage_level2_id = ["VL2b"], bus2_id = ["B2b"],
        rated_u1          = [KV],
        rated_u2          = [KV],
        r                 = [0.0],
        x                 = [BASE_MVA / B_PST * Z_BASE],  # Ω
        g                 = [0.0],
        b                 = [0.0],
    )

    # Attach a phase tap changer to PST_T
    # Steps: −10 to +10  (each step = 3° → range ±30°)
    # neutral step = 10 (middle), so step 0 = −30°, step 10 = 0°, step 20 = +30°
    ptc_df, steps_df = build_pst_dataframes()
    net.create_phase_tap_changers(ptc_df, steps_df)

    # ── Generators ────────────────────────────────────────────────────────────
    net.create_generators(
        id               = ["G1",    "G2"],
        voltage_level_id = ["VL1",   "VL3"],
        bus_id           = ["B1",    "B3"],
        energy_source    = ["OTHER", "OTHER"],
        min_p            = [0.0,     0.0],
        max_p            = [300.0,   150.0],
        target_p         = [200.0,   50.0],
        target_v         = [KV,      KV],
        target_q=[0, 1],
        voltage_regulator_on = [True, False],
    )

    # ── Load ──────────────────────────────────────────────────────────────────
    net.create_loads(
        id               = ["D3"],
        voltage_level_id = ["VL3"],
        bus_id           = ["B3"],
        p0               = [250.0],
        q0               = [0.0],
    )

    return net


# ─────────────────────────────────────────────────────────────────────────────
# 2. Extract parameters
# ─────────────────────────────────────────────────────────────────────────────

def extract_parameters(network: pn.Network) -> dict:
    gens  = network.get_generators()
    loads = network.get_loads()

    Pg_min = {g: float(gens.loc[g, "min_p"]) for g in gens.index}
    Pg_max = {g: float(gens.loc[g, "max_p"]) for g in gens.index}
    Pd_bus = {
        "B1":  0.0,
        "B2a": 0.0,
        "B2b": 0.0,
        "B3":  float(loads.loc["D3", "p0"]),
    }
    cost = {"G1": 30.0, "G2": 45.0}

    M = sum(Pg_max.values())  # 450 MW

    return dict(
        Pg_min=Pg_min, Pg_max=Pg_max, Pd_bus=Pd_bus,
        cost=cost, M=M,
    )

# ─────────────────────────────────────────────────────────────────────────────
# 3. Automaton helper
# ─────────────────────────────────────────────────────────────────────────────

def add_automaton(model, P, P_nat, z, beta, Pmax, M, tag):
    """Big-M automaton for one branch."""
    # §5.3 P = P_nat when connected
    model.add_linear_constraint(
        P - P_nat - M + M * z, poi.Leq, 0.0, name=f"s53u_{tag}")
    model.add_linear_constraint(
        P - P_nat + M - M * z, poi.Geq, 0.0, name=f"s53l_{tag}")
    # §5.4 P = 0 when tripped
    model.add_linear_constraint(
        P - Pmax * z, poi.Leq, 0.0, name=f"s54u_{tag}")
    model.add_linear_constraint(
        P + Pmax * z, poi.Geq, 0.0, name=f"s54l_{tag}")
    # A1± forward
    model.add_linear_constraint(
        P_nat - M + M * z, poi.Leq,  Pmax, name=f"A1p_{tag}")
    model.add_linear_constraint(
        P_nat + M - M * z, poi.Geq, -Pmax, name=f"A1n_{tag}")
    # A2± backward
    model.add_linear_constraint(
        Pmax - Pmax * z - M * beta - P_nat,
        poi.Leq, 0.0, name=f"A2p_{tag}")
    model.add_linear_constraint(
        P_nat + Pmax - Pmax * z - M + M * beta,
        poi.Leq, 0.0, name=f"A2n_{tag}")
    # A2 coupling
    model.add_linear_constraint(
        beta + z, poi.Leq, 1.0, name=f"A2c_{tag}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. MILP solver
# ─────────────────────────────────────────────────────────────────────────────

def solve_milp(params: dict,
               pmax_l12a: float   = PMAX_L12A,
               phi_fixed: float   = None) -> dict:
    """
    DC-OPF with 4-bus theta formulation, PST angle φ, and automata.

    Bus angles
    ----------
      θ_B1 = 0  (reference, implicit)
      θ_B2a, θ_B2b, θ_B3  are decision variables

    Branch flow equations
    ---------------------
      P_L12a  = B_L12A  · (0 − θ_B3)           = −B_L12A·θ_B3
      P_L1_2a = B_L1_2A · (0 − θ_B2a)          = −B_L1_2A·θ_B2a
      P_PST   = B_PST   · (θ_B2a − θ_B2b + φ)
      P_L2b_3 = B_L2B_3 · (θ_B2b − θ_B3)

    Note on PST sign convention
    ---------------------------
      φ > 0 increases the effective angle seen by the PST transformer,
      increasing P_PST and therefore the flow through the PST path.
      This offloads L12a.

    Parameters
    ----------
    phi_fixed : if not None, fix φ to this value (radians) and make it
                a parameter rather than a decision variable — useful for
                testing what happens at a specific PST tap position.
    """
    Pg_min = params["Pg_min"]
    Pg_max = params["Pg_max"]
    Pd_bus = params["Pd_bus"]
    cost   = params["cost"]
    M      = params["M"]

    model = xpress.Model()
    model.set_model_attribute(poi.ModelAttribute.Silent, True)
    obj = poi.ExprBuilder()

    # ── Generator outputs ─────────────────────────────────────────────────────
    Pg = {
        g: model.add_variable(lb=Pg_min[g], ub=Pg_max[g], name=f"Pg_{g}")
        for g in ["G1", "G2"]
    }

    # ── Bus angles (θ_B1 = 0 implicit reference) ─────────────────────────────
    theta = {
        b: model.add_variable(lb=-THETA_MAX, ub=THETA_MAX, name=f"theta_{b}")
        for b in ["B2a", "B2b", "B3"]
    }

    # ── Branch flow variables — natural (physics) and actual ──────────────────
    # L12a direct path
    P_L12a     = model.add_variable(lb=-M, ub=M, name="P_L12a")
    P_nat_L12a = model.add_variable(lb=-M, ub=M, name="P_nat_L12a")

    # L1_2a entry leg of PST path
    P_L1_2a     = model.add_variable(lb=-M, ub=M, name="P_L1_2a")
    P_nat_L1_2a = model.add_variable(lb=-M, ub=M, name="P_nat_L1_2a")

    # PST transformer (no automaton — tripped by upstream L1_2a automaton)
    P_PST = model.add_variable(lb=-PMAX_PST, ub=PMAX_PST, name="P_PST")

    # L2b_3 exit leg (no separate automaton — cascades with entry leg)
    P_L2b3 = model.add_variable(lb=-PMAX_L2B_3, ub=PMAX_L2B_3, name="P_L2b3")

    # ── PST angle ─────────────────────────────────────────────────────────────
    if phi_fixed is not None:
        # Fixed tap — model it as a parameter via tight bounds
        phi = model.add_variable(lb=phi_fixed, ub=phi_fixed, name="phi")
        psi = model.add_variable(lb=abs(phi_fixed), ub=abs(phi_fixed), name="psi")
    else:
        phi = model.add_variable(lb=-PHI_MAX_RAD, ub=PHI_MAX_RAD, name="phi")
        psi = model.add_variable(lb=0.0, ub=PHI_MAX_RAD, name="psi")

    # ── Automaton binaries — L12a and L1_2a ──────────────────────────────────
    z_a    = model.add_variable(domain=poi.VariableDomain.Binary, name="z_a")
    beta_a = model.add_variable(domain=poi.VariableDomain.Binary, name="beta_a")
    z_in   = model.add_variable(domain=poi.VariableDomain.Binary, name="z_in")
    beta_in= model.add_variable(domain=poi.VariableDomain.Binary, name="beta_in")

    # ── Feasibility slacks ────────────────────────────────────────────────────
    ll_B3 = model.add_variable(lb=0.0, ub=Pd_bus["B3"], name="lostload_B3")
    cu_G1 = model.add_variable(lb=0.0, ub=Pg_max["G1"], name="curtail_G1")

    # ── Natural flow equations (theta) ────────────────────────────────────────

    # P_nat_L12a = B_L12A·(0 − θ_B3)
    # → P_nat_L12a + B_L12A·θ_B3 = 0
    model.add_linear_constraint(
        P_nat_L12a + B_L12A * theta["B3"],
        poi.Eq, 0.0, name="nat_L12a",
    )

    # P_nat_L1_2a = B_L1_2A·(0 − θ_B2a)
    # → P_nat_L1_2a + B_L1_2A·θ_B2a = 0
    model.add_linear_constraint(
        P_nat_L1_2a + B_L1_2A * theta["B2a"],
        poi.Eq, 0.0, name="nat_L1_2a",
    )

    # P_PST = B_PST·(θ_B2a − θ_B2b + φ)
    # → P_PST − B_PST·θ_B2a + B_PST·θ_B2b − B_PST·φ = 0
    model.add_linear_constraint(
        P_PST - B_PST * theta["B2a"] + B_PST * theta["B2b"] - B_PST * phi,
        poi.Eq, 0.0, name="pst_flow",
    )

    # P_L2b3 = B_L2B_3·(θ_B2b − θ_B3)
    # → P_L2b3 − B_L2B_3·θ_B2b + B_L2B_3·θ_B3 = 0
    model.add_linear_constraint(
        P_L2b3 - B_L2B_3 * theta["B2b"] + B_L2B_3 * theta["B3"],
        poi.Eq, 0.0, name="flow_L2b3",
    )

    # ── Nodal power balances ──────────────────────────────────────────────────
    #
    # B1:  G1 − P_L12a − P_L1_2a − cu_G1 = 0
    model.add_linear_constraint(
        Pg["G1"] - P_L12a - P_L1_2a - cu_G1,
        poi.Eq, Pd_bus["B1"], name="bal_B1",
    )

    # B2a: P_L1_2a − P_PST = 0
    #      (transit bus: all inflow from L1_2a goes into PST, no load/gen)
    model.add_linear_constraint(
        P_L1_2a - P_PST,
        poi.Eq, Pd_bus["B2a"], name="bal_B2a",
    )

    # B2b: P_PST − P_L2b3 = 0
    #      (transit bus: PST output feeds the exit line)
    model.add_linear_constraint(
        P_PST - P_L2b3,
        poi.Eq, Pd_bus["B2b"], name="bal_B2b",
    )

    # B3:  G2 + P_L12a + P_L2b3 + ll_B3 = 250
    model.add_linear_constraint(
        Pg["G2"] + P_L12a + P_L2b3 + ll_B3,
        poi.Eq, Pd_bus["B3"], name="bal_B3",
    )

    # ── Automaton on L12a ─────────────────────────────────────────────────────
    add_automaton(
        model, P_L12a, P_nat_L12a, z_a, beta_a, pmax_l12a, M, "L12a")

    # ── Automaton on L1_2a (entry leg of PST path) ────────────────────────────
    # When z_in = 0: P_L1_2a = 0 → B2a/B2b island with no feed
    # The PST balance (B2a: P_L1_2a = P_PST) will force P_PST = 0 too
    # which forces P_L2b3 = 0 — entire PST path goes dark
    add_automaton(
        model, P_L1_2a, P_nat_L1_2a, z_in, beta_in, PMAX_L1_2A, M, "L1_2a")

    # ── PST coupling: φ = 0 when entry leg tripped ───────────────────────────
    # When z_in = 0 the PST is de-energised — no point having a phase shift
    model.add_linear_constraint(
        phi - PHI_MAX_RAD * z_in, poi.Leq, 0.0, name="pst_coup_up")
    model.add_linear_constraint(
        phi + PHI_MAX_RAD * z_in, poi.Geq, 0.0, name="pst_coup_lo")

    # ── |φ| linearisation ────────────────────────────────────────────────────
    model.add_linear_constraint(psi - phi, poi.Geq, 0.0, name="psi_pos")
    model.add_linear_constraint(psi + phi, poi.Geq, 0.0, name="psi_neg")

    # ── Objective ─────────────────────────────────────────────────────────────
    for g in ["G1", "G2"]:
        obj += cost[g] * Pg[g]
    obj += VOLL    * ll_B3
    obj += C_CURTAIL * cu_G1
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
        "status"        : str(status),
        "total_cost"    : model.get_model_attribute(poi.ModelAttribute.ObjectiveValue),
        "Pg"            : {g: val(Pg[g]) for g in ["G1", "G2"]},
        "theta_deg"     : {b: math.degrees(val(theta[b])) for b in ["B2a","B2b","B3"]},
        "P_L12a"        : val(P_L12a),
        "P_nat_L12a"    : val(P_nat_L12a),
        "P_L1_2a"       : val(P_L1_2a),
        "P_nat_L1_2a"   : val(P_nat_L1_2a),
        "P_PST"         : val(P_PST),
        "P_L2b3"        : val(P_L2b3),
        "phi_deg"       : math.degrees(phi_val),
        "psi_rad"       : val(psi),
        "z_a"           : int(round(val(z_a))),
        "beta_a"        : int(round(val(beta_a))),
        "z_in"          : int(round(val(z_in))),
        "beta_in"       : int(round(val(beta_in))),
        "lostload_B3"   : val(ll_B3),
        "curtail_G1"    : val(cu_G1),
        "pmax_l12a"     : pmax_l12a,
        "M"             : M,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5. Validate with pypowsybl DC load-flow
# ─────────────────────────────────────────────────────────────────────────────

def validate(results: dict) -> None:
    net = build_network()

    for g, pval in results["Pg"].items():
        net.update_generators(id=g, target_p=pval)

    if results["z_a"] == 0:
        net.update_lines(id="L12a", connected1=False, connected2=False)
    if results["z_in"] == 0:
        net.update_lines(id="L1_2a", connected1=False, connected2=False)

    # Set PST tap position corresponding to optimal phi
    # Each step = 3°, neutral tap = 10
    phi_deg  = results["phi_deg"]
    tap_step = 10 + round(phi_deg / 3.0)
    tap_step = max(0, min(20, tap_step))

    net.update_phase_tap_changers(id="PST_T", tap=tap_step)

    lf = ppw.loadflow.run_dc(net)
    print("\n── pypowsybl DC LF validation ───────────────────────────────────")
    for c in lf:
        print(f"  Component {c.connected_component_num}: {c.status}")

    lines  = net.get_lines(all_attributes=True)
    trafos = net.get_2_windings_transformers(all_attributes=True)
    gens   = net.get_generators(all_attributes=True)

    if results["z_a"] == 1:
        print(f"  L12a   : {lines.loc['L12a', 'p1']:+.2f} MW  "
              f"(OPF: {results['P_L12a']:+.2f} MW)")
    if results["z_in"] == 1:
        print(f"  L1_2a  : {lines.loc['L1_2a', 'p1']:+.2f} MW  "
              f"(OPF: {results['P_L1_2a']:+.2f} MW)")
        print(f"  PST_T  : {trafos.loc['PST_T', 'p1']:+.2f} MW  "
              f"(OPF: {results['P_PST']:+.2f} MW, tap={tap_step}, φ≈{phi_deg:.1f}°)")
        print(f"  L2b_3  : {lines.loc['L2b_3', 'p1']:+.2f} MW  "
              f"(OPF: {results['P_L2b3']:+.2f} MW)")
    for g in gens.index:
        print(f"  {g}     : {gens.loc[g, 'p']:+.2f} MW  "
              f"(OPF: {results['Pg'].get(g, 0):+.2f} MW)")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Pretty printer
# ─────────────────────────────────────────────────────────────────────────────

def print_results(r: dict, title: str = "") -> None:
    sep = "─" * 66
    STATUS = {1: "CONNECTED", 0: "TRIPPED ⚡"}

    print(f"\n{'═'*66}")
    if title:
        print(f"  {title}")
        print(f"{'═'*66}")

    gen_cost  = sum(r["Pg"][g]*c for g,c in zip(["G1","G2"],[30,45]))
    voll_cost = VOLL * r["lostload_B3"]
    pst_cost  = C_PST * r["psi_rad"]

    print(f"  Status  : {r['status']}")
    print(f"  Cost    : ${r['total_cost']:>12,.2f} /h  "
          f"(gen={gen_cost:,.0f}  VOLL={voll_cost:,.0f}  PST={pst_cost:.1f})")
    print(sep)
    print(f"  {'Gen':<6} {'Pg (MW)':>9}  {'Min':>6}  {'Max':>6}  {'$/MWh':>6}")
    print(sep)
    for g, mn, mx, c in [("G1",0,300,30),("G2",0,150,45)]:
        pg  = r["Pg"][g]
        lim = " ◄" if (abs(pg-mn)<0.5 or abs(pg-mx)<0.5) else ""
        print(f"  {g:<6} {pg:>9.2f}  {mn:>6}  {mx:>6}  {c:>6}{lim}")
    print(sep)
    print(f"  Bus angles :  θ_B1=0°  "
          f"θ_B2a={r['theta_deg']['B2a']:+.3f}°  "
          f"θ_B2b={r['theta_deg']['B2b']:+.3f}°  "
          f"θ_B3={r['theta_deg']['B3']:+.3f}°")
    print(sep)
    print(f"  PST angle φ = {r['phi_deg']:+.2f}°  (limit ±{PHI_MAX_DEG}°)")
    print(sep)
    print(f"  {'Branch':<10} {'Nat.flow':>9}  {'Act.flow':>9}  "
          f"{'Pmax':>6}  Status")
    print(sep)

    def row(label, nat, act, z, beta, pmax):
        sta  = STATUS[z]
        flag = "  ← OVERLOAD" if abs(nat) > pmax-0.01 and z==0 else ""
        print(f"  {label:<10} {nat:>+9.2f}  {act:>+9.2f}  "
              f"{pmax:>6.0f}  {sta}{flag}")

    row("L12a",  r["P_nat_L12a"], r["P_L12a"],  r["z_a"],  r["beta_a"],  r["pmax_l12a"])
    row("L1_2a", r["P_nat_L1_2a"],r["P_L1_2a"], r["z_in"], r["beta_in"], PMAX_L1_2A)
    print(f"  {'PST_T':<10} {'(θ model)':>9}  {r['P_PST']:>+9.2f}  "
          f"{PMAX_PST:>6.0f}  CONNECTED")
    print(f"  {'L2b_3':<10} {'(θ model)':>9}  {r['P_L2b3']:>+9.2f}  "
          f"{PMAX_L2B_3:>6.0f}  CONNECTED")

    if r["lostload_B3"] > 0.01:
        print(f"  ⚠  Lost load B3 = {r['lostload_B3']:.2f} MW")
    if r["curtail_G1"]  > 0.01:
        print(f"  ⚠  Curtail G1   = {r['curtail_G1']:.2f} MW")
    print(f"{'═'*66}")


# ─────────────────────────────────────────────────────────────────────────────
# 7. Scenario sweep — vary Pmax_L12a to stress-test PST
# ─────────────────────────────────────────────────────────────────────────────

def sweep_pmax(params: dict) -> None:
    """
    Reduce L12a thermal limit.
    The PST should activate to shift excess flow from L12a onto the PST path.
    At very tight limits, the automaton trips L12a and all load is served via
    the PST path (or lost load if PST path also saturates).
    """
    print(f"\n{'═'*95}")
    print("  Sweep: L12a thermal limit")
    print(f"{'═'*95}")
    hdr = (f"  {'Pmax_a':>7}  {'φ°':>7}  {'z_a':>4}  {'z_in':>4}  "
           f"{'P_L12a':>8}  {'P_L1_2a':>8}  {'P_PST':>7}  "
           f"{'Pg1':>7}  {'Pg2':>7}  "
           f"{'LL':>7}  {'Cost':>12}  Note")
    print(hdr)
    print("  " + "─" * 92)

    prev_z_a = 1
    for pmax_a in [200, 160, 130, 110, 95, 80, 60, 40]:
        try:
            r    = solve_milp(params, pmax_l12a=float(pmax_a))
            note = ""
            if r["z_a"]==0 and prev_z_a==1:
                note = "← L12a TRIPS"
            elif abs(r["phi_deg"]) > 1.0 and r["z_a"]==1:
                note = "← PST active"
            print(
                f"  {pmax_a:>7}  {r['phi_deg']:>+7.2f}  "
                f"{r['z_a']:>4}  {r['z_in']:>4}  "
                f"{r['P_L12a']:>+8.2f}  {r['P_L1_2a']:>+8.2f}  "
                f"{r['P_PST']:>+7.2f}  "
                f"{r['Pg']['G1']:>7.2f}  {r['Pg']['G2']:>7.2f}  "
                f"{r['lostload_B3']:>7.2f}  {r['total_cost']:>12.2f}  {note}"
            )
            prev_z_a = r["z_a"]
        except RuntimeError as e:
            print(f"  {pmax_a:>7}  {'INFEASIBLE':>85}  ({e})")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# 8. Sweep — compare fixed PST taps to optimal
# ─────────────────────────────────────────────────────────────────────────────

def sweep_fixed_tap(params: dict) -> None:
    """
    Compare fixed PST angles (as if it were a manually-set tap) versus
    the optimal angle found by the OPF.
    Shows the value of optimising φ vs a fixed tap position.
    """
    print(f"\n{'═'*80}")
    print("  Sweep: fixed PST tap vs optimal  (Pmax_a = 110 MW)")
    print(f"{'═'*80}")
    hdr = (f"  {'φ (°)':>8}  {'Mode':>10}  {'P_L12a':>8}  {'P_PST':>8}  "
           f"{'z_a':>4}  {'LL':>7}  {'Cost':>12}")
    print(hdr)
    print("  " + "─" * 77)

    # optimal
    try:
        r_opt = solve_milp(params, pmax_l12a=110.0)
        print(f"  {r_opt['phi_deg']:>+8.2f}  {'OPTIMAL':>10}  "
              f"{r_opt['P_L12a']:>+8.2f}  {r_opt['P_PST']:>+8.2f}  "
              f"{r_opt['z_a']:>4}  {r_opt['lostload_B3']:>7.2f}  "
              f"{r_opt['total_cost']:>12.2f}")
    except RuntimeError as e:
        print(f"  {'OPTIMAL FAILED':>80}  ({e})")

    print("  " + "─" * 77)

    # fixed taps
    for phi_deg in [-30, -20, -10, 0, 10, 20, 30]:
        phi_rad = math.radians(phi_deg)
        try:
            r = solve_milp(params, pmax_l12a=110.0, phi_fixed=phi_rad)
            print(f"  {phi_deg:>+8.1f}  {'FIXED TAP':>10}  "
                  f"{r['P_L12a']:>+8.2f}  {r['P_PST']:>+8.2f}  "
                  f"{r['z_a']:>4}  {r['lostload_B3']:>7.2f}  "
                  f"{r['total_cost']:>12.2f}")
        except RuntimeError as e:
            print(f"  {phi_deg:>+8.1f}  {'FIXED':>10}  {'INFEASIBLE':>60}  ({e})")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  DC OPF — Valid PST Topology                                 ║")
    print("╠══════════════════════════════════════════════════════════════╣")
    print("║  Topology:                                                   ║")
    print("║    B1(S1) ──L12a──────────────────────── B3(S3)             ║")
    print("║    B1(S1) ──L1_2a── B2a─[PST]─B2b ──L2b_3── B3(S3)        ║")
    print("║  PST inside S2: VL2a→VL2b (same substation) ✓               ║")
    print("║  Lines cross substations ✓                                   ║")
    print(f"║  L12a: b={B_L12A:.0f} MW/rad  L1_2a: b={B_L1_2A:.0f} MW/rad              ║")
    print(f"║  PST:  b={B_PST:.0f} MW/rad  L2b_3: b={B_L2B_3:.0f} MW/rad              ║")
    print("╚══════════════════════════════════════════════════════════════╝\n")

    net    = build_network()
    params = extract_parameters(net)

    # ── Case 1: no congestion, PST near neutral ───────────────────────────────
    r1 = solve_milp(params, pmax_l12a=200.0)
    print_results(r1, "Case 1 — No congestion (Pmax_a=200 MW)")
    validate(r1)

    # ── Case 2: L12a tight, PST shifts flow to PST path ──────────────────────
    r2 = solve_milp(params, pmax_l12a=110.0)
    print_results(r2, "Case 2 — L12a tight (110 MW): PST activated")
    validate(r2)

    # ── Case 3: L12a very tight, automaton may trip ───────────────────────────
    r3 = solve_milp(params, pmax_l12a=60.0)
    print_results(r3, "Case 3 — L12a very tight (60 MW): possible trip")
    validate(r3)

    # ── Sweeps ────────────────────────────────────────────────────────────────
    sweep_pmax(params)
    sweep_fixed_tap(params)
