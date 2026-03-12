"""
DC OPF — PST Flow Control (LP, no automaton)
=============================================

Both parallel paths between S1 and S3 pass through S2.
L12a bypasses the PST and connects directly to B2b.
L1_2a feeds into B2a, then the PST shifts flow from B2a to B2b.
Both paths merge at B2b and exit to S3 via a single line L2b_3.

Valid pypowsybl topology
-------------------------
  Lines (different substations):
    L12a  : B1(S1,VL1) → B2b(S2,VL2b)   direct path, bypasses PST
    L1_2a : B1(S1,VL1) → B2a(S2,VL2a)   entry leg, feeds PST
    L2b_3 : B2b(S2,VL2b) → B3(S3,VL3)   single exit line

  Transformer (same substation S2):
    PST_T : B2a(S2,VL2a) → B2b(S2,VL2b)  with PhaseTapChanger ✓

Network diagram
---------------
                   ┌──── L12a ─────────────────────────────┐
                   │                                        ↓
G1(S1,B1) ─────B1─┤                                       B2b ──── L2b_3 ──── B3 ──── G2/Load
                   │                                        ↑
                   └──── L1_2a ────── B2a ──[PST_T φ]─────┘
                                      └──── (both in S2) ──┘

Theta formulation
-----------------
  θ_B1 = 0  (slack reference, G1)
  θ_B2a, θ_B2b, θ_B3  free

  P_L12a  = b_a   · (θ_B1 − θ_B2b) = −b_a  · θ_B2b
  P_L1_2a = b_in  · (θ_B1 − θ_B2a) = −b_in · θ_B2a
  P_PST   = b_pst · (θ_B2a − θ_B2b + φ)
  P_L2b3  = b_out · (θ_B2b − θ_B3)

Nodal balances
--------------
  B1  : G1 − P_L12a − P_L1_2a = 0
  B2a : P_L1_2a − P_PST = 0           (transit, no gen/load)
  B2b : P_L12a + P_PST − P_L2b3 = 0  (convergence, no gen/load)
  B3  : G2 + P_L2b3 = Pd_B3

PST effect
----------
  φ > 0 increases effective angle at PST input → more flow via PST path
  φ < 0 reduces PST path flow → more flow via direct L12a
  At φ = 0: flow splits proportionally to branch susceptances
  The OPF chooses φ to minimise cost subject to thermal limits on all branches.
"""

import math
import pypowsybl.network as pn
import pypowsybl.loadflow as plf
import pypowsybl as ppw
import pyoptinterface as poi
from pyoptinterface import xpress
from common_pst import build_pst_dataframes

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

BASE_MVA    = 100.0
KV          = 400.0
Z_BASE      = KV**2 / BASE_MVA      # 1600 Ω

B_L12A      = BASE_MVA / 0.2        # 500  MW/rad  direct path
B_L1_2A     = BASE_MVA / 0.4        # 250  MW/rad  entry leg
B_PST       = BASE_MVA / 0.1        # 1000 MW/rad  PST transformer
B_L2B_3     = BASE_MVA / 0.2        # 500  MW/rad  exit line

PMAX_L12A   = 130.0   # MW
PMAX_L1_2A  = 150.0   # MW
PMAX_PST    = 150.0   # MW
PMAX_L2B_3  = 200.0   # MW

PHI_MAX_DEG = 30.0
PHI_MAX_RAD = math.radians(PHI_MAX_DEG)
C_PST       = 5.0     # $/rad wear cost

THETA_MAX   = math.pi / 3


# ─────────────────────────────────────────────────────────────────────────────
# 1.  pypowsybl network
# ─────────────────────────────────────────────────────────────────────────────

def build_network() -> pn.Network:
    net = pn.create_empty("pst_focus")

    # ── Substations ───────────────────────────────────────────────────────────
    net.create_substations(id=["S1", "S2", "S3"], country=["FR", "FR", "FR"])

    # ── Voltage levels ────────────────────────────────────────────────────────
    # S2 hosts VL2a and VL2b — same substation, PST transformer is valid
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

    # L12a: S1→S2  direct path arriving at B2b (bypasses PST)
    net.create_lines(
        id=["L12a"],
        voltage_level1_id=["VL1"],  bus1_id=["B1"],
        voltage_level2_id=["VL2b"], bus2_id=["B2b"],   # ← arrives at B2b
        r=[0.0], x=[BASE_MVA / B_L12A * Z_BASE],
        g1=[0.0], b1=[0.0], g2=[0.0], b2=[0.0],
    )

    # L1_2a: S1→S2  entry leg arriving at B2a (feeds PST)
    net.create_lines(
        id=["L1_2a"],
        voltage_level1_id=["VL1"],  bus1_id=["B1"],
        voltage_level2_id=["VL2a"], bus2_id=["B2a"],   # ← arrives at B2a
        r=[0.0], x=[BASE_MVA / B_L1_2A * Z_BASE],
        g1=[0.0], b1=[0.0], g2=[0.0], b2=[0.0],
    )

    # L2b_3: S2→S3  single exit line from B2b
    net.create_lines(
        id=["L2b_3"],
        voltage_level1_id=["VL2b"], bus1_id=["B2b"],
        voltage_level2_id=["VL3"],  bus2_id=["B3"],
        r=[0.0], x=[BASE_MVA / B_L2B_3 * Z_BASE],
        g1=[0.0], b1=[0.0], g2=[0.0], b2=[0.0],
    )

    # ── PST transformer (SAME substation S2 → valid) ──────────────────────────
    # VL2a → VL2b, both in S2
    net.create_2_windings_transformers(
        id                = ["PST_T"],
        voltage_level1_id = ["VL2a"], bus1_id = ["B2a"],
        voltage_level2_id = ["VL2b"], bus2_id = ["B2b"],
        rated_u1          = [KV], rated_u2 = [KV],
        r=[0.0], x=[BASE_MVA / B_PST * Z_BASE], g=[0.0], b=[0.0],
    )

    # Phase tap changer: 21 steps (0..20), neutral=10, each step = 3°
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
        id=["D3"], voltage_level_id=["VL3"], bus_id=["B3"],
        p0=[250.0], q0=[0.0],
    )

    return net


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Parameters
# ─────────────────────────────────────────────────────────────────────────────

def extract_parameters(network: pn.Network) -> dict:
    gens  = network.get_generators()
    loads = network.get_loads()
    return dict(
        Pg_min = {g: float(gens.loc[g, "min_p"]) for g in gens.index},
        Pg_max = {g: float(gens.loc[g, "max_p"]) for g in gens.index},
        Pd_B3  = float(loads.loc["D3", "p0"]),
        cost   = {"G1": 30.0, "G2": 45.0},
    )


# ─────────────────────────────────────────────────────────────────────────────
# 3.  LP solver  (pure LP — no binaries, no automaton)
# ─────────────────────────────────────────────────────────────────────────────

def solve_lp(params: dict,
             pmax_l12a: float  = PMAX_L12A,
             pmax_l1_2a: float = PMAX_L1_2A,
             pmax_pst: float   = PMAX_PST,
             pmax_l2b3: float  = PMAX_L2B_3,
             phi_fixed: float  = None) -> dict:
    """
    DC-OPF with PST angle φ as a continuous decision variable.

    This is a pure LP — no binary variables, no automaton.
    Thermal limits are simple box constraints on branch flows.

    Parameters
    ----------
    phi_fixed : if not None, fix φ to this radian value (simulate a fixed tap)
    """
    Pg_min = params["Pg_min"]
    Pg_max = params["Pg_max"]
    Pd_B3  = params["Pd_B3"]
    cost   = params["cost"]

    model = xpress.Model()
    model.set_model_attribute(poi.ModelAttribute.Silent, True)
    obj = poi.ExprBuilder()

    # ── Generator outputs ─────────────────────────────────────────────────────
    Pg = {g: model.add_variable(lb=Pg_min[g], ub=Pg_max[g], name=f"Pg_{g}")
          for g in ["G1", "G2"]}

    # ── Bus angles (θ_B1 = 0, implicit slack reference) ──────────────────────
    theta = {b: model.add_variable(lb=-THETA_MAX, ub=THETA_MAX, name=f"th_{b}")
             for b in ["B2a", "B2b", "B3"]}

    # ── Branch flows with thermal limits ─────────────────────────────────────
    P_L12a  = model.add_variable(lb=-pmax_l12a,  ub=pmax_l12a,  name="P_L12a")
    P_L1_2a = model.add_variable(lb=-pmax_l1_2a, ub=pmax_l1_2a, name="P_L1_2a")
    P_PST   = model.add_variable(lb=-pmax_pst,   ub=pmax_pst,   name="P_PST")
    P_L2b3  = model.add_variable(lb=-pmax_l2b3,  ub=pmax_l2b3,  name="P_L2b3")

    # ── PST angle ─────────────────────────────────────────────────────────────
    if phi_fixed is not None:
        phi = model.add_variable(lb=phi_fixed, ub=phi_fixed, name="phi")
        psi = model.add_variable(lb=abs(phi_fixed), ub=abs(phi_fixed), name="psi")
    else:
        phi = model.add_variable(lb=-PHI_MAX_RAD, ub=PHI_MAX_RAD, name="phi")
        psi = model.add_variable(lb=0.0, ub=PHI_MAX_RAD, name="psi")

    # ── Flow equations (theta formulation) ───────────────────────────────────
    #
    # P_L12a = b_a·(0 − θ_B2b)  →  P_L12a + b_a·θ_B2b = 0
    con_L12a = model.add_linear_constraint(
        P_L12a + B_L12A * theta["B2b"],
        poi.Eq, 0.0, name="flow_L12a",
    )

    # P_L1_2a = b_in·(0 − θ_B2a)  →  P_L1_2a + b_in·θ_B2a = 0
    con_L1_2a = model.add_linear_constraint(
        P_L1_2a + B_L1_2A * theta["B2a"],
        poi.Eq, 0.0, name="flow_L1_2a",
    )

    # P_PST = b_pst·(θ_B2a − θ_B2b + φ)
    # → P_PST − b_pst·θ_B2a + b_pst·θ_B2b − b_pst·φ = 0
    con_PST = model.add_linear_constraint(
        P_PST - B_PST * theta["B2a"] + B_PST * theta["B2b"] - B_PST * phi,
        poi.Eq, 0.0, name="flow_PST",
    )

    # P_L2b3 = b_out·(θ_B2b − θ_B3)
    # → P_L2b3 − b_out·θ_B2b + b_out·θ_B3 = 0
    con_L2b3 = model.add_linear_constraint(
        P_L2b3 - B_L2B_3 * theta["B2b"] + B_L2B_3 * theta["B3"],
        poi.Eq, 0.0, name="flow_L2b3",
    )

    # ── Nodal power balances ──────────────────────────────────────────────────
    #
    # B1:  G1 − P_L12a − P_L1_2a = 0
    con_B1 = model.add_linear_constraint(
        Pg["G1"] - P_L12a - P_L1_2a,
        poi.Eq, 0.0, name="bal_B1",
    )

    # B2a: P_L1_2a − P_PST = 0   (transit bus)
    con_B2a = model.add_linear_constraint(
        P_L1_2a - P_PST,
        poi.Eq, 0.0, name="bal_B2a",
    )

    # B2b: P_L12a + P_PST − P_L2b3 = 0   (convergence bus)
    con_B2b = model.add_linear_constraint(
        P_L12a + P_PST - P_L2b3,
        poi.Eq, 0.0, name="bal_B2b",
    )

    # B3:  G2 + P_L2b3 = Pd_B3
    con_B3 = model.add_linear_constraint(
        Pg["G2"] + P_L2b3,
        poi.Eq, Pd_B3, name="bal_B3",
    )

    # ── |φ| linearisation ────────────────────────────────────────────────────
    model.add_linear_constraint(psi - phi, poi.Geq, 0.0, name="psi_pos")
    model.add_linear_constraint(psi + phi, poi.Geq, 0.0, name="psi_neg")

    # ── Objective ─────────────────────────────────────────────────────────────
    for g in ["G1", "G2"]:
        obj += cost[g] * Pg[g]
    obj += C_PST * psi
    model.set_objective(obj, poi.ObjectiveSense.Minimize)

    # ── Solve ─────────────────────────────────────────────────────────────────
    model.optimize()
    status = model.get_model_attribute(poi.ModelAttribute.TerminationStatus)
    if status != poi.TerminationStatusCode.OPTIMAL:
        raise RuntimeError(f"LP not optimal: {status}")

    def val(v): return model.get_value(v)
    def dual(c): return model.get_constraint_attribute(c, poi.ConstraintAttribute.Dual)

    phi_val = val(phi)
    return {
        "status"      : str(status),
        "total_cost"  : model.get_model_attribute(poi.ModelAttribute.ObjectiveValue),
        "Pg"          : {g: val(Pg[g]) for g in ["G1", "G2"]},
        "theta_deg"   : {b: math.degrees(val(theta[b])) for b in ["B2a","B2b","B3"]},
        "P_L12a"      : val(P_L12a),
        "P_L1_2a"     : val(P_L1_2a),
        "P_PST"       : val(P_PST),
        "P_L2b3"      : val(P_L2b3),
        "phi_deg"     : math.degrees(phi_val),
        "phi_rad"     : phi_val,
        "psi_rad"     : val(psi),
        # Shadow prices (LMPs) from balance constraint duals
        "LMP"         : {
            "B1":  dual(con_B1),
            "B2a": dual(con_B2a),
            "B2b": dual(con_B2b),
            "B3":  dual(con_B3),
        },
        # Shadow prices on flow equations (congestion rents)
        "mu"          : {
            "L12a":  dual(con_L12a),
            "L1_2a": dual(con_L1_2a),
            "PST":   dual(con_PST),
            "L2b3":  dual(con_L2b3),
        },
        "pmax_l12a"   : pmax_l12a,
        "pmax_l1_2a"  : pmax_l1_2a,
        "pmax_pst"    : pmax_pst,
        "pmax_l2b3"   : pmax_l2b3,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Validate with pypowsybl DC load-flow
# ─────────────────────────────────────────────────────────────────────────────

def validate(results: dict) -> None:
    net = build_network()
    net.save("pst1.xiidm")

    for g, pval in results["Pg"].items():
        net.update_generators(id=g, target_p=pval)

    # Map optimal φ → tap step  (step 10 = neutral = 0°, each step = 3°)
    phi_deg  = results["phi_deg"]
    tap      = 10 + round(phi_deg / 3.0)
    tap      = max(0, min(20, tap))
    net.update_phase_tap_changers(id="PST_T", tap=tap)

    lf_params = plf.Parameters(distributed_slack=False)
    lf = ppw.loadflow.run_dc(net, parameters=lf_params)

    print("\n── pypowsybl DC LF validation ───────────────────────────────────")
    for c in lf:
        print(f"  Component {c.connected_component_num}: {c.status}")

    lines  = net.get_lines(all_attributes=True)
    trafos = net.get_2_windings_transformers(all_attributes=True)
    gens   = net.get_generators(all_attributes=True)

    lf_l12a  = lines.loc["L12a",  "p1"]
    lf_l1_2a = lines.loc["L1_2a", "p1"]
    lf_pst   = trafos.loc["PST_T","p1"]
    lf_l2b3  = lines.loc["L2b_3", "p1"]

    print(f"  {'Branch':<10} {'OPF (MW)':>10}  {'LF (MW)':>10}  {'Δ':>8}")
    for name, opf, lf_val in [
        ("L12a",  results["P_L12a"],  lf_l12a),
        ("L1_2a", results["P_L1_2a"], lf_l1_2a),
        ("PST_T", results["P_PST"],   lf_pst),
        ("L2b_3", results["P_L2b3"],  lf_l2b3),
    ]:
        print(f"  {name:<10} {opf:>+10.2f}  {lf_val:>+10.2f}  {opf-lf_val:>+8.3f}")

    print(f"  PST tap used: {tap}  (φ_opt={phi_deg:+.1f}°, "
          f"tap represents φ≈{(tap-10)*3:+.0f}°)")
    for g in gens.index:
        print(f"  {g}: OPF={results['Pg'][g]:+.2f} MW  "
              f"LF={gens.loc[g,'p']:+.2f} MW")


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Pretty printer
# ─────────────────────────────────────────────────────────────────────────────

def print_results(r: dict, title: str = "") -> None:
    sep = "─" * 64
    print(f"\n{'═'*64}")
    if title:
        print(f"  {title}")
        print(f"{'═'*64}")

    gen_cost = sum(r["Pg"][g]*c for g,c in zip(["G1","G2"],[30,45]))
    pst_cost = C_PST * r["psi_rad"]
    print(f"  Status  : {r['status']}")
    print(f"  Cost    : ${r['total_cost']:>10,.2f} /h  "
          f"(gen=${gen_cost:,.0f}  PST_wear=${pst_cost:.2f})")
    print(sep)

    # Generators
    print(f"  {'Gen':<6} {'Pg (MW)':>9}  {'Min':>5}  {'Max':>5}  {'$/MWh':>6}")
    print(sep)
    for g, mn, mx, c in [("G1",0,300,30), ("G2",0,150,45)]:
        pg  = r["Pg"][g]
        lim = " ◄" if (abs(pg-mn)<0.5 or abs(pg-mx)<0.5) else ""
        print(f"  {g:<6} {pg:>9.2f}  {mn:>5}  {mx:>5}  {c:>6}{lim}")
    print(sep)

    # Angles
    print(f"  Angles:  θ_B1=  0.000°  "
          f"θ_B2a={r['theta_deg']['B2a']:>+7.3f}°  "
          f"θ_B2b={r['theta_deg']['B2b']:>+7.3f}°  "
          f"θ_B3={r['theta_deg']['B3']:>+7.3f}°")
    print(f"  PST φ  = {r['phi_deg']:>+6.2f}°  "
          f"(limit ±{PHI_MAX_DEG:.0f}°,  wear=${pst_cost:.2f}/h)")
    print(sep)

    # Branch flows
    data = [
        ("L12a",  r["P_L12a"],  r["pmax_l12a"],  "S1→B2b (bypass)"),
        ("L1_2a", r["P_L1_2a"], r["pmax_l1_2a"], "S1→B2a (PST entry)"),
        ("PST_T", r["P_PST"],   r["pmax_pst"],   "B2a→B2b (φ shift)"),
        ("L2b_3", r["P_L2b3"],  r["pmax_l2b3"],  "B2b→S3 (exit)"),
    ]
    print(f"  {'Branch':<10} {'Flow (MW)':>10}  {'Pmax':>6}  {'Load%':>6}  Description")
    print(sep)
    for name, flow, pmax, desc in data:
        load_pct = 100.0 * abs(flow) / pmax
        flag     = " ◄ BINDING" if load_pct > 99.0 else ""
        print(f"  {name:<10} {flow:>+10.2f}  {pmax:>6.0f}  "
              f"{load_pct:>5.1f}%  {desc}{flag}")
    print(sep)

    # LMPs
    print(f"  Locational Marginal Prices (duals of balance constraints):")
    for b, lmp in r["LMP"].items():
        print(f"    LMP {b:<5} = ${lmp:>8.4f}/MWh")
    print(f"  LMP spread B1→B3 = ${r['LMP']['B3'] - r['LMP']['B1']:>+.4f}/MWh"
          f"  (congestion rent)")
    print(f"{'═'*64}")


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Sweep — vary L12a limit to show PST response
# ─────────────────────────────────────────────────────────────────────────────

def sweep_pmax(params: dict) -> None:
    print(f"\n{'═'*88}")
    print("  Sweep: L12a thermal limit  (PST optimised)")
    print(f"{'═'*88}")
    hdr = (f"  {'Pmax_a':>7}  {'φ°':>7}  "
           f"{'P_L12a':>8}  {'P_L1_2a':>8}  {'P_PST':>7}  {'P_L2b3':>8}  "
           f"{'Pg1':>7}  {'Pg2':>7}  "
           f"{'LMP_B1':>8}  {'LMP_B3':>8}  {'Cost':>10}")
    print(hdr)
    print("  " + "─" * 85)

    for pmax_a in [250, 200, 160, 130, 110, 95, 80, 65, 50]:
        try:
            r = solve_lp(params, pmax_l12a=float(pmax_a))
            print(
                f"  {pmax_a:>7}  {r['phi_deg']:>+7.2f}  "
                f"{r['P_L12a']:>+8.2f}  {r['P_L1_2a']:>+8.2f}  "
                f"{r['P_PST']:>+7.2f}  {r['P_L2b3']:>+8.2f}  "
                f"{r['Pg']['G1']:>7.2f}  {r['Pg']['G2']:>7.2f}  "
                f"{r['LMP']['B1']:>8.4f}  {r['LMP']['B3']:>8.4f}  "
                f"{r['total_cost']:>10.2f}"
            )
        except RuntimeError as e:
            print(f"  {pmax_a:>7}  INFEASIBLE  ({e})")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Sweep — compare fixed tap vs optimal φ
# ─────────────────────────────────────────────────────────────────────────────

def sweep_fixed_tap(params: dict, pmax_l12a: float = 110.0) -> None:
    print(f"\n{'═'*80}")
    print(f"  Fixed tap vs optimal φ  (Pmax_a={pmax_l12a:.0f} MW)")
    print(f"{'═'*80}")
    hdr = (f"  {'φ°':>8}  {'Mode':>10}  "
           f"{'P_L12a':>8}  {'P_PST':>8}  "
           f"{'Pg1':>7}  {'Pg2':>7}  "
           f"{'LMP_B1':>8}  {'LMP_B3':>8}  {'Cost':>10}")
    print(hdr)
    print("  " + "─" * 77)

    # Optimal
    r_opt = solve_lp(params, pmax_l12a=pmax_l12a)
    print(f"  {r_opt['phi_deg']:>+8.2f}  {'OPTIMAL':>10}  "
          f"{r_opt['P_L12a']:>+8.2f}  {r_opt['P_PST']:>+8.2f}  "
          f"{r_opt['Pg']['G1']:>7.2f}  {r_opt['Pg']['G2']:>7.2f}  "
          f"{r_opt['LMP']['B1']:>8.4f}  {r_opt['LMP']['B3']:>8.4f}  "
          f"{r_opt['total_cost']:>10.2f}")
    print("  " + "─" * 77)

    # Fixed taps
    for phi_deg in [-30, -20, -10, 0, 10, 20, 30]:
        try:
            r = solve_lp(params, pmax_l12a=pmax_l12a,
                         phi_fixed=math.radians(phi_deg))
            print(f"  {phi_deg:>+8.1f}  {'FIXED':>10}  "
                  f"{r['P_L12a']:>+8.2f}  {r['P_PST']:>+8.2f}  "
                  f"{r['Pg']['G1']:>7.2f}  {r['Pg']['G2']:>7.2f}  "
                  f"{r['LMP']['B1']:>8.4f}  {r['LMP']['B3']:>8.4f}  "
                  f"{r['total_cost']:>10.2f}")
        except RuntimeError as e:
            print(f"  {phi_deg:>+8.1f}  {'FIXED':>10}  INFEASIBLE  ({e})")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  DC OPF — PST Flow Control (LP, no automaton)                ║")
    print("╠══════════════════════════════════════════════════════════════╣")
    print("║  B1(S1) ──L12a──────────────────────── B2b(S2)              ║")
    print("║  B1(S1) ──L1_2a── B2a(S2)─[PST_T φ]── B2b(S2)              ║")
    print("║                          └─── both in S2 ───┘               ║")
    print("║  B2b(S2) ──L2b_3──────────────────────── B3(S3)─G2/Load     ║")
    print("╠══════════════════════════════════════════════════════════════╣")
    print(f"║  L12a  b={B_L12A:>5.0f} MW/rad  Pmax={PMAX_L12A:>5.0f} MW (bypass)          ║")
    print(f"║  L1_2a b={B_L1_2A:>5.0f} MW/rad  Pmax={PMAX_L1_2A:>5.0f} MW (PST entry)      ║")
    print(f"║  PST_T b={B_PST:>5.0f} MW/rad  Pmax={PMAX_PST:>5.0f} MW (φ∈[±{PHI_MAX_DEG:.0f}°])     ║")
    print(f"║  L2b_3 b={B_L2B_3:>5.0f} MW/rad  Pmax={PMAX_L2B_3:>5.0f} MW (exit)            ║")
    print("╚══════════════════════════════════════════════════════════════╝\n")

    net    = build_network()
    params = extract_parameters(net)

    # ── Case 1: no congestion — PST near neutral ──────────────────────────────
    r1 = solve_lp(params, pmax_l12a=250.0)
    print_results(r1, "Case 1 — No congestion (Pmax_a=250 MW)")
    validate(r1)

    # ── Case 2: L12a tight — PST shifts flow to PST path ─────────────────────
    r2 = solve_lp(params, pmax_l12a=110.0)
    print_results(r2, "Case 2 — L12a congested (Pmax_a=110 MW): PST activated")
    validate(r2)

    # ── Case 3: L12a very tight — PST at maximum angle ───────────────────────
    r3 = solve_lp(params, pmax_l12a=65.0)
    print_results(r3, "Case 3 — L12a very tight (Pmax_a=65 MW): PST near limit")
    validate(r3)

    # ── Sweeps ────────────────────────────────────────────────────────────────
    sweep_pmax(params)
    sweep_fixed_tap(params, pmax_l12a=110.0)
