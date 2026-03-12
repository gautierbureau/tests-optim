"""
PST DC-OPF: Theta vs PTDF formulation comparison
=================================================

Topology (bypass PST, 3 substations)
--------------------------------------
         ┌──── L12a (B1→B2b, bypass) ────────────────┐
G1 ── B1─┤                                            ├── B2b ── L2b_3 ── B3 ── G2/Load
         └──── L1_2a (B1→B2a) ── B2a─[PST φ]────────┘
                                      S2: VL2a→VL2b

Network data
------------
  B_L12A = 500 MW/rad, B_L1_2A = 250 MW/rad
  B_PST  = 1000 MW/rad, B_L2B_3 = 500 MW/rad
  G1: 0-300 MW @30$/MWh (slack), G2: 0-150 MW @45$/MWh
  Load: 250 MW at B3, Pmax_L12a = 110 MW
"""

import math
import pandas as pd
import pypowsybl.network as pn
import pypowsybl.loadflow as plf
import pypowsybl.sensitivity as psa
import pypowsybl as ppw
import pyoptinterface as poi
from pyoptinterface import xpress

# ── Constants ─────────────────────────────────────────────────────────────────
BASE_MVA  = 100.0
KV        = 400.0
Z_BASE      = KV**2 / BASE_MVA
B_L12A    = 500.0   # MW/rad
B_L1_2A   = 250.0
B_PST     = 1000.0
B_L2B_3   = 500.0
PMAX_L12A = 110.0
# PMAX_L12A = 130.0
# PMAX_L12A = 250.0
# PMAX_L12A = 80.0
PHI_MAX   = math.radians(30.0)
C_PST     = 5
PD_B3     = 250.0
BRANCHES  = ["L12a", "L1_2a", "PST_T", "L2b_3"]


# ── Network ───────────────────────────────────────────────────────────────────
def build_network() -> pn.Network:
    net = pn.create_empty("pst_compare")
    net.create_substations(id=["S1","S2","S3"], country=["FR","FR","FR"])
    net.create_voltage_levels(
        id            = ["VL1",        "VL2a",       "VL2b",       "VL3"],
        substation_id = ["S1",         "S2",         "S2",         "S3"],
        topology_kind = ["BUS_BREAKER","BUS_BREAKER","BUS_BREAKER","BUS_BREAKER"],
        nominal_v     = [KV, KV, KV, KV],
    )
    net.create_buses(
        id               = ["B1",  "B2a", "B2b", "B3"],
        voltage_level_id = ["VL1", "VL2a","VL2b","VL3"],
    )
    net.create_lines(
        id=["L12a","L1_2a","L2b_3"],
        voltage_level1_id=["VL1",  "VL1",  "VL2b"],
        bus1_id          =["B1",   "B1",   "B2b" ],
        voltage_level2_id=["VL2b", "VL2a", "VL3" ],
        bus2_id          =["B2b",  "B2a",  "B3"  ],
        r=[0.0]*3, x=[BASE_MVA/B_L12A * Z_BASE, BASE_MVA/B_L1_2A * Z_BASE, BASE_MVA/B_L2B_3 * Z_BASE],
        g1=[0.0]*3, b1=[0.0]*3, g2=[0.0]*3, b2=[0.0]*3,
    )
    net.create_2_windings_transformers(
        id=["PST_T"],
        voltage_level1_id=["VL2a"], bus1_id=["B2a"],
        voltage_level2_id=["VL2b"], bus2_id=["B2b"],
        rated_u1=[KV], rated_u2=[KV],
        r=[0.0], x=[BASE_MVA/B_PST * Z_BASE ], g=[0.0], b=[0.0],
    )
    # PST tap changer: 21 steps, -30° to +30°, neutral tap=10
    ptc_df = pd.DataFrame.from_records(
        index="id", columns=["id","target_deadband","regulation_mode","low_tap","tap", "regulating"],
        data=[("PST_T", 0.0, "ACTIVE_POWER_CONTROL", 0, 10, False)],
    )
    steps_df = pd.DataFrame.from_records(
        index="id", columns=["id","b","g","r","x","rho","alpha"],
        data=[("PST_T", 0.0, 0.0, 0.0, 0, 1.0, (i-10)*3.0) for i in range(21)],
    )
    net.create_phase_tap_changers(ptc_df, steps_df)
    net.create_generators(
        id=["G1","G2"], voltage_level_id=["VL1","VL3"], bus_id=["B1","B3"],
        energy_source=["OTHER","OTHER"], min_p=[0.0,0.0], max_p=[300.0,150.0],
        target_p=[200.0,50.0], target_q=[0,0], target_v=[KV,KV], voltage_regulator_on=[True,False],
    )
    net.create_loads(
        id=["D3"], voltage_level_id=["VL3"], bus_id=["B3"], p0=[PD_B3], q0=[0.0],
    )

    net.save("pst1.xiidm")
    return net


# ── Sensitivities (pypowsybl) ─────────────────────────────────────────────────
def get_sensitivities(net: pn.Network) -> dict:
    sa = psa.create_dc_analysis()
    sa.add_branch_flow_factor_matrix(BRANCHES, ["G1", "G2", "D3"],   "PTDF")
    sa.add_branch_flow_factor_matrix(BRANCHES, ["PST_T"],"PSDF")
    res = sa.run(net, parameters=plf.Parameters(distributed_slack=True))
    #res = sa.run(net)
    ptdf = { gen: {br: float(res.get_sensitivity_matrix("PTDF").loc[gen,   br]) for br in BRANCHES} for gen in ["G1", "G2", "D3"]}
    psdf = {br: float(res.get_sensitivity_matrix("PSDF").loc["PST_T", br]) * (180 / math.pi) for br in BRANCHES}
    return {"ptdf": ptdf, "psdf": psdf}


# ── Theta formulation ─────────────────────────────────────────────────────────
def solve_theta() -> dict:
    """
    Variables: Pg_G1, Pg_G2, θ_B2a, θ_B2b, θ_B3, P_L12a, P_L1_2a, P_PST, P_L2b3, φ, ψ
    Flow equations:
      P_L12a  = B_L12A  · (0 − θ_B2b)
      P_L1_2a = B_L1_2A · (0 − θ_B2a)
      P_PST   = B_PST   · (θ_B2a − θ_B2b + φ)
      P_L2b3  = B_L2B_3 · (θ_B2b − θ_B3)
    """
    m = xpress.Model()
    m.set_model_attribute(poi.ModelAttribute.Silent, True)

    Pg1 = m.add_variable(lb=0,   ub=300,  name="Pg1")
    Pg2 = m.add_variable(lb=0,   ub=150,  name="Pg2")
    t2a = m.add_variable(lb=-1,  ub=1,    name="t_B2a")
    t2b = m.add_variable(lb=-1,  ub=1,    name="t_B2b")
    t3  = m.add_variable(lb=-1,  ub=1,    name="t_B3")
    Pla = m.add_variable(lb=-PMAX_L12A,  ub=PMAX_L12A,  name="P_L12a")
    # Pli = m.add_variable(lb=-250, ub=250, name="P_L1_2a")
    # Pp  = m.add_variable(lb=-250, ub=250, name="P_PST")
    # Plo = m.add_variable(lb=-300, ub=300, name="P_L2b3")
    Pli = m.add_variable(lb=-150, ub=150, name="P_L1_2a")
    Pp  = m.add_variable(lb=-150, ub=150, name="P_PST")
    Plo = m.add_variable(lb=-200, ub=200, name="P_L2b3")
    phi = m.add_variable(lb=-PHI_MAX, ub=PHI_MAX, name="phi")
    psi = m.add_variable(lb=0, ub=PHI_MAX, name="psi")

    # Flow definitions
    m.add_linear_constraint(Pla + B_L12A  * t2b,                          poi.Eq, 0, name="fl_L12a")
    m.add_linear_constraint(Pli + B_L1_2A * t2a,                          poi.Eq, 0, name="fl_L1_2a")
    m.add_linear_constraint(Pp  - B_PST*t2a + B_PST*t2b - B_PST*phi,     poi.Eq, 0, name="fl_PST")
    m.add_linear_constraint(Plo - B_L2B_3*t2b + B_L2B_3*t3,              poi.Eq, 0, name="fl_L2b3")

    # Balances
    m.add_linear_constraint(Pg1 - Pla - Pli,  poi.Eq, 0,     name="bal_B1")
    m.add_linear_constraint(Pli - Pp,          poi.Eq, 0,     name="bal_B2a")
    m.add_linear_constraint(Pla + Pp - Plo,    poi.Eq, 0,     name="bal_B2b")
    m.add_linear_constraint(Pg2 + Plo,         poi.Eq, PD_B3, name="bal_B3")

    # |φ|
    m.add_linear_constraint(psi - phi, poi.Geq, 0)
    m.add_linear_constraint(psi + phi, poi.Geq, 0)

    obj = poi.ExprBuilder()
    obj += 30*Pg1 + 45*Pg2 + C_PST*psi
    m.set_objective(obj, poi.ObjectiveSense.Minimize)
    m.optimize()

    def v(x): return m.get_value(x)
    return {
        "form"   : "theta",
        "Pg1"    : v(Pg1), "Pg2"    : v(Pg2),
        "P_L12a" : v(Pla), "P_L1_2a": v(Pli),
        "P_PST"  : v(Pp),  "P_L2b3" : v(Plo),
        "phi_deg": math.degrees(v(phi)),
        "cost"   : m.get_model_attribute(poi.ModelAttribute.ObjectiveValue),
    }


# ── PTDF formulation ──────────────────────────────────────────────────────────
def solve_ptdf(sens: dict) -> dict:
    """
    Variables: Pg_G1, Pg_G2, P_L12a, P_L1_2a, P_PST, P_L2b3, φ, ψ
    No angle variables. Flow defined by:
      P_l = PTDF(l,G2)·(Pg_G2 − Pd_B3) + PSDF(l)·φ
    """
    ptdf = sens["ptdf"]
    psdf = sens["psdf"]
    #pmax = {"L12a": PMAX_L12A, "L1_2a": 250, "PST_T": 250, "L2b_3": 300}
    pmax = {'L12a': PMAX_L12A, 'L1_2a': 150.0, 'PST_T': 150.0, 'L2b_3': 200.0}

    m = xpress.Model()
    m.set_model_attribute(poi.ModelAttribute.Silent, True)

    Pg1 = m.add_variable(lb=0, ub=300, name="Pg1")
    Pg2 = m.add_variable(lb=0, ub=150, name="Pg2")
    P   = {br: m.add_variable(lb=-pmax[br], ub=pmax[br], name=f"P_{br}") for br in BRANCHES}
    phi = m.add_variable(lb=-PHI_MAX, ub=PHI_MAX, name="phi")
    psi = m.add_variable(lb=0, ub=PHI_MAX, name="psi")

    # PTDF constraints: P_l - PTDF·Pg2 - PSDF·φ = PTDF·(-Pd_B3)
    for br in BRANCHES:
        rhs = ptdf["D3"][br] * (-PD_B3)
        m.add_linear_constraint(
            P[br] - ptdf["G1"][br]*Pg1 - ptdf["G2"][br]*Pg2 - psdf[br]*phi,
            poi.Eq, rhs, name=f"ptdf_{br}",
        )

    # Balances (only B1 and B3 needed — transit buses implicit in PTDF)
    m.add_linear_constraint(Pg1 + Pg2, poi.Eq, PD_B3, name="global_balance")

    m.add_linear_constraint(psi - phi, poi.Geq, 0)
    m.add_linear_constraint(psi + phi, poi.Geq, 0)

    obj = poi.ExprBuilder()
    obj += 30*Pg1 + 45*Pg2 + C_PST*psi
    m.set_objective(obj, poi.ObjectiveSense.Minimize)
    m.optimize()

    def v(x): return m.get_value(x)
    return {
        "form"   : "ptdf",
        "Pg1"    : v(Pg1), "Pg2"    : v(Pg2),
        "P_L12a" : v(P["L12a"]),  "P_L1_2a": v(P["L1_2a"]),
        "P_PST"  : v(P["PST_T"]), "P_L2b3" : v(P["L2b_3"]),
        "phi_deg": math.degrees(v(phi)),
        "cost"   : m.get_model_attribute(poi.ModelAttribute.ObjectiveValue),
    }


# ── Print comparison ──────────────────────────────────────────────────────────
def compare(r_th: dict, r_ptdf: dict) -> None:
    keys = ["Pg1","Pg2","P_L12a","P_L1_2a","P_PST","P_L2b3","phi_deg","cost"]
    print(f"\n{'─'*52}")
    print(f"  {'Variable':<12}  {'Theta':>10}  {'PTDF':>10}  {'Δ':>8}")
    print(f"{'─'*52}")
    for k in keys:
        th, pt = r_th[k], r_ptdf[k]
        print(f"  {k:<12}  {th:>10.4f}  {pt:>10.4f}  {abs(th-pt):>8.1e}")
    print(f"{'─'*52}")


# ── Validate both against pypowsybl DC LF ─────────────────────────────────────
def validate(r: dict, net: pn.Network) -> None:
    tap = max(0, min(20, 10 + round(r["phi_deg"] / 3.0)))
    net.update_generators(id="G1", target_p=r["Pg1"])
    net.update_generators(id="G2", target_p=r["Pg2"])
    net.update_phase_tap_changers(id="PST_T", tap=tap)

    lf = ppw.loadflow.run_dc(net, parameters=plf.Parameters(distributed_slack=False))
    lines  = net.get_lines(all_attributes=True)
    trafos = net.get_2_windings_transformers(all_attributes=True)
    gens = net.get_generators(all_attributes=True)
    taps = net.get_phase_tap_changers(all_attributes=True)

    lf_flows = {
        "L12a" : lines.loc["L12a",  "p1"],
        "L1_2a": lines.loc["L1_2a", "p1"],
        "P_PST": trafos.loc["PST_T","p1"],
        "L2b3" : lines.loc["L2b_3", "p1"],
    }
    print(f"\n  pypowsybl LF ({r['form']}, tap={tap}, α={(tap-10)*3:+.0f}°):")
    print(f"  L12a={lf_flows['L12a']:+.2f}  L1_2a={lf_flows['L1_2a']:+.2f}  "
          f"PST={lf_flows['P_PST']:+.2f}  L2b3={lf_flows['L2b3']:+.2f}  "
          f"status={lf[0].status}")
    print(f"Gen1={gens.loc['G1', 'p']:+.2f}  Gen2={gens.loc['G2', 'p']:+.2f} ")
    print(f"tap changer={taps.loc['PST_T', 'tap']:+.0f}  ")
    print(f"x={trafos.loc['PST_T', 'x_at_current_tap']:.2f}  ")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    net  = build_network()
    sens = get_sensitivities(net)

    print("PTDF coefficients (G1=slack, distributed_slack=False):")
    print(f"  {'Branch':<10} {'PTDF_G1':>10}  {'PTDF_G2':>10}  {'PSDF (MW/rad)':>14}")
    for br in BRANCHES:
        print(f"  {br:<10} {sens['ptdf']['G1'][br]:>+10.6f} {sens['ptdf']['G2'][br]:>+10.6f}  {sens['psdf'][br]:>+14.4f}")

    r_th   = solve_theta()
    r_ptdf = solve_ptdf(sens)

    compare(r_th, r_ptdf)
    validate(r_th,   build_network())
    validate(r_ptdf, build_network())
